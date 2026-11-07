"""
delta_bot.client
~~~~~~~~~~~~~~~~

High-level asynchronous REST API client for Delta Exchange.
Orchestrates authentication, clock synchronization, contract specification caching,
and market data endpoints for perpetual futures (BTC, ETH, SOL, XRP).
"""

import asyncio
from email.utils import parsedate_to_datetime
import logging
import re
import time
from typing import Any

import httpx
from pydantic import SecretStr

from delta_bot.auth import DeltaAuthEngine
from delta_bot.config import BotConfig
from delta_bot.exceptions import DeltaAPIError
from delta_bot.http_client import AsyncHttpEngine
from delta_bot.models import (
    Candle,
    ContractSpec,
    ExchangeOrder,
    PositionInfo,
    RateLimitState,
    ServerTimeInfo,
    Ticker,
    WalletBalance,
)

logger = logging.getLogger("delta_bot.client")


class DeltaRestClient:
    """
    High-performance Delta Exchange REST client.
    Thread-safe and async-safe interface for trading algorithms.
    """

    TARGET_BASE_SYMBOLS = ("BTC", "ETH", "SOL", "XRP", "XAUT")

    def __init__(self, config: BotConfig | None = None) -> None:
        self._config: BotConfig = config or BotConfig()
        self._auth_engine: DeltaAuthEngine = DeltaAuthEngine(
            api_key=self._config.delta_api_key,
            api_secret=self._config.delta_api_secret,
            user_agent=self._config.user_agent,
        )
        self._http_engine: AsyncHttpEngine = AsyncHttpEngine(
            config=self._config,
            auth_engine=self._auth_engine,
        )

        # In-memory contract specifications cache
        self._products_by_id: dict[int, ContractSpec] = {}
        self._products_by_symbol: dict[str, ContractSpec] = {}
        self._last_products_cache_epoch: float = 0.0

    @property
    def config(self) -> BotConfig:
        """Active client configuration."""
        return self._config

    @property
    def auth_engine(self) -> DeltaAuthEngine:
        """Underlying HMAC signature and clock offset engine."""
        return self._auth_engine

    @property
    def http_engine(self) -> AsyncHttpEngine:
        """Underlying async HTTP engine with connection pooling and retry mechanics."""
        return self._http_engine

    @property
    def rate_limit_state(self) -> RateLimitState:
        """Current rate limit quota status."""
        return self._http_engine.rate_limit_state

    def update_credentials(
        self,
        api_key: SecretStr | str,
        api_secret: SecretStr | str,
        base_url: str | None = None,
    ) -> None:
        """Updates API credentials and base URL for REST requests at runtime."""
        k = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
        s = api_secret if isinstance(api_secret, SecretStr) else SecretStr(api_secret)
        self._config.delta_api_key = k
        self._config.delta_api_secret = s
        if base_url:
            clean_url = str(base_url).strip().rstrip("/")
            self._config.delta_base_url = clean_url
            self._http_engine._base_url = clean_url
            self._http_engine._client = None
        self._auth_engine.update_credentials(k, s)

    async def get_server_time(self) -> ServerTimeInfo:
        """
        Fetches current server timestamp from Delta Exchange and calculates local clock offset.
        Essential for staying within the 5-second HMAC authentication validity window.
        """
        client = await self._http_engine._get_client()
        t_start_mono = time.perf_counter()
        t_send = time.time()

        # Delta's /health endpoint returns lightweight HTTP 200 with server Request-In-Time header
        resp = await client.get("/health")
        t_recv = time.time()
        t_end_mono = time.perf_counter()
        rtt_ms = max(0.0, (t_end_mono - t_start_mono) * 1000.0)

        headers = resp.headers
        server_epoch_s: float | None = None

        # Check Request-In-Time header (epoch microseconds)
        req_in_time_str = headers.get("request-in-time") or headers.get("Request-In-Time")
        if req_in_time_str:
            try:
                server_epoch_s = float(req_in_time_str) / 1_000_000.0
            except (ValueError, TypeError):
                server_epoch_s = None

        # Fallback to HTTP Date header
        if server_epoch_s is None:
            date_str = headers.get("date") or headers.get("Date")
            if date_str:
                try:
                    dt = parsedate_to_datetime(date_str)
                    server_epoch_s = dt.timestamp()
                except Exception:
                    server_epoch_s = None

        # If headers failed, use local reception time as fallback baseline
        if server_epoch_s is None:
            server_epoch_s = t_recv

        # Midpoint estimation accounts for symmetrical network transit time:
        # estimated_local_time_at_server_receive = t_send + (rtt / 2)
        estimated_local_at_server = t_send + ((t_recv - t_send) / 2.0)
        clock_offset_s = server_epoch_s - estimated_local_at_server

        # Update auth engine offset
        self._auth_engine.update_clock_offset(server_epoch_s, estimated_local_at_server)

        time_info = ServerTimeInfo(
            server_epoch_s=server_epoch_s,
            local_epoch_s=t_recv,
            clock_offset_s=clock_offset_s,
            round_trip_latency_ms=round(rtt_ms, 2),
        )

        logger.info(
            "Clock sync complete: server_epoch=%.4f, offset=%.4fs, RTT=%.2fms (Acceptable drift: %s)",
            time_info.server_epoch_s,
            time_info.clock_offset_s,
            time_info.round_trip_latency_ms,
            time_info.is_drift_acceptable,
        )
        return time_info

    async def get_wallet_balances(self) -> list[WalletBalance]:
        """
        Fetches user wallet balance and available trading margins across all collateral assets.
        Endpoint: GET /v2/wallet/balances (Authenticated)
        """
        logger.debug("Requesting wallet balances from /v2/wallet/balances...")
        result = await self._http_engine.request(
            method="GET",
            path="/v2/wallet/balances",
            is_authenticated=True,
        )

        balances: list[WalletBalance] = []
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    balances.append(WalletBalance.from_api_dict(item))
        elif isinstance(result, dict):
            balances.append(WalletBalance.from_api_dict(result))

        logger.info("Retrieved %d asset balances from Delta Exchange.", len(balances))
        return balances

    async def get_wallet_balance_by_asset(self, asset_symbol: str = "USDT") -> WalletBalance | None:
        """
        Convenience method to retrieve wallet balance for a specific settling collateral asset (default: USDT).
        """
        balances = await self.get_wallet_balances()
        target = asset_symbol.upper()
        for b in balances:
            if b.asset_symbol.upper() == target:
                return b
        return None

    async def get_products(
        self,
        contract_types: list[str] | None = None,
        force_refresh: bool = False,
    ) -> list[ContractSpec]:
        """
        Fetches product metadata from /v2/products and parses/caches contract specifications:
        - Product ID, Symbol (e.g. BTCUSD, ETHUSD, SOLUSD, XRPUSD, XAUTUSD)
        - Tick size (minimum price step)
        - Lot size / contract size
        - Maker and taker fee rates

        Results are automatically indexed in-memory by both product_id and symbol.
        """
        contract_types = contract_types or ["perpetual_futures"]

        # Cache check: return cached specs if available and not forced
        if not force_refresh and self._products_by_symbol:
            cached_specs = [
                s for s in self._products_by_symbol.values() if s.contract_type in contract_types
            ]
            if cached_specs:
                logger.debug(
                    "Serving %d products from local in-memory specification cache.",
                    len(cached_specs),
                )
                return cached_specs

        query_params: dict[str, Any] = {}
        if contract_types:
            query_params["contract_types"] = ",".join(contract_types)

        logger.debug("Fetching products from /v2/products (params=%s)...", query_params)
        raw_result = await self._http_engine.request(
            method="GET",
            path="/v2/products",
            query=query_params,
            is_authenticated=False,
        )

        specs: list[ContractSpec] = []
        if isinstance(raw_result, list):
            for item in raw_result:
                if isinstance(item, dict):
                    spec = ContractSpec.from_api_dict(item)
                    specs.append(spec)
                    self._register_product_spec(spec)

        self._last_products_cache_epoch = time.time()
        logger.info(
            "Cached %d products from Delta Exchange (Contract types: %s).",
            len(specs),
            contract_types,
        )
        return specs

    def _register_product_spec(self, spec: ContractSpec) -> None:
        """Registers a contract specification in memory with automatic USD/USDT alias indexing."""
        self._products_by_id[spec.product_id] = spec
        sym = spec.symbol.strip().upper()
        self._products_by_symbol[sym] = spec

        # Cross-register USD <-> USDT aliases (e.g. BTCUSD <-> BTCUSDT, XAUTUSD <-> XAUTUSDT)
        if sym.endswith("USD") and not sym.endswith("USDT"):
            alias = sym[:-3] + "USDT"
            if alias not in self._products_by_symbol:
                self._products_by_symbol[alias] = spec
        elif sym.endswith("USDT"):
            alias = sym[:-4] + "USD"
            if alias not in self._products_by_symbol:
                self._products_by_symbol[alias] = spec

    def get_cached_contract_spec(self, symbol_or_id: str | int) -> ContractSpec | None:
        """
        Synchronously retrieves a contract specification from the in-memory cache without network latency.
        Accepts integer product_id or string symbol (e.g., 139, 'BTCUSDT', 'BTCUSD', 'XAUTUSD').
        """
        if isinstance(symbol_or_id, int):
            return self._products_by_id.get(symbol_or_id)

        clean_symbol = str(symbol_or_id).strip().upper()
        if clean_symbol.isdigit():
            return self._products_by_id.get(int(clean_symbol))
        
        spec = self._products_by_symbol.get(clean_symbol)
        if spec is not None:
            return spec

        # Suffix alias fallback
        if clean_symbol.endswith("USD") and not clean_symbol.endswith("USDT"):
            alt = clean_symbol[:-3] + "USDT"
            if alt in self._products_by_symbol:
                return self._products_by_symbol[alt]
        elif clean_symbol.endswith("USDT"):
            alt = clean_symbol[:-4] + "USD"
            if alt in self._products_by_symbol:
                return self._products_by_symbol[alt]

        return None

    async def get_contract_spec(self, symbol_or_id: str | int) -> ContractSpec:
        """
        Retrieves a contract specification by symbol or product_id, fetching
        products from REST API if not already cached in memory.
        Includes automatic cross-regional fallback (India <-> Global) if symbol
        is exclusive to a particular regional exchange deployment (e.g. XAUTUSD on Delta India).
        """
        cached = self.get_cached_contract_spec(symbol_or_id)
        if cached is not None:
            return cached
        await self.get_products(contract_types=["perpetual_futures"])
        cached = self.get_cached_contract_spec(symbol_or_id)
        if cached is not None:
            return cached

        # Multi-region fallback: If running on Global Delta and symbol (e.g. XAUTUSD) is not found,
        # or running on India Delta and symbol is not found, check the alternate Delta regional endpoint.
        current_base = getattr(self._config, "delta_base_url", "")
        fallback_urls = []
        if "india" in current_base.lower():
            fallback_urls.append("https://api.delta.exchange")
        else:
            fallback_urls.append("https://api.india.delta.exchange")

        for fb_url in fallback_urls:
            try:
                logger.info(
                    "Attempting cross-regional product specification fallback for '%s' via %s...",
                    symbol_or_id,
                    fb_url,
                )
                async with httpx.AsyncClient(timeout=10.0) as fb_client:
                    fb_resp = await fb_client.get(f"{fb_url}/v2/products?contract_types=perpetual_futures")
                    if fb_resp.status_code == 200:
                        data = fb_resp.json().get("result", [])
                        for item in data:
                            if isinstance(item, dict):
                                fb_spec = ContractSpec.from_api_dict(item)
                                self._register_product_spec(fb_spec)
                        cached = self.get_cached_contract_spec(symbol_or_id)
                        if cached is not None:
                            logger.info(
                                "Successfully resolved '%s' via cross-region fallback (%s): ProductID=%d, Symbol=%s",
                                symbol_or_id,
                                fb_url,
                                cached.product_id,
                                cached.symbol,
                            )
                            return cached
            except Exception as fb_err:
                logger.warning("Cross-regional fallback check on %s failed: %s", fb_url, fb_err)

        raise DeltaAPIError(f"Contract specification for '{symbol_or_id}' not found on Delta Exchange.")

    async def get_target_perpetuals(self) -> dict[str, ContractSpec]:
        """
        Fetches or looks up perpetual futures specifications specifically for the core target assets:
        BTC, ETH, SOL, and XRP perpetuals.
        Returns a mapping of base symbol (e.g. 'BTC') to ContractSpec.
        """
        if not self._products_by_symbol:
            await self.get_products(contract_types=["perpetual_futures"])

        target_map: dict[str, ContractSpec] = {}
        for base_symbol in self.TARGET_BASE_SYMBOLS:
            # Look for matching symbol like BTCUSDT, BTCUSD, etc.
            pattern = re.compile(rf"^{base_symbol}(USDT|USD)$", re.IGNORECASE)
            for sym, spec in self._products_by_symbol.items():
                if spec.contract_type == "perpetual_futures" and pattern.match(sym):
                    target_map[base_symbol] = spec
                    break

        return target_map

    async def get_ticker(self, symbol_or_product_id: str | int) -> Ticker:
        """
        Fetches real-time mark price, best bid/ask, and open interest for a specific instrument.
        Endpoint: GET /v2/tickers/{symbol_or_product_id} (Public)
        """
        clean_target = str(symbol_or_product_id).strip().upper()
        cached_spec = self.get_cached_contract_spec(clean_target)
        target = cached_spec.symbol if cached_spec else clean_target
        path = f"/v2/tickers/{target}"

        logger.debug("Fetching real-time ticker quotes for %s...", target)
        try:
            raw_result = await self._http_engine.request(
                method="GET",
                path=path,
                is_authenticated=False,
            )
        except Exception as e:
            if target != clean_target:
                raw_result = await self._http_engine.request(
                    method="GET",
                    path=f"/v2/tickers/{clean_target}",
                    is_authenticated=False,
                )
            else:
                raise e

        if not isinstance(raw_result, dict):
            raise DeltaAPIError(
                message=f"Unexpected ticker response format for {target}: {type(raw_result).__name__}",
                endpoint=path,
            )

        ticker = Ticker.from_api_dict(raw_result)
        logger.debug(
            "Ticker %s: Mark=%.2f, Bid=%s, Ask=%s, OI=%s",
            ticker.symbol,
            ticker.mark_price,
            ticker.best_bid,
            ticker.best_ask,
            ticker.open_interest,
        )
        return ticker

    async def get_all_tickers(self, contract_types: list[str] | None = None) -> list[Ticker]:
        """
        Fetches real-time ticker information across all active contracts.
        Endpoint: GET /v2/tickers (Public)
        """
        query_params: dict[str, Any] = {}
        if contract_types:
            query_params["contract_types"] = ",".join(contract_types)

        raw_result = await self._http_engine.request(
            method="GET",
            path="/v2/tickers",
            query=query_params,
            is_authenticated=False,
        )

        tickers: list[Ticker] = []
        if isinstance(raw_result, list):
            for item in raw_result:
                if isinstance(item, dict):
                    tickers.append(Ticker.from_api_dict(item))

        return tickers

    async def get_historical_candles(
        self,
        symbol: str,
        resolution: str = "1m",
        limit: int = 500,
        start: int | None = None,
        end: int | None = None,
    ) -> list[Candle]:
        """
        Fetches historical closed candlestick bars from Delta REST API: GET /v2/history/candles.
        Used by the Cold-Start Warm-up Routine to prime the 500-bar RingBuffer on minute zero.
        Returns a chronologically sorted (oldest to newest) list of confirmed Candle instances.
        """
        resolution_seconds_map = {
            "1m": 60,
            "3m": 180,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "2h": 7200,
            "4h": 14400,
            "6h": 21600,
            "12h": 43200,
            "1d": 86400,
        }
        bar_sec = resolution_seconds_map.get(resolution, 60)
        end_sec = int(end if end is not None else time.time())
        # Delta Exchange candle history requires a minimum window (>= 3-4 hours) to return data
        window_sec = max(limit * bar_sec * 3, 86400)
        start_sec = int(start if start is not None else (end_sec - window_sec))

        clean_symbol = symbol.strip().upper()
        cached_spec = self.get_cached_contract_spec(clean_symbol)
        query_symbol = cached_spec.symbol if cached_spec else clean_symbol

        query_params = {
            "symbol": query_symbol,
            "resolution": resolution,
            "start": start_sec,
            "end": end_sec,
        }

        logger.debug(
            "Fetching historical candles for %s (%s) from REST (query_symbol=%s, limit=%d, start=%d, end=%d)...",
            clean_symbol,
            resolution,
            query_symbol,
            limit,
            start_sec,
            end_sec,
        )

        raw_result = await self._http_engine.request(
            method="GET",
            path="/v2/history/candles",
            query=query_params,
            is_authenticated=False,
        )

        # If empty response, attempt alternate USD/USDT alias
        if not raw_result or len(raw_result) == 0:
            alt_symbol = None
            if query_symbol.endswith("USDT"):
                alt_symbol = query_symbol[:-4] + "USD"
            elif query_symbol.endswith("USD"):
                alt_symbol = query_symbol[:-3] + "USDT"

            if alt_symbol and alt_symbol != query_symbol:
                logger.debug("Retrying historical candles with alternative symbol %s...", alt_symbol)
                alt_params = dict(query_params)
                alt_params["symbol"] = alt_symbol
                try:
                    alt_result = await self._http_engine.request(
                        method="GET",
                        path="/v2/history/candles",
                        query=alt_params,
                        is_authenticated=False,
                    )
                    if alt_result and len(alt_result) > 0:
                        raw_result = alt_result
                except Exception:
                    pass

        candles: list[Candle] = []
        if isinstance(raw_result, list):
            for item in raw_result:
                if isinstance(item, dict):
                    t_sec = int(item.get("time", 0))
                    c_usec = t_sec * 1_000_000
                    c = Candle(
                        symbol=clean_symbol,
                        resolution=resolution,
                        open=float(item.get("open", 0.0)),
                        high=float(item.get("high", 0.0)),
                        low=float(item.get("low", 0.0)),
                        close=float(item.get("close", 0.0)),
                        volume=float(item.get("volume", 0.0)),
                        candle_start_time_usec=c_usec,
                        is_closed=True,
                        timestamp_usec=c_usec,
                    )
                    candles.append(c)

        # Delta returns candles latest-first; sort ascending (chronological order)
        candles.sort(key=lambda x: x.candle_start_time_usec)

        # Enforce limit if API returned more
        if len(candles) > limit:
            candles = candles[-limit:]

        logger.info(
            "Retrieved %d historical %s candles for %s from REST.",
            len(candles),
            resolution,
            clean_symbol,
        )
        return candles

    # -------------------------------------------------------------------------
    # Order Execution & Position Management (Phase 4)
    # -------------------------------------------------------------------------

    async def set_order_leverage(self, product_id: int, leverage: int | float) -> dict[str, Any]:
        """
        Sets order leverage for a specific contract product on Delta Exchange.
        Primary Endpoint: POST /v2/products/{product_id}/orders/leverage
        Fallback Endpoint: POST /v2/orders/leverage
        Guarantees that new stop and limit orders adhere to the desired leverage tier (1x - 100x).
        """
        lev_str = str(int(leverage))
        payload = {"leverage": lev_str, "product_id": int(product_id)}

        logger.info("Setting exchange order leverage to %sx for product_id %d...", lev_str, product_id)
        try:
            raw_result = await self._http_engine.request(
                method="POST",
                path=f"/v2/products/{product_id}/orders/leverage",
                payload=payload,
                is_authenticated=True,
            )
        except Exception as exc:
            # Fallback to alternate global path if product-scoped URL is not accepted
            if "404" in str(exc) or "not_found" in str(exc).lower():
                logger.debug("Falling back to /v2/orders/leverage for product_id %d", product_id)
                raw_result = await self._http_engine.request(
                    method="POST",
                    path="/v2/orders/leverage",
                    payload=payload,
                    is_authenticated=True,
                )
            else:
                raise

        if not isinstance(raw_result, dict):
            return {"product_id": product_id, "leverage": int(leverage), "status": "success"}
        return raw_result

    async def place_order(
        self,
        product_id: int,
        size: int,
        side: str,
        order_type: str = "market_order",
        limit_price: float | None = None,
        stop_price: float | None = None,
        stop_order_type: str | None = None,
        stop_trigger_method: str = "mark_price",
        bracket_stop_loss_price: float | None = None,
        bracket_take_profit_price: float | None = None,
        bracket_stop_loss_limit_price: float | None = None,
        bracket_take_profit_limit_price: float | None = None,
        time_in_force: str = "gtc",
        reduce_only: bool = False,
        client_order_id: str | None = None,
        symbol: str = "",
    ) -> ExchangeOrder:
        """
        Submits an order to Delta Exchange.
        Endpoint: POST /v2/orders (Authenticated)

        Supports Exchange-Native Stop Orders with attached brackets:
        - stop_price + stop_order_type ("stop_loss_order")
        - bracket_stop_loss_price
        - bracket_take_profit_price
        """
        payload: dict[str, Any] = {
            "product_id": int(product_id),
            "size": int(size),
            "side": str(side).lower(),
            "order_type": str(order_type).lower(),
            "time_in_force": str(time_in_force).lower(),
        }

        if reduce_only:
            payload["reduce_only"] = True

        if limit_price is not None:
            payload["limit_price"] = str(limit_price)
        if stop_price is not None:
            payload["stop_price"] = str(stop_price)
            payload["stop_order_type"] = stop_order_type or "stop_loss_order"
            payload["stop_trigger_method"] = stop_trigger_method
        if bracket_stop_loss_price is not None:
            payload["bracket_stop_loss_price"] = str(bracket_stop_loss_price)
        if bracket_take_profit_price is not None:
            payload["bracket_take_profit_price"] = str(bracket_take_profit_price)
        if bracket_stop_loss_limit_price is not None:
            payload["bracket_stop_loss_limit_price"] = str(bracket_stop_loss_limit_price)
        if bracket_take_profit_limit_price is not None:
            payload["bracket_take_profit_limit_price"] = str(bracket_take_profit_limit_price)
        if client_order_id:
            payload["client_order_id"] = str(client_order_id)

        logger.info(
            "Submitting exchange order: product_id=%d side=%s size=%d type=%s stop_price=%s",
            product_id,
            side,
            size,
            order_type,
            stop_price,
        )

        raw_result = await self._http_engine.request(
            method="POST",
            path="/v2/orders",
            payload=payload,
            is_authenticated=True,
        )

        if not isinstance(raw_result, dict):
            raise DeltaAPIError(
                message=f"Invalid response format from POST /v2/orders: {type(raw_result).__name__}",
                endpoint="/v2/orders",
            )

        order = ExchangeOrder.from_api_dict(raw_result, symbol=symbol)
        logger.info(
            "Order placed successfully: id=%d state=%s product_id=%d size=%d",
            order.id,
            order.state,
            order.product_id,
            order.size,
        )
        return order

    async def cancel_order(
        self,
        product_id: int,
        order_id: int | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Cancels an individual resting order on Delta Exchange.
        Endpoint: DELETE /v2/orders (Authenticated)
        Requires product_id and either id or client_order_id.
        """
        payload: dict[str, Any] = {"product_id": int(product_id)}
        if order_id is not None:
            payload["id"] = int(order_id)
        elif client_order_id is not None:
            payload["client_order_id"] = str(client_order_id)
        else:
            raise ValueError("Either order_id or client_order_id must be provided to cancel an order.")

        logger.info("Cancelling resting order on exchange: product_id=%d order_id=%s...", product_id, order_id)
        raw_result = await self._http_engine.request(
            method="DELETE",
            path="/v2/orders",
            payload=payload,
            is_authenticated=True,
        )

        if not isinstance(raw_result, dict):
            return {"status": "cancelled", "product_id": product_id, "id": order_id}
        return raw_result

    async def cancel_all_orders(self, product_id: int | None = None) -> list[dict[str, Any]]:
        """
        Cancels all open orders on Delta Exchange (optionally scoped to a product_id).
        Endpoint: DELETE /v2/orders/all (Authenticated)
        """
        payload: dict[str, Any] = {}
        if product_id is not None:
            payload["product_id"] = int(product_id)

        logger.info("Cancelling all open orders (product_id=%s)...", product_id)
        raw_result = await self._http_engine.request(
            method="DELETE",
            path="/v2/orders/all",
            payload=payload if payload else None,
            is_authenticated=True,
        )

        if isinstance(raw_result, list):
            return [item for item in raw_result if isinstance(item, dict)]
        return [{"status": "all_cancelled", "product_id": product_id}]

    async def edit_order(
        self,
        product_id: int,
        order_id: int,
        size: int | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
        bracket_stop_loss_price: float | None = None,
        bracket_take_profit_price: float | None = None,
    ) -> ExchangeOrder:
        """
        Modifies a resting order or bracket SL/TP on Delta Exchange.
        Endpoint: PUT /v2/orders (Authenticated)
        """
        payload: dict[str, Any] = {
            "product_id": int(product_id),
            "id": int(order_id),
        }
        if size is not None:
            payload["size"] = int(size)
        if limit_price is not None:
            payload["limit_price"] = str(limit_price)
        if stop_price is not None:
            payload["stop_price"] = str(stop_price)
        if bracket_stop_loss_price is not None:
            payload["bracket_stop_loss_price"] = str(bracket_stop_loss_price)
        if bracket_take_profit_price is not None:
            payload["bracket_take_profit_price"] = str(bracket_take_profit_price)

        logger.info("Editing resting order id=%d on exchange (product_id=%d)...", order_id, product_id)
        raw_result = await self._http_engine.request(
            method="PUT",
            path="/v2/orders",
            payload=payload,
            is_authenticated=True,
        )

        if not isinstance(raw_result, dict):
            raise DeltaAPIError(
                message=f"Invalid response from PUT /v2/orders: {type(raw_result).__name__}",
                endpoint="/v2/orders",
            )
        return ExchangeOrder.from_api_dict(raw_result)

    async def edit_bracket_order(
        self,
        product_id: int,
        order_id: int,
        bracket_stop_loss_price: float | None = None,
        bracket_take_profit_price: float | None = None,
    ) -> dict[str, Any]:
        """
        Modifies bracket parameters on an open/unfilled entry order.
        Endpoint: PUT /v2/orders/bracket (Authenticated)
        """
        payload: dict[str, Any] = {
            "id": int(order_id),
            "product_id": int(product_id),
        }
        if bracket_stop_loss_price is not None:
            payload["bracket_stop_loss_price"] = str(bracket_stop_loss_price)
        if bracket_take_profit_price is not None:
            payload["bracket_take_profit_price"] = str(bracket_take_profit_price)

        logger.info("Editing bracket for pending order id=%d (product_id=%d)...", order_id, product_id)
        raw_result = await self._http_engine.request(
            method="PUT",
            path="/v2/orders/bracket",
            payload=payload,
            is_authenticated=True,
        )
        if isinstance(raw_result, dict):
            return raw_result
        return {"id": order_id, "product_id": product_id, "status": "success"}

    async def set_position_bracket(
        self,
        product_id: int,
        bracket_stop_loss_price: float | None = None,
        bracket_take_profit_price: float | None = None,
    ) -> dict[str, Any]:
        """
        Attaches or modifies bracket Stop Loss and Take Profit on an active open position.
        Endpoint: POST /v2/orders/bracket (Authenticated)
        """
        payload: dict[str, Any] = {"product_id": int(product_id)}
        if bracket_stop_loss_price is not None:
            payload["bracket_stop_loss_price"] = str(bracket_stop_loss_price)
        if bracket_take_profit_price is not None:
            payload["bracket_take_profit_price"] = str(bracket_take_profit_price)

        logger.info("Setting position bracket for product_id=%d...", product_id)
        raw_result = await self._http_engine.request(
            method="POST",
            path="/v2/orders/bracket",
            payload=payload,
            is_authenticated=True,
        )
        if isinstance(raw_result, dict):
            return raw_result
        return {"product_id": product_id, "status": "success"}

    async def find_active_stop_loss_order(
        self,
        product_id: int,
        side: str,
    ) -> ExchangeOrder | None:
        """
        Locates the active resting child Stop Loss order on Delta Exchange
        associated with an open position.
        """
        open_orders = await self.get_open_orders(product_id=product_id)
        clean_side = side.lower()
        for order in open_orders:
            if order.side.lower() == clean_side and order.stop_price is not None:
                return order
        return None

    async def get_open_orders(self, product_id: int | None = None) -> list[ExchangeOrder]:
        """
        Fetches open/pending orders from Delta Exchange.
        Endpoint: GET /v2/orders?state=open
        """
        query: dict[str, Any] = {"state": "open"}
        if product_id is not None:
            query["product_id"] = str(product_id)

        raw_result = await self._http_engine.request(
            method="GET",
            path="/v2/orders",
            query=query,
            is_authenticated=True,
        )

        orders: list[ExchangeOrder] = []
        if isinstance(raw_result, list):
            for item in raw_result:
                if isinstance(item, dict):
                    orders.append(ExchangeOrder.from_api_dict(item))
        return orders

    async def get_open_positions(self, product_id: int | None = None) -> list[PositionInfo]:
        """
        Fetches active margined positions from Delta Exchange.
        Endpoint: GET /v2/positions/margined
        """
        raw_result = await self._http_engine.request(
            method="GET",
            path="/v2/positions/margined",
            is_authenticated=True,
        )

        positions: list[PositionInfo] = []
        if isinstance(raw_result, list):
            for item in raw_result:
                if isinstance(item, dict):
                    pos = PositionInfo.from_api_dict(item)
                    if product_id is None or pos.product_id == product_id:
                        positions.append(pos)
        return positions

    async def close(self) -> None:
        """Gracefully closes all open HTTP sessions and resources."""
        await self._http_engine.close()

    async def __aenter__(self) -> "DeltaRestClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
