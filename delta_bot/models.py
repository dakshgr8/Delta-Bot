"""
delta_bot.models
~~~~~~~~~~~~~~~~

Domain data structures and typed schemas for Delta Exchange trading bot.
Models represent market data, exchange time synchronization, contract specifications,
and wallet margin states.
"""

from collections import deque
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import time
from typing import Any
import uuid
from pydantic import BaseModel, ConfigDict, Field, computed_field


class RateLimitState(BaseModel):
    """Encapsulates active rate limit budget and reset timing."""

    model_config = ConfigDict(frozen=True)

    limit: int | None = Field(default=None, description="Max allowed requests in window")
    remaining: int | None = Field(default=None, description="Remaining requests in window")
    reset_ms: float | None = Field(default=None, description="Milliseconds until quota reset")
    last_updated_epoch: float = Field(default=0.0, description="Local epoch time of last update")

    @property
    def is_exhausted(self) -> bool:
        """Indicates if rate limits are completely consumed."""
        if self.remaining is not None and self.remaining <= 0:
            return True
        return False

    @property
    def utilization_pct(self) -> float:
        """Percentage of rate limit quota utilized."""
        if self.limit and self.remaining is not None and self.limit > 0:
            used = self.limit - self.remaining
            return (used / self.limit) * 100.0
        return 0.0


class ServerTimeInfo(BaseModel):
    """
    Clock synchronization telemetry comparing Delta server clock to host clock.
    Used to keep signature timestamps within the mandatory 5-second validity window.
    """

    model_config = ConfigDict(frozen=True)

    server_epoch_s: float = Field(description="Delta Exchange server time in epoch seconds")
    local_epoch_s: float = Field(description="Local system time at reception in epoch seconds")
    clock_offset_s: float = Field(
        description="Calculated offset: server_time - local_time. Positive means local clock lags server."
    )
    round_trip_latency_ms: float = Field(
        default=0.0,
        description="Round-trip network time measured during time synchronization request",
    )

    @property
    def is_drift_acceptable(self) -> bool:
        """Returns True if local clock drift is safely within 3.5s (Delta margin is 5.0s)."""
        return abs(self.clock_offset_s) < 3.5


class ContractSpec(BaseModel):
    """
    Standardized trading specification for a Delta Exchange product.
    Includes tick size, lot/contract size, and exchange fee schedules.
    """

    model_config = ConfigDict(frozen=True)

    product_id: int = Field(description="Unique product identifier")
    symbol: str = Field(description="Trading symbol, e.g. BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT")
    contract_type: str = Field(description="Contract category (e.g., perpetual_futures, spot)")
    contract_value: float = Field(
        description="Contract/lot size multiplier in underlying asset units (e.g., 0.001 BTC)"
    )
    tick_size: float = Field(description="Minimum price increment (e.g., 0.5 for BTCUSDT)")
    maker_commission_rate: float = Field(description="Fee rate applied to maker orders (e.g. 0.0001)")
    taker_commission_rate: float = Field(description="Fee rate applied to taker orders (e.g. 0.0001)")
    state: str = Field(default="live", description="Product lifecycle state: live, expired, etc.")
    trading_status: str = Field(default="operational", description="Trading status: operational, halted, etc.")
    underlying_symbol: str = Field(default="", description="Base underlying symbol (e.g. BTC)")
    quoting_symbol: str = Field(default="USDT", description="Quoting currency symbol (e.g. USDT)")
    settling_symbol: str | None = Field(default="USDT", description="Settlement asset symbol")
    description: str = Field(default="", description="Descriptive product title")

    def round_price(self, price: float) -> float:
        """
        Quantizes a target price to the product's valid tick size step using bankers/half-up rounding.
        Guarantees that submitted limit orders adhere strictly to the exchange price grid.
        """
        price_dec = Decimal(str(price))
        tick_dec = Decimal(str(self.tick_size))
        rounded_dec = (price_dec / tick_dec).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * tick_dec
        return float(rounded_dec)

    def asset_qty_to_contracts(self, asset_qty: float) -> int:
        """
        Converts fractional asset quantity (e.g. 0.005 BTC) into integer contracts (lots) for Delta order placement:
            contracts = int(round(asset_qty / contract_value))
        Delta Exchange derivatives order placement (/v2/orders) strictly expects integer contracts.
        """
        if asset_qty <= 0:
            return 0
        qty_dec = Decimal(str(asset_qty))
        val_dec = Decimal(str(self.contract_value))
        contracts_dec = (qty_dec / val_dec).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        return int(contracts_dec)

    def contracts_to_asset_qty(self, contracts: int) -> float:
        """
        Converts integer contracts back to native underlying asset units:
            asset_qty = contracts * contract_value
        """
        return float(Decimal(str(contracts)) * Decimal(str(self.contract_value)))

    def round_size(self, asset_qty: float) -> int:
        """
        Converts target native asset quantity into integer contracts (lots) expected by Delta Exchange:
            int(target_asset_qty / contract_value)
        Enforces integer contracts to prevent exchange 'invalid_order_size' rejections.
        """
        return self.asset_qty_to_contracts(asset_qty)

    @classmethod
    def from_api_dict(cls, data: dict[str, Any]) -> "ContractSpec":
        """Instantiates ContractSpec by parsing raw product metadata from /v2/products."""
        underlying = data.get("underlying_asset", {}) or {}
        quoting = data.get("quoting_asset", {}) or {}
        settling = data.get("settling_asset", {}) or {}

        return cls(
            product_id=int(data["id"]),
            symbol=str(data["symbol"]).upper(),
            contract_type=str(data.get("contract_type", "perpetual_futures")),
            contract_value=float(data.get("contract_value", 1.0)),
            tick_size=float(data.get("tick_size", 0.01)),
            maker_commission_rate=float(data.get("maker_commission_rate", 0.0)),
            taker_commission_rate=float(data.get("taker_commission_rate", 0.0)),
            state=str(data.get("state", "live")),
            trading_status=str(data.get("trading_status", "operational")),
            underlying_symbol=str(underlying.get("symbol", "")),
            quoting_symbol=str(quoting.get("symbol", "USDT")),
            settling_symbol=settling.get("symbol") if settling else None,
            description=str(data.get("description", "")),
        )


class Ticker(BaseModel):
    """
    Consolidated real-time market quotes, mark price, and open interest for a product.
    Parsed from Delta Exchange /v2/tickers.
    """

    model_config = ConfigDict(frozen=True)

    product_id: int = Field(description="Product ID")
    symbol: str = Field(description="Trading symbol")
    mark_price: float = Field(description="Current mark price used for margin & liquidations")
    best_bid: float | None = Field(default=None, description="Top-of-book best bid price")
    best_ask: float | None = Field(default=None, description="Top-of-book best ask price")
    bid_size: float | None = Field(default=None, description="Quantity available at best bid")
    ask_size: float | None = Field(default=None, description="Quantity available at best ask")
    open_interest: float = Field(default=0.0, description="Open interest expressed in underlying or contracts")
    open_interest_usd: float | None = Field(default=None, description="Open interest notional in USD")
    volume_24h: float | None = Field(default=None, description="24-hour traded volume")
    turnover_24h: float | None = Field(default=None, description="24-hour traded turnover")
    high_24h: float | None = Field(default=None, description="24-hour high price")
    low_24h: float | None = Field(default=None, description="24-hour low price")
    spot_price: float | None = Field(default=None, description="Underlying index spot price")
    funding_rate: float | None = Field(default=None, description="Current 8-hour funding rate")
    timestamp_usec: int | None = Field(default=None, description="Server snapshot timestamp in microseconds")

    @computed_field
    @property
    def spread(self) -> float | None:
        """Absolute top-of-book spread (best_ask - best_bid)."""
        if self.best_ask is not None and self.best_bid is not None:
            return round(self.best_ask - self.best_bid, 8)
        return None

    @computed_field
    @property
    def spread_bps(self) -> float | None:
        """Top-of-book spread expressed in basis points (bps) relative to mid price."""
        if self.best_ask is not None and self.best_bid is not None and self.best_bid > 0:
            mid = (self.best_ask + self.best_bid) / 2.0
            diff = self.best_ask - self.best_bid
            return round((diff / mid) * 10_000.0, 2)
        return None

    @classmethod
    def from_api_dict(cls, data: dict[str, Any]) -> "Ticker":
        """Instantiates Ticker by parsing response dictionary from /v2/tickers."""
        quotes = data.get("quotes", {}) or {}

        def _to_float(val: Any) -> float | None:
            if val is None or val == "":
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        return cls(
            product_id=int(data.get("product_id", 0)),
            symbol=str(data.get("symbol", "")),
            mark_price=float(data.get("mark_price", 0.0)),
            best_bid=_to_float(quotes.get("best_bid")),
            best_ask=_to_float(quotes.get("best_ask")),
            bid_size=_to_float(quotes.get("bid_size")),
            ask_size=_to_float(quotes.get("ask_size")),
            open_interest=_to_float(data.get("oi")) or _to_float(data.get("oi_contracts")) or 0.0,
            open_interest_usd=_to_float(data.get("oi_value_usd")),
            volume_24h=_to_float(data.get("volume")),
            turnover_24h=_to_float(data.get("turnover_usd")) or _to_float(data.get("turnover")),
            high_24h=_to_float(data.get("high")),
            low_24h=_to_float(data.get("low")),
            spot_price=_to_float(data.get("spot_price")),
            funding_rate=_to_float(data.get("funding_rate")),
            timestamp_usec=int(data["timestamp"]) if "timestamp" in data and data["timestamp"] else None,
        )


class WalletBalance(BaseModel):
    """
    Account balance and margin allocation for a specific settling collateral asset (e.g., USDT, BTC).
    Parsed from Delta Exchange /v2/wallet/balances.
    """

    model_config = ConfigDict(frozen=True)

    asset_id: int = Field(description="Unique asset identifier")
    asset_symbol: str = Field(description="Asset symbol (e.g. USDT, BTC)")
    balance: float = Field(description="Total wallet balance including locked margin")
    available_balance: float = Field(description="Free balance available for trading / initial margin")
    order_margin: float = Field(default=0.0, description="Margin locked in active pending orders")
    position_margin: float = Field(default=0.0, description="Margin allocated to active open positions")
    blocked_margin: float = Field(default=0.0, description="Aggregated blocked margin (orders + positions)")
    unrealized_pnl: float | None = Field(default=0.0, description="Real-time open position PnL")

    @computed_field
    @property
    def margin_utilization_pct(self) -> float:
        """Percentage of total equity currently consumed by margin requirements."""
        if self.balance > 0:
            return round((self.blocked_margin / self.balance) * 100.0, 2)
        return 0.0

    @classmethod
    def from_api_dict(cls, data: dict[str, Any]) -> "WalletBalance":
        """Instantiates WalletBalance from item in /v2/wallet/balances result."""
        def _to_float(val: Any) -> float:
            if val is None or val == "":
                return 0.0
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        return cls(
            asset_id=int(data.get("asset_id", 0)),
            asset_symbol=str(data.get("asset_symbol", "")),
            balance=_to_float(data.get("balance")),
            available_balance=_to_float(data.get("available_balance")),
            order_margin=_to_float(data.get("order_margin")),
            position_margin=_to_float(data.get("position_margin")),
            blocked_margin=_to_float(data.get("blocked_margin")),
            unrealized_pnl=_to_float(data.get("unrealized_pnl")),
        )


class ExecutionMode(str, Enum):
    """Trading execution mode."""

    LIVE = "LIVE"
    DRY_RUN = "DRY_RUN"
    PAPER = "PAPER"

    @classmethod
    def from_str(cls, val: Any) -> "ExecutionMode":
        """Parses execution mode string cleanly (case-insensitive)."""
        if isinstance(val, cls):
            return val
        raw = val.value if hasattr(val, "value") else str(val)
        cleaned = raw.strip().upper()
        if "." in cleaned:
            cleaned = cleaned.split(".")[-1]
        if cleaned in ("PAPER", "DRY_RUN", "DRYRUN", "SIM", "SIMULATED", "TEST"):
            return cls.PAPER
        if cleaned in ("LIVE", "PROD", "PRODUCTION"):
            return cls.LIVE
        return cls.PAPER


class ExchangeOrder(BaseModel):
    """
    Representation of an order on Delta Exchange.
    Supports native Stop orders and attached Bracket orders.
    """

    model_config = ConfigDict(frozen=True)

    id: int = Field(description="Delta Exchange internal order ID")
    client_order_id: str | None = Field(default=None, description="Client custom tracking ID")
    product_id: int = Field(description="Contract Product ID")
    symbol: str = Field(default="", description="Trading symbol (e.g. BTCUSDT)")
    side: str = Field(description="Order side: 'buy' or 'sell'")
    order_type: str = Field(description="Order type: 'limit_order', 'market_order'")
    size: int = Field(description="Order size in integer contract lots")
    unfilled_size: int = Field(default=0, description="Remaining unfilled contract lots")
    limit_price: float | None = Field(default=None, description="Limit execution price")
    stop_price: float | None = Field(default=None, description="Trigger price for stop order")
    stop_order_type: str | None = Field(default=None, description="Stop order classification")
    stop_trigger_method: str | None = Field(default="mark_price", description="Trigger reference")
    bracket_stop_loss_price: float | None = Field(default=None, description="Attached Bracket Stop Loss price")
    bracket_take_profit_price: float | None = Field(default=None, description="Attached Bracket Take Profit price")
    state: str = Field(default="open", description="State: open, pending, closed, cancelled, rejected")
    created_at: float = Field(default=0.0, description="Creation epoch timestamp in seconds")

    @property
    def is_open(self) -> bool:
        """Returns True if order is active on the exchange."""
        return self.state in ("open", "pending")

    @property
    def is_cancelled(self) -> bool:
        """Returns True if order has been cancelled."""
        return self.state in ("cancelled", "rejected")

    @classmethod
    def from_api_dict(cls, data: dict[str, Any], symbol: str = "") -> "ExchangeOrder":
        """Instantiates ExchangeOrder from raw Delta API response dictionary."""
        def _to_float(val: Any) -> float | None:
            if val is None or val == "":
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        def _to_int(val: Any, default: int = 0) -> int:
            if val is None or val == "":
                return default
            try:
                return int(val)
            except (ValueError, TypeError):
                return default

        return cls(
            id=_to_int(data.get("id")),
            client_order_id=data.get("client_order_id"),
            product_id=_to_int(data.get("product_id")),
            symbol=symbol or str(data.get("product_symbol", "")),
            side=str(data.get("side", "")).lower(),
            order_type=str(data.get("order_type", "")),
            size=_to_int(data.get("size")),
            unfilled_size=_to_int(data.get("unfilled_size")),
            limit_price=_to_float(data.get("limit_price")),
            stop_price=_to_float(data.get("stop_price")),
            stop_order_type=data.get("stop_order_type"),
            stop_trigger_method=data.get("stop_trigger_method", "mark_price"),
            bracket_stop_loss_price=_to_float(data.get("bracket_stop_loss_price")),
            bracket_take_profit_price=_to_float(data.get("bracket_take_profit_price")),
            state=str(data.get("state", "open")).lower(),
            created_at=_to_float(data.get("created_at")) or 0.0,
        )


class PositionInfo(BaseModel):
    """
    Representation of an active open position on Delta Exchange.
    Parsed from GET /v2/positions/margined or /v2/positions.
    """

    model_config = ConfigDict(frozen=True)

    product_id: int = Field(description="Contract Product ID")
    symbol: str = Field(default="", description="Trading symbol")
    size: int = Field(description="Position size in integer contract lots (positive=long, negative=short)")
    entry_price: float = Field(default=0.0, description="Average entry price")
    mark_price: float = Field(default=0.0, description="Current mark price")
    liquidation_price: float | None = Field(default=None, description="Estimated liquidation price")
    margin: float = Field(default=0.0, description="Initial margin allocated to position")
    maintenance_margin: float = Field(default=0.0, description="Maintenance margin threshold")
    realized_pnl: float = Field(default=0.0, description="Realized PnL")
    unrealized_pnl: float = Field(default=0.0, description="Unrealized floating PnL")

    @property
    def is_long(self) -> bool:
        return self.size > 0

    @property
    def is_short(self) -> bool:
        return self.size < 0

    @classmethod
    def from_api_dict(cls, data: dict[str, Any], symbol: str = "") -> "PositionInfo":
        """Instantiates PositionInfo from raw Delta positions response."""
        def _to_float(val: Any) -> float:
            if val is None or val == "":
                return 0.0
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        def _to_opt_float(val: Any) -> float | None:
            if val is None or val == "":
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        return cls(
            product_id=int(data.get("product_id", 0)),
            symbol=symbol or str(data.get("product_symbol", "")),
            size=int(data.get("size", 0)),
            entry_price=_to_float(data.get("entry_price")),
            mark_price=_to_float(data.get("mark_price")),
            liquidation_price=_to_opt_float(data.get("liquidation_price")),
            margin=_to_float(data.get("margin")),
            maintenance_margin=_to_float(data.get("maintenance_margin")),
            realized_pnl=_to_float(data.get("realized_pnl")),
            unrealized_pnl=_to_float(data.get("unrealized_pnl")),
        )


class MarginCheckResult(BaseModel):
    """
    Detailed telemetry produced by the dual-constraint margin & leverage sizing engine.
    Guarantees orders never exceed initial margin limits or trigger insufficient_margin rejections.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    product_id: int
    leverage: float
    available_margin: float
    required_margin: float
    risk_budget_usd: float
    risk_based_lots: int
    margin_based_lots: int
    final_lots: int
    is_margin_sufficient: bool
    rejection_reason: str | None = None


class Candle(BaseModel):
    """
    Standardized OHLCV candlestick bar for real-time strategy calculation.
    Enforces clear state separation between forming/open candles and confirmed closed candles.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(description="Trading symbol (e.g., BTCUSDT)")
    resolution: str = Field(default="1m", description="Bar timeframe resolution, e.g. 1m, 5m")
    open: float = Field(description="Bar opening price")
    high: float = Field(description="Bar highest price")
    low: float = Field(description="Bar lowest price")
    close: float = Field(description="Bar closing / current price")
    volume: float = Field(default=0.0, description="Bar traded volume in underlying or contracts")
    candle_start_time_usec: int = Field(description="Candle start timestamp in epoch microseconds")
    is_closed: bool = Field(default=False, description="True once bar interval is complete and confirmed")
    timestamp_usec: int | None = Field(default=None, description="Last update timestamp in epoch microseconds")

    @property
    def candle_start_epoch_s(self) -> float:
        """Candle start time in epoch seconds."""
        return self.candle_start_time_usec / 1_000_000.0

    @property
    def hl2(self) -> float:
        """Median price: (High + Low) / 2."""
        return (self.high + self.low) / 2.0

    @property
    def hlc3(self) -> float:
        """Typical price: (High + Low + Close) / 3."""
        return (self.high + self.low + self.close) / 3.0

    @property
    def ohlc4(self) -> float:
        """Weighted price: (Open + High + Low + Close) / 4."""
        return (self.open + self.high + self.low + self.close) / 4.0

    @property
    def is_bullish(self) -> bool:
        """True if Close >= Open."""
        return self.close >= self.open

    @property
    def is_bearish(self) -> bool:
        """True if Close < Open."""
        return self.close < self.open

    @property
    def range(self) -> float:
        """Absolute price range: High - Low."""
        return round(self.high - self.low, 8)

    @property
    def body(self) -> float:
        """Absolute candle body: |Close - Open|."""
        return round(abs(self.close - self.open), 8)

    @classmethod
    def from_api_dict(cls, data: dict[str, Any], is_closed: bool = False) -> "Candle":
        """Instantiates Candle from Delta candlestick WebSocket payload."""
        start_usec = int(data.get("candle_start_time") or data.get("timestamp") or 0)
        return cls(
            symbol=str(data.get("symbol", "")),
            resolution=str(data.get("resolution", "1m")),
            open=float(data.get("open", 0.0)),
            high=float(data.get("high", 0.0)),
            low=float(data.get("low", 0.0)),
            close=float(data.get("close", 0.0)),
            volume=float(data.get("volume", 0.0)),
            candle_start_time_usec=start_usec,
            is_closed=is_closed,
            timestamp_usec=int(data["timestamp"]) if "timestamp" in data and data["timestamp"] else None,
        )


class OrderBookLevel(BaseModel):
    """Single price-quantity level in the orderbook."""

    model_config = ConfigDict(frozen=True)

    limit_price: float = Field(description="Orderbook limit price")
    size: float = Field(description="Contract volume at this price level")
    depth: float = Field(default=0.0, description="Cumulative depth at this price level")


class OrderBook(BaseModel):
    """
    Level 2 Orderbook snapshot parsed from Delta Exchange 'l2_orderbook' feed.
    Used for spread monitoring and execution slippage estimation.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(description="Trading symbol")
    product_id: int | None = Field(default=None, description="Product ID")
    sequence_no: int | None = Field(default=None, description="Sequence number for packet tracking")
    bids: list[OrderBookLevel] = Field(default_factory=list, description="Bids sorted descending by price")
    asks: list[OrderBookLevel] = Field(default_factory=list, description="Asks sorted ascending by price")
    percentage_spread: float | None = Field(default=None, description="Spread as percentage of mid price")
    timestamp_usec: int | None = Field(default=None, description="Server update timestamp in microseconds")
    is_valid: bool = Field(default=True, description="False if a sequence gap occurred and resync is required")
    needs_resync: bool = Field(default=False, description="True if fresh orderbook snapshot must be fetched")

    @property
    def best_bid(self) -> float | None:
        """Top-of-book best bid price."""
        return self.bids[0].limit_price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        """Top-of-book best ask price."""
        return self.asks[0].limit_price if self.asks else None

    @property
    def spread(self) -> float | None:
        """Top-of-book absolute spread: best_ask - best_bid."""
        if self.best_ask is not None and self.best_bid is not None:
            return round(self.best_ask - self.best_bid, 8)
        return None

    @property
    def mid_price(self) -> float | None:
        """Midpoint price between best bid and best ask."""
        if self.best_ask is not None and self.best_bid is not None:
            return (self.best_ask + self.best_bid) / 2.0
        return None

    def estimate_slippage(self, side: str, size: float) -> tuple[float, float]:
        """
        Simulates filling `size` contracts across orderbook levels.
        Returns:
            (vwap_execution_price, slippage_in_basis_points)
        Raises ValueError if insufficient liquidity is available or if book is invalidated by sequence gaps.
        """
        if not self.is_valid or self.needs_resync:
            raise ValueError(
                f"Orderbook for {self.symbol} is invalid due to a detected sequence gap. "
                "Execution slippage cannot be estimated until book re-synchronizes."
            )

        if size <= 0:
            raise ValueError(f"Size must be positive, got {size}")

        ladder = self.asks if side.lower() == "buy" else self.bids
        if not ladder:
            raise ValueError(f"Orderbook {side} side is empty, cannot estimate slippage")

        remaining = size
        accumulated_cost = 0.0

        for level in ladder:
            take_qty = min(remaining, level.size)
            accumulated_cost += take_qty * level.limit_price
            remaining -= take_qty
            if remaining <= 0:
                break

        if remaining > 0:
            # Orderbook depth exhausted
            raise ValueError(
                f"Insufficient liquidity to fill {size} {self.symbol} contracts (Short by {remaining:.4f})"
            )

        vwap_price = accumulated_cost / size
        top_price = ladder[0].limit_price

        # Calculate slippage relative to top-of-book
        if side.lower() == "buy":
            diff = vwap_price - top_price
        else:
            diff = top_price - vwap_price

        slippage_bps = (diff / top_price) * 10_000.0 if top_price > 0 else 0.0
        return round(vwap_price, 6), round(slippage_bps, 2)

    def copy_invalidated(self) -> "OrderBook":
        """Returns invalidated copy of this orderbook when sequence gap occurs."""
        return OrderBook(
            symbol=self.symbol,
            product_id=self.product_id,
            sequence_no=self.sequence_no,
            bids=self.bids,
            asks=self.asks,
            percentage_spread=self.percentage_spread,
            timestamp_usec=self.timestamp_usec,
            is_valid=False,
            needs_resync=True,
        )

    @classmethod
    def from_api_dict(cls, data: dict[str, Any]) -> "OrderBook":
        """Instantiates OrderBook from 'l2_orderbook' WebSocket message."""
        raw_bids = data.get("buy", []) or []
        raw_asks = data.get("sell", []) or []

        bids = [
            OrderBookLevel(
                limit_price=float(item["limit_price"]),
                size=float(item.get("size", 0.0)),
                depth=float(item.get("depth", 0.0)),
            )
            for item in raw_bids
            if "limit_price" in item
        ]
        asks = [
            OrderBookLevel(
                limit_price=float(item["limit_price"]),
                size=float(item.get("size", 0.0)),
                depth=float(item.get("depth", 0.0)),
            )
            for item in raw_asks
            if "limit_price" in item
        ]

        # Ensure correct sorting: bids descending, asks ascending
        bids.sort(key=lambda x: x.limit_price, reverse=True)
        asks.sort(key=lambda x: x.limit_price)

        pct_spread: float | None = None
        if "percentage_spread" in data and data["percentage_spread"] is not None:
            try:
                pct_spread = float(data["percentage_spread"])
            except ValueError:
                pct_spread = None

        seq_no: int | None = None
        if "last_sequence_no" in data and data["last_sequence_no"] is not None:
            try:
                seq_no = int(data["last_sequence_no"])
            except ValueError:
                seq_no = None

        return cls(
            symbol=str(data.get("symbol", "")),
            product_id=int(data["product_id"]) if "product_id" in data and data["product_id"] else None,
            sequence_no=seq_no,
            bids=bids,
            asks=asks,
            percentage_spread=pct_spread,
            timestamp_usec=int(data["last_updated_at"]) if "last_updated_at" in data and data["last_updated_at"] else None,
        )


class SessionAnchorData(BaseModel):
    """
    Daily Session Anchor (00:00 UTC) data structure.
    Tracks intraday Session VWAP and Previous Day High/Low/Close (PDH/PDL/PDC).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    symbol: str = Field(description="Trading symbol")
    session_date: str = Field(description="UTC session date string: YYYY-MM-DD")
    session_start_epoch_s: float = Field(description="00:00 UTC start epoch seconds")
    prior_day_high: float | None = Field(default=None, description="Previous Day High (PDH)")
    prior_day_low: float | None = Field(default=None, description="Previous Day Low (PDL)")
    prior_day_close: float | None = Field(default=None, description="Previous Day Close (PDC)")
    session_open: float = Field(default=0.0, description="Session opening price at 00:00 UTC")
    session_high: float = Field(default=0.0, description="Current day session high")
    session_low: float = Field(default=float("inf"), description="Current day session low")
    session_close: float = Field(default=0.0, description="Current price / latest close")
    session_volume: float = Field(default=0.0, description="Cumulative session volume")
    cumulative_pv: float = Field(default=0.0, description="Sum of (Typical Price * Volume)")
    cumulative_v: float = Field(default=0.0, description="Sum of Volume for VWAP")
    session_vwap: float = Field(default=0.0, description="Real-time Intraday Session VWAP")

    def update(
        self,
        price_or_candle: Candle | float,
        volume: float = 0.0,
        high: float | None = None,
        low: float | None = None,
    ) -> None:
        """Updates session high/low/volume and recalculates Session VWAP."""
        if isinstance(price_or_candle, Candle):
            c = price_or_candle
            price = c.close
            vol = c.volume
            h = c.high
            l = c.low
            typical_price = c.hlc3
        else:
            price = float(price_or_candle)
            vol = float(volume)
            h = high if high is not None else price
            l = low if low is not None else price
            typical_price = (h + l + price) / 3.0

        if self.session_open == 0.0:
            self.session_open = price
        if h > self.session_high:
            self.session_high = h
        if l < self.session_low:
            self.session_low = l

        self.session_close = price
        self.session_volume += vol

        if vol > 0:
            self.cumulative_pv += typical_price * vol
            self.cumulative_v += vol
            self.session_vwap = round(self.cumulative_pv / self.cumulative_v, 6)
        elif self.session_vwap == 0.0:
            self.session_vwap = price

    update_from_candle = update

    def rollover(
        self,
        next_session_date: str,
        next_session_start_s: float,
        next_open: float,
    ) -> "SessionAnchorData":
        """
        Executes 00:00 UTC session rollover:
        Locks current session high/low as PDH/PDL and resets intraday VWAP accumulator.
        """
        pdh = self.session_high if self.session_high > 0 else None
        pdl = self.session_low if self.session_low != float("inf") else None
        pdc = self.session_close if self.session_close > 0 else (next_open if next_open > 0 else None)

        return SessionAnchorData(
            symbol=self.symbol,
            session_date=next_session_date,
            session_start_epoch_s=next_session_start_s,
            prior_day_high=pdh,
            prior_day_low=pdl,
            prior_day_close=pdc,
            session_open=next_open,
            session_high=next_open,
            session_low=next_open,
            session_close=next_open,
            session_volume=0.0,
            cumulative_pv=0.0,
            cumulative_v=0.0,
            session_vwap=next_open,
        )


class CircularBarCache:
    """
    In-memory RingBuffer / Circular Bar Cache retaining the last N closed bars (default: 500).
    Supplies the Strategy Engine (Phase 3) with confirmed candle history and indicators.
    """

    def __init__(
        self,
        symbol: str = "UNKNOWN",
        resolution: str = "1m",
        capacity: int = 500,
        maxlen: int | None = None,
    ) -> None:
        effective_capacity = maxlen if maxlen is not None else capacity
        self.symbol: str = symbol
        self.resolution: str = resolution
        self.capacity: int = effective_capacity
        self._bars: deque[Candle] = deque(maxlen=effective_capacity)
        self.open_candle: Candle | None = None

    def seed_historical_bars(self, candles: list[Candle]) -> int:
        """
        Seeds circular buffer with historical closed bars from REST warm-up routine.
        Guarantees strategy indicators (EMA, ATR, ADX) are primed on minute zero without cold start delay.
        """
        count = 0
        for c in candles:
            self.append(c)
            count += 1
        return count

    def append(self, candle: Candle) -> None:
        """Appends a confirmed closed candle to the ring buffer."""
        # Ensure candle marked closed
        if not candle.is_closed:
            closed_candle = Candle(
                symbol=candle.symbol,
                resolution=candle.resolution,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
                candle_start_time_usec=candle.candle_start_time_usec,
                is_closed=True,
                timestamp_usec=candle.timestamp_usec,
            )
        else:
            closed_candle = candle

        self._bars.append(closed_candle)

    def set_open_bar(self, candle: Candle) -> None:
        """Updates the current forming open candle."""
        self.open_candle = candle

    def get_closed_bars(self) -> list[Candle]:
        """Returns ordered list of confirmed closed bars from oldest to newest."""
        return list(self._bars)

    def last_closed(self) -> Candle | None:
        """Returns the most recently closed candle or None if cache is empty."""
        return self._bars[-1] if self._bars else None

    @property
    def count(self) -> int:
        """Number of closed bars currently stored in cache."""
        return len(self._bars)

    @property
    def size(self) -> int:
        """Alias for count."""
        return len(self._bars)

    def __len__(self) -> int:
        return len(self._bars)

    @property
    def is_full(self) -> bool:
        """True if the circular buffer has reached its capacity limit."""
        return len(self._bars) >= self.capacity

    def to_dict(self) -> dict[str, list[float]]:
        """
        Exports cache to dictionary of arrays suitable for indicator calculations (EMA, RSI, ATR).
        """
        opens: list[float] = []
        highs: list[float] = []
        lows: list[float] = []
        closes: list[float] = []
        volumes: list[float] = []
        timestamps: list[float] = []

        for bar in self._bars:
            opens.append(bar.open)
            highs.append(bar.high)
            lows.append(bar.low)
            closes.append(bar.close)
            volumes.append(bar.volume)
            timestamps.append(bar.candle_start_epoch_s)

        return {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "timestamp": timestamps,
        }

    def clear(self) -> None:
        """Clears all closed bars and forming bar from cache."""
        self._bars.clear()
        self.open_candle = None

    def __len__(self) -> int:
        return len(self._bars)

    def __repr__(self) -> str:
        return (
            f"CircularBarCache(symbol={self.symbol!r}, resolution={self.resolution!r}, "
            f"count={len(self._bars)}/{self.capacity})"
        )


# -----------------------------------------------------------------------------
# Phase 5: State Reconciliation, Private Feeds & Liquidation Models
# -----------------------------------------------------------------------------

class UserTrade(BaseModel):
    """
    Execution fill report from Delta Exchange 'user_trades' private channel.
    Tracks execution price, filled contracts, fees paid, and maker/taker role.
    """

    model_config = ConfigDict(frozen=True)

    trade_id: int | str = Field(description="Unique trade fill ID")
    order_id: int = Field(description="Associated parent or bracket order ID")
    product_id: int = Field(description="Contract Product ID")
    symbol: str = Field(default="", description="Trading symbol (e.g. BTCUSDT)")
    side: str = Field(description="Order side: 'buy' or 'sell'")
    size: int = Field(description="Filled contract lots")
    price: float = Field(description="Fill execution price")
    commission: float = Field(default=0.0, description="Fee paid in USD or collateral asset")
    role: str = Field(default="taker", description="Execution role: 'maker' or 'taker'")
    created_at: float = Field(default_factory=time.time, description="Fill timestamp in epoch seconds")

    @classmethod
    def from_api_dict(cls, data: dict[str, Any]) -> "UserTrade":
        """Instantiates UserTrade from raw WebSocket or REST payload."""
        def _to_float(v: Any) -> float:
            try:
                return float(v) if v is not None and v != "" else 0.0
            except (ValueError, TypeError):
                return 0.0

        raw_id = data.get("id") or data.get("trade_id") or str(uuid.uuid4().int)[:10]
        return cls(
            trade_id=raw_id,
            order_id=int(data.get("order_id", 0)),
            product_id=int(data.get("product_id", 0)),
            symbol=str(data.get("symbol") or data.get("product_symbol", "")),
            side=str(data.get("side", "")).lower(),
            size=int(data.get("size", 0)),
            price=_to_float(data.get("price") or data.get("fill_price")),
            commission=_to_float(data.get("commission") or data.get("fee")),
            role=str(data.get("role") or data.get("order_role", "taker")).lower(),
            created_at=_to_float(data.get("created_at") or data.get("timestamp")) or time.time(),
        )


class OrderUpdate(BaseModel):
    """
    Real-time order state event streamed from Delta Exchange 'orders' private channel.
    Tracks transitions across: open, pending, closed, cancelled, rejected.
    """

    model_config = ConfigDict(frozen=True)

    id: int = Field(description="Exchange order ID")
    client_order_id: str | None = Field(default=None, description="Client order ID (max 32 chars)")
    product_id: int = Field(description="Contract Product ID")
    symbol: str = Field(default="", description="Trading symbol")
    side: str = Field(description="Order side: 'buy' or 'sell'")
    order_type: str = Field(default="market_order", description="Order type")
    state: str = Field(description="Order state: open, pending, closed, cancelled, rejected")
    size: int = Field(description="Original order size in contract lots")
    unfilled_size: int = Field(default=0, description="Remaining unfilled contract lots")
    limit_price: float | None = Field(default=None, description="Limit price if applicable")
    stop_price: float | None = Field(default=None, description="Stop trigger price if stop order")
    bracket_stop_loss_price: float | None = Field(default=None, description="Attached bracket SL price")
    bracket_take_profit_price: float | None = Field(default=None, description="Attached bracket TP price")
    stop_order_type: str | None = Field(default=None, description="stop_loss_order or take_profit_order")
    stop_trigger_method: str | None = Field(default="mark_price", description="Trigger method (e.g. mark_price)")
    paid_commission: float = Field(default=0.0, description="Cumulative commission paid")
    created_at: float = Field(default_factory=time.time, description="Event creation timestamp in epoch seconds")

    @property
    def is_closed(self) -> bool:
        """Returns True if order is completely filled/closed."""
        return self.state in ("closed", "filled")

    @property
    def is_cancelled(self) -> bool:
        """Returns True if order was cancelled or rejected."""
        return self.state in ("cancelled", "rejected")

    @property
    def is_open(self) -> bool:
        """Returns True if order is active or resting on the book."""
        return self.state in ("open", "pending")

    @classmethod
    def from_api_dict(cls, data: dict[str, Any]) -> "OrderUpdate":
        """Instantiates OrderUpdate from raw Delta orders frame."""
        def _to_float_opt(v: Any) -> float | None:
            try:
                return float(v) if v is not None and v != "" else None
            except (ValueError, TypeError):
                return None

        def _to_float(v: Any) -> float:
            try:
                return float(v) if v is not None and v != "" else 0.0
            except (ValueError, TypeError):
                return 0.0

        return cls(
            id=int(data.get("id", 0)),
            client_order_id=data.get("client_order_id"),
            product_id=int(data.get("product_id", 0)),
            symbol=str(data.get("symbol") or data.get("product_symbol", "")),
            side=str(data.get("side", "")).lower(),
            order_type=str(data.get("order_type", "market_order")),
            state=str(data.get("state", "open")).lower(),
            size=int(data.get("size", 0)),
            unfilled_size=int(data.get("unfilled_size", 0)),
            limit_price=_to_float_opt(data.get("limit_price")),
            stop_price=_to_float_opt(data.get("stop_price")),
            bracket_stop_loss_price=_to_float_opt(data.get("bracket_stop_loss_price")),
            bracket_take_profit_price=_to_float_opt(data.get("bracket_take_profit_price")),
            stop_order_type=data.get("stop_order_type"),
            stop_trigger_method=data.get("stop_trigger_method", "mark_price"),
            paid_commission=_to_float(data.get("paid_commission") or data.get("fee")),
            created_at=_to_float(data.get("created_at") or data.get("timestamp")) or time.time(),
        )


class GhostPosition(BaseModel):
    """
    Representation of an open position discovered on Delta Exchange that is untracked
    by the bot's local memory. Tagged under external management to prevent unintended market closes.
    """

    model_config = ConfigDict(frozen=True)

    product_id: int = Field(description="Contract Product ID")
    symbol: str = Field(description="Trading symbol")
    size: int = Field(description="Position size in contract lots")
    entry_price: float = Field(description="Average entry price")
    liquidation_price: float | None = Field(default=None, description="Liquidation price")
    discovered_at: float = Field(default_factory=time.time, description="Timestamp when position was discovered")
    tag: str = Field(default="EXTERNAL_UNTRACKED", description="Management classification tag")


class LiquidationAlert(BaseModel):
    """
    Critical safety event triggered when mark price comes dangerously close to liquidation price.
    Prompts immediate emergency flatten execution.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(description="Contract symbol (e.g. BTCUSDT)")
    product_id: int = Field(description="Contract Product ID")
    mark_price: float = Field(description="Current mark price")
    liquidation_price: float = Field(description="Exchange liquidation threshold price")
    distance: float = Field(description="Absolute price distance to liquidation: |mark - liq|")
    threshold_distance: float = Field(description="Safety threshold distance (1.5 * ATR)")
    atr: float = Field(description="Current ATR value used for threshold calculation")
    timestamp: float = Field(default_factory=time.time, description="Epoch timestamp of alert")


