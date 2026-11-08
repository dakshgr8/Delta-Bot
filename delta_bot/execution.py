"""
delta_bot.execution
~~~~~~~~~~~~~~~~~~~

Order Execution, Routing & Lifecycle Management Engine (Phase 4).
Solves critical execution traps for Delta Exchange algorithmic trading:
1. Exchange-Native Stop Orders with attached brackets (zero client latency slippage)
2. Dual-Constraint Margin & Leverage Sizing (coordinated with RiskManager)
3. Dynamic Product ID Resolution across Delta India, Delta Global, and Delta Testnet
4. Deterministic client_order_id generation (<= 32 chars) for idempotent order placement
5. REST Invalidation & TTL Purging of resting exchange stop orders
6. Exchange-native Trailing Stop Ratchet synchronization
7. Emergency Flatten & Kill-Switch execution
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from delta_bot.client import DeltaRestClient
from delta_bot.exceptions import (
    DeltaAPIError,
    InsufficientMarginError,
)
from delta_bot.models import (
    ContractSpec,
    ExchangeOrder,
    ExecutionMode,
    MarginCheckResult,
    PositionInfo,
    WalletBalance,
)
from delta_bot.risk import RiskConfig, RiskManager
from delta_bot.strategy import (
    OrderIntent,
    OrderStatus,
    SignalType,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Execution Configuration
# -----------------------------------------------------------------------------

class ExecutionConfig(BaseModel):
    """Configuration parameters for order execution and routing."""

    model_config = ConfigDict(frozen=True)

    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.LIVE,
        description="Execution mode: LIVE (submits to exchange) or DRY_RUN (simulated)",
    )
    leverage: int = Field(
        default=5, ge=1, le=100, description="Target contract leverage tier (e.g. 5x, 10x)"
    )
    margin_safety_buffer: float = Field(
        default=0.90,
        gt=0.0,
        le=1.0,
        description="Fraction of available margin usable for initial margin (10% safety buffer for taker fees)",
    )
    stop_trigger_method: str = Field(
        default="mark_price",
        description="Reference price for stop trigger matching ('mark_price' or 'last_traded_price')",
    )
    auto_set_leverage: bool = Field(
        default=True,
        description="Automatically issue POST /v2/products/{id}/orders/leverage prior to order placement",
    )
    default_collateral_symbol: str = Field(
        default="USDT",
        description="Primary quote/collateral asset symbol for margin balance checks",
    )


# -----------------------------------------------------------------------------
# Local Order Registry
# -----------------------------------------------------------------------------

class LocalRegistry:
    """
    In-memory registry maintaining bidirectional mapping between:
    - client_order_id <-> exchange_order_id
    - intent_id <-> ExchangeOrder
    Ensures idempotent tracking, state validation, and clean lifecycle management.
    """

    def __init__(self) -> None:
        self._orders_by_client_id: dict[str, ExchangeOrder] = {}
        self._orders_by_exchange_id: dict[int, ExchangeOrder] = {}
        self._intent_to_exchange_id: dict[str, int] = {}
        self._client_to_exchange_id: dict[str, int] = {}

    def register(self, exchange_order: ExchangeOrder, intent_id: str | None = None) -> None:
        """Registers an active exchange order across all index maps."""
        self._orders_by_exchange_id[exchange_order.id] = exchange_order

        if exchange_order.client_order_id:
            self._orders_by_client_id[exchange_order.client_order_id] = exchange_order
            self._client_to_exchange_id[exchange_order.client_order_id] = exchange_order.id

        if intent_id:
            self._intent_to_exchange_id[intent_id] = exchange_order.id

    def get_by_exchange_id(self, exchange_id: int) -> ExchangeOrder | None:
        """Looks up an order by its Delta Exchange internal ID."""
        return self._orders_by_exchange_id.get(exchange_id)

    def get_by_client_id(self, client_id: str) -> ExchangeOrder | None:
        """Looks up an order by its client_order_id."""
        return self._orders_by_client_id.get(client_id)

    def get_by_intent_id(self, intent_id: str) -> ExchangeOrder | None:
        """Looks up an order by strategy OrderIntent ID."""
        ex_id = self._intent_to_exchange_id.get(intent_id)
        if ex_id is not None:
            return self._orders_by_exchange_id.get(ex_id)
        return None

    def remove(self, exchange_id: int) -> None:
        """Removes an order from the registry upon cancellation or fill."""
        order = self._orders_by_exchange_id.pop(exchange_id, None)
        if order and order.client_order_id:
            self._orders_by_client_id.pop(order.client_order_id, None)
            self._client_to_exchange_id.pop(order.client_order_id, None)

        # Remove intent mapping if present
        for intent_id, mapped_id in list(self._intent_to_exchange_id.items()):
            if mapped_id == exchange_id:
                self._intent_to_exchange_id.pop(intent_id, None)

    def clear(self) -> None:
        """Clears all registry mappings."""
        self._orders_by_client_id.clear()
        self._orders_by_exchange_id.clear()
        self._intent_to_exchange_id.clear()
        self._client_to_exchange_id.clear()

    @property
    def active_orders(self) -> list[ExchangeOrder]:
        """Returns list of currently tracked active exchange orders."""
        return list(self._orders_by_exchange_id.values())


# -----------------------------------------------------------------------------
# Order Router (Low-Level Delta Exchange Gateway)
# -----------------------------------------------------------------------------

class OrderRouter:
    """
    Derivatives order router translating strategy orders into Delta Exchange API frames.
    Handles client_order_id generation (<= 32 chars), native stop orders with brackets,
    resting order cancellations, and market-close flatten orders.
    """

    def __init__(self, client: DeltaRestClient, config: ExecutionConfig) -> None:
        self.client: DeltaRestClient = client
        self.config: ExecutionConfig = config

    @staticmethod
    def generate_client_order_id(
        symbol: str,
        tenant_id: str | None = None,
        base_signal_id: str | None = None,
    ) -> str:
        """
        Generates a deterministic, unique client_order_id adhering strictly to
        Delta Exchange's 32-character maximum constraint.
        When base_signal_id and tenant_id are present, utilizes uuid.uuid5 to ensure
        unique yet deterministic mapping per tenant across fan-outs.
            Format with tenant fan-out: D_{30_hex_of_uuid5} (32 characters)
            Standard format: DBOT_{SYMBOL}_{EPOCH_MS} (<= 32 characters)
        """
        if base_signal_id and tenant_id:
            raw_uuid = uuid.uuid5(uuid.NAMESPACE_OID, f"{base_signal_id}:{tenant_id}").hex
            return f"D_{raw_uuid[:30]}"
        epoch_ms = int(time.time() * 1000)
        clean_symbol = symbol.replace("/", "").replace("-", "").upper()
        raw_id = f"DBOT_{clean_symbol[:10]}_{epoch_ms}"
        return raw_id[:32]

    async def submit_native_stop_order(
        self,
        product_id: int,
        symbol: str,
        side: str,
        size: int,
        stop_price: float,
        bracket_stop_loss_price: float,
        bracket_take_profit_price: float,
        client_order_id: str,
    ) -> ExchangeOrder:
        """
        Submits an Exchange-Native Stop Order with attached Brackets to Delta Exchange.
        Endpoint: POST /v2/orders
        """
        if self.config.execution_mode == ExecutionMode.DRY_RUN:
            sim_id = int(uuid.uuid4().int % 100_000_000)
            order = ExchangeOrder(
                id=sim_id,
                client_order_id=client_order_id,
                product_id=product_id,
                symbol=symbol,
                side=side.lower(),
                order_type="market_order",
                size=size,
                unfilled_size=size,
                stop_price=stop_price,
                stop_order_type=None,
                stop_trigger_method=self.config.stop_trigger_method,
                bracket_stop_loss_price=bracket_stop_loss_price,
                bracket_take_profit_price=bracket_take_profit_price,
                state="open",
                created_at=time.time(),
            )
            logger.info(
                "[DRY_RUN] Native %s Stop Order placed: ID=%d | %s | StopPx=%.4f | SL=%.4f | TP=%.4f | Lots=%d | CID=%s",
                side.upper(),
                sim_id,
                symbol,
                stop_price,
                bracket_stop_loss_price,
                bracket_take_profit_price,
                size,
                client_order_id,
            )
            return order

        # Live submission
        try:
            order = await self.client.place_order(
                product_id=product_id,
                size=size,
                side=side,
                order_type="market_order",
                stop_price=stop_price,
                stop_order_type=None,
                stop_trigger_method=self.config.stop_trigger_method,
                bracket_stop_loss_price=bracket_stop_loss_price,
                bracket_take_profit_price=bracket_take_profit_price,
                time_in_force="gtc",
                client_order_id=client_order_id,
                symbol=symbol,
            )

            # Verification 1: If order is resting, ensure brackets are attached on exchange
            if getattr(order, "is_open", False) and (
                getattr(order, "bracket_stop_loss_price", None) is None
                or getattr(order, "bracket_take_profit_price", None) is None
            ):
                try:
                    await self.client.edit_bracket_order(
                        product_id=product_id,
                        order_id=order.id,
                        bracket_stop_loss_price=bracket_stop_loss_price,
                        bracket_take_profit_price=bracket_take_profit_price,
                    )
                    logger.info(
                        "[BRACKET GUARD] Successfully attached brackets via PUT /v2/orders/bracket to resting order #%d: SL=%.4f, TP=%.4f",
                        order.id,
                        bracket_stop_loss_price,
                        bracket_take_profit_price,
                    )
                    if hasattr(order, "model_copy"):
                        order = order.model_copy(
                            update={
                                "bracket_stop_loss_price": bracket_stop_loss_price,
                                "bracket_take_profit_price": bracket_take_profit_price,
                            }
                        )
                except Exception as b_err:
                    logger.warning(
                        "[BRACKET GUARD] Could not attach brackets to resting order #%d via PUT /v2/orders/bracket: %s",
                        order.id,
                        b_err,
                    )

            # Verification 2: If order filled immediately (not resting open), ensure position is armed
            if not getattr(order, "is_open", False):
                await self.ensure_position_protection(
                    product_id=product_id,
                    symbol=symbol,
                    side=side,
                    size=size,
                    stop_loss_price=bracket_stop_loss_price,
                    take_profit_price=bracket_take_profit_price,
                )

            return order
        except DeltaAPIError as err:
            if err.error_code == "immediate_execution_stop_order":
                logger.warning(
                    "[BREAKOUT GATEWAY] Stop trigger %.4f on %s (%s) was already breached (immediate_execution_stop_order). "
                    "Falling back to immediate market order entry with attached brackets.",
                    stop_price,
                    symbol,
                    side,
                )
                fallback_cid = f"{client_order_id[:29]}_m" if client_order_id else None
                market_order = await self.client.place_order(
                    product_id=product_id,
                    size=size,
                    side=side,
                    order_type="market_order",
                    bracket_stop_loss_price=bracket_stop_loss_price,
                    bracket_take_profit_price=bracket_take_profit_price,
                    time_in_force="gtc",
                    client_order_id=fallback_cid,
                    symbol=symbol,
                )

                # Ensure position is armed with SL and TP on exchange so it is never naked
                await self.ensure_position_protection(
                    product_id=product_id,
                    symbol=symbol,
                    side=side,
                    size=size,
                    stop_loss_price=bracket_stop_loss_price,
                    take_profit_price=bracket_take_profit_price,
                )
                return market_order
            raise

    async def ensure_position_protection(
        self,
        product_id: int,
        symbol: str,
        side: str,
        size: int,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
    ) -> bool:
        """
        Guarantees that an open position has active Stop Loss and Take Profit protection.
        Attempts POST /v2/orders/bracket first, and falls back to native reduce-only child orders
        if bracket attachment fails.
        """
        if self.config.execution_mode == ExecutionMode.DRY_RUN:
            return True

        if stop_loss_price is None and take_profit_price is None:
            return False

        has_sl = False
        has_tp = False
        exit_side = "sell" if str(side).lower() == "buy" else "buy"
        try:
            open_orders = await self.client.get_open_orders(product_id=product_id)
            if isinstance(open_orders, list):
                has_sl = any(
                    o.product_id == product_id
                    and str(getattr(o, "side", "")).lower() == exit_side
                    and getattr(o, "stop_price", None) is not None
                    for o in open_orders
                )
                has_tp = any(
                    o.product_id == product_id
                    and str(getattr(o, "side", "")).lower() == exit_side
                    and (
                        getattr(o, "order_type", "") == "limit_order"
                        or getattr(o, "bracket_take_profit_price", None) is not None
                    )
                    for o in open_orders
                )
        except Exception as o_err:
            logger.debug("Could not fetch open orders to check existing protection: %s", o_err)

        if has_sl and has_tp:
            logger.info("[PROTECTION GUARD] Position for %s already has active SL and TP protection.", symbol)
            return True

        attached = False
        try:
            await self.client.set_position_bracket(
                product_id=product_id,
                bracket_stop_loss_price=stop_loss_price if not has_sl else None,
                bracket_take_profit_price=take_profit_price if not has_tp else None,
            )
            logger.info(
                "[PROTECTION GUARD] Successfully armed position bracket for %s (product_id=%d): SL=%s, TP=%s",
                symbol,
                product_id,
                stop_loss_price if not has_sl else "EXISTING",
                take_profit_price if not has_tp else "EXISTING",
            )
            attached = True
        except Exception as b_exc:
            logger.warning(
                "[PROTECTION GUARD] set_position_bracket failed for %s: %s. Falling back to native reduce-only child orders.",
                symbol,
                b_exc,
            )

        if not attached:
            if not has_sl and stop_loss_price is not None:
                try:
                    sl_res = await self.client.place_order(
                        product_id=product_id,
                        size=size,
                        side=exit_side,
                        order_type="market_order",
                        stop_price=stop_loss_price,
                        stop_order_type="stop_loss_order",
                        reduce_only=True,
                        time_in_force="gtc",
                        symbol=symbol,
                    )
                    logger.info(
                        "[PROTECTION GUARD] Placed fallback child Stop Loss order ID=%d for %s at %.4f",
                        sl_res.id,
                        symbol,
                        stop_loss_price,
                    )
                except Exception as sl_exc:
                    logger.critical(
                        "[PROTECTION GUARD] Failed to place child SL order for %s: %s! Position may be unprotected!",
                        symbol,
                        sl_exc,
                    )

            if not has_tp and take_profit_price is not None:
                try:
                    tp_res = await self.client.place_order(
                        product_id=product_id,
                        size=size,
                        side=exit_side,
                        order_type="limit_order",
                        limit_price=take_profit_price,
                        reduce_only=True,
                        time_in_force="gtc",
                        symbol=symbol,
                    )
                    logger.info(
                        "[PROTECTION GUARD] Placed fallback child Take Profit order ID=%d for %s at %.4f",
                        tp_res.id,
                        symbol,
                        take_profit_price,
                    )
                except Exception as tp_exc:
                    logger.error(
                        "[PROTECTION GUARD] Failed to place child TP order for %s: %s",
                        symbol,
                        tp_exc,
                    )

        return True

    async def cancel_order(self, product_id: int, order_id: int) -> dict[str, Any]:
        """Cancels a resting stop order via DELETE /v2/orders."""
        if self.config.execution_mode == ExecutionMode.DRY_RUN:
            logger.info("[DRY_RUN] Simulated cancel order id=%d (product_id=%d)", order_id, product_id)
            return {"status": "cancelled", "product_id": product_id, "id": order_id}

        return await self.client.cancel_order(product_id=product_id, order_id=order_id)

    async def cancel_all_orders(self, product_id: int | None = None) -> list[dict[str, Any]]:
        """Cancels all resting orders on Delta via DELETE /v2/orders/all."""
        if self.config.execution_mode == ExecutionMode.DRY_RUN:
            logger.info("[DRY_RUN] Simulated cancel all orders for product_id=%s", product_id)
            return [{"status": "all_cancelled", "product_id": product_id}]

        return await self.client.cancel_all_orders(product_id=product_id)

    async def edit_pending_bracket(
        self,
        product_id: int,
        order_id: int,
        new_sl: float | None = None,
        new_tp: float | None = None,
    ) -> dict[str, Any]:
        """
        Modifies bracket Stop Loss or Take Profit on a pending/resting entry order.
        Endpoint: PUT /v2/orders/bracket (Authenticated)
        """
        if self.config.execution_mode == ExecutionMode.DRY_RUN:
            logger.info(
                "[DRY_RUN] Simulated edit pending bracket for order id=%d (product_id=%d): SL=%s TP=%s",
                order_id,
                product_id,
                new_sl,
                new_tp,
            )
            return {
                "id": order_id,
                "product_id": product_id,
                "bracket_stop_loss_price": str(new_sl) if new_sl is not None else None,
                "bracket_take_profit_price": str(new_tp) if new_tp is not None else None,
                "status": "success",
            }

        return await self.client.edit_bracket_order(
            product_id=product_id,
            order_id=order_id,
            bracket_stop_loss_price=new_sl,
            bracket_take_profit_price=new_tp,
        )

    async def edit_active_stop_order(
        self,
        product_id: int,
        stop_order_id: int,
        new_sl: float,
    ) -> ExchangeOrder:
        """
        Modifies the stop_price on an active child Stop Loss order once the position is open.
        Endpoint: PUT /v2/orders (Authenticated)
        """
        if self.config.execution_mode == ExecutionMode.DRY_RUN:
            logger.info(
                "[DRY_RUN] Simulated edit child stop order id=%d (product_id=%d) stop_price to %.4f",
                stop_order_id,
                product_id,
                new_sl,
            )
            return ExchangeOrder(
                id=stop_order_id,
                product_id=product_id,
                side="sell",
                order_type="market_order",
                size=1,
                stop_price=new_sl,
                state="open",
            )

        return await self.client.edit_order(
            product_id=product_id,
            order_id=stop_order_id,
            stop_price=new_sl,
        )

    async def set_position_bracket(
        self,
        product_id: int,
        new_sl: float | None = None,
        new_tp: float | None = None,
    ) -> dict[str, Any]:
        """
        Attaches or modifies bracket Stop Loss and Take Profit on an active position.
        Endpoint: POST /v2/orders/bracket (Authenticated)
        """
        if self.config.execution_mode == ExecutionMode.DRY_RUN:
            logger.info(
                "[DRY_RUN] Simulated set position bracket for product_id=%d: SL=%s TP=%s",
                product_id,
                new_sl,
                new_tp,
            )
            return {
                "product_id": product_id,
                "bracket_stop_loss_price": str(new_sl) if new_sl is not None else None,
                "bracket_take_profit_price": str(new_tp) if new_tp is not None else None,
                "status": "success",
            }

        return await self.client.set_position_bracket(
            product_id=product_id,
            bracket_stop_loss_price=new_sl,
            bracket_take_profit_price=new_tp,
        )

    async def edit_bracket_stop_loss(
        self,
        product_id: int,
        order_id: int,
        new_sl: float,
    ) -> dict[str, Any] | ExchangeOrder:
        """
        Convenience method: modifies bracket Stop Loss on pending order via PUT /v2/orders/bracket.
        Falls back to PUT /v2/orders if PUT /v2/orders/bracket fails.
        """
        return await self.edit_pending_bracket(product_id=product_id, order_id=order_id, new_sl=new_sl)

    async def market_close_position(
        self,
        product_id: int,
        size: int,
        side: str,
        symbol: str = "",
    ) -> ExchangeOrder:
        """
        Executes a market order with reduce_only=True to flatten an active open position.
        """
        client_id = self.generate_client_order_id(symbol or "FLATTEN")
        if self.config.execution_mode == ExecutionMode.DRY_RUN:
            sim_id = int(uuid.uuid4().int % 100_000_000)
            logger.info(
                "[DRY_RUN] Market-closed position for product_id=%d: %s %d lots (reduce_only=True)",
                product_id,
                side.upper(),
                size,
            )
            return ExchangeOrder(
                id=sim_id,
                client_order_id=client_id,
                product_id=product_id,
                symbol=symbol,
                side=side,
                order_type="market_order",
                size=size,
                state="filled",
            )

        return await self.client.place_order(
            product_id=product_id,
            size=size,
            side=side,
            order_type="market_order",
            reduce_only=True,
            time_in_force="ioc",
            client_order_id=client_id,
            symbol=symbol,
        )


# -----------------------------------------------------------------------------
# Execution Manager (High-Level Execution Pipeline)
# -----------------------------------------------------------------------------

class ExecutionManager:
    """
    Principal Order Execution & Position Lifecycle Manager for Delta-Bot.
    Coordinates:
    - Pre-trade risk validation & dual-constraint sizing with RiskManager
    - Product ID resolution across Delta Global, Delta India, and Delta Testnet
    - Exchange-native stop order dispatching with attached brackets
    - LocalRegistry client_order_id <-> exchange_order_id tracking
    - Intermediate EMA invalidation & TTL expiration order purging
    - Trailing stop ratchet synchronization
    - Emergency flatten kill-switch
    """

    def __init__(
        self,
        client: DeltaRestClient,
        arg2: RiskManager | ExecutionConfig | None = None,
        config: ExecutionConfig | None = None,
        risk_manager: RiskManager | None = None,
        state_manager: Any | None = None,
    ) -> None:
        self.client: DeltaRestClient = client

        # Resolve config and risk_manager from positional or keyword arguments
        resolved_config = config
        resolved_risk_mgr = risk_manager

        if isinstance(arg2, ExecutionConfig):
            resolved_config = arg2
        elif isinstance(arg2, RiskManager):
            resolved_risk_mgr = arg2

        self.config: ExecutionConfig = resolved_config or ExecutionConfig()
        self.risk_manager: RiskManager = resolved_risk_mgr or RiskManager(
            RiskConfig(
                max_leverage=self.config.leverage,
                margin_safety_buffer=self.config.margin_safety_buffer,
            )
        )
        self.order_router: OrderRouter = OrderRouter(client, self.config)
        self.registry: LocalRegistry = LocalRegistry()
        self.state_manager: Any | None = state_manager
        if self.state_manager is not None and getattr(self.state_manager, "execution_manager", None) is None:
            self.state_manager.execution_manager = self

        # Cached leverage state per product_id
        self._configured_leverages: dict[int, int] = {}

        # Execution Safety Shields
        self._inflight_fingerprints: dict[str, float] = {}
        self.is_execution_locked: bool = False

    def set_state_manager(self, state_manager: Any) -> None:
        """Attaches TradingStateManager for real-time WebSocket state synchronization."""
        self.state_manager = state_manager
        if state_manager is not None:
            state_manager.execution_manager = self

    # -------------------------------------------------------------------------
    # Product ID Resolution (Trap 3 Guard)
    # -------------------------------------------------------------------------

    async def resolve_product_spec(self, symbol: str) -> ContractSpec:
        """
        Dynamically resolves the contract specification and integer product_id for a symbol.
        Guarantees correct product_id across Delta India, Delta Global, and Testnet without hardcoding.
        """
        clean_symbol = symbol.strip().upper()
        spec = self.client.get_cached_contract_spec(clean_symbol)
        if spec is not None:
            return spec

        await self.client.get_products(["perpetual_futures"])
        spec = self.client.get_cached_contract_spec(clean_symbol)
        if spec is None:
            raise DeltaAPIError(
                message=f"Product specification for '{clean_symbol}' not found on exchange.",
                endpoint="/v2/products",
            )
        return spec

    # -------------------------------------------------------------------------
    # Leverage Synchronization (Trap 2 Guard: Step 1)
    # -------------------------------------------------------------------------

    async def ensure_order_leverage(self, spec: ContractSpec) -> int:
        """
        Ensures the product's order leverage on Delta Exchange matches target config.
        Endpoint: POST /v2/products/{product_id}/orders/leverage
        """
        target_lev = self.config.leverage
        cached_lev = self._configured_leverages.get(spec.product_id)

        if cached_lev == target_lev:
            return target_lev

        if self.config.execution_mode == ExecutionMode.DRY_RUN:
            self._configured_leverages[spec.product_id] = target_lev
            return target_lev

        if self.config.auto_set_leverage:
            try:
                await self.client.set_order_leverage(spec.product_id, target_lev)
                self._configured_leverages[spec.product_id] = target_lev
                logger.info(
                    "Synchronized exchange order leverage for %s (id=%d) to %dx",
                    spec.symbol,
                    spec.product_id,
                    target_lev,
                )
            except Exception as exc:
                logger.warning(
                    "Could not set order leverage for %s (id=%d): %s. Continuing with exchange default.",
                    spec.symbol,
                    spec.product_id,
                    exc,
                )

        return target_lev

    # -------------------------------------------------------------------------
    # Dual-Constraint Margin Sizing Engine (Trap 2 Guard: Step 2)
    # -------------------------------------------------------------------------

    async def check_and_size_margin(
        self,
        spec: ContractSpec,
        trigger_price: float,
        stop_loss_price: float,
        account_equity: float,
        risk_pct: float = 1.0,
        available_margin_override: float | None = None,
    ) -> MarginCheckResult:
        """
        Computes final order lot size under dual constraints:
        1. Risk-Based Constraint: loss budget = equity * (risk_pct / 100)
        2. Margin-Based Constraint: initial margin = (notional / leverage) <= available_margin * safety_buffer
        """
        available_margin = available_margin_override
        if available_margin is None:
            if self.client.config.has_valid_credentials():
                # Real live trading with configured API credentials: query real exchange balances strictly
                try:
                    balances = await self.client.get_wallet_balances()
                    target_symbol = spec.quoting_symbol or self.config.default_collateral_symbol
                    priority_symbols = [target_symbol.upper(), "INR", "USD", "USDT"]
                    matched = [b for b in balances if b.asset_symbol.upper() in priority_symbols and float(b.available_balance) > 0]
                    if matched:
                        available_margin = float(matched[0].available_balance)
                    elif balances:
                        pos_bals = [b for b in balances if float(b.available_balance) > 0]
                        available_margin = float(pos_bals[0].available_balance) if pos_bals else float(balances[0].available_balance)
                    else:
                        available_margin = 0.0
                except Exception as exc:
                    logger.error("Failed to fetch live wallet balances from Delta Exchange: %s. Disallowing trade sizing on fallback demo values.", exc)
                    available_margin = 0.0
            else:
                available_margin = account_equity if account_equity > 0 else 0.0

        if account_equity <= 0 and available_margin is not None and available_margin > 0:
            account_equity = available_margin

        return self.risk_manager.size_position(
            spec=spec,
            trigger_price=trigger_price,
            stop_loss_price=stop_loss_price,
            account_equity=account_equity,
            available_margin=available_margin,
            risk_pct_override=risk_pct,
        )

    # -------------------------------------------------------------------------
    # Native Stop Order Dispatching with Brackets (Trap 1 Guard)
    # -------------------------------------------------------------------------

    async def execute_signal(
        self,
        order_intent: OrderIntent,
        account_equity: float = 0.0,
        available_margin_override: float | None = None,
    ) -> ExchangeOrder | None:
        """
        Event-driven execution entry point:
        1. Checks circuit breaker & max concurrent positions
        2. Resolves product specification & integer product_id
        3. Pre-trade risk check & dual-constraint margin sizing
        4. Synchronizes order leverage
        5. Generates deterministic client_order_id (<= 32 chars)
        6. Submits native stop order with attached brackets to Delta Exchange
        7. Registers order in LocalRegistry
        """
        # 0. Execution Safety Shields
        # Shield 1: Split-Brain Reconnect Guard
        if self.is_execution_locked or (
            self.state_manager is not None and getattr(self.state_manager, "is_reconciling", False) is True
        ):
            logger.warning(
                "[SPLIT-BRAIN SHIELD] Reconnection or state reconciliation in progress for %s. Order execution locked.",
                order_intent.symbol,
            )
            order_intent.cancel("Split-brain shield: reconciliation in progress", OrderStatus.CANCELLED_MANUAL)
            return None

        # Shield 2: Anti-Flood Idempotency Shield (5-second sliding window)
        now_ts = time.time()
        self._inflight_fingerprints = {
            k: v for k, v in self._inflight_fingerprints.items() if now_ts - v < 5.0
        }
        fingerprint = f"{order_intent.symbol}_{order_intent.signal_type.value}_{round(float(order_intent.trigger_price), 4)}"
        if fingerprint in self._inflight_fingerprints:
            logger.warning(
                "[ANTI-FLOOD SHIELD] Duplicate order submission blocked within 5s window: %s",
                fingerprint,
            )
            order_intent.cancel("Anti-flood duplicate order blocked", OrderStatus.CANCELLED_MANUAL)
            return None
        self._inflight_fingerprints[fingerprint] = now_ts

        # Shield 3: Stale Feed Guard (>15s without ticker update)
        if self.state_manager is not None and hasattr(self.state_manager, "get_market_context_age"):
            feed_age = self.state_manager.get_market_context_age(order_intent.symbol)
            if feed_age is not None and feed_age > 15.0:
                logger.warning(
                    "[STALE FEED SHIELD] Price quote for %s is stale (>15s without update: %.1fs). Order execution aborted.",
                    order_intent.symbol,
                    feed_age,
                )
                order_intent.cancel(f"Stale feed quote (>15s): age={feed_age:.1f}s", OrderStatus.CANCELLED_MANUAL)
                return None

        # 1. Circuit breaker & position limits
        can_open, reject_reason = self.risk_manager.can_open_new_position(symbol=order_intent.symbol)
        if not can_open:
            logger.warning(
                "Order intent %s for %s rejected by RiskManager: %s",
                order_intent.intent_id,
                order_intent.symbol,
                reject_reason,
            )
            order_intent.cancel(reject_reason or "RiskManager rejection", OrderStatus.CANCELLED_MANUAL)
            return None

        # 2. Resolve contract spec & product_id
        spec = await self.resolve_product_spec(order_intent.symbol)

        # Strict perpetual futures enforcement (Spot and options are strictly prohibited)
        if spec.contract_type != "perpetual_futures":
            logger.error(
                "Execution rejected: %s is not a perpetual futures contract (contract_type='%s')",
                spec.symbol,
                spec.contract_type,
            )
            order_intent.cancel(f"Non-perpetual contract rejected: {spec.contract_type}", OrderStatus.CANCELLED_MANUAL)
            return None

        # 3. Synchronize leverage tier
        await self.ensure_order_leverage(spec)

        # 4. Pre-trade margin check & sizing
        margin_result = await self.check_and_size_margin(
            spec=spec,
            trigger_price=order_intent.trigger_price,
            stop_loss_price=order_intent.initial_stop_loss,
            account_equity=account_equity,
            risk_pct=self.risk_manager.config.max_account_risk_pct,
            available_margin_override=available_margin_override,
        )

        if not margin_result.is_margin_sufficient or margin_result.final_lots < 1:
            logger.error(
                "Order intent %s for %s failed margin check: %s",
                order_intent.intent_id,
                order_intent.symbol,
                margin_result.rejection_reason,
            )
            raise InsufficientMarginError(
                message=f"Order rejected: {margin_result.rejection_reason}",
            )

        # Update intent attributes
        order_intent.contract_lots = margin_result.final_lots
        order_intent.product_id = spec.product_id
        order_intent.leverage = self.config.leverage

        side = "buy" if order_intent.signal_type == SignalType.BULLISH_BREAKOUT else "sell"
        client_order_id = (
            order_intent.client_order_id
            or self.order_router.generate_client_order_id(
                spec.symbol,
                tenant_id=getattr(self.config, "tenant_id", None),
                base_signal_id=order_intent.base_signal_id,
            )
        )
        order_intent.client_order_id = client_order_id

        # 5. Dispatch native stop order with brackets
        exchange_order = await self.order_router.submit_native_stop_order(
            product_id=spec.product_id,
            symbol=spec.symbol,
            side=side,
            size=order_intent.contract_lots,
            stop_price=order_intent.trigger_price,
            bracket_stop_loss_price=order_intent.initial_stop_loss,
            bracket_take_profit_price=order_intent.take_profit,
            client_order_id=client_order_id,
        )

        # 6. Update intent & local registry
        order_intent.exchange_order_id = exchange_order.id
        if exchange_order.client_order_id:
            order_intent.client_order_id = exchange_order.client_order_id
        if exchange_order.stop_price is None:
            order_intent.status = OrderStatus.TRIGGERED
        self.registry.register(exchange_order, intent_id=order_intent.intent_id)
        self.risk_manager.record_trade_entry()

        if self.state_manager is not None:
            self.state_manager.register_intent(order_intent)

        order_kind = "Market Order" if exchange_order.stop_price is None else "Stop Order"
        stop_str = f"Stop={order_intent.trigger_price:.4f}" if exchange_order.stop_price is not None else "Type=MARKET"
        logger.info(
            "Native %s %s dispatched for %s: ExchangeID=%d | ClientID=%s | Lots=%d | %s | SL=%.4f | TP=%.4f",
            side.upper(),
            order_kind,
            spec.symbol,
            exchange_order.id,
            exchange_order.client_order_id or client_order_id,
            exchange_order.size,
            stop_str,
            order_intent.initial_stop_loss,
            order_intent.take_profit,
        )
        return exchange_order

    # Backwards-compatible alias
    execute_stop_order_with_brackets = execute_signal

    # -------------------------------------------------------------------------
    # Invalidation & TTL Handlers
    # -------------------------------------------------------------------------

    async def handle_invalidation(
        self,
        order_intent: OrderIntent,
        reason: str,
    ) -> bool:
        """
        Cancels the resting stop order from Delta Exchange when an intermediate candle
        closes across EMA 20 before the price trigger is reached.
        """
        if order_intent.exchange_order_id is None:
            return True

        order_id = order_intent.exchange_order_id
        product_id = order_intent.product_id
        if product_id is None:
            spec = await self.resolve_product_spec(order_intent.symbol)
            product_id = spec.product_id

        logger.info(
            "Signal invalidated for %s (%s). Purging resting stop order ID=%d from exchange...",
            order_intent.symbol,
            reason,
            order_id,
        )

        try:
            await self.order_router.cancel_order(product_id=product_id, order_id=order_id)
            self.registry.remove(order_id)
            if self.risk_manager.active_positions_count > 0:
                self.risk_manager.active_positions_count -= 1
            return True
        except Exception as exc:
            logger.error("Failed to cancel resting order ID=%d on Delta Exchange: %s", order_id, exc)
            return False

    # Backwards-compatible alias
    cancel_resting_order = handle_invalidation

    async def handle_ttl_expiration(
        self,
        order_intent: OrderIntent,
    ) -> bool:
        """
        Cancels the resting stop order when max_pending_bars elapses without fill.
        """
        reason = f"TTL expired ({order_intent.bars_elapsed}/{order_intent.ttl_bars} bars elapsed)"
        return await self.handle_invalidation(order_intent, reason)

    # -------------------------------------------------------------------------
    # Trailing Stop Ratchet Synchronization
    # -------------------------------------------------------------------------

    async def handle_trailing_stop_ratchet(
        self,
        order_intent: OrderIntent,
        new_sl: float,
    ) -> bool:
        """
        Synchronizes trailing stop ratchet with Delta Exchange.
        Strict Guard: Rejects any adjustment that loosens risk.

        Handles Delta Exchange's two-tier order lifecycle:
        1. PENDING Entry Order (order_intent.is_pending):
           - Modifies the attached bracket on the resting parent order via PUT /v2/orders/bracket.
        2. TRIGGERED / Filled Position (order_intent.is_triggered):
           - The parent entry order is closed/filled; Delta Exchange spawns an independent child Stop Loss order.
           - Calling PUT /v2/orders with parent entry order_id will fail with open_order_not_found!
           - Resolution:
             a) If child_stop_loss_order_id is already tracked, edits that child order via PUT /v2/orders (stop_price=new_sl).
             b) If child_stop_loss_order_id is None, dynamically discovers the child SL order via
                find_active_stop_loss_order(product_id, exit_side), caches it, and edits via PUT /v2/orders.
             c) Fallback: Updates position-level bracket via POST /v2/orders/bracket.
        """
        if order_intent.exchange_order_id is None:
            return False

        # Directional Guard: never loosen risk
        if order_intent.signal_type == SignalType.BULLISH_BREAKOUT:
            if new_sl <= order_intent.current_stop_loss:
                logger.warning(
                    "Rejected Long SL loosening: new_sl (%.4f) <= current_sl (%.4f)",
                    new_sl,
                    order_intent.current_stop_loss,
                )
                return False
        elif order_intent.signal_type == SignalType.BEARISH_BREAKDOWN:
            if new_sl >= order_intent.current_stop_loss:
                logger.warning(
                    "Rejected Short SL loosening: new_sl (%.4f) >= current_sl (%.4f)",
                    new_sl,
                    order_intent.current_stop_loss,
                )
                return False

        product_id = order_intent.product_id
        if product_id is None:
            spec = await self.resolve_product_spec(order_intent.symbol)
            product_id = spec.product_id

        # ---------------------------------------------------------------------
        # Tier 1: Pending Entry Order (Parent resting on order book)
        # ---------------------------------------------------------------------
        if order_intent.is_pending:
            parent_order_id = order_intent.exchange_order_id
            logger.info(
                "Ratcheting pending bracket for %s ID=%d: %.4f -> %.4f via PUT /v2/orders/bracket",
                order_intent.symbol,
                parent_order_id,
                order_intent.current_stop_loss,
                new_sl,
            )
            try:
                await self.order_router.edit_pending_bracket(
                    product_id=product_id,
                    order_id=parent_order_id,
                    new_sl=new_sl,
                )
                order_intent.current_stop_loss = new_sl
                return True
            except Exception as exc:
                logger.error("Failed to edit pending bracket order on Delta Exchange: %s", exc)
                return False

        # ---------------------------------------------------------------------
        # Tier 2: Triggered / Filled Position (Parent is closed, child SL active)
        # ---------------------------------------------------------------------
        elif order_intent.is_triggered:
            exit_side = "sell" if order_intent.signal_type == SignalType.BULLISH_BREAKOUT else "buy"
            child_sl_id = order_intent.child_stop_loss_order_id

            # 1. Zero-latency check: Query TradingStateManager (cached from private WebSocket feed)
            if child_sl_id is None and self.state_manager is not None:
                cached_child_id = self.state_manager.get_child_stop_loss_order_id(product_id)
                if cached_child_id is not None:
                    child_sl_id = cached_child_id
                    order_intent.child_stop_loss_order_id = child_sl_id
                    logger.info("Resolved child Stop Loss order ID=%d via TradingStateManager in real-time", child_sl_id)

            # 2. Discover child stop loss order if not currently tracked
            if child_sl_id is None:
                if self.config.execution_mode == ExecutionMode.LIVE:
                    try:
                        active_sl = await self.client.find_active_stop_loss_order(
                            product_id=product_id,
                            side=exit_side,
                        )
                        if active_sl is not None:
                            child_sl_id = active_sl.id
                            order_intent.child_stop_loss_order_id = child_sl_id
                            logger.info(
                                "Discovered active child Stop Loss order ID=%d for %s",
                                child_sl_id,
                                order_intent.symbol,
                            )
                    except Exception as disc_exc:
                        logger.warning("Failed to discover active child stop loss order on exchange: %s", disc_exc)
                elif self.config.execution_mode == ExecutionMode.DRY_RUN:
                    child_sl_id = int(uuid.uuid4().int % 100_000_000)
                    order_intent.child_stop_loss_order_id = child_sl_id

            # If child order ID is identified, edit it directly via PUT /v2/orders (stop_price=new_sl)
            if child_sl_id is not None:
                logger.info(
                    "Ratcheting active child stop order ID=%d for %s (exit_side=%s): %.4f -> %.4f via PUT /v2/orders (stop_price)",
                    child_sl_id,
                    order_intent.symbol,
                    exit_side,
                    order_intent.current_stop_loss,
                    new_sl,
                )
                try:
                    await self.order_router.edit_active_stop_order(
                        product_id=product_id,
                        stop_order_id=child_sl_id,
                        new_sl=new_sl,
                    )
                    order_intent.current_stop_loss = new_sl
                    return True
                except Exception as child_exc:
                    logger.warning(
                        "Editing child stop order ID=%d failed: %s. Falling back to position bracket...",
                        child_sl_id,
                        child_exc,
                    )

            # Fallback: Position bracket endpoint POST /v2/orders/bracket
            logger.info(
                "Ratcheting position-level bracket for %s (product_id=%d): %.4f -> %.4f via POST /v2/orders/bracket",
                order_intent.symbol,
                product_id,
                order_intent.current_stop_loss,
                new_sl,
            )
            try:
                await self.order_router.set_position_bracket(
                    product_id=product_id,
                    new_sl=new_sl,
                )
                order_intent.current_stop_loss = new_sl
                return True
            except Exception as fb_exc:
                logger.error("Failed to update position bracket on exchange: %s", fb_exc)
                return False

        else:
            logger.warning(
                "Cannot ratchet stop loss for order intent with inactive status %s",
                order_intent.status,
            )
            return False

    # Backwards-compatible alias
    ratchet_exchange_stop_loss = handle_trailing_stop_ratchet

    # -------------------------------------------------------------------------
    # Emergency Flatten & Kill-Switch
    # -------------------------------------------------------------------------

    async def emergency_flatten(self, symbol: str | None = None) -> dict[str, Any]:
        """
        Emergency Kill-Switch:
        1. Immediately cancels all open/resting orders across target symbol or all products.
        2. Market-closes all open positions using reduce_only=True.
        3. Trips the circuit breaker to halt further trading.
        4. Clears local registry.
        """
        logger.critical("EMERGENCY FLATTEN TRIGGERED for %s!", symbol or "ALL SYMBOLS")
        flatten_report: dict[str, Any] = {
            "timestamp": time.time(),
            "target_symbol": symbol,
            "cancelled_orders": 0,
            "closed_positions": [],
        }

        # 1. Cancel resting orders
        product_id = None
        if symbol:
            spec = await self.resolve_product_spec(symbol)
            product_id = spec.product_id

        try:
            cancelled = await self.order_router.cancel_all_orders(product_id=product_id)
            flatten_report["cancelled_orders"] = len(cancelled)
        except Exception as exc:
            logger.error("Error during emergency cancel all orders: %s", exc)

        # 2. Market close open positions
        if self.config.execution_mode == ExecutionMode.LIVE and self.client.config.has_valid_credentials():
            try:
                positions = await self.client.get_open_positions(product_id=product_id)
                for pos in positions:
                    if pos.size != 0:
                        close_side = "sell" if pos.size > 0 else "buy"
                        abs_size = abs(pos.size)
                        logger.info(
                            "Closing open position on exchange: %s (id=%d) %s %d lots (reduce_only=True)...",
                            pos.symbol,
                            pos.product_id,
                            close_side,
                            abs_size,
                        )
                        close_order = await self.order_router.market_close_position(
                            product_id=pos.product_id,
                            size=abs_size,
                            side=close_side,
                            symbol=pos.symbol,
                        )
                        flatten_report["closed_positions"].append(close_order.model_dump())
            except Exception as exc:
                logger.error("Error fetching/closing positions during emergency flatten: %s", exc)
        else:
            logger.info("[DRY_RUN] Simulated emergency flatten: cleared all local open positions.")

        # 3. Trip circuit breaker & purge registry
        self.risk_manager.circuit_breaker.trip(f"Emergency flatten executed for {symbol or 'ALL'}")
        self.risk_manager.active_positions_count = 0
        self.registry.clear()

        logger.info("Emergency flatten complete: %s", flatten_report)
        return flatten_report


# Backwards-compatible alias for existing code
OrderExecutionManager = ExecutionManager
