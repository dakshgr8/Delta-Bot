"""
delta_bot.state
~~~~~~~~~~~~~~~

State Reconciliation and In-Memory Position Synchronization Engine (Phase 5).
Coordinates:
- Full Order Intent Lifecycle: OrderIntent -> Entry Stop Order -> Filled Position -> Active Child SL/TP Bracket Orders
- Fill Detection & Immediate Child Bracket Leg Binding (zero-latency trailing ratchets)
- Dual-Source Synchronization (Hybrid Push/Pull Architecture):
    * Primary: Real-time event-driven updates from Private WebSocket (orders, positions, user_trades)
    * Secondary: Periodic REST reconciliation polling (/v2/positions/margined, /v2/orders?state=open)
- Liquidation Proximity Guard: Alerts and auto-flattens when mark price comes within 1.5 * ATR of liquidation
- Ghost Position Isolation: Protects untracked exchange positions from unintended automated closure
"""

import asyncio
from collections.abc import Awaitable, Callable
import logging
import time
from typing import Any

from delta_bot.client import DeltaRestClient
from delta_bot.models import (
    GhostPosition,
    LiquidationAlert,
    OrderUpdate,
    PositionInfo,
    UserTrade,
)
from delta_bot.strategy import OrderIntent, OrderStatus, SignalType

logger = logging.getLogger("delta_bot.state")


def _symbols_match(sym1: str, sym2: str) -> bool:
    """Matches symbols with automatic USD <-> USDT alias tolerance."""
    s1 = sym1.strip().upper()
    s2 = sym2.strip().upper()
    if s1 == s2:
        return True
    if s1.endswith("USDT") and s1[:-4] + "USD" == s2:
        return True
    if s1.endswith("USD") and s1[:-3] + "USDT" == s2:
        return True
    return False


class TradingStateManager:
    """
    Central in-memory trading state machine for Delta-Bot.
    Reconciles private WebSocket push events with periodic REST pull snapshots to guarantee
    zero-latency trailing stop ratchets and prevent execution state drift.
    """

    def __init__(
        self,
        client: DeltaRestClient,
        execution_manager: Any | None = None,
        proximity_atr_mult: float = 1.5,
        poll_interval_sec: float = 30.0,
    ) -> None:
        self.client: DeltaRestClient = client
        self.execution_manager: Any | None = execution_manager
        self.risk_manager: Any | None = None
        self.proximity_atr_mult: float = proximity_atr_mult
        self.poll_interval_sec: float = poll_interval_sec

        # In-Memory State Registry
        self._intents_by_id: dict[str, OrderIntent] = {}
        self._intents_by_exchange_id: dict[int, OrderIntent] = {}
        self._intents_by_child_sl_id: dict[int, OrderIntent] = {}
        self._intents_by_child_tp_id: dict[int, OrderIntent] = {}
        self._open_positions: dict[int, PositionInfo] = {}  # product_id -> PositionInfo
        self._ghost_positions: dict[int, GhostPosition] = {}  # product_id -> GhostPosition

        # Real-time Market Context Cache for Liquidation Guard
        self._cached_mark_prices: dict[str, float] = {}  # symbol -> mark_price
        self._cached_atrs: dict[str, float] = {}  # symbol -> latest atr
        self._cached_market_timestamps: dict[str, float] = {}  # symbol -> epoch timestamp
        self.is_reconciling: bool = False

        # Background Periodic Sync Worker
        self._reconcile_task: asyncio.Task[None] | None = None
        self._running: bool = False

        # Event Hooks
        self._fill_callbacks: list[Callable[[OrderIntent, float, int], Any]] = []
        self._liquidation_callbacks: list[Callable[[LiquidationAlert], Any]] = []
        self._ghost_callbacks: list[Callable[[GhostPosition], Any]] = []

    def get_risk_manager(self) -> Any | None:
        """Returns the active RiskManager instance whether via direct attachment or execution_manager."""
        if self.risk_manager is not None:
            return self.risk_manager
        if self.execution_manager and getattr(self.execution_manager, "risk_manager", None):
            return self.execution_manager.risk_manager
        return None

    # -------------------------------------------------------------------------
    # Event Hook Registration
    # -------------------------------------------------------------------------

    def on_fill(self, cb: Callable[[OrderIntent, float, int], Any]) -> None:
        """Registers a listener called when an entry order fills (intent, fill_price, filled_size)."""
        self._fill_callbacks.append(cb)

    def on_liquidation_alert(self, cb: Callable[[LiquidationAlert], Any]) -> None:
        """Registers a listener called when a position is within the liquidation safety margin."""
        self._liquidation_callbacks.append(cb)

    def on_ghost_position(self, cb: Callable[[GhostPosition], Any]) -> None:
        """Registers a listener called when an untracked external position is discovered."""
        self._ghost_callbacks.append(cb)

    # -------------------------------------------------------------------------
    # Intent Registration & Lookup
    # -------------------------------------------------------------------------

    def register_intent(self, intent: OrderIntent) -> None:
        """
        Indexes an OrderIntent in local memory.
        Maps by intent_id, exchange_order_id, and any existing child bracket IDs.
        """
        self._intents_by_id[intent.intent_id] = intent
        if intent.exchange_order_id is not None:
            self._intents_by_exchange_id[intent.exchange_order_id] = intent
        if intent.child_stop_loss_order_id is not None:
            self._intents_by_child_sl_id[intent.child_stop_loss_order_id] = intent
        if intent.child_take_profit_order_id is not None:
            self._intents_by_child_tp_id[intent.child_take_profit_order_id] = intent

        logger.debug(
            "Registered OrderIntent [%s] in TradingStateManager (ExchangeID=%s, Status=%s)",
            intent.intent_id,
            intent.exchange_order_id,
            intent.status.value,
        )

    def bind_exchange_order_id(self, intent_id: str, exchange_order_id: int) -> None:
        """Binds the parent entry order ID to the tracked intent."""
        intent = self._intents_by_id.get(intent_id)
        if intent:
            intent.exchange_order_id = exchange_order_id
            self._intents_by_exchange_id[exchange_order_id] = intent

    def get_intent(self, intent_id: str) -> OrderIntent | None:
        """Retrieves an OrderIntent by intent_id."""
        return self._intents_by_id.get(intent_id)

    def get_intent_by_exchange_id(self, exchange_order_id: int) -> OrderIntent | None:
        """Retrieves an OrderIntent by parent entry exchange order ID."""
        return self._intents_by_exchange_id.get(exchange_order_id)

    def get_active_intent_for_product(self, product_id: int) -> OrderIntent | None:
        """Retrieves the active (PENDING or TRIGGERED) OrderIntent for a product."""
        for intent in self._intents_by_id.values():
            if intent.product_id == product_id and intent.is_active:
                return intent
        return None

    def get_child_stop_loss_order_id(self, product_id: int) -> int | None:
        """
        Returns the cached child Stop Loss order ID for an active position.
        Crucial for zero-latency trailing stop ratchets without REST round-trips.
        """
        intent = self.get_active_intent_for_product(product_id)
        if intent and intent.child_stop_loss_order_id is not None:
            return intent.child_stop_loss_order_id
        return None

    def get_position(self, product_id: int) -> PositionInfo | None:
        """Retrieves cached PositionInfo for a product ID."""
        return self._open_positions.get(product_id)

    def get_all_positions(self) -> list[PositionInfo]:
        """Returns all open positions tracked in memory."""
        return [p for p in self._open_positions.values() if p.size != 0]

    def get_ghost_positions(self) -> list[GhostPosition]:
        """Returns all untracked external positions discovered on the exchange."""
        return list(self._ghost_positions.values())

    # -------------------------------------------------------------------------
    # Market Context Telemetry (for Liquidation Guard)
    # -------------------------------------------------------------------------

    async def update_market_context(
        self,
        symbol: str,
        mark_price: float,
        atr: float | None = None,
        high: float | None = None,
        low: float | None = None,
        **kwargs: Any,
    ) -> LiquidationAlert | None:
        """Updates real-time mark price and ATR for liquidation proximity evaluation."""
        clean_symbol = symbol.strip().upper()
        self._cached_mark_prices[clean_symbol] = mark_price
        now_ts = time.time()
        self._cached_market_timestamps[clean_symbol] = now_ts
        if atr is not None and atr > 0:
            self._cached_atrs[clean_symbol] = atr

        # Cross-cache USD/USDT alias
        alt_sym = None
        if clean_symbol.endswith("USD") and not clean_symbol.endswith("USDT"):
            alt_sym = clean_symbol[:-3] + "USDT"
        elif clean_symbol.endswith("USDT"):
            alt_sym = clean_symbol[:-4] + "USD"

        if alt_sym:
            self._cached_mark_prices[alt_sym] = mark_price
            self._cached_market_timestamps[alt_sym] = now_ts
            if atr is not None and atr > 0:
                self._cached_atrs[alt_sym] = atr

        # Run liquidation proximity evaluation
        return await self.check_liquidation_proximity(clean_symbol)

    def get_market_context_age(self, symbol: str) -> float | None:
        """Returns age in seconds of the latest market context / ticker update for a symbol."""
        clean_symbol = symbol.strip().upper()
        ts = self._cached_market_timestamps.get(clean_symbol)
        if ts is None:
            return None
        return time.time() - ts

    # -------------------------------------------------------------------------
    # WebSocket Event Dispatchers (Push Pipeline)
    # -------------------------------------------------------------------------

    async def handle_order_update(self, update: OrderUpdate | dict[str, Any]) -> None:
        """
        Processes real-time order state events streamed from 'orders' private channel:
        1. Identifies entry STOP order fills -> Transitions OrderIntent to TRIGGERED
        2. Captures spawned child Stop Loss / Take Profit order IDs -> Binds to OrderIntent
        3. Handles cancellations / rejections
        """
        if isinstance(update, dict):
            update = OrderUpdate.from_api_dict(update)

        order_id = update.id
        logger.debug(
            "StateEngine: handle_order_update ID=%d | %s | State=%s | StopPx=%s | SL=%s",
            order_id,
            update.symbol,
            update.state,
            update.stop_price,
            update.bracket_stop_loss_price,
        )

        # 1. Check if this is an update on a known parent entry order
        intent = self._intents_by_exchange_id.get(order_id)
        if intent is not None:
            if update.is_closed and intent.is_pending:
                fill_px = update.stop_price or update.limit_price or intent.trigger_price
                fill_size = update.size
                intent.record_fill(fill_price=fill_px, filled_size=fill_size, timestamp=update.created_at)
                logger.info(
                    "OrderIntent [%s] (%s) transitioned to TRIGGERED: Filled %d lots at %.4f",
                    intent.intent_id,
                    intent.symbol,
                    fill_size,
                    fill_px,
                )
                await self._emit_fill(intent, fill_px, fill_size)

            elif update.is_cancelled and intent.is_pending:
                intent.cancel(f"Exchange order {update.state}: ID={order_id}")
                logger.info("OrderIntent [%s] cancelled on exchange: %s", intent.intent_id, update.state)
                rm = self.get_risk_manager()
                if rm is not None and rm.active_positions_count > 0:
                    rm.active_positions_count -= 1
            return

        # 2. Check if this is an update on an already-bound child Stop Loss or Take Profit order
        intent_sl = self._intents_by_child_sl_id.get(order_id)
        intent_tp = self._intents_by_child_tp_id.get(order_id)
        target_intent = intent_sl or intent_tp

        if target_intent is not None and target_intent.is_active:
            if update.is_closed:
                status = OrderStatus.CLOSED_SL if target_intent == intent_sl else OrderStatus.CLOSED_TP
                exit_px = update.stop_price or update.limit_price or (target_intent.current_stop_loss if status == OrderStatus.CLOSED_SL else target_intent.take_profit)
                target_intent.close_position(status, exit_px, update.created_at)
                logger.info(
                    "OrderIntent [%s] (%s) closed via child bracket fill: %s at %.4f",
                    target_intent.intent_id,
                    target_intent.symbol,
                    status.value,
                    exit_px,
                )
                if not getattr(target_intent, "is_exit_recorded", False):
                    target_intent.is_exit_recorded = True
                    rm = self.get_risk_manager()
                    if rm is not None:
                        rm.record_trade_exit(
                            pnl=target_intent.realized_pnl or 0.0,
                            current_equity=0.0,
                            symbol=target_intent.symbol,
                        )
                return

        # 3. Check if this is a newly spawned child Stop Loss / Take Profit order
        # Delta creates child stop orders with stop_order_type="stop_loss_order" or stop_price set
        is_child_sl = (
            update.stop_order_type == "stop_loss_order"
            or (update.stop_price is not None and update.is_open)
        )

        matched_intent = self.get_active_intent_for_product(update.product_id)
        if matched_intent is not None:
            # Determine expected exit side: Long position exits via SELL stop; Short exits via BUY stop
            expected_exit_side = "sell" if matched_intent.signal_type == SignalType.BULLISH_BREAKOUT else "buy"

            if is_child_sl and update.side.lower() == expected_exit_side:
                matched_intent.child_stop_loss_order_id = update.id
                self._intents_by_child_sl_id[update.id] = matched_intent
                logger.info(
                    "Bound spawned child Stop Loss order ID=%d to OrderIntent [%s] (%s)",
                    update.id,
                    matched_intent.intent_id,
                    matched_intent.symbol,
                )
            elif update.bracket_take_profit_price is not None or (update.order_type in ("limit_order", "market_order") and update.side.lower() == expected_exit_side):
                matched_intent.child_take_profit_order_id = update.id
                self._intents_by_child_tp_id[update.id] = matched_intent
                logger.info(
                    "Bound spawned child Take Profit order ID=%d to OrderIntent [%s] (%s)",
                    update.id,
                    matched_intent.intent_id,
                    matched_intent.symbol,
                )

    async def handle_user_trade(self, trade: UserTrade | dict[str, Any]) -> None:
        """
        Processes real-time execution fills streamed from 'user_trades' private channel:
        Ensures immediate transition to TRIGGERED and records fill pricing telemetry.
        """
        if isinstance(trade, dict):
            trade = UserTrade.from_api_dict(trade)

        logger.debug(
            "StateEngine: handle_user_trade TradeID=%s | OrderID=%d | %s | Price=%.4f | Size=%d",
            trade.trade_id,
            trade.order_id,
            trade.symbol,
            trade.price,
            trade.size,
        )

        intent = self._intents_by_exchange_id.get(trade.order_id)
        if intent is not None and intent.is_pending:
            intent.record_fill(fill_price=trade.price, filled_size=trade.size, timestamp=trade.created_at)
            logger.info(
                "OrderIntent [%s] (%s) confirmed fill via user_trade: %d lots at %.4f",
                intent.intent_id,
                intent.symbol,
                trade.size,
                trade.price,
            )
            await self._emit_fill(intent, trade.price, trade.size)

    async def handle_position_update(self, position: PositionInfo | dict[str, Any]) -> None:
        """
        Processes real-time position updates streamed from 'positions' private channel:
        Updates cached position metrics and triggers the Liquidation Proximity Guard.
        """
        if isinstance(position, dict):
            position = PositionInfo.from_api_dict(position)

        if position.size == 0:
            self._open_positions.pop(position.product_id, None)
            tracked_intent = self.get_active_intent_for_product(position.product_id)
            if tracked_intent and tracked_intent.is_active:
                exit_px = position.mark_price or tracked_intent.current_stop_loss
                status = (
                    OrderStatus.CLOSED_TP
                    if (tracked_intent.take_profit and abs(exit_px - tracked_intent.take_profit) < abs(exit_px - tracked_intent.current_stop_loss))
                    else OrderStatus.CLOSED_SL
                )
                tracked_intent.close_position(status, exit_px, time.time())
                logger.info(
                    "OrderIntent [%s] (%s) closed via position size=0 update: %s at %.4f",
                    tracked_intent.intent_id,
                    tracked_intent.symbol,
                    status.value,
                    exit_px,
                )
                if not getattr(tracked_intent, "is_exit_recorded", False):
                    tracked_intent.is_exit_recorded = True
                    rm = self.get_risk_manager()
                    if rm is not None:
                        rm.record_trade_exit(
                            pnl=tracked_intent.realized_pnl or 0.0,
                            current_equity=0.0,
                            symbol=tracked_intent.symbol,
                        )
        else:
            self._open_positions[position.product_id] = position
            if position.symbol and position.mark_price > 0:
                self._cached_mark_prices[position.symbol] = position.mark_price

        logger.debug(
            "StateEngine: Position cached for %s (id=%d): Size=%d, Entry=%.4f, Mark=%.4f, Liq=%s",
            position.symbol,
            position.product_id,
            position.size,
            position.entry_price,
            position.mark_price,
            position.liquidation_price,
        )

        # Evaluate liquidation proximity
        await self.check_liquidation_proximity(position.symbol)

    # -------------------------------------------------------------------------
    # Liquidation Watcher & Proximity Guard
    # -------------------------------------------------------------------------

    async def check_liquidation_proximity(self, symbol: str) -> LiquidationAlert | None:
        """
        Liquidation Proximity Guard:
        Continuously computes distance between current mark price and liquidation_price.
        If mark price comes within 1.5 * ATR of liquidation_price:
            1. Emits critical safety alert.
            2. Automatically triggers emergency_flatten() on ExecutionManager.
        """
        clean_symbol = symbol.strip().upper()
        # Find product position
        pos = None
        for p in self._open_positions.values():
            if _symbols_match(p.symbol, clean_symbol):
                pos = p
                break

        if pos is None or pos.size == 0 or pos.liquidation_price is None:
            return None

        mark_px = self._cached_mark_prices.get(clean_symbol, pos.mark_price)
        atr = self._cached_atrs.get(clean_symbol, 0.0)

        # Proximity threshold strictly requires verified market ATR
        if atr <= 0:
            return None

        distance = abs(mark_px - pos.liquidation_price)
        threshold_distance = self.proximity_atr_mult * atr

        if distance <= threshold_distance:
            alert = LiquidationAlert(
                symbol=clean_symbol,
                product_id=pos.product_id,
                mark_price=mark_px,
                liquidation_price=pos.liquidation_price,
                distance=distance,
                threshold_distance=threshold_distance,
                atr=atr,
                timestamp=time.time(),
            )
            logger.critical(
                "🚨 LIQUIDATION PROXIMITY ALERT for %s! Mark=%.4f is within %.4f of Liq=%.4f "
                "(Distance=%.4f <= %.1f*ATR=%.4f). TRIGGERING EMERGENCY FLATTEN!",
                clean_symbol,
                mark_px,
                distance,
                pos.liquidation_price,
                distance,
                self.proximity_atr_mult,
                threshold_distance,
            )

            # Trigger registered alert listeners
            for cb in self._liquidation_callbacks:
                try:
                    res = cb(alert)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception as exc:
                    logger.error("Error in liquidation alert callback: %s", exc)

            # Auto-execute emergency flatten if ExecutionManager is wired
            if self.execution_manager is not None:
                try:
                    logger.critical("Auto-executing emergency_flatten for %s via ExecutionManager...", clean_symbol)
                    await self.execution_manager.emergency_flatten(symbol=clean_symbol)
                except Exception as exc:
                    logger.error("Failed to execute emergency_flatten during liquidation alert: %s", exc)

            return alert

        return None

    # -------------------------------------------------------------------------
    # REST Fallback Reconciliation Engine (Pull Pipeline)
    # -------------------------------------------------------------------------

    async def reconcile_with_rest(self) -> dict[str, Any]:
        """
        Secondary Dual-Source Sync (Hybrid Push/Pull Architecture):
        Polls Delta Exchange REST endpoints (/v2/positions/margined and /v2/orders?state=open)
        to reconcile any discrepancies caused by dropped packets or missed socket frames.

        Tasks:
        1. Updates open positions and identifies Ghost Positions.
        2. Detects filled or cancelled entry orders that missed socket events.
        3. Discovers and binds resting child Stop Loss / Take Profit orders.
        """
        self.is_reconciling = True
        try:
            report: dict[str, Any] = {
                "timestamp": time.time(),
                "synced_positions": 0,
                "ghost_positions_found": 0,
                "unfilled_orders_checked": 0,
                "child_orders_bound": 0,
            }

            logger.debug("Starting periodic REST state reconciliation...")

            # 1. Fetch live open positions from REST
            try:
                positions = await self.client.get_open_positions()
                active_product_ids = set()

                for pos in positions:
                    if pos.size != 0:
                        self._open_positions[pos.product_id] = pos
                        active_product_ids.add(pos.product_id)
                        report["synced_positions"] += 1
                    else:
                        self._open_positions.pop(pos.product_id, None)

                # Prune positions that no longer exist on Delta
                for pid in list(self._open_positions.keys()):
                    if pid not in active_product_ids:
                        self._open_positions.pop(pid, None)
                        tracked_intent = self.get_active_intent_for_product(pid)
                        if tracked_intent and tracked_intent.is_active:
                            exit_px = tracked_intent.current_stop_loss or tracked_intent.fill_price or 0.0
                            status = OrderStatus.CLOSED_SL
                            tracked_intent.close_position(status, exit_px, time.time())
                            logger.info(
                                "REST reconciliation: closed orphaned intent [%s] for product %d",
                                tracked_intent.intent_id,
                                pid,
                            )
                            if not getattr(tracked_intent, "is_exit_recorded", False):
                                tracked_intent.is_exit_recorded = True
                                rm = self.get_risk_manager()
                                if rm is not None:
                                    rm.record_trade_exit(
                                        pnl=tracked_intent.realized_pnl or 0.0,
                                        current_equity=0.0,
                                        symbol=tracked_intent.symbol,
                                    )

                for pos in positions:
                    if pos.size != 0:
                        # Ghost Position Check: Is this position tracked by a local OrderIntent?
                        tracked_intent = self.get_active_intent_for_product(pos.product_id)
                        if tracked_intent is None:
                            # Untracked exchange position discovered!
                            ghost = GhostPosition(
                                product_id=pos.product_id,
                                symbol=pos.symbol,
                                size=pos.size,
                                entry_price=pos.entry_price,
                                liquidation_price=pos.liquidation_price,
                            )
                            self._ghost_positions[pos.product_id] = ghost
                            report["ghost_positions_found"] += 1
                            logger.critical(
                                "👻 GHOST POSITION DETECTED: %s (id=%d) has %d open lots at %.4f "
                                "not tracked by local bot memory! Tagged as EXTERNAL_UNTRACKED.",
                                pos.symbol,
                                pos.product_id,
                                pos.size,
                                pos.entry_price,
                            )
                            for cb in self._ghost_callbacks:
                                try:
                                    res = cb(ghost)
                                    if asyncio.iscoroutine(res):
                                        await res
                                except Exception as exc:
                                    logger.error("Error in ghost position callback: %s", exc)
                        else:
                            # If tracked intent is still marked PENDING, REST confirms it filled!
                            if tracked_intent.is_pending:
                                tracked_intent.record_fill(
                                    fill_price=pos.entry_price,
                                    filled_size=abs(pos.size),
                                )
                                logger.info(
                                    "REST reconciliation confirmed fill for OrderIntent [%s]: %d lots at %.4f",
                                    tracked_intent.intent_id,
                                    abs(pos.size),
                                    pos.entry_price,
                                )
                                await self._emit_fill(tracked_intent, pos.entry_price, abs(pos.size))

            except Exception as exc:
                logger.warning("REST position reconciliation error: %s", exc)

            # 2. Fetch live open orders from REST
            try:
                open_orders = await self.client.get_open_orders()
                open_order_map = {o.id: o for o in open_orders}
                report["unfilled_orders_checked"] = len(open_orders)

                # Reconcile pending intents: if parent order is missing from open_orders, check fill status
                for parent_order_id, intent in list(self._intents_by_exchange_id.items()):
                    if intent.is_pending and parent_order_id not in open_order_map:
                        # Parent order is no longer open! Check if position is open
                        pos = self._open_positions.get(intent.product_id or 0)
                        if pos and pos.size != 0:
                            intent.record_fill(fill_price=pos.entry_price, filled_size=abs(pos.size))
                            logger.info("REST sync reconciled filled parent order ID=%d for %s", parent_order_id, intent.symbol)
                        else:
                            intent.cancel(f"Order ID={parent_order_id} no longer present on Delta open order book")
                            logger.info("REST sync reconciled cancelled parent order ID=%d for %s", parent_order_id, intent.symbol)
                            rm = self.get_risk_manager()
                            if rm is not None and rm.active_positions_count > 0:
                                rm.active_positions_count -= 1

                # Discover and bind resting child Stop Loss orders for active triggered positions
                for intent in list(self._intents_by_id.values()):
                    if intent.is_triggered and intent.child_stop_loss_order_id is None:
                        exit_side = "sell" if intent.signal_type == SignalType.BULLISH_BREAKOUT else "buy"
                        for o in open_orders:
                            if (
                                o.product_id == intent.product_id
                                and o.side.lower() == exit_side
                                and o.stop_price is not None
                            ):
                                intent.child_stop_loss_order_id = o.id
                                self._intents_by_child_sl_id[o.id] = intent
                                report["child_orders_bound"] += 1
                                logger.info(
                                    "REST sync discovered and bound child Stop Loss order ID=%d for %s",
                                    o.id,
                                    intent.symbol,
                                )
                                break

                # 3. Position Protection Audit: Ensure NO open position is left naked without SL
                for pos in positions:
                    if pos.size != 0:
                        exit_side = "sell" if pos.size > 0 else "buy"
                        has_sl = any(
                            o.product_id == pos.product_id
                            and o.side.lower() == exit_side
                            and o.stop_price is not None
                            for o in open_orders
                        )
                        tracked_intent = self.get_active_intent_for_product(pos.product_id)
                        if not has_sl:
                            sl_px = tracked_intent.current_stop_loss if tracked_intent else None
                            tp_px = tracked_intent.take_profit if tracked_intent else None
                            spec = self.client.get_cached_contract_spec(pos.symbol)
                            if sl_px is None and pos.entry_price > 0:
                                raw_sl = pos.entry_price * (0.985 if pos.size > 0 else 1.015)
                                raw_tp = pos.entry_price * (1.03 if pos.size > 0 else 0.97)
                                sl_px = spec.round_price(raw_sl) if spec else round(raw_sl, 4)
                                tp_px = spec.round_price(raw_tp) if spec else round(raw_tp, 4)
                            elif spec:
                                if sl_px is not None:
                                    sl_px = spec.round_price(sl_px)
                                if tp_px is not None:
                                    tp_px = spec.round_price(tp_px)

                            if sl_px is not None:
                                logger.critical(
                                    "🚨 NAKED POSITION DETECTED! %s (id=%d, size=%d) has NO active Stop Loss! Attaching protection bracket...",
                                    pos.symbol,
                                    pos.product_id,
                                    pos.size,
                                )
                                try:
                                    await self.client.set_position_bracket(
                                        product_id=pos.product_id,
                                        bracket_stop_loss_price=sl_px,
                                        bracket_take_profit_price=tp_px,
                                    )
                                    report["naked_positions_repaired"] = report.get("naked_positions_repaired", 0) + 1
                                    logger.info(
                                        "🛡️ Repaired naked position for %s: Attached SL=%.4f, TP=%s",
                                        pos.symbol,
                                        sl_px,
                                        tp_px,
                                    )
                                except Exception as patch_exc:
                                    logger.warning(
                                        "set_position_bracket failed for %s: %s. Placing fallback native reduce-only child orders...",
                                        pos.symbol,
                                        patch_exc,
                                    )
                                    # Fallback 1: Native reduce-only child stop loss order
                                    try:
                                        sl_res = await self.client.place_order(
                                            product_id=pos.product_id,
                                            size=abs(pos.size),
                                            side=exit_side,
                                            order_type="market_order",
                                            stop_price=sl_px,
                                            stop_order_type="stop_loss_order",
                                            reduce_only=True,
                                            time_in_force="gtc",
                                            symbol=pos.symbol,
                                        )
                                        report["naked_positions_repaired"] = report.get("naked_positions_repaired", 0) + 1
                                        logger.info(
                                            "🛡️ Repaired naked position for %s via fallback child SL order #%d @ %.4f",
                                            pos.symbol,
                                            sl_res.id,
                                            sl_px,
                                        )
                                    except Exception as sl_err:
                                        logger.error("Failed to place fallback child SL for %s: %s", pos.symbol, sl_err)

                                    # Fallback 2: Native reduce-only child take profit order (if missing)
                                    has_tp = any(
                                        o.product_id == pos.product_id
                                        and o.side.lower() == exit_side
                                        and o.order_type == "limit_order"
                                        for o in open_orders
                                    )
                                    if not has_tp and tp_px is not None:
                                        try:
                                            tp_res = await self.client.place_order(
                                                product_id=pos.product_id,
                                                size=abs(pos.size),
                                                side=exit_side,
                                                order_type="limit_order",
                                                limit_price=tp_px,
                                                reduce_only=True,
                                                time_in_force="gtc",
                                                symbol=pos.symbol,
                                            )
                                            logger.info(
                                                "🛡️ Repaired naked position for %s via fallback child TP order #%d @ %.4f",
                                                pos.symbol,
                                                tp_res.id,
                                                tp_px,
                                            )
                                        except Exception as tp_err:
                                            logger.error("Failed to place fallback child TP for %s: %s", pos.symbol, tp_err)

            except Exception as exc:
                logger.warning("REST open orders reconciliation error: %s", exc)

            logger.debug("REST state reconciliation complete: %s", report)
            return report
        finally:
            self.is_reconciling = False

    # -------------------------------------------------------------------------
    # Background Periodic Worker Task
    # -------------------------------------------------------------------------

    async def start_periodic_reconciliation(self) -> None:
        """Launches the background polling task that runs reconcile_with_rest every 30s."""
        if self._running:
            return
        self._running = True
        logger.info("Starting background REST state reconciliation worker (interval: %.1fs)...", self.poll_interval_sec)
        self._reconcile_task = asyncio.create_task(self._reconcile_loop(), name="state_reconcile_loop")

    async def stop_periodic_reconciliation(self) -> None:
        """Stops the background reconciliation polling task."""
        self._running = False
        if self._reconcile_task and not self._reconcile_task.done():
            self._reconcile_task.cancel()
            try:
                await self._reconcile_task
            except asyncio.CancelledError:
                logger.debug("State reconciliation task cancelled successfully")
        logger.info("Stopped background REST state reconciliation worker.")

    async def _reconcile_loop(self) -> None:
        """Continuous polling loop executing reconciliation at regular intervals."""
        while self._running:
            try:
                await asyncio.sleep(self.poll_interval_sec)
                await self.reconcile_with_rest()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error during periodic reconciliation loop: %s", exc)

    async def _emit_fill(self, intent: OrderIntent, fill_price: float, filled_size: int) -> None:
        """Helper to invoke on_fill callbacks."""
        for cb in self._fill_callbacks:
            try:
                res = cb(intent, fill_price, filled_size)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as exc:
                logger.error("Error in on_fill callback: %s", exc)
