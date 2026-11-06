"""
delta_bot.paper
~~~~~~~~~~~~~~~

In-Memory Virtual Order Matching & Paper Trading Engine for Delta-Bot (Phase 6).
Provides:
- Virtual Collateral Wallet: Tracks initial balance, available margin, floating equity,
  realized/unrealized PnL, win rate, and high-watermark drawdown without risking real funds.
- Exchange-Native Stop & Bracket Order Simulation:
  Simulates fill triggers, child bracket order spawning, and orderbook depth slippage.
- Real-Time State Machine Integration:
  Dispatches synthetic OrderUpdate, UserTrade, and PositionInfo events into TradingStateManager
  so the execution, risk, and state pipelines operate identically between paper and live modes.
"""

from collections.abc import Callable
import logging
import time
from typing import Any
import uuid

from pydantic import BaseModel, ConfigDict, Field

from delta_bot.models import (
    ContractSpec,
    ExchangeOrder,
    OrderBook,
    OrderUpdate,
    PositionInfo,
    UserTrade,
)
from delta_bot.strategy import OrderIntent, OrderStatus, SignalType

logger = logging.getLogger("delta_bot.paper")


class VirtualWallet(BaseModel):
    """
    In-memory virtual collateral wallet for paper trading.
    Calculates floating equity, margin utilization, win rate, and max drawdown.
    """

    model_config = ConfigDict(frozen=False)

    initial_balance: float = Field(default=10000.0, description="Starting USD collateral")
    balance: float = Field(default=10000.0, description="Realized cash balance")
    blocked_margin: float = Field(default=0.0, description="Margin allocated to open positions & orders")
    realized_pnl: float = Field(default=0.0, description="Cumulative realized PnL")
    unrealized_pnl: float = Field(default=0.0, description="Floating floating PnL")
    peak_equity: float = Field(default=10000.0, description="High-watermark equity")
    max_drawdown_pct: float = Field(default=0.0, description="Maximum peak-to-trough drawdown %")
    total_trades: int = Field(default=0, description="Total completed round-trip trades")
    winning_trades: int = Field(default=0, description="Trades closed with positive PnL")
    losing_trades: int = Field(default=0, description="Trades closed with zero or negative PnL")

    @property
    def equity(self) -> float:
        """Current account equity: balance + unrealized floating PnL."""
        return self.balance + self.unrealized_pnl

    @property
    def available_balance(self) -> float:
        """Usable collateral margin remaining for new orders."""
        return max(0.0, self.balance - self.blocked_margin)

    @property
    def win_rate_pct(self) -> float:
        """Percentage of closed trades that were profitable."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100.0

    @property
    def net_roi_pct(self) -> float:
        """Return on investment percentage relative to initial balance."""
        if self.initial_balance <= 0:
            return 0.0
        return ((self.equity - self.initial_balance) / self.initial_balance) * 100.0

    def update_drawdown(self) -> None:
        """Updates peak equity and maximum drawdown percentage."""
        current_eq = self.equity
        if current_eq > self.peak_equity:
            self.peak_equity = current_eq

        if self.peak_equity > 0:
            dd = ((self.peak_equity - current_eq) / self.peak_equity) * 100.0
            if dd > self.max_drawdown_pct:
                self.max_drawdown_pct = dd


class VirtualOrder(BaseModel):
    """Resting stop order simulated in paper trading memory."""

    order_id: int
    client_order_id: str
    product_id: int
    symbol: str
    side: str
    size: int
    stop_price: float
    bracket_stop_loss_price: float | None = None
    bracket_take_profit_price: float | None = None
    leverage: int = 5
    blocked_margin: float = 0.0
    intent_id: str = ""
    status: str = "open"
    created_at: float = Field(default_factory=time.time)


class VirtualPosition(BaseModel):
    """Active open position simulated in paper trading memory."""

    product_id: int
    symbol: str
    side: str  # "buy" (long) or "sell" (short)
    size: int
    entry_price: float
    mark_price: float
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    child_stop_loss_order_id: int | None = None
    child_take_profit_order_id: int | None = None
    allocated_margin: float = 0.0
    unrealized_pnl: float = 0.0
    intent_id: str = ""
    opened_at: float = Field(default_factory=time.time)

    @property
    def is_long(self) -> bool:
        return self.side.lower() == "buy"


class PaperExchangeSimulator:
    """
    Virtual order matching simulator for paper trading.
    Provides exchange-like mechanics with realistic slippage, fee deduction,
    and automatic state dispatching.
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        default_slippage_bps: float = 1.0,
        state_manager: Any | None = None,
    ) -> None:
        self.wallet: VirtualWallet = VirtualWallet(
            initial_balance=initial_balance,
            balance=initial_balance,
            peak_equity=initial_balance,
        )
        self.default_slippage_bps: float = default_slippage_bps
        self.state_manager: Any | None = state_manager

        # Simulator State
        self._pending_orders: dict[int, VirtualOrder] = {}  # order_id -> VirtualOrder
        self._open_positions: dict[int, VirtualPosition] = {}  # product_id -> VirtualPosition
        self._specs_by_product_id: dict[int, ContractSpec] = {}
        self._next_id: int = 10000000

        # Optional event callbacks
        self._on_fill_cb: Callable[[OrderUpdate, UserTrade], Any] | None = None

    def set_state_manager(self, state_manager: Any) -> None:
        """Attaches TradingStateManager for real-time simulated push events."""
        self.state_manager = state_manager

    def register_contract_spec(self, spec: ContractSpec) -> None:
        """Caches contract specification for lot-to-asset conversions and fees."""
        self._specs_by_product_id[spec.product_id] = spec

    def _generate_id(self) -> int:
        self._next_id += 1
        return self._next_id

    # -------------------------------------------------------------------------
    # Order Placement & Cancellation
    # -------------------------------------------------------------------------

    def submit_stop_order(
        self,
        product_id: int,
        symbol: str,
        side: str,
        size: int,
        stop_price: float,
        bracket_stop_loss_price: float | None = None,
        bracket_take_profit_price: float | None = None,
        client_order_id: str = "",
        leverage: int = 5,
        intent_id: str = "",
    ) -> ExchangeOrder:
        """
        Places a simulated native stop order with brackets into paper memory.
        Calculates and reserves initial margin from virtual wallet.
        """
        spec = self._specs_by_product_id.get(product_id)
        contract_val = spec.contract_value if spec else 0.001

        # Initial margin requirement = (lots * contract_value * price) / leverage
        notional = size * contract_val * stop_price
        req_margin = notional / max(1, leverage)

        if req_margin > self.wallet.available_balance:
            logger.warning(
                "[PAPER] Insufficient margin: Required=$%.2f > Available=$%.2f",
                req_margin,
                self.wallet.available_balance,
            )
            # Allow order placement in paper mode with clamped size or warning

        # Reserve margin
        self.wallet.blocked_margin += req_margin
        order_id = self._generate_id()
        cid = client_order_id or f"PAPER_{symbol}_{int(time.time()*1000)}"

        v_order = VirtualOrder(
            order_id=order_id,
            client_order_id=cid,
            product_id=product_id,
            symbol=symbol,
            side=side.lower(),
            size=size,
            stop_price=stop_price,
            bracket_stop_loss_price=bracket_stop_loss_price,
            bracket_take_profit_price=bracket_take_profit_price,
            leverage=leverage,
            blocked_margin=req_margin,
            intent_id=intent_id,
        )
        self._pending_orders[order_id] = v_order

        logger.info(
            "[PAPER] Placed Stop Order: ID=%d | %s %s %d lots | StopPx=%.4f | SL=%s | TP=%s | Margin=$%.2f",
            order_id,
            symbol,
            side.upper(),
            size,
            stop_price,
            bracket_stop_loss_price,
            bracket_take_profit_price,
            req_margin,
        )

        return ExchangeOrder(
            id=order_id,
            client_order_id=cid,
            product_id=product_id,
            symbol=symbol,
            side=side.lower(),
            order_type="market_order",
            size=size,
            unfilled_size=size,
            stop_price=stop_price,
            stop_order_type="stop_loss_order",
            stop_trigger_method="mark_price",
            bracket_stop_loss_price=bracket_stop_loss_price,
            bracket_take_profit_price=bracket_take_profit_price,
            state="open",
            created_at=time.time(),
        )

    def cancel_order(self, product_id: int, order_id: int) -> dict[str, Any]:
        """Cancels a resting paper stop order and releases blocked margin."""
        v_order = self._pending_orders.pop(order_id, None)
        if v_order:
            self.wallet.blocked_margin = max(0.0, self.wallet.blocked_margin - v_order.blocked_margin)
            logger.info("[PAPER] Cancelled resting order ID=%d for %s (Released $%.2f margin)",
                        order_id, v_order.symbol, v_order.blocked_margin)
            return {"status": "cancelled", "id": order_id, "product_id": product_id}
        return {"status": "not_found", "id": order_id, "product_id": product_id}

    def cancel_all_orders(self, product_id: int | None = None) -> list[dict[str, Any]]:
        """Cancels all pending orders across target product or all products."""
        cancelled = []
        for oid, o in list(self._pending_orders.items()):
            if product_id is None or o.product_id == product_id:
                cancelled.append(self.cancel_order(o.product_id, oid))
        return cancelled

    def edit_order_stop_loss(self, product_id: int, order_id: int, new_sl: float) -> bool:
        """
        Modifies Stop Loss price on a resting order or an active position.
        """
        # 1. Check resting orders
        if order_id in self._pending_orders:
            self._pending_orders[order_id].bracket_stop_loss_price = new_sl
            logger.info("[PAPER] Modified bracket SL on resting order ID=%d to %.4f", order_id, new_sl)
            return True

        # 2. Check active positions
        for pos in self._open_positions.values():
            if pos.product_id == product_id:
                pos.stop_loss_price = new_sl
                logger.info("[PAPER] Ratcheted position SL for %s to %.4f", pos.symbol, new_sl)
                return True

        return False

    # -------------------------------------------------------------------------
    # Matching Engine & Execution Simulator
    # -------------------------------------------------------------------------

    async def on_tick(
        self,
        symbol: str,
        mark_price: float,
        high: float | None = None,
        low: float | None = None,
        orderbook: OrderBook | None = None,
    ) -> list[dict[str, Any]]:
        """
        Evaluates ticks against resting stop orders and active positions:
        1. Triggers pending Stop Orders -> fills position & spawns child brackets
        2. Evaluates active positions -> executes Take Profit or Stop Loss exits
        3. Updates floating unrealized PnL and high-watermark drawdown
        """
        clean_symbol = symbol.strip().upper()
        high_px = high if high is not None else mark_price
        low_px = low if low is not None else mark_price
        executions: list[dict[str, Any]] = []

        # ---------------------------------------------------------------------
        # 1. Evaluate Resting Stop Orders
        # ---------------------------------------------------------------------
        for oid, o in list(self._pending_orders.items()):
            if o.symbol.upper() != clean_symbol:
                continue

            triggered = False
            if o.side == "buy" and (mark_price >= o.stop_price or high_px >= o.stop_price):
                triggered = True
            elif o.side == "sell" and (mark_price <= o.stop_price or low_px <= o.stop_price):
                triggered = True

            if triggered:
                # Compute execution price with slippage
                slippage_bps = self.default_slippage_bps
                if orderbook is not None and orderbook.is_valid:
                    try:
                        _, ob_bps = orderbook.estimate_slippage(o.side, o.size)
                        slippage_bps = max(slippage_bps, ob_bps)
                    except Exception as ob_err:
                        logger.debug("Failed orderbook slippage calculation: %s", ob_err)

                slippage_factor = 1.0 + (slippage_bps / 10000.0) if o.side == "buy" else 1.0 - (slippage_bps / 10000.0)
                fill_price = o.stop_price * slippage_factor

                # Execute fill
                del self._pending_orders[oid]
                child_sl_id = self._generate_id()
                child_tp_id = self._generate_id()

                # Deduct taker fee
                spec = self._specs_by_product_id.get(o.product_id)
                fee_rate = spec.taker_commission_rate if spec else 0.0001
                contract_val = spec.contract_value if spec else 0.001
                fee_usd = fill_price * o.size * contract_val * fee_rate
                self.wallet.balance -= fee_usd

                # Create open position
                pos = VirtualPosition(
                    product_id=o.product_id,
                    symbol=o.symbol,
                    side=o.side,
                    size=o.size,
                    entry_price=fill_price,
                    mark_price=mark_price,
                    stop_loss_price=o.bracket_stop_loss_price,
                    take_profit_price=o.bracket_take_profit_price,
                    child_stop_loss_order_id=child_sl_id,
                    child_take_profit_order_id=child_tp_id,
                    allocated_margin=o.blocked_margin,
                    intent_id=o.intent_id,
                )
                self._open_positions[o.product_id] = pos

                logger.info(
                    "[PAPER] STOP ORDER TRIGGERED & FILLED: ID=%d | %s %s %d lots at %.4f (Slippage: %.2fbps, Fee: $%.4f)",
                    oid,
                    o.symbol,
                    o.side.upper(),
                    o.size,
                    fill_price,
                    slippage_bps,
                    fee_usd,
                )

                # Prepare event updates
                order_update = OrderUpdate(
                    id=oid,
                    client_order_id=o.client_order_id,
                    product_id=o.product_id,
                    symbol=o.symbol,
                    side=o.side,
                    order_type="market_order",
                    state="closed",
                    size=o.size,
                    stop_price=fill_price,
                )
                user_trade = UserTrade(
                    trade_id=f"PTR_{oid}",
                    order_id=oid,
                    product_id=o.product_id,
                    symbol=o.symbol,
                    side=o.side,
                    size=o.size,
                    price=fill_price,
                    commission=fee_usd,
                )
                child_sl_update = OrderUpdate(
                    id=child_sl_id,
                    product_id=o.product_id,
                    symbol=o.symbol,
                    side="sell" if o.side == "buy" else "buy",
                    order_type="market_order",
                    stop_order_type="stop_loss_order",
                    stop_price=o.bracket_stop_loss_price or (fill_price * 0.98),
                    state="open",
                    size=o.size,
                )

                # Push to TradingStateManager in real time
                if self.state_manager is not None:
                    await self.state_manager.handle_order_update(order_update)
                    await self.state_manager.handle_user_trade(user_trade)
                    await self.state_manager.handle_order_update(child_sl_update)

                executions.append({"event": "fill", "order_id": oid, "fill_price": fill_price, "size": o.size})

        # ---------------------------------------------------------------------
        # 2. Evaluate Active Positions (Take Profit & Stop Loss)
        # ---------------------------------------------------------------------
        for pid, pos in list(self._open_positions.items()):
            if pos.symbol.upper() != clean_symbol:
                continue

            pos.mark_price = mark_price
            spec = self._specs_by_product_id.get(pid)
            contract_val = spec.contract_value if spec else 0.001

            # Floating PnL calculation
            if pos.is_long:
                pos.unrealized_pnl = (mark_price - pos.entry_price) * pos.size * contract_val
            else:
                pos.unrealized_pnl = (pos.entry_price - mark_price) * pos.size * contract_val

            closed = False
            exit_price = mark_price
            exit_reason = ""

            # Check TP
            if pos.take_profit_price is not None:
                if pos.is_long and (mark_price >= pos.take_profit_price or high_px >= pos.take_profit_price):
                    closed = True
                    exit_price = pos.take_profit_price
                    exit_reason = "TAKE_PROFIT"
                elif not pos.is_long and (mark_price <= pos.take_profit_price or low_px <= pos.take_profit_price):
                    closed = True
                    exit_price = pos.take_profit_price
                    exit_reason = "TAKE_PROFIT"

            # Check SL
            if not closed and pos.stop_loss_price is not None:
                if pos.is_long and (mark_price <= pos.stop_loss_price or low_px <= pos.stop_loss_price):
                    closed = True
                    exit_price = pos.stop_loss_price
                    exit_reason = "STOP_LOSS"
                elif not pos.is_long and (mark_price >= pos.stop_loss_price or high_px >= pos.stop_loss_price):
                    closed = True
                    exit_price = pos.stop_loss_price
                    exit_reason = "STOP_LOSS"

            if closed:
                del self._open_positions[pid]
                # Calculate final trade PnL
                if pos.is_long:
                    trade_pnl = (exit_price - pos.entry_price) * pos.size * contract_val
                else:
                    trade_pnl = (pos.entry_price - exit_price) * pos.size * contract_val

                # Deduct exit fee
                fee_rate = spec.taker_commission_rate if spec else 0.0001
                exit_fee = exit_price * pos.size * contract_val * fee_rate
                net_pnl = trade_pnl - exit_fee

                # Update wallet metrics
                self.wallet.realized_pnl += net_pnl
                self.wallet.balance += net_pnl
                self.wallet.blocked_margin = max(0.0, self.wallet.blocked_margin - pos.allocated_margin)
                self.wallet.total_trades += 1
                if net_pnl > 0:
                    self.wallet.winning_trades += 1
                else:
                    self.wallet.losing_trades += 1

                logger.info(
                    "[PAPER] POSITION CLOSED (%s): %s %s %d lots | Entry=%.4f -> Exit=%.4f | Net PnL=$%.2f | Balance=$%.2f",
                    exit_reason,
                    pos.symbol,
                    pos.side.upper(),
                    pos.size,
                    pos.entry_price,
                    exit_price,
                    net_pnl,
                    self.wallet.balance,
                )

                # Emit close to state manager
                if self.state_manager is not None:
                    # Clear position in state manager
                    closed_pos_info = PositionInfo(
                        product_id=pid,
                        symbol=pos.symbol,
                        size=0,
                        entry_price=0.0,
                        mark_price=exit_price,
                    )
                    await self.state_manager.handle_position_update(closed_pos_info)

                executions.append({
                    "event": "position_closed",
                    "product_id": pid,
                    "symbol": pos.symbol,
                    "reason": exit_reason,
                    "pnl": net_pnl,
                })

        # ---------------------------------------------------------------------
        # 3. Update Overall Portfolio Floating PnL & High Watermark Drawdown
        # ---------------------------------------------------------------------
        total_upnl = sum(p.unrealized_pnl for p in self._open_positions.values())
        self.wallet.unrealized_pnl = total_upnl
        self.wallet.update_drawdown()

        return executions

    def close_position(self, product_id: int) -> dict[str, Any] | None:
        """Market closes a specific active open position in paper trading memory."""
        pos = self._open_positions.pop(product_id, None)
        if pos is None:
            return None

        self.wallet.blocked_margin = max(0.0, self.wallet.blocked_margin - pos.allocated_margin)
        self.wallet.balance += pos.unrealized_pnl
        self.wallet.realized_pnl += pos.unrealized_pnl
        self.wallet.total_trades += 1
        if pos.unrealized_pnl > 0:
            self.wallet.winning_trades += 1
        else:
            self.wallet.losing_trades += 1

        self.wallet.unrealized_pnl = sum(p.unrealized_pnl for p in self._open_positions.values())
        self.wallet.update_drawdown()

        logger.info(
            "[PAPER] Closed position for %s: %d lots at %.4f (PnL: $%.2f)",
            pos.symbol,
            pos.size,
            pos.mark_price,
            pos.unrealized_pnl,
        )
        return {
            "symbol": pos.symbol,
            "product_id": product_id,
            "size": pos.size,
            "pnl": pos.unrealized_pnl,
            "exit_price": pos.mark_price,
        }

    def flatten_all(self) -> list[dict[str, Any]]:
        """Market closes all active open positions and releases margin."""
        results = []
        for pid, pos in list(self._open_positions.items()):
            del self._open_positions[pid]
            self.wallet.blocked_margin = max(0.0, self.wallet.blocked_margin - pos.allocated_margin)
            self.wallet.balance += pos.unrealized_pnl
            self.wallet.realized_pnl += pos.unrealized_pnl
            self.wallet.total_trades += 1
            if pos.unrealized_pnl > 0:
                self.wallet.winning_trades += 1
            else:
                self.wallet.losing_trades += 1

            logger.info("[PAPER] Flattened position for %s: %d lots at %.4f (PnL: $%.2f)",
                        pos.symbol, pos.size, pos.mark_price, pos.unrealized_pnl)
            results.append({"symbol": pos.symbol, "pnl": pos.unrealized_pnl})

        self.wallet.unrealized_pnl = 0.0
        self.wallet.update_drawdown()
        return results

    def get_summary(self) -> dict[str, Any]:
        """Provides a complete dictionary summary of virtual performance."""
        return {
            "initial_balance": self.wallet.initial_balance,
            "balance": self.wallet.balance,
            "equity": self.wallet.equity,
            "available_margin": self.wallet.available_balance,
            "blocked_margin": self.wallet.blocked_margin,
            "realized_pnl": self.wallet.realized_pnl,
            "unrealized_pnl": self.wallet.unrealized_pnl,
            "net_roi_pct": self.wallet.net_roi_pct,
            "win_rate_pct": self.wallet.win_rate_pct,
            "max_drawdown_pct": self.wallet.max_drawdown_pct,
            "total_trades": self.wallet.total_trades,
            "winning_trades": self.wallet.winning_trades,
            "losing_trades": self.wallet.losing_trades,
            "open_positions_count": len(self._open_positions),
            "pending_orders_count": len(self._pending_orders),
        }
