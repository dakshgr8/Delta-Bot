"""
delta_bot.strategy
~~~~~~~~~~~~~~~~~~

Strategy and Signal Generation Engine (Phase 3) for Delta-Bot.
Implements the "EMA 20 + Session VWAP + PDH/PDL Volatility Breakout Strategy (V2 Upgraded)".

Integrates seamlessly with:
- Phase 1 REST Client (historical warm-up routine via /v2/history/candles)
- Phase 2 CircularBarCache (500-bar RingBuffer) and SessionAnchorData (00:00 UTC anchor)
- Pure NumPy IndicatorPipeline (EMA 20, VWAP, ATR 14, SMA(ATR, 20), ADX 14 with +DI/-DI)
- Integer Contract Lot Sizing adhering to Delta Exchange derivative specs.

Strictly zero lookahead bias: Signals and indicator calculations evaluate exclusively
on confirmed bar closes.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import inspect
import logging
import math
import time
import uuid
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from delta_bot.client import DeltaRestClient
from delta_bot.indicators import IndicatorPipeline, IndicatorSnapshot
from delta_bot.models import Candle, CircularBarCache, ContractSpec, SessionAnchorData

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Signal & Order Enums
# -----------------------------------------------------------------------------

class SignalType(str, Enum):
    """Trading signal classification emitted on confirmed bar close."""

    BULLISH_BREAKOUT = "BULLISH_BREAKOUT"
    BEARISH_BREAKDOWN = "BEARISH_BREAKDOWN"
    NO_SIGNAL = "NO_SIGNAL"


class OrderStatus(str, Enum):
    """Lifecycle status for an OrderIntent."""

    PENDING = "PENDING"  # Awaiting price breach of trigger stop level
    TRIGGERED = "TRIGGERED"  # Stop price breached, position opened/active
    CANCELLED_TTL_EXPIRED = "CANCELLED_TTL_EXPIRED"  # Exceeded max_pending_bars
    CANCELLED_INVALIDATED = "CANCELLED_INVALIDATED"  # Intermediate candle closed across EMA 20
    CANCELLED_MANUAL = "CANCELLED_MANUAL"  # Manually cancelled or replaced
    CLOSED_TP = "CLOSED_TP"  # Position closed via Take Profit
    CLOSED_SL = "CLOSED_SL"  # Position closed via Stop Loss / Trailing Stop


# -----------------------------------------------------------------------------
# Strategy Configuration & Order Intent Models
# -----------------------------------------------------------------------------

class StrategyConfig(BaseModel):
    """
    Configuration parameters for the V2 Upgraded Volatility Breakout Strategy.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(description="Target contract symbol (e.g. BTCUSD, ETHUSD, XAUTUSD)")
    resolution: str = Field(default="5m", description="Candle timeframe resolution")

    # Indicator Parameters
    ema_period: int = Field(default=20, ge=2, description="EMA period for trend baseline")
    atr_period: int = Field(default=14, ge=2, description="ATR period for volatility measurement")
    atr_sma_period: int = Field(default=20, ge=2, description="SMA period applied over ATR series")
    adx_period: int = Field(default=14, ge=2, description="ADX period for regime classification")
    adx_threshold: float = Field(
        default=20.0, ge=0.0, description="Minimum ADX required to confirm non-consolidation"
    )
    pd_condition_mode: str = Field(
        default="either",
        description="Previous day level condition mode: 'either' (Trading-Academy V2 trend continuation) or 'strict_breakout'",
    )

    # Entry Trigger & Buffer Parameters
    price_buffer_ticks: float = Field(
        default=1.0, ge=0.0, description="Buffer in tick sizes added above High or below Low"
    )
    max_pending_bars: int = Field(
        default=3, ge=1, description="TTL in bars before an untriggered pending order expires"
    )

    # Risk Management & Position Sizing
    risk_per_trade_pct: float = Field(
        default=1.0, gt=0.0, le=10.0, description="Account equity risk percentage per trade (e.g. 1.0%)"
    )
    rr_ratio: float = Field(
        default=2.0, gt=0.0, description="Risk-to-reward ratio for Take Profit target"
    )
    max_risk_usd: float | None = Field(
        default=None, description="Optional absolute maximum USD dollar loss cap per trade"
    )
    min_contracts: int = Field(
        default=1, ge=1, description="Minimum contract lots to submit if within risk limits"
    )
    min_stop_loss_pct: float = Field(
        default=0.05,
        ge=0.0,
        description="Minimum stop loss distance percentage (e.g. 0.05%) to skip noise stop-outs and prevent oversized sizing",
    )
    min_stop_loss_atr_mult: float = Field(
        default=0.25,
        ge=0.0,
        description="Minimum stop loss distance as a multiple of ATR 14 to filter out tight consolidation noise",
    )

    # Trailing Stop & Break-Even Configuration
    trailing_stop: bool = Field(
        default=True, description="Enable bar-by-bar trailing stop ratchet once triggered"
    )
    trailing_mode: str = Field(
        default="candle",
        description="Trailing stop calculation mode: 'candle' (Trading-Academy V2 candle extreme buffer) or 'atr'",
    )
    trail_trigger_rr: float = Field(
        default=1.0,
        ge=0.0,
        description="Minimum RR profit required before trailing stop activates (1.0 = 1:1 RR, 0 = immediate)",
    )
    trailing_stop_atr_mult: float = Field(
        default=1.0, gt=0.0, description="ATR multiplier used to ratchet trailing stop loss in atr mode"
    )
    enable_breakeven: bool = Field(
        default=True, description="Auto-ratchet Stop Loss to break-even (+1 tick) upon reaching target R profit"
    )
    breakeven_trigger_r: float = Field(
        default=1.0, gt=0.0, description="R-multiple floating profit required to trigger break-even ratchet (e.g. 1.0R or 1.5R)"
    )
    enable_weekend_shield: bool = Field(
        default=True, description="Whether to block new breakout signal generation on Saturday and Sunday (UTC weekend filter)"
    )


class OrderIntent(BaseModel):
    """
    Representation of an order intent generated by the strategy engine.
    Tracks entry triggers, protective stop loss, take profit targets, and integer contract sizing.
    """

    intent_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    base_signal_id: str | None = Field(default=None, description="Parent signal ID for multi-tenant fan-out traceability")
    client_order_id: str | None = Field(default=None, description="Deterministic exchange client order ID")
    symbol: str = Field(description="Target contract symbol")
    signal_type: SignalType = Field(description="Signal direction (BULLISH_BREAKOUT or BEARISH_BREAKDOWN)")
    status: OrderStatus = Field(default=OrderStatus.PENDING)

    # Price levels
    trigger_price: float = Field(description="Buy Stop (High + buf) or Sell Stop (Low - buf)")
    initial_stop_loss: float = Field(description="Initial protective Stop Loss level")
    current_stop_loss: float = Field(description="Current Stop Loss level (ratcheted if trailing)")
    take_profit: float = Field(description="Take Profit target price based on rr_ratio")
    rr_ratio: float = Field(default=2.0, description="Risk-to-reward ratio")

    # Position Sizing & Lots
    contract_lots: int = Field(description="Integer contract lots (Delta derivatives size)")
    target_asset_qty: float = Field(description="Underlying native asset quantity (lots * contract_value)")
    capital_at_risk_usd: float = Field(description="Projected USD dollar capital at risk")

    # Signal Bar Context
    signal_candle_high: float = Field(description="High of the signal bar")
    signal_candle_low: float = Field(description="Low of the signal bar")
    signal_candle_close: float = Field(description="Close of the signal bar")
    signal_bar_time: float = Field(description="Timestamp in epoch seconds of signal bar")

    # Order Lifecycle & Invalidation
    exchange_order_id: int | None = Field(default=None, description="Delta Exchange parent order ID")
    child_stop_loss_order_id: int | None = Field(
        default=None, description="Exchange order ID of spawned child Stop Loss order on position fill"
    )
    child_take_profit_order_id: int | None = Field(
        default=None, description="Exchange order ID of spawned child Take Profit order on position fill"
    )
    product_id: int | None = Field(default=None, description="Delta Exchange contract product ID")
    leverage: float = Field(default=1.0, description="Order leverage used on Delta Exchange")
    created_at_epoch_s: float = Field(description="Epoch timestamp when intent was created")
    ttl_bars: int = Field(default=3, description="Maximum confirmed bars before order expires")
    bars_elapsed: int = Field(default=0, description="Number of confirmed bars elapsed since placement")
    cancellation_reason: str | None = Field(default=None, description="Reason if cancelled")
    triggered_at: float | None = Field(default=None, description="Epoch timestamp when triggered")
    closed_at: float | None = Field(default=None, description="Epoch timestamp when closed")
    exit_price: float | None = Field(default=None, description="Price at which position exited")
    fill_price: float | None = Field(default=None, description="Actual execution fill price from private user feed")
    filled_size: int | None = Field(default=None, description="Actual executed contract lots from private user feed")
    realized_pnl: float | None = Field(default=None, description="Realized PnL of closed position in USD")
    is_breakeven_locked: bool = Field(default=False, description="Flag indicating SL has been advanced to break-even")
    r1_hit: bool = Field(default=False, description="Flag indicating trail_trigger_rr (+1.0R) profit has been breached")
    is_exit_recorded: bool = Field(default=False, description="Flag indicating exit has been accounted for in RiskManager")

    @property
    def is_pending(self) -> bool:
        """Returns True if order is pending entry trigger."""
        return self.status == OrderStatus.PENDING

    @property
    def is_triggered(self) -> bool:
        """Returns True if order has triggered and position is currently open."""
        return self.status == OrderStatus.TRIGGERED

    @property
    def is_active(self) -> bool:
        """Returns True if order is pending or position is active."""
        return self.status in (OrderStatus.PENDING, OrderStatus.TRIGGERED)

    @property
    def is_closed(self) -> bool:
        """Returns True if order was cancelled or position was closed."""
        return self.status in (
            OrderStatus.CANCELLED_TTL_EXPIRED,
            OrderStatus.CANCELLED_INVALIDATED,
            OrderStatus.CANCELLED_MANUAL,
            OrderStatus.CLOSED_TP,
            OrderStatus.CLOSED_SL,
        )

    def cancel(self, reason: str, status: OrderStatus = OrderStatus.CANCELLED_MANUAL) -> None:
        """Marks the order intent as cancelled with an explicit reason."""
        self.status = status
        self.cancellation_reason = reason
        logger.info(
            "OrderIntent [%s] %s cancelled: %s (status=%s)",
            self.intent_id,
            self.symbol,
            reason,
            status.value,
        )

    def trigger(self, trigger_bar: Candle) -> None:
        """Marks the order intent as triggered by price action."""
        self.status = OrderStatus.TRIGGERED
        self.triggered_at = trigger_bar.candle_start_epoch_s
        logger.info(
            "OrderIntent [%s] %s TRIGGERED at price %.4f (bar H=%.4f, L=%.4f)",
            self.intent_id,
            self.symbol,
            self.trigger_price,
            trigger_bar.high,
            trigger_bar.low,
        )

    def record_fill(
        self,
        fill_price: float,
        filled_size: int,
        timestamp: float | None = None,
    ) -> None:
        """
        Records an execution fill received from the private WebSocket or REST feed.
        Transitions order intent directly to TRIGGERED state and captures execution metrics.
        """
        self.status = OrderStatus.TRIGGERED
        self.fill_price = fill_price
        self.filled_size = filled_size
        self.triggered_at = timestamp or time.time()
        logger.info(
            "OrderIntent [%s] %s FILLED: size=%d price=%.4f (Status: TRIGGERED)",
            self.intent_id,
            self.symbol,
            filled_size,
            fill_price,
        )

    def close_position(self, status: OrderStatus, exit_price: float, timestamp: float) -> None:
        """Marks the active position as closed via Take Profit or Stop Loss."""
        if self.is_closed:
            return
        self.status = status
        self.exit_price = exit_price
        self.closed_at = timestamp
        entry_px = self.fill_price or self.trigger_price
        if entry_px and self.target_asset_qty:
            if self.signal_type == SignalType.BULLISH_BREAKOUT:
                self.realized_pnl = round((exit_price - entry_px) * self.target_asset_qty, 4)
            else:
                self.realized_pnl = round((entry_px - exit_price) * self.target_asset_qty, 4)
        else:
            self.realized_pnl = 0.0
        logger.info(
            "OrderIntent [%s] %s position CLOSED: %s at exit_price=%.4f (PnL: $%.2f)",
            self.intent_id,
            self.symbol,
            status.value,
            exit_price,
            self.realized_pnl or 0.0,
        )

    def ratchet_sl(self, new_sl: float) -> bool:
        """
        Updates the current protective stop loss if the new level reduces risk.
        Long: SL can only move UP.
        Short: SL can only move DOWN.
        Returns True if ratcheted, False otherwise.
        """
        if self.signal_type == SignalType.BULLISH_BREAKOUT:
            if new_sl > self.current_stop_loss:
                old_sl = self.current_stop_loss
                self.current_stop_loss = new_sl
                logger.info(
                    "OrderIntent [%s] %s LONG trailing SL ratcheted UP: %.4f -> %.4f",
                    self.intent_id,
                    self.symbol,
                    old_sl,
                    new_sl,
                )
                return True
        elif self.signal_type == SignalType.BEARISH_BREAKDOWN:
            if new_sl < self.current_stop_loss:
                old_sl = self.current_stop_loss
                self.current_stop_loss = new_sl
                logger.info(
                    "OrderIntent [%s] %s SHORT trailing SL ratcheted DOWN: %.4f -> %.4f",
                    self.intent_id,
                    self.symbol,
                    old_sl,
                    new_sl,
                )
                return True
        return False


# -----------------------------------------------------------------------------
# Breakout Strategy Engine
# -----------------------------------------------------------------------------

class BreakoutStrategyEngine:
    """
    Principal quantitative breakout strategy engine for Delta Exchange.
    Coordinates indicator updates, signal generation, pending order trigger checks,
    TTL expiration, EMA invalidation, trailing stop ratchets, and contract lot sizing.
    """

    def __init__(
        self,
        config: StrategyConfig,
        indicator_pipeline: IndicatorPipeline | None = None,
    ) -> None:
        self.config: StrategyConfig = config
        self.symbol: str = config.symbol.upper()

        self.indicator_pipeline: IndicatorPipeline = (
            indicator_pipeline
            if indicator_pipeline is not None
            else IndicatorPipeline(
                symbol=self.symbol,
                ema_period=config.ema_period,
                atr_period=config.atr_period,
                atr_sma_period=config.atr_sma_period,
                adx_period=config.adx_period,
                adx_threshold=config.adx_threshold,
            )
        )

        # Order state management
        self.pending_order: OrderIntent | None = None
        self.active_position: OrderIntent | None = None
        self.order_history: list[OrderIntent] = []

        # Freshness guards (True if condition was met on prior bar t-1)
        self._prior_bullish_condition: bool = False
        self._prior_bearish_condition: bool = False

        # Callback registry
        self._on_signal_cb: Callable[[SignalType, OrderIntent], Any] | None = None
        self._on_order_trigger_cb: Callable[[OrderIntent, Candle], Any] | None = None
        self._on_order_cancel_cb: Callable[[OrderIntent, str], Any] | None = None
        self._on_sl_ratchet_cb: Callable[[OrderIntent, float, float], Any] | None = None
        self._on_position_close_cb: Callable[[OrderIntent, OrderStatus, float], Any] | None = None

    # Callback Registration Methods
    def on_signal(self, cb: Callable[[SignalType, OrderIntent], Any]) -> None:
        """Register callback for newly generated order intent signals."""
        self._on_signal_cb = cb

    def on_order_trigger(self, cb: Callable[[OrderIntent, Candle], Any]) -> None:
        """Register callback for pending order triggers."""
        self._on_order_trigger_cb = cb

    def on_order_cancel(self, cb: Callable[[OrderIntent, str], Any]) -> None:
        """Register callback for pending order cancellations."""
        self._on_order_cancel_cb = cb

    def on_sl_ratchet(self, cb: Callable[[OrderIntent, float, float], Any]) -> None:
        """Register callback for trailing stop ratchets."""
        self._on_sl_ratchet_cb = cb

    def on_position_close(self, cb: Callable[[OrderIntent, OrderStatus, float], Any]) -> None:
        """Register callback for position TP/SL closures."""
        self._on_position_close_cb = cb

    async def _emit(self, callback: Callable[..., Any] | None, *args: Any) -> None:
        """Safely dispatches sync or async callbacks without unhandled exceptions."""
        if callback is None:
            return
        try:
            res = callback(*args)
            if inspect.isawaitable(res):
                await res
        except Exception as exc:
            logger.error("Error executing callback %s: %s", callback, exc, exc_info=True)

    # -------------------------------------------------------------------------
    # REST Historical Warm-Up Routine
    # -------------------------------------------------------------------------

    async def warm_up_from_rest(
        self,
        client: DeltaRestClient,
        bar_cache: CircularBarCache,
        session_anchor: SessionAnchorData,
        limit: int = 200,
    ) -> int:
        """
        Fetches historical closed candles from Delta REST API to eliminate cold-start delay.
        Populates CircularBarCache, updates SessionAnchorData, and computes initial indicators.
        Returns count of historical candles successfully primed.
        """
        logger.info(
            "Executing REST warm-up for %s (%s, limit=%d)...",
            self.symbol,
            self.config.resolution,
            limit,
        )

        candles = await client.get_historical_candles(
            symbol=self.symbol,
            resolution=self.config.resolution,
            limit=limit,
        )

        if not candles:
            logger.warning("REST warm-up returned 0 historical candles for %s", self.symbol)
            return 0

        # Prime circular bar cache
        seeded = bar_cache.seed_historical_bars(candles)

        # Update session anchor with historical candles
        for candle in candles:
            session_anchor.update(candle)

        # Compute initial indicators
        snapshot = self.indicator_pipeline.compute_from_cache(bar_cache, session_anchor)
        if snapshot is not None:
            logger.info(
                "REST warm-up complete for %s. Seeded %d bars. Indicators primed: EMA=%.2f, VWAP=%.2f, ATR=%.2f, ADX=%.2f",
                self.symbol,
                seeded,
                snapshot.ema_20,
                snapshot.session_vwap,
                snapshot.atr_14,
                snapshot.adx_14,
            )
        else:
            logger.warning("REST warm-up seeded %d bars, but failed to generate snapshot", seeded)

        return seeded

    # -------------------------------------------------------------------------
    # Confirmed Candle Close Lifecycle Evaluation
    # -------------------------------------------------------------------------

    async def on_candle_close(
        self,
        candle: Candle,
        session_anchor: SessionAnchorData,
        contract_spec: ContractSpec,
        bar_cache: CircularBarCache,
        account_equity: float = 0.0,
    ) -> OrderIntent | None:
        """
        Core lifecycle entry point invoked strictly on confirmed candle closes.

        Order of Operations:
        1. Process Active Position (TP, SL, Trailing Ratchet)
        2. Process Existing Pending Order (Trigger, EMA Invalidation, TTL Expiration)
        3. Update Indicators via IndicatorPipeline over confirmed closed bars
        4. Signal Evaluation & Freshness Check (Bullish Breakout vs Bearish Breakdown)
        5. Position Sizing & OrderIntent Generation
        """
        if not candle.is_closed:
            logger.debug("Ignoring non-closed candle for %s", candle.symbol)
            return None

        # ---------------------------------------------------------------------
        # Step 0: Sync Strategy Order State with External Reality (WS / REST)
        # ---------------------------------------------------------------------
        if self.pending_order is not None:
            if self.pending_order.is_triggered:
                self.active_position = self.pending_order
                self.pending_order = None
            elif self.pending_order.is_closed:
                self.pending_order = None

        if self.active_position is not None and self.active_position.is_closed:
            self.active_position = None

        # ---------------------------------------------------------------------
        # Step 1: Compute Indicators strictly on confirmed closed bar first
        # ---------------------------------------------------------------------
        snapshot = self.indicator_pipeline.compute_from_cache(bar_cache, session_anchor)
        if snapshot is None:
            logger.debug("Insufficient bar history for indicator calculation on %s", self.symbol)
            return None

        # ---------------------------------------------------------------------
        # Step 2: Active Position Management (TP, SL, Trailing Ratchet)
        # ---------------------------------------------------------------------
        if self.active_position is not None:
            pos = self.active_position
            if pos.signal_type == SignalType.BULLISH_BREAKOUT:
                # Check Take Profit (simulated offline mode only)
                if pos.exchange_order_id is None and candle.high >= pos.take_profit:
                    pos.close_position(OrderStatus.CLOSED_TP, pos.take_profit, candle.candle_start_epoch_s)
                    await self._emit(self._on_position_close_cb, pos, OrderStatus.CLOSED_TP, pos.take_profit)
                    self.active_position = None
                # Check Stop Loss (simulated offline mode only)
                elif pos.exchange_order_id is None and candle.low <= pos.current_stop_loss:
                    pos.close_position(OrderStatus.CLOSED_SL, pos.current_stop_loss, candle.candle_start_epoch_s)
                    await self._emit(self._on_position_close_cb, pos, OrderStatus.CLOSED_SL, pos.current_stop_loss)
                    self.active_position = None
                else:
                    entry_px = pos.fill_price if (pos.fill_price and pos.fill_price > 0) else pos.trigger_price
                    r_dist = abs(entry_px - pos.initial_stop_loss)
                    if r_dist <= 0 and pos.take_profit and self.config.rr_ratio > 0:
                        r_dist = abs(pos.take_profit - entry_px) / self.config.rr_ratio

                    # Check +1.0R / trail_trigger_rr profit breach
                    if not pos.r1_hit and self.config.trail_trigger_rr > 0 and r_dist > 0 and entry_px > 0:
                        trigger_dist = r_dist * self.config.trail_trigger_rr
                        if candle.high >= entry_px + trigger_dist:
                            pos.r1_hit = True
                            logger.info(
                                "🚀 %s LONG reached +%.1fR profit target (High=%.4f >= Target=%.4f)",
                                pos.symbol,
                                self.config.trail_trigger_rr,
                                candle.high,
                                entry_px + trigger_dist,
                            )

                    # Automated Break-Even condition check
                    be_triggered = bool(
                        self.config.enable_breakeven
                        and r_dist > 0
                        and (
                            pos.r1_hit
                            or candle.high >= entry_px + (self.config.breakeven_trigger_r * r_dist)
                        )
                    )
                    be_price = contract_spec.round_price(entry_px + contract_spec.tick_size) if be_triggered else None

                    # Unified Trailing Stop & Break-Even Evaluation (Single Ratchet Emission)
                    candidate_sl: float | None = None
                    should_trail = self.config.trailing_stop and (self.config.trail_trigger_rr <= 0 or pos.r1_hit)
                    if should_trail:
                        if self.config.trailing_mode == "atr":
                            raw_trailing = candle.low - (self.config.trailing_stop_atr_mult * snapshot.atr_14)
                        else:
                            # Trading-Academy V2 candle extreme trailing mode
                            buffer_val = self.config.price_buffer_ticks * contract_spec.tick_size
                            raw_trailing = candle.low - buffer_val

                        if pos.r1_hit or be_triggered:
                            # Lock at least breakeven (+1 tick), and trail higher with candle lows above breakeven
                            candidate_sl = max(be_price or (entry_px + contract_spec.tick_size), raw_trailing)
                        else:
                            candidate_sl = raw_trailing

                    elif be_triggered and not pos.is_breakeven_locked:
                        candidate_sl = be_price

                    if candidate_sl is not None:
                        quantized_sl = contract_spec.round_price(candidate_sl)
                        if quantized_sl > pos.current_stop_loss:
                            old_sl = pos.current_stop_loss
                            if pos.ratchet_sl(quantized_sl):
                                if be_price is not None and quantized_sl >= be_price:
                                    pos.is_breakeven_locked = True
                                await self._emit(self._on_sl_ratchet_cb, pos, old_sl, pos.current_stop_loss)

            elif pos.signal_type == SignalType.BEARISH_BREAKDOWN:
                # Check Take Profit (simulated offline mode only)
                if pos.exchange_order_id is None and candle.low <= pos.take_profit:
                    pos.close_position(OrderStatus.CLOSED_TP, pos.take_profit, candle.candle_start_epoch_s)
                    await self._emit(self._on_position_close_cb, pos, OrderStatus.CLOSED_TP, pos.take_profit)
                    self.active_position = None
                # Check Stop Loss (simulated offline mode only)
                elif pos.exchange_order_id is None and candle.high >= pos.current_stop_loss:
                    pos.close_position(OrderStatus.CLOSED_SL, pos.current_stop_loss, candle.candle_start_epoch_s)
                    await self._emit(self._on_position_close_cb, pos, OrderStatus.CLOSED_SL, pos.current_stop_loss)
                    self.active_position = None
                else:
                    entry_px = pos.fill_price if (pos.fill_price and pos.fill_price > 0) else pos.trigger_price
                    r_dist = abs(entry_px - pos.initial_stop_loss)
                    if r_dist <= 0 and pos.take_profit and self.config.rr_ratio > 0:
                        r_dist = abs(pos.take_profit - entry_px) / self.config.rr_ratio

                    # Check +1.0R / trail_trigger_rr profit breach
                    if not pos.r1_hit and self.config.trail_trigger_rr > 0 and r_dist > 0 and entry_px > 0:
                        trigger_dist = r_dist * self.config.trail_trigger_rr
                        if candle.low <= entry_px - trigger_dist:
                            pos.r1_hit = True
                            logger.info(
                                "🚀 %s SHORT reached +%.1fR profit target (Low=%.4f <= Target=%.4f)",
                                pos.symbol,
                                self.config.trail_trigger_rr,
                                candle.low,
                                entry_px - trigger_dist,
                            )

                    # Automated Break-Even condition check
                    be_triggered = bool(
                        self.config.enable_breakeven
                        and r_dist > 0
                        and (
                            pos.r1_hit
                            or candle.low <= entry_px - (self.config.breakeven_trigger_r * r_dist)
                        )
                    )
                    be_price = contract_spec.round_price(entry_px - contract_spec.tick_size) if be_triggered else None

                    # Unified Trailing Stop & Break-Even Evaluation (Single Ratchet Emission)
                    candidate_sl: float | None = None
                    should_trail = self.config.trailing_stop and (self.config.trail_trigger_rr <= 0 or pos.r1_hit)
                    if should_trail:
                        if self.config.trailing_mode == "atr":
                            raw_trailing = candle.high + (self.config.trailing_stop_atr_mult * snapshot.atr_14)
                        else:
                            # Trading-Academy V2 candle extreme trailing mode
                            buffer_val = self.config.price_buffer_ticks * contract_spec.tick_size
                            raw_trailing = candle.high + buffer_val

                        if pos.r1_hit or be_triggered:
                            # Lock at least breakeven (-1 tick), and trail lower with candle highs below breakeven
                            candidate_sl = min(be_price or (entry_px - contract_spec.tick_size), raw_trailing)
                        else:
                            candidate_sl = raw_trailing

                    elif be_triggered and not pos.is_breakeven_locked:
                        candidate_sl = be_price

                    if candidate_sl is not None and candidate_sl > 0:
                        quantized_sl = contract_spec.round_price(candidate_sl)
                        if quantized_sl < pos.current_stop_loss:
                            old_sl = pos.current_stop_loss
                            if pos.ratchet_sl(quantized_sl):
                                if be_price is not None and quantized_sl <= be_price:
                                    pos.is_breakeven_locked = True
                                await self._emit(self._on_sl_ratchet_cb, pos, old_sl, pos.current_stop_loss)

            # Only one open position at a time -> Do not open/place new orders
            if self.active_position is not None:
                return None

        # ---------------------------------------------------------------------
        # Step 3: Pending Order Trigger, Invalidation & TTL Evaluation
        # ---------------------------------------------------------------------
        if self.pending_order is not None:
            order = self.pending_order
            triggered = False

            # In simulated offline mode without exchange order id, simulate trigger and immediate bar exits
            if order.exchange_order_id is None:
                if order.signal_type == SignalType.BULLISH_BREAKOUT:
                    # Trigger condition: Bar high breaches trigger price
                    if candle.high >= order.trigger_price:
                        order.trigger(candle)
                        self.active_position = order
                        self.pending_order = None
                        triggered = True
                        await self._emit(self._on_order_trigger_cb, order, candle)

                        # Evaluate if the trigger bar hits TP or SL
                        if candle.high >= order.take_profit:
                            order.close_position(OrderStatus.CLOSED_TP, order.take_profit, candle.candle_start_epoch_s)
                            await self._emit(self._on_position_close_cb, order, OrderStatus.CLOSED_TP, order.take_profit)
                            self.active_position = None
                        elif candle.low <= order.current_stop_loss:
                            order.close_position(OrderStatus.CLOSED_SL, order.current_stop_loss, candle.candle_start_epoch_s)
                            await self._emit(self._on_position_close_cb, order, OrderStatus.CLOSED_SL, order.current_stop_loss)
                            self.active_position = None

                elif order.signal_type == SignalType.BEARISH_BREAKDOWN:
                    # Trigger condition: Bar low breaches trigger price
                    if candle.low <= order.trigger_price:
                        order.trigger(candle)
                        self.active_position = order
                        self.pending_order = None
                        triggered = True
                        await self._emit(self._on_order_trigger_cb, order, candle)

                        # Evaluate if the trigger bar hits TP or SL
                        if candle.low <= order.take_profit:
                            order.close_position(OrderStatus.CLOSED_TP, order.take_profit, candle.candle_start_epoch_s)
                            await self._emit(self._on_position_close_cb, order, OrderStatus.CLOSED_TP, order.take_profit)
                            self.active_position = None
                        elif candle.high >= order.current_stop_loss:
                            order.close_position(OrderStatus.CLOSED_SL, order.current_stop_loss, candle.candle_start_epoch_s)
                            await self._emit(self._on_position_close_cb, order, OrderStatus.CLOSED_SL, order.current_stop_loss)
                            self.active_position = None

            if not triggered and self.pending_order is not None:
                # Invalidation: intermediate candle closes across EMA 20
                if order.signal_type == SignalType.BULLISH_BREAKOUT and candle.close < snapshot.ema_20:
                    reason = f"Intermediate candle close ({candle.close:.4f}) below EMA 20 ({snapshot.ema_20:.4f})"
                    order.cancel(reason, OrderStatus.CANCELLED_INVALIDATED)
                    self.pending_order = None
                    await self._emit(self._on_order_cancel_cb, order, reason)
                elif order.signal_type == SignalType.BEARISH_BREAKDOWN and candle.close > snapshot.ema_20:
                    reason = f"Intermediate candle close ({candle.close:.4f}) above EMA 20 ({snapshot.ema_20:.4f})"
                    order.cancel(reason, OrderStatus.CANCELLED_INVALIDATED)
                    self.pending_order = None
                    await self._emit(self._on_order_cancel_cb, order, reason)

                # TTL Expiration Check
                if self.pending_order is not None:
                    order.bars_elapsed += 1
                    if order.bars_elapsed >= order.ttl_bars:
                        reason = f"Order TTL expired ({order.bars_elapsed}/{order.ttl_bars} bars elapsed without trigger)"
                        order.cancel(reason, OrderStatus.CANCELLED_TTL_EXPIRED)
                        self.pending_order = None
                        await self._emit(self._on_order_cancel_cb, order, reason)
            return None

        # ---------------------------------------------------------------------
        # Step 4: Signal Generation & Regime Filtering
        # ---------------------------------------------------------------------
        # Weekend Shield: Suppress new signal evaluation on Saturday and Sunday UTC
        if self.config.enable_weekend_shield:
            candle_dt = datetime.fromtimestamp(candle.candle_start_epoch_s, tz=timezone.utc)
            if candle_dt.weekday() in (5, 6):
                day_name = "Saturday" if candle_dt.weekday() == 5 else "Sunday"
                logger.debug(
                    "Signal evaluation skipped on %s: weekend shield active (%s UTC)",
                    self.symbol,
                    day_name,
                )
                return None

        # Strategy Rules:
        # 1. Close > EMA 20 (Bull) or Close < EMA 20 (Bear)
        # 2. Close > Session VWAP (Bull) or Close < Session VWAP (Bear)
        # 3. Previous Day Level Condition:
        #    - 'either' (Trading-Academy V2): Close > PDH or Close > PDL (Bull); Close < PDH or Close < PDL (Bear)
        #    - 'strict_breakout': Close > PDH (Bull); Close < PDL (Bear)
        # 4. ADX >= 20.0 (Consolidation Guard) AND ATR >= SMA(ATR, 20) (Volatility Expansion Check)
        # 5. Fresh Signal: Condition met on bar t, but was FALSE on bar t-1.

        has_pdh = snapshot.pdh is not None and snapshot.pdh > 0
        has_pdl = snapshot.pdl is not None and snapshot.pdl > 0

        if self.config.pd_condition_mode == "strict_breakout":
            c_pd_bull = bool(has_pdh and candle.close > snapshot.pdh)
            c_pd_bear = bool(has_pdl and candle.close < snapshot.pdl)
        else:
            # Trading-Academy V2 "either" mode:
            # Bull: (Close > PDH) or (Close > PDL) (if both present, or whichever is present)
            c_pd_bull = bool(
                (has_pdh and candle.close > snapshot.pdh)
                or (has_pdl and candle.close > snapshot.pdl)
            )
            c_pd_bear = bool(
                (has_pdh and candle.close < snapshot.pdh)
                or (has_pdl and candle.close < snapshot.pdl)
            )

        bullish_condition = bool(
            (has_pdh or has_pdl)
            and (candle.close > snapshot.ema_20)
            and (candle.close > snapshot.session_vwap)
            and c_pd_bull
            and (snapshot.adx_14 >= self.config.adx_threshold)
            and snapshot.is_volatility_expanded
        )

        bearish_condition = bool(
            (has_pdh or has_pdl)
            and (candle.close < snapshot.ema_20)
            and (candle.close < snapshot.session_vwap)
            and c_pd_bear
            and (snapshot.adx_14 >= self.config.adx_threshold)
            and snapshot.is_volatility_expanded
        )

        # Freshness determination
        fresh_bullish = bullish_condition and (not self._prior_bullish_condition)
        fresh_bearish = bearish_condition and (not self._prior_bearish_condition)

        # Update prior state for next bar evaluation
        self._prior_bullish_condition = bullish_condition
        self._prior_bearish_condition = bearish_condition

        # Suppress signal generation if already engaged in active position or pending order
        if self.active_position is not None or self.pending_order is not None:
            logger.debug(
                "Signal evaluation skipped on %s: active_pos=%s, pending_order=%s",
                self.symbol,
                self.active_position is not None,
                self.pending_order is not None,
            )
            return None

        # ---------------------------------------------------------------------
        # Step 5: Position Sizing & OrderIntent Construction
        # ---------------------------------------------------------------------
        new_intent: OrderIntent | None = None
        buffer_val = self.config.price_buffer_ticks * contract_spec.tick_size

        if fresh_bullish:
            # Bullish Entry: Buy Stop at Signal Candle High + Buffer
            raw_trigger = candle.high + buffer_val
            trigger_price = contract_spec.round_price(raw_trigger)

            # Initial Stop Loss at Signal Candle Low
            raw_sl = candle.low
            initial_sl = contract_spec.round_price(raw_sl)

            # Ensure valid risk distance and skip microscopic stoploss
            per_unit_risk = trigger_price - initial_sl
            if per_unit_risk <= (contract_spec.tick_size * 2):
                logger.warning(
                    "Bullish breakout on %s skipped: per_unit_risk (%.4f) too tight (<= 2 ticks)",
                    self.symbol,
                    per_unit_risk,
                )
                return None

            sl_pct = (per_unit_risk / trigger_price * 100.0) if trigger_price > 0 else 0.0
            if self.config.min_stop_loss_pct > 0 and sl_pct < self.config.min_stop_loss_pct:
                logger.warning(
                    "Bullish breakout on %s skipped: Stop-Loss distance (%.4f = %.3f%%) is below min_stop_loss_pct (%.2f%%)",
                    self.symbol,
                    per_unit_risk,
                    sl_pct,
                    self.config.min_stop_loss_pct,
                )
                return None

            if (
                self.config.min_stop_loss_atr_mult > 0
                and snapshot.atr_14 > 0
                and per_unit_risk < (self.config.min_stop_loss_atr_mult * snapshot.atr_14)
            ):
                logger.warning(
                    "Bullish breakout on %s skipped: Stop-Loss distance (%.4f) is below %.2fx ATR_14 (%.4f)",
                    self.symbol,
                    per_unit_risk,
                    self.config.min_stop_loss_atr_mult,
                    self.config.min_stop_loss_atr_mult * snapshot.atr_14,
                )
                return None

            # Take Profit target: Trigger + (RR * Risk)
            raw_tp = trigger_price + (self.config.rr_ratio * per_unit_risk)
            take_profit = contract_spec.round_price(raw_tp)

            # Capital at risk and integer contract sizing
            capital_at_risk = account_equity * (self.config.risk_per_trade_pct / 100.0)
            if self.config.max_risk_usd is not None:
                capital_at_risk = min(capital_at_risk, self.config.max_risk_usd)

            target_asset_qty = capital_at_risk / per_unit_risk
            contract_lots = contract_spec.round_size(target_asset_qty)

            if contract_lots < self.config.min_contracts:
                min_lot_risk = self.config.min_contracts * contract_spec.contract_value * per_unit_risk
                if account_equity > 0 and min_lot_risk > (capital_at_risk * 1.05):
                    logger.warning(
                        "Bullish breakout on %s skipped: min %d lot risk ($%.2f) exceeds risk budget ($%.2f)",
                        self.symbol,
                        self.config.min_contracts,
                        min_lot_risk,
                        capital_at_risk,
                    )
                    return None
                contract_lots = self.config.min_contracts

            actual_asset_qty = contract_spec.contracts_to_asset_qty(contract_lots)
            actual_risk_usd = contract_lots * contract_spec.contract_value * per_unit_risk

            new_intent = OrderIntent(
                symbol=self.symbol,
                signal_type=SignalType.BULLISH_BREAKOUT,
                status=OrderStatus.PENDING,
                trigger_price=trigger_price,
                initial_stop_loss=initial_sl,
                current_stop_loss=initial_sl,
                take_profit=take_profit,
                rr_ratio=self.config.rr_ratio,
                contract_lots=contract_lots,
                target_asset_qty=round(actual_asset_qty, 6),
                capital_at_risk_usd=round(actual_risk_usd, 2),
                signal_candle_high=candle.high,
                signal_candle_low=candle.low,
                signal_candle_close=candle.close,
                signal_bar_time=candle.candle_start_epoch_s,
                created_at_epoch_s=candle.candle_start_epoch_s,
                ttl_bars=self.config.max_pending_bars,
            )

        elif fresh_bearish:
            # Bearish Entry: Sell Stop at Signal Candle Low - Buffer
            raw_trigger = candle.low - buffer_val
            trigger_price = contract_spec.round_price(raw_trigger)

            # Initial Stop Loss at Signal Candle High
            raw_sl = candle.high
            initial_sl = contract_spec.round_price(raw_sl)

            # Ensure valid risk distance and skip microscopic stoploss
            per_unit_risk = initial_sl - trigger_price
            if per_unit_risk <= (contract_spec.tick_size * 2):
                logger.warning(
                    "Bearish breakdown on %s skipped: per_unit_risk (%.4f) too tight (<= 2 ticks)",
                    self.symbol,
                    per_unit_risk,
                )
                return None

            sl_pct = (per_unit_risk / trigger_price * 100.0) if trigger_price > 0 else 0.0
            if self.config.min_stop_loss_pct > 0 and sl_pct < self.config.min_stop_loss_pct:
                logger.warning(
                    "Bearish breakdown on %s skipped: Stop-Loss distance (%.4f = %.3f%%) is below min_stop_loss_pct (%.2f%%)",
                    self.symbol,
                    per_unit_risk,
                    sl_pct,
                    self.config.min_stop_loss_pct,
                )
                return None

            if (
                self.config.min_stop_loss_atr_mult > 0
                and snapshot.atr_14 > 0
                and per_unit_risk < (self.config.min_stop_loss_atr_mult * snapshot.atr_14)
            ):
                logger.warning(
                    "Bearish breakdown on %s skipped: Stop-Loss distance (%.4f) is below %.2fx ATR_14 (%.4f)",
                    self.symbol,
                    per_unit_risk,
                    self.config.min_stop_loss_atr_mult,
                    self.config.min_stop_loss_atr_mult * snapshot.atr_14,
                )
                return None

            # Take Profit target: Trigger - (RR * Risk)
            raw_tp = trigger_price - (self.config.rr_ratio * per_unit_risk)
            take_profit = contract_spec.round_price(raw_tp)

            # Capital at risk and integer contract sizing
            capital_at_risk = account_equity * (self.config.risk_per_trade_pct / 100.0)
            if self.config.max_risk_usd is not None:
                capital_at_risk = min(capital_at_risk, self.config.max_risk_usd)

            target_asset_qty = capital_at_risk / per_unit_risk
            contract_lots = contract_spec.round_size(target_asset_qty)

            if contract_lots < self.config.min_contracts:
                min_lot_risk = self.config.min_contracts * contract_spec.contract_value * per_unit_risk
                if account_equity > 0 and min_lot_risk > (capital_at_risk * 1.05):
                    logger.warning(
                        "Bearish breakdown on %s skipped: min %d lot risk ($%.2f) exceeds risk budget ($%.2f)",
                        self.symbol,
                        self.config.min_contracts,
                        min_lot_risk,
                        capital_at_risk,
                    )
                    return None
                contract_lots = self.config.min_contracts

            actual_asset_qty = contract_spec.contracts_to_asset_qty(contract_lots)
            actual_risk_usd = contract_lots * contract_spec.contract_value * per_unit_risk

            new_intent = OrderIntent(
                symbol=self.symbol,
                signal_type=SignalType.BEARISH_BREAKDOWN,
                status=OrderStatus.PENDING,
                trigger_price=trigger_price,
                initial_stop_loss=initial_sl,
                current_stop_loss=initial_sl,
                take_profit=take_profit,
                rr_ratio=self.config.rr_ratio,
                contract_lots=contract_lots,
                target_asset_qty=round(actual_asset_qty, 6),
                capital_at_risk_usd=round(actual_risk_usd, 2),
                signal_candle_high=candle.high,
                signal_candle_low=candle.low,
                signal_candle_close=candle.close,
                signal_bar_time=candle.candle_start_epoch_s,
                created_at_epoch_s=candle.candle_start_epoch_s,
                ttl_bars=self.config.max_pending_bars,
            )

        if new_intent is not None:
            self.pending_order = new_intent
            self.order_history.append(new_intent)
            logger.info(
                "Fresh %s Signal Generated on %s: Trigger=%.4f, SL=%.4f, TP=%.4f, Lots=%d, Risk=$%.2f",
                new_intent.signal_type.value,
                self.symbol,
                new_intent.trigger_price,
                new_intent.initial_stop_loss,
                new_intent.take_profit,
                new_intent.contract_lots,
                new_intent.capital_at_risk_usd,
            )
            await self._emit(self._on_signal_cb, new_intent.signal_type, new_intent)

        return new_intent
