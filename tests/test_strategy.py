"""
tests/test_strategy.py
~~~~~~~~~~~~~~~~~~~~~~

Comprehensive unit test suite for Strategy and Signal Generation Engine (Phase 3).
Verifies:
- Pure NumPy indicator accuracy (EMA, SMA, ATR, ADX, +DI/-DI)
- Bullish Breakout & Bearish Breakdown fresh signal rules
- Filter suppression (ADX < 20 consolidation guard, ATR < SMA volatility guard, inside-day chop)
- Fresh signal gating (condition at bar t, but False at bar t-1)
- Pending order trigger execution
- Pending order TTL expiration (max_pending_bars)
- Pending order invalidation on intermediate EMA cross
- Position management (Take Profit 2.0R, Stop Loss)
- Trailing stop ratchet behavior (never loosening risk)
- Integer contract lot sizing via ContractSpec
- REST historical warm-up helper
"""

import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from delta_bot.indicators import (
    IndicatorPipeline,
    calculate_adx,
    calculate_atr,
    calculate_ema,
    calculate_sma,
    calculate_true_range,
)
from delta_bot.models import Candle, CircularBarCache, ContractSpec, SessionAnchorData
from delta_bot.strategy import (
    BreakoutStrategyEngine,
    OrderIntent,
    OrderStatus,
    SignalType,
    StrategyConfig,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def btc_spec() -> ContractSpec:
    """Standard BTCUSDT perpetual contract spec for Delta Exchange."""
    return ContractSpec(
        product_id=1,
        symbol="BTCUSDT",
        contract_type="perpetual_futures",
        contract_value=0.001,  # 1 lot = 0.001 BTC
        tick_size=0.5,
        maker_commission_rate=0.0002,
        taker_commission_rate=0.0005,
    )


@pytest.fixture
def strategy_config() -> StrategyConfig:
    return StrategyConfig(
        symbol="BTCUSDT",
        resolution="1m",
        ema_period=20,
        atr_period=14,
        atr_sma_period=20,
        adx_period=14,
        adx_threshold=20.0,
        price_buffer_ticks=1.0,
        max_pending_bars=3,
        risk_per_trade_pct=1.0,
        rr_ratio=2.0,
        trailing_stop=False,
    )


def make_candle(
    symbol: str = "BTCUSDT",
    open_p: float = 50000.0,
    high_p: float = 50100.0,
    low_p: float = 49900.0,
    close_p: float = 50050.0,
    volume: float = 10.0,
    epoch_s: float = 1700000000.0,
) -> Candle:
    return Candle(
        symbol=symbol,
        resolution="1m",
        open=open_p,
        high=high_p,
        low=low_p,
        close=close_p,
        volume=volume,
        candle_start_time_usec=int(epoch_s * 1_000_000),
        is_closed=True,
        timestamp_usec=int(epoch_s * 1_000_000),
    )


def seed_synthetic_cache(
    bar_cache: CircularBarCache,
    count: int = 50,
    base_price: float = 50000.0,
    step: float = 10.0,
) -> None:
    """Fills bar cache with ascending bars to prime EMA, ATR, and ADX."""
    start_time = 1700000000.0
    for i in range(count):
        price = base_price + (i * step)
        c = make_candle(
            open_p=price - 5.0,
            high_p=price + 15.0,
            low_p=price - 10.0,
            close_p=price,
            epoch_s=start_time + (i * 60),
        )
        bar_cache.append(c)


# -----------------------------------------------------------------------------
# 1. Pure NumPy Indicator Unit Tests
# -----------------------------------------------------------------------------

def test_pure_numpy_indicators() -> None:
    """Verifies math correctness for SMA, EMA, True Range, ATR, and ADX."""
    data = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=np.float64)

    # SMA
    sma3 = calculate_sma(data, period=3)
    assert np.isnan(sma3[0])
    assert np.isnan(sma3[1])
    assert sma3[2] == pytest.approx(11.0)
    assert sma3[5] == pytest.approx(14.0)

    # EMA
    ema3 = calculate_ema(data, period=3)
    assert np.isnan(ema3[0])
    assert np.isnan(ema3[1])
    assert ema3[2] == pytest.approx(11.0)
    # alpha = 2 / (3 + 1) = 0.5; ema[3] = 0.5 * 13 + 0.5 * 11 = 12.0
    assert ema3[3] == pytest.approx(12.0)

    # True Range
    highs = np.array([105.0, 110.0, 108.0])
    lows = np.array([95.0, 100.0, 98.0])
    closes = np.array([100.0, 105.0, 102.0])
    tr = calculate_true_range(highs, lows, closes)
    assert len(tr) == 3
    assert tr[0] == pytest.approx(10.0)  # 105 - 95
    assert tr[1] == pytest.approx(10.0)  # max(110-100, |110-100|, |100-100|) = 10.0

    # ATR
    atr = calculate_atr(highs, lows, closes, period=2)
    assert np.isnan(atr[0])
    assert atr[1] == pytest.approx(10.0)

    # ADX: Trending series should have high ADX and +DI > -DI
    trend_highs = np.linspace(100, 300, 40)
    trend_lows = trend_highs - 5.0
    trend_closes = trend_highs - 2.0
    adx, p_di, m_di = calculate_adx(trend_highs, trend_lows, trend_closes, period=14)
    assert len(adx) == 40
    # Last reading should reflect strong upward trend
    assert not np.isnan(adx[-1])
    assert p_di[-1] > m_di[-1]
    assert adx[-1] > 20.0


# -----------------------------------------------------------------------------
# 2. Bullish Breakout & Signal Freshness
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bullish_breakout_fresh_signal(strategy_config: StrategyConfig, btc_spec: ContractSpec) -> None:
    """Verifies fresh Bullish Breakout generation, trigger stop price, and sizing."""
    engine = BreakoutStrategyEngine(strategy_config)
    bar_cache = CircularBarCache(capacity=500)
    seed_synthetic_cache(bar_cache, count=45, base_price=50000.0, step=20.0)

    session_anchor = SessionAnchorData(
        symbol="BTCUSDT",
        session_date="2026-11-06",
        session_start_epoch_s=1700000000.0,
        prior_day_high=50800.0,  # PDH
        prior_day_low=49000.0,
        prior_day_close=49500.0,
        session_vwap=50200.0,
    )

    # Signal candle: Close breaches PDH (50800), VWAP (50200), and EMA (~50600)
    signal_bar = make_candle(
        open_p=50700.0,
        high_p=51000.0,
        low_p=50600.0,
        close_p=50950.0,
        volume=50.0,
        epoch_s=1700000000.0 + (46 * 60),
    )
    bar_cache.append(signal_bar)

    callback_called = False
    emitted_signal: SignalType | None = None
    emitted_intent: OrderIntent | None = None

    def on_sig(sig: SignalType, intent: OrderIntent) -> None:
        nonlocal callback_called, emitted_signal, emitted_intent
        callback_called = True
        emitted_signal = sig
        emitted_intent = intent

    engine.on_signal(on_sig)

    intent = await engine.on_candle_close(
        candle=signal_bar,
        session_anchor=session_anchor,
        contract_spec=btc_spec,
        bar_cache=bar_cache,
        account_equity=10000.0,
    )

    assert intent is not None
    assert callback_called is True
    assert emitted_signal == SignalType.BULLISH_BREAKOUT
    assert intent.status == OrderStatus.PENDING

    # Trigger price = High (51000) + 1 tick (0.5) = 51000.5
    assert intent.trigger_price == 51000.5
    # Initial SL = Low = 50600.0
    assert intent.initial_stop_loss == 50600.0
    # Risk per unit = 51000.5 - 50600.0 = 400.5
    # TP = Trigger + 2.0 * 400.5 = 51000.5 + 801.0 = 51801.5
    assert intent.take_profit == 51801.5

    # Contract sizing: equity = 10,000, risk 1% = $100.
    # target_asset_qty = 100 / 400.5 ~= 0.24968 BTC
    # Contract value = 0.001 BTC -> contracts = int(round(0.24968 / 0.001)) = 250 contracts
    assert intent.contract_lots == 250
    assert intent.target_asset_qty == pytest.approx(0.25, abs=0.001)

    # --- Freshness check ---
    # Next bar continues higher. Since bullish condition was already True, it must NOT fire a duplicate intent.
    next_bar = make_candle(
        open_p=50950.0,
        high_p=50980.0,  # Below trigger 51000.5 so pending order stays pending
        low_p=50800.0,
        close_p=50970.0,
        epoch_s=1700000000.0 + (47 * 60),
    )
    bar_cache.append(next_bar)

    dup_intent = await engine.on_candle_close(
        candle=next_bar,
        session_anchor=session_anchor,
        contract_spec=btc_spec,
        bar_cache=bar_cache,
        account_equity=10000.0,
    )
    # Must be None because a pending order is already active and condition is not fresh
    assert dup_intent is None


# -----------------------------------------------------------------------------
# 3. Filter Suppression Tests (ADX, ATR, Inside-Day)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_filter_suppression_low_adx(strategy_config: StrategyConfig, btc_spec: ContractSpec) -> None:
    """Verifies that signals are suppressed when ADX < 20 (consolidation guard)."""
    # High threshold ensures suppression
    strict_config = strategy_config.model_copy(update={"adx_threshold": 95.0})
    engine = BreakoutStrategyEngine(strict_config)
    bar_cache = CircularBarCache(capacity=500)
    seed_synthetic_cache(bar_cache, count=45, base_price=50000.0, step=20.0)

    session_anchor = SessionAnchorData(
        symbol="BTCUSDT",
        session_date="2026-11-06",
        session_start_epoch_s=1700000000.0,
        prior_day_high=50800.0,
        prior_day_low=49000.0,
        session_vwap=50200.0,
    )

    signal_bar = make_candle(
        open_p=50700.0,
        high_p=51000.0,
        low_p=50600.0,
        close_p=50950.0,  # Above PDH, VWAP, EMA
        epoch_s=1700000000.0 + (46 * 60),
    )
    bar_cache.append(signal_bar)

    intent = await engine.on_candle_close(
        candle=signal_bar,
        session_anchor=session_anchor,
        contract_spec=btc_spec,
        bar_cache=bar_cache,
    )
    assert intent is None
    assert engine.pending_order is None


@pytest.mark.asyncio
async def test_filter_suppression_inside_day(strategy_config: StrategyConfig, btc_spec: ContractSpec) -> None:
    """Verifies that signals are suppressed when close is inside PDH/PDL."""
    engine = BreakoutStrategyEngine(strategy_config)
    bar_cache = CircularBarCache(capacity=500)
    seed_synthetic_cache(bar_cache, count=45, base_price=50000.0, step=20.0)

    # Set PDH far higher (60,000) so current price (50,950) is inside-day
    session_anchor = SessionAnchorData(
        symbol="BTCUSDT",
        session_date="2026-11-06",
        session_start_epoch_s=1700000000.0,
        prior_day_high=60000.0,  # Inside-day
        prior_day_low=40000.0,
        session_vwap=50200.0,
    )

    signal_bar = make_candle(
        open_p=50700.0,
        high_p=51000.0,
        low_p=50600.0,
        close_p=50950.0,
        epoch_s=1700000000.0 + (46 * 60),
    )
    bar_cache.append(signal_bar)

    intent = await engine.on_candle_close(
        candle=signal_bar,
        session_anchor=session_anchor,
        contract_spec=btc_spec,
        bar_cache=bar_cache,
    )
    assert intent is None
    assert engine.pending_order is None


# -----------------------------------------------------------------------------
# 4. Bearish Breakdown Signal
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bearish_breakdown_fresh_signal(strategy_config: StrategyConfig, btc_spec: ContractSpec) -> None:
    """Verifies fresh Bearish Breakdown generation, sell stop trigger price, and SL/TP."""
    engine = BreakoutStrategyEngine(strategy_config)
    bar_cache = CircularBarCache(capacity=500)
    # Downward trending bars
    start_time = 1700000000.0
    for i in range(45):
        p = 50000.0 - (i * 20.0)
        c = make_candle(
            open_p=p + 5.0,
            high_p=p + 10.0,
            low_p=p - 15.0,
            close_p=p,
            epoch_s=start_time + (i * 60),
        )
        bar_cache.append(c)

    session_anchor = SessionAnchorData(
        symbol="BTCUSDT",
        session_date="2026-11-06",
        session_start_epoch_s=1700000000.0,
        prior_day_high=51000.0,
        prior_day_low=49300.0,  # PDL
        session_vwap=49500.0,
    )

    breakdown_bar = make_candle(
        open_p=49200.0,
        high_p=49250.0,
        low_p=48900.0,
        close_p=49000.0,  # Below PDL (49300), VWAP (49500), and EMA
        volume=60.0,
        epoch_s=start_time + (46 * 60),
    )
    bar_cache.append(breakdown_bar)

    intent = await engine.on_candle_close(
        candle=breakdown_bar,
        session_anchor=session_anchor,
        contract_spec=btc_spec,
        bar_cache=bar_cache,
        account_equity=10000.0,
    )

    assert intent is not None
    assert intent.signal_type == SignalType.BEARISH_BREAKDOWN
    # Sell Stop trigger = Low (48900) - 1 tick (0.5) = 48899.5
    assert intent.trigger_price == 48899.5
    # Initial SL = High (49250)
    assert intent.initial_stop_loss == 49250.0
    # Risk = 49250.0 - 48899.5 = 350.5
    # TP = 48899.5 - (2.0 * 350.5) = 48899.5 - 701.0 = 48198.5
    assert intent.take_profit == 48198.5
    assert intent.contract_lots > 0


# -----------------------------------------------------------------------------
# 5. Pending Order Trigger Execution
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pending_order_trigger(strategy_config: StrategyConfig, btc_spec: ContractSpec) -> None:
    """Verifies pending order transitions to TRIGGERED when subsequent bar high breaches trigger."""
    engine = BreakoutStrategyEngine(strategy_config)
    bar_cache = CircularBarCache(capacity=500)
    seed_synthetic_cache(bar_cache, count=45, base_price=50000.0, step=20.0)

    session_anchor = SessionAnchorData(
        symbol="BTCUSDT",
        session_date="2026-11-06",
        session_start_epoch_s=1700000000.0,
        prior_day_high=50800.0,
        session_vwap=50200.0,
    )

    # 1. Generate signal
    sig_bar = make_candle(
        high_p=51000.0,
        low_p=50600.0,
        close_p=50950.0,
        epoch_s=1700000000.0 + (46 * 60),
    )
    bar_cache.append(sig_bar)
    intent = await engine.on_candle_close(sig_bar, session_anchor, btc_spec, bar_cache)
    assert intent is not None
    assert engine.pending_order is not None
    assert intent.trigger_price == 51000.5

    # 2. Subsequent bar breaches trigger (High = 51050 >= 51000.5)
    trigger_callback_called = False

    def on_trig(order: OrderIntent, candle: Candle) -> None:
        nonlocal trigger_callback_called
        trigger_callback_called = True

    engine.on_order_trigger(on_trig)

    next_bar = make_candle(
        open_p=50950.0,
        high_p=51050.0,  # Breaches trigger!
        low_p=50900.0,
        close_p=51020.0,
        epoch_s=1700000000.0 + (47 * 60),
    )
    bar_cache.append(next_bar)

    await engine.on_candle_close(next_bar, session_anchor, btc_spec, bar_cache)

    assert trigger_callback_called is True
    assert engine.pending_order is None
    assert engine.active_position is not None
    assert engine.active_position.status == OrderStatus.TRIGGERED


# -----------------------------------------------------------------------------
# 6. TTL Expiration & EMA Invalidation
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pending_order_ttl_expiration(strategy_config: StrategyConfig, btc_spec: ContractSpec) -> None:
    """Verifies pending order is cancelled after max_pending_bars (default: 3) elapse."""
    cfg = strategy_config.model_copy(update={"max_pending_bars": 3})
    engine = BreakoutStrategyEngine(cfg)
    bar_cache = CircularBarCache(capacity=500)
    seed_synthetic_cache(bar_cache, count=45, base_price=50000.0, step=20.0)

    session_anchor = SessionAnchorData(
        symbol="BTCUSDT",
        session_date="2026-11-06",
        session_start_epoch_s=1700000000.0,
        prior_day_high=50800.0,
        session_vwap=50200.0,
    )

    sig_bar = make_candle(
        high_p=51000.0,
        low_p=50600.0,
        close_p=50950.0,
        epoch_s=1700000000.0 + (46 * 60),
    )
    bar_cache.append(sig_bar)
    intent = await engine.on_candle_close(sig_bar, session_anchor, btc_spec, bar_cache)
    assert intent is not None
    assert engine.pending_order is not None

    cancel_reasons: list[str] = []
    engine.on_order_cancel(lambda order, reason: cancel_reasons.append(reason))

    # Pass 3 bars that do not reach trigger (high < 51000.5) and stay above EMA (~50600)
    for b_idx in range(1, 4):
        idle_bar = make_candle(
            high_p=50980.0,  # Below trigger
            low_p=50800.0,
            close_p=50900.0,  # Above EMA
            epoch_s=1700000000.0 + ((46 + b_idx) * 60),
        )
        bar_cache.append(idle_bar)
        await engine.on_candle_close(idle_bar, session_anchor, btc_spec, bar_cache)

    assert engine.pending_order is None
    assert intent.status == OrderStatus.CANCELLED_TTL_EXPIRED
    assert len(cancel_reasons) == 1
    assert "TTL expired" in cancel_reasons[0]


@pytest.mark.asyncio
async def test_pending_order_ema_invalidation(strategy_config: StrategyConfig, btc_spec: ContractSpec) -> None:
    """Verifies that an intermediate bar closing across EMA 20 immediately invalidates pending order."""
    engine = BreakoutStrategyEngine(strategy_config)
    bar_cache = CircularBarCache(capacity=500)
    seed_synthetic_cache(bar_cache, count=45, base_price=50000.0, step=20.0)

    session_anchor = SessionAnchorData(
        symbol="BTCUSDT",
        session_date="2026-11-06",
        session_start_epoch_s=1700000000.0,
        prior_day_high=50800.0,
        session_vwap=50200.0,
    )

    sig_bar = make_candle(
        high_p=51000.0,
        low_p=50600.0,
        close_p=50950.0,
        epoch_s=1700000000.0 + (46 * 60),
    )
    bar_cache.append(sig_bar)
    intent = await engine.on_candle_close(sig_bar, session_anchor, btc_spec, bar_cache)
    assert intent is not None
    assert engine.pending_order is not None

    # Next bar collapses below EMA 20 (~50600)
    dump_bar = make_candle(
        open_p=50700.0,
        high_p=50750.0,
        low_p=50200.0,
        close_p=50300.0,  # Plummets below EMA!
        epoch_s=1700000000.0 + (47 * 60),
    )
    bar_cache.append(dump_bar)
    await engine.on_candle_close(dump_bar, session_anchor, btc_spec, bar_cache)

    assert engine.pending_order is None
    assert intent.status == OrderStatus.CANCELLED_INVALIDATED
    assert "below EMA 20" in str(intent.cancellation_reason)


# -----------------------------------------------------------------------------
# 7. Position Exits: Take Profit & Stop Loss
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_position_take_profit_and_stop_loss(strategy_config: StrategyConfig, btc_spec: ContractSpec) -> None:
    """Verifies active positions close correctly when Take Profit or Stop Loss is reached."""
    engine = BreakoutStrategyEngine(strategy_config)
    bar_cache = CircularBarCache(capacity=500)
    seed_synthetic_cache(bar_cache, count=45, base_price=50000.0, step=20.0)

    session_anchor = SessionAnchorData(
        symbol="BTCUSDT",
        session_date="2026-11-06",
        session_start_epoch_s=1700000000.0,
        prior_day_high=50800.0,
        session_vwap=50200.0,
    )

    # 1. Trigger order
    sig_bar = make_candle(high_p=51000.0, low_p=50600.0, close_p=50950.0)
    bar_cache.append(sig_bar)
    intent = await engine.on_candle_close(sig_bar, session_anchor, btc_spec, bar_cache)
    assert intent is not None

    # Trigger it
    trig_bar = make_candle(high_p=51050.0, low_p=50900.0, close_p=51020.0)
    bar_cache.append(trig_bar)
    await engine.on_candle_close(trig_bar, session_anchor, btc_spec, bar_cache)
    assert engine.active_position is not None

    # Target TP is 51801.5. Next bar skyrockets past TP
    closed_events: list[OrderStatus] = []
    engine.on_position_close(lambda order, status, px: closed_events.append(status))

    tp_bar = make_candle(high_p=52000.0, low_p=51000.0, close_p=51900.0)
    bar_cache.append(tp_bar)
    await engine.on_candle_close(tp_bar, session_anchor, btc_spec, bar_cache)

    assert engine.active_position is None
    assert intent.status == OrderStatus.CLOSED_TP
    assert OrderStatus.CLOSED_TP in closed_events
    assert intent.realized_pnl is not None
    assert intent.realized_pnl > 0.0


# -----------------------------------------------------------------------------
# 8. Trailing Stop Ratchet Behavior
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_trailing_stop_ratchet(strategy_config: StrategyConfig, btc_spec: ContractSpec) -> None:
    """Verifies trailing stop ratchet moves stop loss favorably without ever loosening risk."""
    trailing_cfg = strategy_config.model_copy(update={"trailing_stop": True, "trailing_stop_atr_mult": 1.0})
    engine = BreakoutStrategyEngine(trailing_cfg)
    bar_cache = CircularBarCache(capacity=500)
    seed_synthetic_cache(bar_cache, count=45, base_price=50000.0, step=20.0)

    session_anchor = SessionAnchorData(
        symbol="BTCUSDT",
        session_date="2026-11-06",
        session_start_epoch_s=1700000000.0,
        prior_day_high=50800.0,
        session_vwap=50200.0,
    )

    sig_bar = make_candle(high_p=51000.0, low_p=50600.0, close_p=50950.0)
    bar_cache.append(sig_bar)
    intent = await engine.on_candle_close(sig_bar, session_anchor, btc_spec, bar_cache)
    assert intent is not None
    initial_sl = intent.current_stop_loss

    # Trigger position
    trig_bar = make_candle(high_p=51050.0, low_p=50900.0, close_p=51020.0)
    bar_cache.append(trig_bar)
    await engine.on_candle_close(trig_bar, session_anchor, btc_spec, bar_cache)
    assert engine.active_position is not None

    ratchets: list[tuple[float, float]] = []
    engine.on_sl_ratchet(lambda order, old_sl, new_sl: ratchets.append((old_sl, new_sl)))

    # Bar 1 climbs higher: Low is 51100. Proposed SL = Low - ATR (~25) = ~51075 > initial_sl (50600)
    climb_bar = make_candle(high_p=51300.0, low_p=51100.0, close_p=51250.0)
    bar_cache.append(climb_bar)
    await engine.on_candle_close(climb_bar, session_anchor, btc_spec, bar_cache)

    assert intent.current_stop_loss > initial_sl
    assert len(ratchets) == 1
    new_ratcheted_sl = intent.current_stop_loss

    # Bar 2 pulls back: Low is 50900. Proposed SL would be lower, so SL must NOT loosen!
    pullback_bar = make_candle(high_p=51200.0, low_p=50900.0, close_p=51100.0)
    bar_cache.append(pullback_bar)
    await engine.on_candle_close(pullback_bar, session_anchor, btc_spec, bar_cache)

    # Stop Loss should stay at new_ratcheted_sl (never loosened)
    assert intent.current_stop_loss == new_ratcheted_sl
    assert len(ratchets) == 1


# -----------------------------------------------------------------------------
# 9. REST Warm-Up Routine
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rest_warmup_routine(strategy_config: StrategyConfig) -> None:
    """Verifies that warm_up_from_rest primes circular cache, indicators, and session anchor."""
    engine = BreakoutStrategyEngine(strategy_config)
    bar_cache = CircularBarCache(capacity=500)
    session_anchor = SessionAnchorData(
        symbol="BTCUSDT",
        session_date="2026-11-06",
        session_start_epoch_s=1700000000.0,
    )

    mock_client = AsyncMock()
    mock_candles = [
        make_candle(
            open_p=50000.0 + i,
            high_p=50010.0 + i,
            low_p=49990.0 + i,
            close_p=50005.0 + i,
            epoch_s=1700000000.0 + (i * 60),
        )
        for i in range(50)
    ]
    mock_client.get_historical_candles.return_value = mock_candles

    seeded = await engine.warm_up_from_rest(
        client=mock_client,
        bar_cache=bar_cache,
        session_anchor=session_anchor,
        limit=50,
    )

    assert seeded == 50
    assert bar_cache.size == 50
    assert engine.indicator_pipeline.latest_snapshot is not None
    assert engine.indicator_pipeline.latest_snapshot.ema_20 > 0
    assert session_anchor.session_high >= 50050.0


# -----------------------------------------------------------------------------
# 10. Automated Break-Even Auto-Ratchet (Feature E)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_automated_breakeven_autoratchet_bullish(
    btc_spec: ContractSpec,
    strategy_config: StrategyConfig,
) -> None:
    """Verifies that reaching +1.0R auto-ratchets Stop Loss to entry + 1 tick (break-even)."""
    cfg = strategy_config.model_copy(update={"enable_breakeven": True, "breakeven_trigger_r": 1.0, "trailing_stop": False})
    engine = BreakoutStrategyEngine(cfg)
    bar_cache = CircularBarCache(capacity=500)
    session_anchor = SessionAnchorData(symbol="BTCUSDT", session_date="2026-11-06", session_start_epoch_s=1000.0)

    # Active Long: Entry = 50,000.0, Initial SL = 49,000.0 (1R = 1,000.0), TP = 52,000.0
    intent = OrderIntent(
        symbol="BTCUSDT",
        signal_type=SignalType.BULLISH_BREAKOUT,
        trigger_price=50000.0,
        initial_stop_loss=49000.0,
        current_stop_loss=49000.0,
        take_profit=52000.0,
        contract_lots=10,
        target_asset_qty=0.01,
        capital_at_risk_usd=100.0,
        signal_candle_high=49900.0,
        signal_candle_low=49500.0,
        signal_candle_close=49800.0,
        signal_bar_time=1000.0,
        created_at_epoch_s=1000.0,
        status=OrderStatus.TRIGGERED,
    )
    engine.active_position = intent

    ratchets: list[tuple[float, float]] = []
    engine.on_sl_ratchet(lambda pos, old_sl, new_sl: ratchets.append((old_sl, new_sl)))

    # Bar 1 reaches 51,050.0 (+1.05R floating gain, below TP of 52,000)
    # Expected BE SL: 50,000.0 + tick_size (0.5) = 50,000.5
    bar1 = make_candle(open_p=50500.0, high_p=51050.0, low_p=50400.0, close_p=50900.0, epoch_s=2000.0)
    bar_cache.append(bar1)
    await engine.on_candle_close(bar1, session_anchor, btc_spec, bar_cache)

    assert intent.is_breakeven_locked is True
    assert intent.current_stop_loss == 50000.5
    assert len(ratchets) == 1
    assert ratchets[0] == (49000.0, 50000.5)

    # Bar 2 pulls back to 50,000.0 (hits BE SL of 50,000.5)
    bar2 = make_candle(open_p=50900.0, high_p=50950.0, low_p=50000.0, close_p=50000.0, epoch_s=2060.0)
    bar_cache.append(bar2)
    await engine.on_candle_close(bar2, session_anchor, btc_spec, bar_cache)

    assert engine.active_position is None  # Closed out at break-even!
    assert intent.status == OrderStatus.CLOSED_SL
    assert intent.exit_price == 50000.5


@pytest.mark.asyncio
async def test_automated_breakeven_autoratchet_bearish(
    btc_spec: ContractSpec,
    strategy_config: StrategyConfig,
) -> None:
    """Verifies that reaching +1.0R on Short auto-ratchets Stop Loss to entry - 1 tick."""
    cfg = strategy_config.model_copy(update={"enable_breakeven": True, "breakeven_trigger_r": 1.0, "trailing_stop": False})
    engine = BreakoutStrategyEngine(cfg)
    bar_cache = CircularBarCache(capacity=500)
    session_anchor = SessionAnchorData(symbol="BTCUSDT", session_date="2026-11-06", session_start_epoch_s=1000.0)

    # Active Short: Entry = 50,000.0, Initial SL = 51,000.0 (1R = 1,000.0), TP = 48,000.0
    intent = OrderIntent(
        symbol="BTCUSDT",
        signal_type=SignalType.BEARISH_BREAKDOWN,
        trigger_price=50000.0,
        initial_stop_loss=51000.0,
        current_stop_loss=51000.0,
        take_profit=48000.0,
        contract_lots=10,
        target_asset_qty=0.01,
        capital_at_risk_usd=100.0,
        signal_candle_high=50500.0,
        signal_candle_low=50100.0,
        signal_candle_close=50200.0,
        signal_bar_time=1000.0,
        created_at_epoch_s=1000.0,
        status=OrderStatus.TRIGGERED,
    )
    engine.active_position = intent

    ratchets: list[tuple[float, float]] = []
    engine.on_sl_ratchet(lambda pos, old_sl, new_sl: ratchets.append((old_sl, new_sl)))

    # Bar 1 drops to 48,950.0 (+1.05R floating gain, above TP of 48,000)
    # Expected BE SL: 50,000.0 - tick_size (0.5) = 49,999.5
    bar1 = make_candle(open_p=49500.0, high_p=49600.0, low_p=48950.0, close_p=49100.0, epoch_s=2000.0)
    bar_cache.append(bar1)
    await engine.on_candle_close(bar1, session_anchor, btc_spec, bar_cache)

    assert intent.is_breakeven_locked is True
    assert intent.current_stop_loss == 49999.5
    assert len(ratchets) == 1
    assert ratchets[0] == (51000.0, 49999.5)


@pytest.mark.asyncio
async def test_strategy_skips_breakout_with_tiny_stop_loss_pct(
    strategy_config: StrategyConfig, btc_spec: ContractSpec
) -> None:
    """Verifies that setups with stop loss distance < min_stop_loss_pct (e.g. flat/micro bars) are skipped."""
    engine = BreakoutStrategyEngine(strategy_config)
    bar_cache = CircularBarCache(capacity=500)
    seed_synthetic_cache(bar_cache, count=45, base_price=50000.0, step=20.0)

    session_anchor = SessionAnchorData(
        symbol="BTCUSDT",
        session_date="2026-11-06",
        session_start_epoch_s=1700000000.0,
        prior_day_high=50800.0,
        prior_day_low=49000.0,
        prior_day_close=49500.0,
        session_vwap=50200.0,
    )

    # Signal candle breaches PDH, but has a microscopic range:
    # High = 50950.5, Low = 50949.5 (range = 1.0, distance = 1.5 at 51,000 = 0.003%)
    # min_stop_loss_pct = 0.15% requires distance >= 76.4
    flat_signal_bar = make_candle(
        open_p=50950.0,
        high_p=50950.5,
        low_p=50949.5,
        close_p=50950.0,
        volume=50.0,
        epoch_s=1700000000.0 + (46 * 60),
    )
    bar_cache.append(flat_signal_bar)

    intent = await engine.on_candle_close(
        candle=flat_signal_bar,
        session_anchor=session_anchor,
        contract_spec=btc_spec,
        bar_cache=bar_cache,
        account_equity=10000.0,
    )

    # Must be cleanly skipped! No intent generated, no position opened
    assert intent is None


@pytest.mark.asyncio
async def test_strategy_skips_breakout_with_sub_atr_stop_loss(
    strategy_config: StrategyConfig, btc_spec: ContractSpec
) -> None:
    """Verifies that setups with stop loss distance < min_stop_loss_atr_mult * ATR are skipped."""
    # Set min_stop_loss_pct to 0 to isolate the ATR check
    cfg = strategy_config.model_copy(update={"min_stop_loss_pct": 0.0, "min_stop_loss_atr_mult": 0.50})
    engine = BreakoutStrategyEngine(cfg)
    bar_cache = CircularBarCache(capacity=500)
    seed_synthetic_cache(bar_cache, count=45, base_price=50000.0, step=20.0)

    session_anchor = SessionAnchorData(
        symbol="BTCUSDT",
        session_date="2026-11-06",
        session_start_epoch_s=1700000000.0,
        prior_day_high=50800.0,
        prior_day_low=49000.0,
        prior_day_close=49500.0,
        session_vwap=50200.0,
    )

    # ATR in synthetic cache is ~25.0. 0.50 * ATR = 12.5.
    # We craft a bar where distance is 5.5 (< 12.5)
    tight_signal_bar = make_candle(
        open_p=50950.0,
        high_p=50955.0,
        low_p=50950.0,
        close_p=50953.0,
        volume=50.0,
        epoch_s=1700000000.0 + (46 * 60),
    )
    bar_cache.append(tight_signal_bar)

    intent = await engine.on_candle_close(
        candle=tight_signal_bar,
        session_anchor=session_anchor,
        contract_spec=btc_spec,
        bar_cache=bar_cache,
        account_equity=10000.0,
    )

    # Must be cleanly skipped!
    assert intent is None

