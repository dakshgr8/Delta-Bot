"""
tests/test_immediate_cut_prevention.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Verifies safeguards preventing premature trade cutting and brokerage loss:
1. Mark price inversion guard prevents setting Stop Loss across current mark price.
2. Symbol extraction in PositionInfo and ExchangeOrder supports multiple Delta schema formats.
3. Strategy engine clamps candidate SL to strictly prevent crossing confirmed candle close.
4. Exchange history endpoint returns orders and fills for isolated tenant diagnostics.
"""

from unittest.mock import AsyncMock, MagicMock
import pytest
import time
from fastapi.testclient import TestClient

from delta_bot.client import DeltaRestClient
from delta_bot.config import BotConfig
from delta_bot.execution import ExecutionConfig, ExecutionManager
from delta_bot.models import ContractSpec, ExecutionMode, PositionInfo, ExchangeOrder
from delta_bot.risk import RiskConfig, RiskManager
from delta_bot.server import create_app, log_handler
from delta_bot.state import TradingStateManager
from delta_bot.strategy import OrderIntent, OrderStatus, SignalType, BreakoutStrategyEngine, StrategyConfig, Candle


@pytest.fixture
def btc_spec() -> ContractSpec:
    return ContractSpec(
        product_id=27,
        symbol="BTCUSD",
        contract_type="perpetual_futures",
        contract_value=0.001,
        tick_size=0.5,
        maker_commission_rate=0.0002,
        taker_commission_rate=0.0005,
        quoting_symbol="USD",
    )


@pytest.fixture
def sample_intent() -> OrderIntent:
    return OrderIntent(
        symbol="BTCUSD",
        signal_type=SignalType.BULLISH_BREAKOUT,
        status=OrderStatus.TRIGGERED,
        trigger_price=73500.0,
        initial_stop_loss=73000.0,
        current_stop_loss=73000.0,
        take_profit=74500.0,
        rr_ratio=2.0,
        contract_lots=10,
        target_asset_qty=0.01,
        capital_at_risk_usd=5.0,
        signal_candle_high=73500.0,
        signal_candle_low=73000.0,
        signal_candle_close=73400.0,
        signal_bar_time=time.time(),
        created_at_epoch_s=time.time(),
        product_id=27,
        exchange_order_id=12345,
        child_stop_loss_order_id=67890,
    )


@pytest.mark.asyncio
async def test_mark_price_inversion_guard_rejects_long_sl_above_mark(btc_spec: ContractSpec, sample_intent: OrderIntent) -> None:
    """Verifies that an SL ratchet at or above current mark price is rejected to prevent instant market liquidation."""
    client = MagicMock(spec=DeltaRestClient)
    client.get_cached_contract_spec.return_value = btc_spec
    state_mgr = MagicMock(spec=TradingStateManager)
    # Mark price is currently 73500.0
    state_mgr._cached_mark_prices = {"BTCUSD": 73500.0}

    exec_mgr = ExecutionManager(
        client=client,
        config=ExecutionConfig(execution_mode=ExecutionMode.LIVE),
        risk_manager=MagicMock(spec=RiskManager),
        state_manager=state_mgr,
    )

    # Attempt to ratchet SL to 73500.0 (or 73499.8 >= mark - tick_size = 73499.5)
    # Should be rejected
    rejected = await exec_mgr.handle_trailing_stop_ratchet(
        order_intent=sample_intent,
        new_sl=73500.0,
        old_sl=73000.0,
    )
    assert rejected is False
    assert sample_intent.current_stop_loss == 73000.0

    # Safe ratchet: SL is safely below mark price (e.g. 73400.0 < 73499.5)
    client.edit_order = AsyncMock(return_value=MagicMock())
    allowed = await exec_mgr.handle_trailing_stop_ratchet(
        order_intent=sample_intent,
        new_sl=73400.0,
        old_sl=73000.0,
    )
    assert allowed is True
    assert sample_intent.current_stop_loss == 73400.0


@pytest.mark.asyncio
async def test_mark_price_inversion_guard_rejects_short_sl_below_mark(btc_spec: ContractSpec) -> None:
    """Verifies that a Short SL ratchet at or below current mark price is rejected to prevent instant liquidation."""
    client = MagicMock(spec=DeltaRestClient)
    client.get_cached_contract_spec.return_value = btc_spec
    state_mgr = MagicMock(spec=TradingStateManager)
    # Mark price is currently 73500.0
    state_mgr._cached_mark_prices = {"BTCUSD": 73500.0}

    exec_mgr = ExecutionManager(
        client=client,
        config=ExecutionConfig(execution_mode=ExecutionMode.LIVE),
        risk_manager=MagicMock(spec=RiskManager),
        state_manager=state_mgr,
    )

    short_intent = OrderIntent(
        symbol="BTCUSD",
        signal_type=SignalType.BEARISH_BREAKDOWN,
        status=OrderStatus.TRIGGERED,
        trigger_price=73500.0,
        initial_stop_loss=74000.0,
        current_stop_loss=74000.0,
        take_profit=72500.0,
        rr_ratio=2.0,
        contract_lots=10,
        target_asset_qty=0.01,
        capital_at_risk_usd=5.0,
        signal_candle_high=74000.0,
        signal_candle_low=73500.0,
        signal_candle_close=73600.0,
        signal_bar_time=time.time(),
        created_at_epoch_s=time.time(),
        product_id=27,
        exchange_order_id=12345,
        child_stop_loss_order_id=67890,
    )

    # Attempt to ratchet Short SL to 73500.0 (<= mark + tick_size = 73500.5)
    # Should be rejected
    rejected = await exec_mgr.handle_trailing_stop_ratchet(
        order_intent=short_intent,
        new_sl=73500.0,
        old_sl=74000.0,
    )
    assert rejected is False
    assert short_intent.current_stop_loss == 74000.0

    # Safe ratchet: Short SL is safely above mark price (73600.0 > 73500.5)
    client.edit_order = AsyncMock(return_value=MagicMock())
    allowed = await exec_mgr.handle_trailing_stop_ratchet(
        order_intent=short_intent,
        new_sl=73600.0,
        old_sl=74000.0,
    )
    assert allowed is True
    assert short_intent.current_stop_loss == 73600.0


def test_position_info_and_exchange_order_symbol_extraction() -> None:
    """Verifies that symbol is parsed correctly whether under 'symbol', 'product_symbol', or nested 'product.symbol'."""
    # Case 1: symbol directly in payload
    p1 = PositionInfo.from_api_dict({"product_id": 27, "symbol": "BTCUSD", "size": 5})
    assert p1.symbol == "BTCUSD"

    # Case 2: product_symbol in payload
    p2 = PositionInfo.from_api_dict({"product_id": 27, "product_symbol": "ETHUSD", "size": 10})
    assert p2.symbol == "ETHUSD"

    # Case 3: nested product dict
    p3 = PositionInfo.from_api_dict({"product_id": 27, "product": {"symbol": "SOLUSD"}, "size": 20})
    assert p3.symbol == "SOLUSD"

    # Same for ExchangeOrder
    o1 = ExchangeOrder.from_api_dict({"id": 1, "product_id": 27, "symbol": "BTCUSD", "size": 5})
    assert o1.symbol == "BTCUSD"

    o2 = ExchangeOrder.from_api_dict({"id": 2, "product_id": 27, "product_symbol": "ETHUSD", "size": 10})
    assert o2.symbol == "ETHUSD"

    o3 = ExchangeOrder.from_api_dict({"id": 3, "product_id": 27, "product": {"symbol": "SOLUSD"}, "size": 20})
    assert o3.symbol == "SOLUSD"


def test_ring_buffer_filters_heartbeats_and_expands_capacity() -> None:
    """Verifies that noisy websocket keep-alive messages do not push critical trading logs out of ring buffer."""
    assert log_handler.capacity >= 1000
    start_count = len(log_handler.buffer)

    import logging
    record = logging.LogRecord(
        name="delta_bot.websocket",
        level=logging.DEBUG,
        pathname="",
        lineno=0,
        msg="Received keep-alive message [heartbeat]",
        args=(),
        exc_info=None,
    )
    log_handler.emit(record)
    # Filtered out -> count did not change
    assert len(log_handler.buffer) == start_count

    # Non-heartbeat log is kept
    record2 = logging.LogRecord(
        name="delta_bot.execution",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Order executed successfully",
        args=(),
        exc_info=None,
    )
    log_handler.emit(record2)
    assert len(log_handler.buffer) == start_count + 1


def test_tenant_exchange_history_endpoint() -> None:
    """Verifies that the exchange history endpoint queries Delta orders and fills."""
    from delta_bot.paper import PaperExchangeSimulator
    from delta_bot.orchestrator import TradingBotOrchestrator

    cfg = BotConfig(execution_mode="paper")
    sim = PaperExchangeSimulator(initial_balance=10000.0)
    orch = TradingBotOrchestrator(config=cfg, mode=ExecutionMode.PAPER, paper_simulator=sim)
    app = create_app(orch)
    client = TestClient(app)

    auth_header = {"Authorization": "Bearer ChangeMe123!Secure"}
    resp = client.get("/api/tenants/default/exchange-history", headers=auth_header)
    assert resp.status_code == 200
    data = resp.json()
    assert data["tenant_id"] == "default"
    assert "orders" in data
    assert "fills" in data


def make_candle(
    symbol: str = "BTCUSD",
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
    symbol: str = "BTCUSD",
    count: int = 50,
    base_price: float = 50000.0,
    step: float = 10.0,
) -> None:
    start_time = 1700000000.0
    for i in range(count):
        price = base_price + (i * step)
        c = make_candle(
            symbol=symbol,
            open_p=price - 5.0,
            high_p=price + 15.0,
            low_p=price - 10.0,
            close_p=price,
            epoch_s=start_time + (i * 60),
        )
        bar_cache.append(c)


@pytest.mark.asyncio
async def test_breakeven_and_trailing_only_after_1r_hit_long(btc_spec: ContractSpec) -> None:
    """Verifies that for a Long position:
    1. SL stays at initial SL while floating profit is < 1.0R (no breakeven, no trailing).
    2. Once high reaches >= 1.0R, r1_hit is flagged, SL locks break-even (+1 tick).
    3. Afterwards, trailing stop trails higher with price action above break-even.
    """
    from delta_bot.models import CircularBarCache, SessionAnchorData

    cfg = StrategyConfig(
        symbol="BTCUSD",
        enable_breakeven=True,
        breakeven_trigger_r=1.0,
        trailing_stop=True,
        trailing_mode="candle",
        price_buffer_ticks=2,
        trail_trigger_rr=1.0,
    )
    engine = BreakoutStrategyEngine(cfg)
    bar_cache = CircularBarCache(capacity=500)
    seed_synthetic_cache(bar_cache, symbol="BTCUSD", count=45, base_price=50000.0, step=20.0)

    anchor = SessionAnchorData(
        symbol="BTCUSD",
        session_date="2026-11-12",
        session_start_epoch_s=1700000000.0,
        prior_day_high=50800.0,
        session_vwap=50200.0,
    )

    # Signal bar: High 51000.0, Low 50600.0, Close 50950.0
    # Entry Trigger = 51000.5, SL = 50600.0
    # Risk distance R = 51000.5 - 50600.0 = 400.5
    # 1.0R Target = 51000.5 + 400.5 = 51401.0
    sig_bar = make_candle(symbol="BTCUSD", open_p=50800.0, high_p=51000.0, low_p=50600.0, close_p=50950.0, epoch_s=1700003000.0)
    bar_cache.append(sig_bar)
    intent = await engine.on_candle_close(sig_bar, anchor, btc_spec, bar_cache)
    assert intent is not None
    assert intent.signal_type == SignalType.BULLISH_BREAKOUT
    initial_sl = intent.current_stop_loss
    assert initial_sl == 50600.0

    # Trigger position
    trig_bar = make_candle(symbol="BTCUSD", open_p=50950.0, high_p=51050.0, low_p=50900.0, close_p=51020.0, epoch_s=1700003060.0)
    bar_cache.append(trig_bar)
    await engine.on_candle_close(trig_bar, anchor, btc_spec, bar_cache)
    assert engine.active_position is not None
    assert engine.active_position.is_triggered
    # Protect against offline TP auto-closure during trailing test
    engine.active_position.take_profit = 99999.0

    ratchets: list[tuple[float, float]] = []
    engine.on_sl_ratchet(lambda order, old_sl, new_sl: ratchets.append((old_sl, new_sl)))

    # Step 1: Sub-1R Candle (High=51300.0 < 51401.0 Target -> ~0.75R profit)
    # Stop loss MUST NOT ratchet, must stay at initial SL (50600.0)
    sub_1r_bar = make_candle(symbol="BTCUSD", open_p=51020.0, high_p=51300.0, low_p=51000.0, close_p=51250.0, epoch_s=1700003120.0)
    bar_cache.append(sub_1r_bar)
    await engine.on_candle_close(sub_1r_bar, anchor, btc_spec, bar_cache)

    assert engine.active_position.r1_hit is False
    assert engine.active_position.is_breakeven_locked is False
    assert engine.active_position.current_stop_loss == initial_sl == 50600.0
    assert len(ratchets) == 0

    # Step 2: 1.0R Target Hit (High=51450.0 >= 51401.0 Target)
    # Candle low is 51010.0 (near entry).
    # Break-even price is 51000.5 + 0.5 (1 tick) = 51001.0
    # Stop loss must be ratcheted to at least break-even!
    hit_1r_bar = make_candle(symbol="BTCUSD", open_p=51250.0, high_p=51450.0, low_p=51010.0, close_p=51400.0, epoch_s=1700003180.0)
    bar_cache.append(hit_1r_bar)
    await engine.on_candle_close(hit_1r_bar, anchor, btc_spec, bar_cache)

    assert engine.active_position.r1_hit is True
    assert engine.active_position.is_breakeven_locked is True
    assert engine.active_position.current_stop_loss >= 51001.0
    assert len(ratchets) == 1

    # Step 3: Trailing Afterwards (Climbing to 51750.0, Low=51500.0)
    # Raw trailing = low - 2 ticks = 51500.0 - 1.0 = 51499.0 > Breakeven
    # SL must ratchet higher, trailing with the candle low!
    trail_bar = make_candle(symbol="BTCUSD", open_p=51400.0, high_p=51750.0, low_p=51500.0, close_p=51700.0, epoch_s=1700003240.0)
    bar_cache.append(trail_bar)
    await engine.on_candle_close(trail_bar, anchor, btc_spec, bar_cache)

    assert engine.active_position.current_stop_loss >= 51499.0
    assert len(ratchets) == 2


@pytest.mark.asyncio
async def test_breakeven_and_trailing_only_after_1r_hit_short(btc_spec: ContractSpec) -> None:
    """Verifies that for a Short position:
    1. SL stays at initial SL while floating profit is < 1.0R (no breakeven, no trailing).
    2. Once low reaches <= 1.0R, r1_hit is flagged, SL locks break-even (-1 tick).
    3. Afterwards, trailing stop trails lower with price action below break-even.
    """
    from delta_bot.models import CircularBarCache, SessionAnchorData

    cfg = StrategyConfig(
        symbol="BTCUSD",
        enable_breakeven=True,
        breakeven_trigger_r=1.0,
        trailing_stop=True,
        trailing_mode="candle",
        price_buffer_ticks=2,
        trail_trigger_rr=1.0,
    )
    engine = BreakoutStrategyEngine(cfg)
    bar_cache = CircularBarCache(capacity=500)
    # Descending bars
    start_time = 1700000000.0
    for i in range(45):
        price = 55000.0 - (i * 20.0)
        c = make_candle(
            symbol="BTCUSD",
            open_p=price + 5.0,
            high_p=price + 10.0,
            low_p=price - 15.0,
            close_p=price,
            epoch_s=start_time + (i * 60),
        )
        bar_cache.append(c)

    anchor = SessionAnchorData(
        symbol="BTCUSD",
        session_date="2026-11-12",
        session_start_epoch_s=1700000000.0,
        prior_day_low=54500.0,
        session_vwap=54800.0,
    )

    # Signal bar generates a Short order intent:
    # High 54400.0, Low 54000.0 -> Sell Stop = 53999.5, SL = 54400.0
    # Risk distance R = 54400.0 - 53999.5 = 400.5
    # 1.0R Target = 53999.5 - 400.5 = 53599.0
    sig_bar = make_candle(symbol="BTCUSD", open_p=54200.0, high_p=54400.0, low_p=54000.0, close_p=54050.0, epoch_s=1700003000.0)
    bar_cache.append(sig_bar)
    intent = await engine.on_candle_close(sig_bar, anchor, btc_spec, bar_cache)
    assert intent is not None
    assert intent.signal_type == SignalType.BEARISH_BREAKDOWN
    initial_sl = intent.current_stop_loss
    assert initial_sl == 54400.0

    # Trigger bar triggers entry
    trig_bar = make_candle(symbol="BTCUSD", open_p=54050.0, high_p=54100.0, low_p=53950.0, close_p=53980.0, epoch_s=1700003060.0)
    bar_cache.append(trig_bar)
    await engine.on_candle_close(trig_bar, anchor, btc_spec, bar_cache)
    assert engine.active_position is not None
    assert engine.active_position.is_triggered
    # Protect against offline TP auto-closure during trailing test
    engine.active_position.take_profit = 1000.0

    ratchets: list[tuple[float, float]] = []
    engine.on_sl_ratchet(lambda order, old_sl, new_sl: ratchets.append((old_sl, new_sl)))

    # Step 1: Sub-1R Candle (Low=53700.0 > 53599.0 Target -> ~0.75R profit)
    # Stop loss MUST NOT ratchet, must stay at initial SL (54400.0)
    sub_1r_bar = make_candle(symbol="BTCUSD", open_p=53980.0, high_p=54000.0, low_p=53700.0, close_p=53750.0, epoch_s=1700003120.0)
    bar_cache.append(sub_1r_bar)
    await engine.on_candle_close(sub_1r_bar, anchor, btc_spec, bar_cache)

    assert engine.active_position.r1_hit is False
    assert engine.active_position.is_breakeven_locked is False
    assert engine.active_position.current_stop_loss == initial_sl == 54400.0
    assert len(ratchets) == 0

    # Step 2: 1.0R Target Hit (Low=53550.0 <= 53599.0 Target)
    # Candle high is 53990.0 (near entry).
    # Break-even price is 53999.5 - 0.5 (1 tick) = 53999.0
    # Stop loss must be ratcheted to at least break-even!
    hit_1r_bar = make_candle(symbol="BTCUSD", open_p=53750.0, high_p=53990.0, low_p=53550.0, close_p=53600.0, epoch_s=1700003180.0)
    bar_cache.append(hit_1r_bar)
    await engine.on_candle_close(hit_1r_bar, anchor, btc_spec, bar_cache)

    assert engine.active_position.r1_hit is True
    assert engine.active_position.is_breakeven_locked is True
    assert engine.active_position.current_stop_loss <= 53999.0
    assert len(ratchets) == 1

    # Step 3: Trailing Afterwards (Dropping to 53250.0, High=53500.0)
    # Raw trailing = high + 2 ticks = 53500.0 + 1.0 = 53501.0 < Breakeven
    # SL must ratchet lower, trailing with the candle high!
    trail_bar = make_candle(symbol="BTCUSD", open_p=53600.0, high_p=53500.0, low_p=53250.0, close_p=53300.0, epoch_s=1700003240.0)
    bar_cache.append(trail_bar)
    await engine.on_candle_close(trail_bar, anchor, btc_spec, bar_cache)

    assert engine.active_position.current_stop_loss <= 53501.0
    assert len(ratchets) == 2


@pytest.mark.asyncio
async def test_tenant_live_reconciliation_only_breakevens_after_1r(btc_spec: ContractSpec) -> None:
    """Verifies that in Tenant live reconciliation:
    1. A position with floating profit < 1.0R does NOT get ratcheted to breakeven.
    2. Once mark price reaches >= 1.0R, r1_hit is set and breakeven ratchet is safely dispatched.
    """
    from delta_bot.tenant import TenantInstance, TenantConfig

    tenant_cfg = TenantConfig(
        tenant_id="rupdave-test",
        name="RupDave Test",
        delta_api_key="test_key",
        delta_api_secret="test_secret",
        is_live=True,
        symbols=["BTCUSD"],
        trail_trigger_rr=1.0,
        enable_breakeven=True,
    )

    tenant = TenantInstance(tenant_cfg)
    tenant.contract_specs = {"BTCUSD": btc_spec}
    tenant.execution_manager = MagicMock()
    tenant.execution_manager.handle_trailing_stop_ratchet = AsyncMock(return_value=True)

    # Setup active long position in strategy
    # Entry = 73500.0, Initial SL = 73000.0 (R = 500.0, 1R target = 74000.0)
    strat = tenant.strategies["BTCUSD"]
    intent = OrderIntent(
        symbol="BTCUSD",
        signal_type=SignalType.BULLISH_BREAKOUT,
        status=OrderStatus.TRIGGERED,
        trigger_price=73500.0,
        fill_price=73500.0,
        initial_stop_loss=73000.0,
        current_stop_loss=73000.0,
        take_profit=74500.0,
        rr_ratio=2.0,
        contract_lots=5,
        target_asset_qty=0.005,
        capital_at_risk_usd=2.5,
        signal_candle_high=73500.0,
        signal_candle_low=73000.0,
        signal_candle_close=73400.0,
        signal_bar_time=time.time(),
        created_at_epoch_s=time.time(),
        product_id=27,
        exchange_order_id=123,
        child_stop_loss_order_id=456,
    )
    strat.active_position = intent

    # Mock Delta Exchange live positions and orders
    pos1 = PositionInfo(
        product_id=27,
        symbol="BTCUSD",
        size=5,
        entry_price=73500.0,
        mark_price=73700.0,  # Sub-1R (Mark 73700 < 74000)
        margin=10.0,
        liquidation_price=70000.0,
        bankruptcy_price=69000.0,
        unrealized_pnl=1.0,
        realized_pnl=0.0,
    )
    ord1 = ExchangeOrder(
        id=456,
        product_id=27,
        symbol="BTCUSD",
        order_type="stop_market",
        stop_price=73000.0,
        side="sell",
        state="open",
        size=5,
        unfilled_size=5,
    )

    tenant.rest_client.get_wallet_balances = AsyncMock(return_value=[])
    tenant.rest_client.get_open_positions = AsyncMock(return_value=[pos1])
    tenant.rest_client.get_open_orders = AsyncMock(return_value=[ord1])

    # Reconcile when mark price is 73700.0 (< 1.0R)
    await tenant.sync_live_metrics()

    # Must NOT have hit 1R, and must NOT have ratcheted SL
    assert intent.r1_hit is False
    assert intent.current_stop_loss == 73000.0
    assert tenant.execution_manager.handle_trailing_stop_ratchet.await_count == 0

    # Now mark price surges past 1.0R (Mark = 74100.0 >= 74000.0 Target)
    pos2 = PositionInfo(
        product_id=27,
        symbol="BTCUSD",
        size=5,
        entry_price=73500.0,
        mark_price=74100.0,  # Past 1R (Mark 74100 >= 74000)
        margin=10.0,
        liquidation_price=70000.0,
        bankruptcy_price=69000.0,
        unrealized_pnl=3.0,
        realized_pnl=0.0,
    )
    tenant.rest_client.get_open_positions = AsyncMock(return_value=[pos2])

    await tenant.sync_live_metrics()

    # Must have unlocked 1R and dispatched breakeven ratchet!
    assert intent.r1_hit is True
    assert intent.is_breakeven_locked is True
    assert intent.current_stop_loss == 73500.5  # Entry + 1 tick
    assert tenant.execution_manager.handle_trailing_stop_ratchet.await_count == 1


@pytest.mark.asyncio
async def test_orphaned_reduce_only_orders_purged_when_no_position(btc_spec: ContractSpec) -> None:
    """Verifies that reduce-only / child exit orders lingering with no active position are purged,
    and are never mistakenly treated as resting entry stop orders with new brackets attached."""
    from delta_bot.tenant import TenantInstance, TenantConfig

    tenant_cfg = TenantConfig(
        tenant_id="rupdave-test",
        name="RupDave Test",
        delta_api_key="test_key",
        delta_api_secret="test_secret",
        is_live=True,
        symbols=["BTCUSD"],
    )

    tenant = TenantInstance(tenant_cfg)
    tenant.contract_specs = {"BTCUSD": btc_spec}
    tenant.rest_client.get_wallet_balances = AsyncMock(return_value=[])
    tenant.rest_client.get_open_positions = AsyncMock(return_value=[])  # No position open!
    tenant.rest_client.cancel_order = AsyncMock(return_value=True)
    tenant.rest_client.edit_bracket_order = AsyncMock()

    # Orphaned stop loss order from past closed trade
    orphaned_sl = ExchangeOrder(
        id=777,
        product_id=27,
        symbol="BTCUSD",
        side="sell",
        order_type="market_order",
        stop_price=72000.0,
        stop_order_type="stop_loss_order",
        reduce_only=True,
        state="open",
        size=10,
    )
    tenant.rest_client.get_open_orders = AsyncMock(return_value=[orphaned_sl])

    await tenant.sync_live_metrics()

    # Must cancel the orphaned order
    tenant.rest_client.cancel_order.assert_awaited_once_with(product_id=27, order_id=777)
    # Must NOT attach brackets to an orphaned child order!
    assert tenant.rest_client.edit_bracket_order.await_count == 0


@pytest.mark.asyncio
async def test_position_close_cancels_counterpart_child_orders(btc_spec: ContractSpec) -> None:
    """Verifies that when a position closes on TP, the counterpart child SL order is cancelled on exchange."""
    from delta_bot.tenant import TenantInstance, TenantConfig

    tenant_cfg = TenantConfig(
        tenant_id="rupdave-test",
        name="RupDave Test",
        delta_api_key="test_key",
        delta_api_secret="test_secret",
        is_live=True,
        symbols=["BTCUSD"],
    )

    tenant = TenantInstance(tenant_cfg)
    tenant.contract_specs = {"BTCUSD": btc_spec}
    tenant.execution_manager = MagicMock()
    tenant.execution_manager.client = MagicMock()
    tenant.execution_manager.client.cancel_order = AsyncMock(return_value=True)

    intent = OrderIntent(
        symbol="BTCUSD",
        signal_type=SignalType.BULLISH_BREAKOUT,
        status=OrderStatus.TRIGGERED,
        trigger_price=73500.0,
        initial_stop_loss=73000.0,
        current_stop_loss=73500.5,
        take_profit=74500.0,
        rr_ratio=2.0,
        contract_lots=5,
        target_asset_qty=0.005,
        capital_at_risk_usd=2.5,
        signal_candle_high=73500.0,
        signal_candle_low=73000.0,
        signal_candle_close=73400.0,
        signal_bar_time=time.time(),
        created_at_epoch_s=time.time(),
        product_id=27,
        child_stop_loss_order_id=888,
        child_take_profit_order_id=999,
    )

    # Position closes by TP
    await tenant._handle_position_close(intent, OrderStatus.CLOSED_TP, 74500.0)

    # Counterpart child SL order #888 MUST be cancelled!
    tenant.execution_manager.client.cancel_order.assert_awaited_once_with(product_id=27, order_id=888)


@pytest.mark.asyncio
async def test_stop_order_slippage_guard_rejects_excessive_runaway(btc_spec: ContractSpec) -> None:
    """Verifies that if Delta returns immediate_execution_stop_order, but price has run away far,
    the market order fallback is aborted to prevent chasing at peak prices."""
    from delta_bot.execution import OrderRouter
    from delta_bot.exceptions import DeltaAPIError

    mock_client = MagicMock(spec=DeltaRestClient)
    # First call to place_order fails with immediate_execution_stop_order
    mock_client.place_order = AsyncMock(
        side_effect=DeltaAPIError(
            message="Immediate execution stop order",
            status_code=400,
            error_code="immediate_execution_stop_order",
        )
    )
    # Mark price has run away by 1000 points (from 73500 to 74500)
    mock_ticker = MagicMock()
    mock_ticker.mark_price = 74500.0
    mock_client.get_ticker = AsyncMock(return_value=mock_ticker)

    router = OrderRouter(mock_client, ExecutionConfig(execution_mode=ExecutionMode.LIVE))

    with pytest.raises(DeltaAPIError):
        await router.submit_native_stop_order(
            product_id=27,
            symbol="BTCUSD",
            side="buy",
            size=5,
            stop_price=73500.0,
            bracket_stop_loss_price=73000.0,
            bracket_take_profit_price=74500.0,
            client_order_id="test_cid",
        )

    # place_order must have been called only once (the stop order), never the fallback market order!
    assert mock_client.place_order.await_count == 1



