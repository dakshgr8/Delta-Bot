"""
tests/test_orchestrator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit & Integration Test Suite for Delta-Bot Phase 6 Orchestrator & Deployment Engine.
Covers:
1. Operational mode switching (Paper vs Live)
2. Credential validation & ConfigurationError enforcement
3. Multi-asset initialization & ContractSpec caching
4. End-to-end Paper Breakout flow (Candle Close -> Intent -> Stop Order -> Tick Fill -> State Update)
5. Paper Take Profit execution & VirtualWallet accounting
6. Dynamic Trailing Stop Ratchet event handling
7. Intermediate candle EMA invalidation cancellation
8. Daily Session Rollover (00:00 UTC) & circuit breaker reset
9. Graceful shutdown without flatten (preserves exchange brackets)
10. Graceful shutdown with emergency flatten (market close)
11. Health telemetry snapshot & ASCII dashboard formatting
12. RiskManager circuit breaker pre-trade rejection
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from delta_bot.config import BotConfig
from delta_bot.exceptions import ConfigurationError
from delta_bot.execution import ExecutionManager
from delta_bot.models import (
    Candle,
    ContractSpec,
    ExecutionMode,
    OrderBook,
    OrderBookLevel,
    PositionInfo,
    SessionAnchorData,
    Ticker,
)
from delta_bot.orchestrator import TradingBotOrchestrator
from delta_bot.paper import PaperExchangeSimulator, VirtualOrder, VirtualPosition, VirtualWallet
from delta_bot.risk import RiskManager
from delta_bot.state import TradingStateManager
from delta_bot.strategy import (
    BreakoutStrategyEngine,
    OrderIntent,
    OrderStatus,
    SignalType,
    StrategyConfig,
)


def make_dummy_spec(
    symbol: str = "BTCUSDT", product_id: int = 1, contract_value: float = 0.001
) -> ContractSpec:
    """Helper creating typed ContractSpec for test fixtures."""
    return ContractSpec(
        product_id=product_id,
        symbol=symbol,
        contract_type="perpetual_futures",
        contract_value=contract_value,
        tick_size=0.5,
        maker_commission_rate=0.0002,
        taker_commission_rate=0.0005,
        initial_margin_rate=0.02,
        maintenance_margin_rate=0.01,
    )


def make_closed_candle(
    symbol: str = "BTCUSDT",
    close: float = 68000.0,
    high: float = 68500.0,
    low: float = 67500.0,
    open_px: float = 67800.0,
    volume: float = 10.0,
    timestamp: float = 1712750400.0,
) -> Candle:
    """Helper creating typed closed Candle."""
    return Candle(
        symbol=symbol,
        resolution="1m",
        candle_start_time_usec=int(timestamp * 1_000_000),
        candle_start_epoch_s=timestamp,
        open=open_px,
        high=high,
        low=low,
        close=close,
        volume=volume,
        is_closed=True,
    )


def make_dummy_intent(
    symbol: str = "BTCUSDT",
    signal_type: SignalType = SignalType.BULLISH_BREAKOUT,
    trigger_price: float = 68000.0,
    initial_stop_loss: float = 67000.0,
    take_profit: float = 70000.0,
    contract_lots: int = 10,
    exchange_order_id: int | None = None,
) -> OrderIntent:
    """Helper creating fully populated typed OrderIntent."""
    return OrderIntent(
        symbol=symbol,
        signal_type=signal_type,
        status=OrderStatus.PENDING,
        trigger_price=trigger_price,
        initial_stop_loss=initial_stop_loss,
        current_stop_loss=initial_stop_loss,
        take_profit=take_profit,
        rr_ratio=2.0,
        contract_lots=contract_lots,
        target_asset_qty=0.01,
        capital_at_risk_usd=100.0,
        signal_candle_high=trigger_price - 50.0,
        signal_candle_low=initial_stop_loss,
        signal_candle_close=trigger_price - 100.0,
        signal_bar_time=1712750400.0,
        created_at_epoch_s=1712750400.0,
        exchange_order_id=exchange_order_id,
    )


# -----------------------------------------------------------------------------
# 1. Initialization & Configuration Tests
# -----------------------------------------------------------------------------

def test_orchestrator_initialization_paper_mode() -> None:
    """Verifies that Paper mode initializes simulator, suppresses private WS, and creates strategies."""
    cfg = BotConfig(
        execution_mode="paper",
        target_symbols_str="BTCUSDT,ETHUSDT",
    )
    orch = TradingBotOrchestrator(config=cfg, mode=ExecutionMode.PAPER)

    assert orch.mode == ExecutionMode.PAPER
    assert orch.paper_simulator is not None
    assert orch.execution_manager is None
    assert orch.private_ws is None
    assert len(orch.strategies) == 2
    assert "BTCUSDT" in orch.strategies
    assert "ETHUSDT" in orch.strategies
    assert not orch.is_running
    assert not orch._shutdown_completed


def test_orchestrator_initialization_live_mode_missing_credentials() -> None:
    """Verifies that LIVE mode without valid credentials raises ConfigurationError."""
    cfg = BotConfig(
        delta_api_key="",
        delta_api_secret="",
        execution_mode="live",
    )
    with pytest.raises(ConfigurationError) as exc_info:
        TradingBotOrchestrator(config=cfg, mode=ExecutionMode.LIVE)

    assert "requires valid DELTA_API_KEY" in str(exc_info.value)


def test_orchestrator_initialization_live_mode_with_credentials() -> None:
    """Verifies that LIVE mode with valid credentials initializes ExecutionManager and PrivateWS."""
    cfg = BotConfig(
        delta_api_key="valid_live_key_9999",
        delta_api_secret="valid_live_secret_8888",
        execution_mode="live",
        target_symbols_str="SOLUSDT",
    )
    orch = TradingBotOrchestrator(config=cfg, mode=ExecutionMode.LIVE)

    assert orch.mode == ExecutionMode.LIVE
    assert orch.execution_manager is not None
    assert orch.private_ws is not None
    assert orch.paper_simulator is None
    assert "SOLUSDT" in orch.strategies


# -----------------------------------------------------------------------------
# 2. End-to-End Paper Breakout Flow
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_end_to_end_paper_breakout_flow() -> None:
    """
    Tests end-to-end event chain in Paper mode:
    Candle Close -> Breakout Signal -> Risk Check -> Stop Order -> Tick Triggers Fill -> State Update
    """
    cfg = BotConfig(
        execution_mode="paper",
        target_symbols_str="BTCUSDT",
        default_risk_pct=1.0,
    )
    paper_sim = PaperExchangeSimulator(initial_balance=10000.0)
    orch = TradingBotOrchestrator(config=cfg, mode=ExecutionMode.PAPER, paper_simulator=paper_sim)

    # Seed contract spec
    spec = make_dummy_spec("BTCUSDT", product_id=1)
    orch.contract_specs["BTCUSDT"] = spec
    paper_sim.register_contract_spec(spec)

    # Mock strategy to return a synthetic OrderIntent upon candle close
    intent = make_dummy_intent(
        symbol="BTCUSDT",
        signal_type=SignalType.BULLISH_BREAKOUT,
        trigger_price=68000.0,
        initial_stop_loss=67000.0,
        take_profit=70000.0,
        contract_lots=10,
    )

    strategy_mock = MagicMock()
    strategy_mock.on_candle_close = AsyncMock(return_value=intent)
    orch.strategies["BTCUSDT"] = strategy_mock

    candle = make_closed_candle(symbol="BTCUSDT", close=67900.0, high=67950.0, low=67000.0)

    # 1. Trigger candle close
    await orch._handle_candle_close("BTCUSDT", candle)

    # Verify stop order was placed in PaperExchangeSimulator
    assert len(paper_sim._pending_orders) == 1
    v_order = list(paper_sim._pending_orders.values())[0]
    assert v_order.stop_price == 68000.0
    assert v_order.side == "buy"
    assert intent.exchange_order_id == v_order.order_id

    # Verify intent registered in state manager
    registered_intent = orch.state_manager.get_intent_by_exchange_id(v_order.order_id)
    assert registered_intent is not None
    assert registered_intent.intent_id == intent.intent_id

    # 2. Trigger price tick crossing the stop price (68050 >= 68000)
    ticker = Ticker(
        symbol="BTCUSDT",
        product_id=1,
        mark_price=68050.0,
        bid_price=68045.0,
        ask_price=68050.0,
    )
    await orch._handle_ticker("BTCUSDT", ticker)

    # Verify resting order was filled and moved to open positions
    assert len(paper_sim._pending_orders) == 0
    assert len(paper_sim._open_positions) == 1

    pos = paper_sim._open_positions[1]
    assert pos.symbol == "BTCUSDT"
    assert pos.size == v_order.size
    assert pos.entry_price >= 68000.0

    # Verify StateManager received child stop loss order
    child_sl_id = orch.state_manager.get_child_stop_loss_order_id(product_id=1)
    assert child_sl_id is not None
    assert child_sl_id == pos.child_stop_loss_order_id


@pytest.mark.asyncio
async def test_orchestrator_paper_take_profit_execution() -> None:
    """Verifies that active paper position triggers take-profit, updates PnL, and adjusts wallet."""
    paper_sim = PaperExchangeSimulator(initial_balance=10000.0)
    orch = TradingBotOrchestrator(
        config=BotConfig(execution_mode="paper", target_symbols_str="ETHUSDT"),
        mode=ExecutionMode.PAPER,
        paper_simulator=paper_sim,
    )
    spec = make_dummy_spec("ETHUSDT", product_id=2, contract_value=0.01)
    orch.contract_specs["ETHUSDT"] = spec
    paper_sim.register_contract_spec(spec)

    # Submit stop order directly into paper sim
    v_order = paper_sim.submit_stop_order(
        product_id=2,
        symbol="ETHUSDT",
        side="buy",
        size=100,
        stop_price=3500.0,
        bracket_stop_loss_price=3400.0,
        bracket_take_profit_price=3700.0,
    )

    # Tick 1: Fill order at 3500
    await paper_sim.on_tick("ETHUSDT", mark_price=3505.0)
    assert len(paper_sim._open_positions) == 1
    assert len(paper_sim._pending_orders) == 0

    # Tick 2: Trigger Take Profit at 3710.0 (TP was 3700.0)
    execs = await paper_sim.on_tick("ETHUSDT", mark_price=3710.0)

    assert len(paper_sim._open_positions) == 0
    assert any(e["event"] == "position_closed" and e["reason"] == "TAKE_PROFIT" for e in execs)

    # Wallet verification: trade was profitable
    assert paper_sim.wallet.total_trades == 1
    assert paper_sim.wallet.winning_trades == 1
    assert paper_sim.wallet.realized_pnl > 0.0
    assert paper_sim.wallet.balance > 10000.0
    assert paper_sim.wallet.win_rate_pct == 100.0


# -----------------------------------------------------------------------------
# 3. Dynamic Events: Trailing Stop Ratchet & Invalidation
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_sl_ratchet_paper_and_live() -> None:
    """Verifies SL ratchet routing in both Paper and Live modes."""
    # 1. Paper Mode Ratchet
    paper_sim = PaperExchangeSimulator(initial_balance=10000.0)
    orch_paper = TradingBotOrchestrator(
        config=BotConfig(execution_mode="paper", target_symbols_str="BTCUSDT"),
        mode=ExecutionMode.PAPER,
        paper_simulator=paper_sim,
    )
    spec = make_dummy_spec("BTCUSDT", product_id=1)
    orch_paper.contract_specs["BTCUSDT"] = spec
    paper_sim.register_contract_spec(spec)

    v_order = paper_sim.submit_stop_order(
        product_id=1,
        symbol="BTCUSDT",
        side="buy",
        size=10,
        stop_price=68000.0,
        bracket_stop_loss_price=67000.0,
    )

    intent = make_dummy_intent(
        symbol="BTCUSDT",
        signal_type=SignalType.BULLISH_BREAKOUT,
        trigger_price=68000.0,
        initial_stop_loss=67000.0,
        take_profit=70000.0,
        exchange_order_id=v_order.id,
    )

    # Dispatch ratchet: 67000 -> 67500
    await orch_paper._handle_sl_ratchet(intent, old_sl=67000.0, new_sl=67500.0)
    assert paper_sim._pending_orders[v_order.id].bracket_stop_loss_price == 67500.0

    # 2. Live Mode Ratchet
    cfg_live = BotConfig(
        delta_api_key="validkey123",
        delta_api_secret="validsecret123",
        execution_mode="live",
        target_symbols_str="BTCUSDT",
    )
    orch_live = TradingBotOrchestrator(config=cfg_live, mode=ExecutionMode.LIVE)
    orch_live.execution_manager.handle_trailing_stop_ratchet = AsyncMock()

    await orch_live._handle_sl_ratchet(intent, old_sl=67000.0, new_sl=67500.0)
    orch_live.execution_manager.handle_trailing_stop_ratchet.assert_awaited_once_with(
        order_intent=intent,
        old_sl=67000.0,
        new_sl=67500.0,
    )


@pytest.mark.asyncio
async def test_orchestrator_order_cancel_invalidation() -> None:
    """Verifies that intermediate EMA invalidation cancels resting orders."""
    paper_sim = PaperExchangeSimulator(initial_balance=10000.0)
    orch = TradingBotOrchestrator(
        config=BotConfig(execution_mode="paper", target_symbols_str="BTCUSDT"),
        mode=ExecutionMode.PAPER,
        paper_simulator=paper_sim,
    )
    spec = make_dummy_spec("BTCUSDT", product_id=1)
    orch.contract_specs["BTCUSDT"] = spec
    paper_sim.register_contract_spec(spec)

    v_order = paper_sim.submit_stop_order(
        product_id=1,
        symbol="BTCUSDT",
        side="buy",
        size=10,
        stop_price=68000.0,
    )
    assert len(paper_sim._pending_orders) == 1

    intent = make_dummy_intent(
        symbol="BTCUSDT",
        signal_type=SignalType.BULLISH_BREAKOUT,
        trigger_price=68000.0,
        initial_stop_loss=67000.0,
        exchange_order_id=v_order.id,
    )

    # Cancel order on invalidation
    await orch._handle_order_cancel(intent, reason="Intermediate candle closed below EMA 20")
    assert len(paper_sim._pending_orders) == 0


# -----------------------------------------------------------------------------
# 4. Session Rollover & Circuit Breaker Reset
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_session_rollover_resets_risk() -> None:
    """Verifies that 00:00 UTC rollover updates PDH/PDL and resets circuit breaker baseline."""
    orch = TradingBotOrchestrator(
        config=BotConfig(execution_mode="paper", target_symbols_str="SOLUSDT"),
        mode=ExecutionMode.PAPER,
    )

    # Simulate equity drawdown and trip circuit breaker
    orch.risk_manager.circuit_breaker.trip("Simulated max drawdown breach")
    assert orch.risk_manager.is_circuit_breaker_tripped

    # Create rollover session anchor
    session = SessionAnchorData(
        symbol="SOLUSDT",
        session_date="2026-04-12",
        session_start_epoch_s=1712880000.0,
        prior_day_high=180.50,
        prior_day_low=165.20,
        prior_day_close=175.00,
        session_open=175.00,
        session_high=175.00,
        session_low=175.00,
        session_close=175.00,
    )

    # Trigger session rollover
    await orch._handle_session_rollover("SOLUSDT", session)

    # Circuit breaker should now be reset
    assert not orch.risk_manager.is_circuit_breaker_tripped
    assert orch.risk_manager.circuit_breaker.session_start_equity == 10000.0


# -----------------------------------------------------------------------------
# 5. Graceful Shutdown & Flatten Policy
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_shutdown_without_flatten() -> None:
    """
    Tests shutdown with FLATTEN_ON_SHUTDOWN=False:
    Resting stop orders are cancelled, but open positions remain preserved with brackets.
    """
    paper_sim = PaperExchangeSimulator(initial_balance=10000.0)
    orch = TradingBotOrchestrator(
        config=BotConfig(
            execution_mode="paper",
            target_symbols_str="BTCUSDT",
            flatten_on_shutdown=False,
        ),
        mode=ExecutionMode.PAPER,
        paper_simulator=paper_sim,
    )
    spec = make_dummy_spec("BTCUSDT", product_id=1)
    orch.contract_specs["BTCUSDT"] = spec
    paper_sim.register_contract_spec(spec)

    # Create 1 resting stop order
    paper_sim.submit_stop_order(product_id=1, symbol="BTCUSDT", side="buy", size=5, stop_price=70000.0)
    assert len(paper_sim._pending_orders) == 1

    # Create 1 active open position manually
    paper_sim._open_positions[1] = VirtualPosition(
        product_id=1,
        symbol="BTCUSDT",
        side="buy",
        size=10,
        entry_price=68000.0,
        mark_price=68000.0,
        unrealized_pnl=0.0,
    )
    assert len(paper_sim._open_positions) == 1

    orch.public_ws.disconnect = AsyncMock()
    orch.rest_client.close = AsyncMock()

    await orch.shutdown(reason="Test shutdown without flatten")

    # Resting order cancelled
    assert len(paper_sim._pending_orders) == 0
    # Active position kept intact
    assert len(paper_sim._open_positions) == 1
    assert orch._shutdown_completed
    orch.public_ws.disconnect.assert_awaited_once()
    orch.rest_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_orchestrator_shutdown_with_flatten() -> None:
    """
    Tests shutdown with FLATTEN_ON_SHUTDOWN=True:
    Resting stop orders are cancelled AND open positions are emergency flattened.
    """
    paper_sim = PaperExchangeSimulator(initial_balance=10000.0)
    orch = TradingBotOrchestrator(
        config=BotConfig(
            execution_mode="paper",
            target_symbols_str="BTCUSDT",
            flatten_on_shutdown=True,
        ),
        mode=ExecutionMode.PAPER,
        paper_simulator=paper_sim,
    )
    spec = make_dummy_spec("BTCUSDT", product_id=1)
    orch.contract_specs["BTCUSDT"] = spec
    paper_sim.register_contract_spec(spec)

    # Place and fill order
    paper_sim.submit_stop_order(product_id=1, symbol="BTCUSDT", side="buy", size=10, stop_price=68000.0)
    await paper_sim.on_tick("BTCUSDT", mark_price=68050.0)
    assert len(paper_sim._open_positions) == 1

    orch.public_ws.disconnect = AsyncMock()
    orch.rest_client.close = AsyncMock()

    await orch.shutdown(reason="Emergency shutdown with flatten")

    # Position flattened
    assert len(paper_sim._open_positions) == 0
    assert len(paper_sim._pending_orders) == 0
    assert orch._shutdown_completed


# -----------------------------------------------------------------------------
# 6. Telemetry & Dashboard Formatting
# -----------------------------------------------------------------------------

def test_orchestrator_telemetry_snapshot_and_dashboard() -> None:
    """Verifies that health telemetry snapshot and dashboard string format properly."""
    paper_sim = PaperExchangeSimulator(initial_balance=10000.0)
    orch = TradingBotOrchestrator(
        config=BotConfig(execution_mode="paper", target_symbols_str="BTCUSDT,ETHUSDT"),
        mode=ExecutionMode.PAPER,
        paper_simulator=paper_sim,
    )
    orch.is_running = True
    orch.start_time = time.time() - 3665  # 1 hour 1 minute 5 seconds ago

    snapshot = orch.get_telemetry_snapshot()
    assert snapshot["mode"] == "PAPER"
    assert snapshot["is_running"] is True
    assert snapshot["uptime_seconds"] >= 3660.0
    assert snapshot["balance"] == 10000.0
    assert snapshot["equity"] == 10000.0
    assert snapshot["circuit_breaker_tripped"] is False

    dashboard = orch.format_telemetry_dashboard()
    assert "DELTA-BOT HEALTH TELEMETRY" in dashboard
    assert "Mode: PAPER" in dashboard
    assert "Collateral: $10,000.00 Equity" in dashboard
    assert "Circuit Breaker: NORMAL" in dashboard


# -----------------------------------------------------------------------------
# 7. RiskManager Pre-Trade Circuit Breaker Rejection
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_risk_rejection_prevents_order() -> None:
    """Verifies that a tripped circuit breaker rejects breakout signals prior to order creation."""
    paper_sim = PaperExchangeSimulator(initial_balance=10000.0)
    orch = TradingBotOrchestrator(
        config=BotConfig(execution_mode="paper", target_symbols_str="BTCUSDT"),
        mode=ExecutionMode.PAPER,
        paper_simulator=paper_sim,
    )
    spec = make_dummy_spec("BTCUSDT", product_id=1)
    orch.contract_specs["BTCUSDT"] = spec

    # Trip circuit breaker
    orch.risk_manager.circuit_breaker.trip("Daily drawdown limit breached")

    intent = make_dummy_intent(
        symbol="BTCUSDT",
        signal_type=SignalType.BULLISH_BREAKOUT,
        trigger_price=68000.0,
        initial_stop_loss=67000.0,
        take_profit=70000.0,
        contract_lots=10,
    )
    strategy_mock = MagicMock()
    strategy_mock.on_candle_close = AsyncMock(return_value=intent)
    orch.strategies["BTCUSDT"] = strategy_mock

    candle = make_closed_candle("BTCUSDT")

    # Evaluate candle close
    await orch._handle_candle_close("BTCUSDT", candle)

    # Order must NOT be placed in paper simulator
    assert len(paper_sim._pending_orders) == 0
    # Intent must be marked CANCELLED
    assert intent.status == OrderStatus.CANCELLED_MANUAL
    assert "Daily drawdown limit breached" in intent.cancellation_reason
