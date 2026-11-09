"""
tests/test_live_account_safeguard_fixes.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regression test suite verifying fixes for live-account financial loss and runtime flaws:
- Trailing stop ratchet accepts optional `old_sl` argument without raising TypeError.
- Invalidation handler accepts default reason parameter without TypeError.
- Sizing strictly floors contract calculation to avoid risk budget overrun.
- Currency normalization correctly prevents INR balance 87x inflation in derivative sizing.
- State REST reconciliation clears orphaned active OrderIntents and notifies RiskManager.
- 00:00 UTC session rollover properly triggers registered callbacks.
- Orchestrator properly enables the daily drawdown circuit breaker by default.
"""

from unittest.mock import AsyncMock, MagicMock
import pytest
import time

from delta_bot.client import DeltaRestClient
from delta_bot.config import BotConfig
from delta_bot.execution import ExecutionConfig, OrderExecutionManager
from delta_bot.models import ContractSpec, ExecutionMode, WalletBalance
from delta_bot.orchestrator import TradingBotOrchestrator
from delta_bot.risk import RiskConfig, RiskManager
from delta_bot.state import TradingStateManager
from delta_bot.strategy import OrderIntent, OrderStatus, SignalType
from delta_bot.websocket_client import CandleAggregator


@pytest.fixture
def btc_spec() -> ContractSpec:
    return ContractSpec(
        product_id=139,
        symbol="BTCUSDT",
        contract_type="perpetual_futures",
        contract_value=0.001,
        tick_size=0.5,
        maker_commission_rate=0.0002,
        taker_commission_rate=0.0005,
        quoting_symbol="USDT",
    )


@pytest.fixture
def sample_intent() -> OrderIntent:
    return OrderIntent(
        symbol="BTCUSDT",
        signal_type=SignalType.BULLISH_BREAKOUT,
        status=OrderStatus.PENDING,
        trigger_price=50000.0,
        initial_stop_loss=49000.0,
        current_stop_loss=49000.0,
        take_profit=52000.0,
        rr_ratio=2.0,
        contract_lots=10,
        target_asset_qty=0.01,
        capital_at_risk_usd=10.0,
        signal_candle_high=50000.0,
        signal_candle_low=49000.0,
        signal_candle_close=49800.0,
        signal_bar_time=1700000000.0,
        created_at_epoch_s=1700000000.0,
        ttl_bars=3,
        exchange_order_id=12345,
        product_id=139,
    )


@pytest.mark.asyncio
async def test_trailing_stop_ratchet_accepts_old_sl_argument(
    btc_spec: ContractSpec, sample_intent: OrderIntent
) -> None:
    """Verifies handle_trailing_stop_ratchet accepts `old_sl` keyword argument without TypeError."""
    mock_client = AsyncMock(spec=DeltaRestClient)
    mock_client.get_cached_contract_spec.return_value = btc_spec
    mock_client.config = BotConfig()

    manager = OrderExecutionManager(mock_client, ExecutionConfig(execution_mode=ExecutionMode.DRY_RUN))
    # Should not raise TypeError: unexpected keyword argument 'old_sl'
    success = await manager.handle_trailing_stop_ratchet(
        order_intent=sample_intent,
        new_sl=49500.0,
        old_sl=49000.0,
    )
    assert success is True
    assert sample_intent.current_stop_loss == 49500.0


@pytest.mark.asyncio
async def test_handle_invalidation_default_reason(
    btc_spec: ContractSpec, sample_intent: OrderIntent
) -> None:
    """Verifies handle_invalidation can be called without explicit reason or with custom reason."""
    mock_client = AsyncMock(spec=DeltaRestClient)
    mock_client.get_cached_contract_spec.return_value = btc_spec
    mock_client.config = BotConfig()

    manager = OrderExecutionManager(mock_client, ExecutionConfig(execution_mode=ExecutionMode.DRY_RUN))
    # Test without positional argument
    success = await manager.handle_invalidation(sample_intent)
    assert success is True


def test_risk_sizing_floors_contracts_without_overrunning_budget(btc_spec: ContractSpec) -> None:
    """Verifies risk calculation floors lot sizing so total risk does not exceed capital_at_risk."""
    rm = RiskManager(initial_equity=1000.0)
    # Entry at 50000, Stop Loss at 49000 -> Distance = 1000 USD
    # Risk budget = 1000 * 1% = 10 USD
    # Contract value = 0.001 BTC. Risk per lot = 1000 * 0.001 = 1.00 USD
    # Target qty by risk = 10 / 1000 = 0.01 BTC.
    # If target qty gives fractional lots, math.floor ensures risk <= 10.0
    result = rm.size_position(
        spec=btc_spec,
        trigger_price=50000.0,
        stop_loss_price=49000.0,
        account_equity=1000.0,
        available_margin=500.0,
    )
    assert result.final_lots == 10
    actual_risk = result.final_lots * btc_spec.contract_value * (50000.0 - 49000.0)
    assert actual_risk <= result.risk_budget_usd


@pytest.mark.asyncio
async def test_margin_check_normalizes_inr_currency(btc_spec: ContractSpec) -> None:
    """Verifies check_and_size_margin converts INR balances to USD equivalent (not 1:1)."""
    mock_client = AsyncMock(spec=DeltaRestClient)
    mock_client.get_cached_contract_spec.return_value = btc_spec
    mock_client.config = BotConfig()
    # Return wallet balance with only INR asset (e.g. 87000 INR = ~1000 USD)
    mock_client.get_wallet_balances.return_value = [
        WalletBalance(asset_id=1, asset_symbol="INR", balance=87000.0, available_balance=87000.0)
    ]

    manager = OrderExecutionManager(mock_client, ExecutionConfig(execution_mode=ExecutionMode.DRY_RUN))
    result = await manager.check_and_size_margin(
        spec=btc_spec,
        trigger_price=50000.0,
        stop_loss_price=49000.0,
        account_equity=1000.0,
    )
    # Available margin should be scaled (~1000 USD), not 87000 USD!
    assert result.available_margin < 2000.0


@pytest.mark.asyncio
async def test_state_rest_reconciliation_cleans_orphaned_intent_and_exits_risk(
    sample_intent: OrderIntent,
) -> None:
    """Verifies REST reconciliation marks closed positions as CLOSED_SL and updates RiskManager."""
    mock_client = AsyncMock(spec=DeltaRestClient)
    # REST reports no open positions
    mock_client.get_open_positions.return_value = []
    mock_client.get_open_orders.return_value = []

    state_mgr = TradingStateManager(client=mock_client)
    sample_intent.status = OrderStatus.TRIGGERED  # Active open position
    sample_intent.fill_price = 50000.0
    state_mgr._open_positions[139] = MagicMock(product_id=139, size=10, entry_price=50000.0)
    state_mgr.register_intent(sample_intent)

    mock_exec_mgr = MagicMock()
    mock_risk_mgr = MagicMock()
    mock_exec_mgr.risk_manager = mock_risk_mgr
    state_mgr.execution_manager = mock_exec_mgr

    # Reconcile with REST
    await state_mgr.reconcile_with_rest()

    assert sample_intent.status == OrderStatus.CLOSED_SL
    mock_risk_mgr.record_trade_exit.assert_called_once()


@pytest.mark.asyncio
async def test_session_rollover_triggers_callback_when_date_changes() -> None:
    """Verifies that crossing 00:00 UTC fires the session rollover callback."""
    rollover_fired = []

    async def on_rollover(symbol: str, anchor) -> None:
        rollover_fired.append((symbol, anchor.session_date))

    aggregator = CandleAggregator(on_session_rollover=on_rollover)
    anchor = aggregator.get_session_anchor("BTCUSDT")
    anchor.session_date = "2026-11-08"

    # Ingest candle for 2026-11-08 23:59:00 UTC
    c1 = {
        "symbol": "BTCUSDT",
        "resolution": "1m",
        "candle_start_time": 1794182340_000_000,
        "open": 50000.0,
        "high": 50100.0,
        "low": 49950.0,
        "close": 50050.0,
        "volume": 10.0,
    }
    await aggregator.process_candle_update(c1)

    # Ingest candle crossing midnight: 2026-11-09 00:00:00 UTC
    c2 = {
        "symbol": "BTCUSDT",
        "resolution": "1m",
        "candle_start_time": 1794182400_000_000,
        "open": 50050.0,
        "high": 50200.0,
        "low": 50040.0,
        "close": 50150.0,
        "volume": 15.0,
    }
    await aggregator.process_candle_update(c2)

    assert len(rollover_fired) == 1
    assert rollover_fired[0][0] == "BTCUSDT"
    assert rollover_fired[0][1] == "2026-11-09"


def test_orchestrator_initializes_with_circuit_breaker_enabled() -> None:
    """Verifies TradingBotOrchestrator configures RiskManager with circuit breaker enabled by default."""
    config = BotConfig(execution_mode="paper", max_drawdown_pct=5.0)
    orchestrator = TradingBotOrchestrator(config=config)
    assert orchestrator.risk_manager.config.enable_drawdown_circuit_breaker is True
    assert orchestrator.risk_manager.config.daily_max_drawdown_pct == 5.0
