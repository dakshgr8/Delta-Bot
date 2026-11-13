"""
Test Suite: Stale Intent Reconciliation and Exchange Flat State Unblocking Fixes.
Verifies that:
1. sync_live_metrics purges stale OrderIntents from state_manager when positions close or orders disappear.
2. sync_live_metrics cancels and clears strategy.pending_order when the order is no longer resting on Delta.
3. _execute_tenant_intent detects when exchange is flat, purges stale intents, and unblocks trade execution.
4. fanout_order_intent detects when exchange is flat, purges stale intents, and unblocks trade fanout.
5. emergency_flatten completely flushes state_manager intents and strategy active/pending states.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from delta_bot.config import BotConfig
from delta_bot.models import (
    Candle,
    CircularBarCache,
    ContractSpec,
    ExchangeOrder,
    PositionInfo,
    SessionAnchorData,
    WalletBalance,
)
from delta_bot.orchestrator import TradingBotOrchestrator
from delta_bot.state import TradingStateManager
from delta_bot.strategy import (
    BreakoutStrategyEngine,
    OrderIntent,
    OrderStatus,
    SignalType,
    StrategyConfig,
)
from delta_bot.tenant import TenantConfig, TenantInstance, TenantManager


@pytest.fixture
def btc_spec() -> ContractSpec:
    return ContractSpec(
        symbol="BTCUSD",
        product_id=27,
        contract_type="perpetual_futures",
        tick_size=0.5,
        contract_value=0.001,
        min_size=1,
        maker_commission_rate=0.0002,
        taker_commission_rate=0.0005,
    )


@pytest.fixture
def sol_spec() -> ContractSpec:
    return ContractSpec(
        symbol="SOLUSD",
        product_id=14823,
        contract_type="perpetual_futures",
        tick_size=0.0001,
        contract_value=1.0,
        min_size=1,
        maker_commission_rate=0.0002,
        taker_commission_rate=0.0005,
    )


@pytest.mark.asyncio
async def test_sync_live_metrics_purges_stale_intents_when_position_closes(sol_spec: ContractSpec) -> None:
    """Verifies state_manager and strategy active_position are cleared and marked closed when exchange position closes."""
    cfg = TenantConfig(
        tenant_id="rupdave-22",
        name="RupDave Live",
        mode="REAL",
        delta_api_key="key",
        delta_api_secret="secret",
        symbols=["SOLUSD"],
    )
    tenant = TenantInstance(config=cfg, contract_specs={"SOLUSD": sol_spec})

    pos = PositionInfo(
        product_id=14823,
        symbol="SOLUSD",
        size=11,
        entry_price=80.0,
        mark_price=81.0,
        unrealized_pnl=11.0,
        margin=10.0,
    )
    bal = WalletBalance(
        asset_id=1,
        asset_symbol="USD",
        balance=190.47,
        available_balance=190.47,
    )

    tenant.rest_client.get_wallet_balances = AsyncMock(return_value=[bal])
    tenant.rest_client.get_open_positions = AsyncMock(return_value=[pos])
    tenant.rest_client.get_open_orders = AsyncMock(return_value=[])

    # 1. First sync with active position
    await tenant.sync_live_metrics()
    strat = tenant.strategies["SOLUSD"]
    assert strat.active_position is not None
    assert tenant.state_manager.get_active_intent_for_product(14823) is not None

    # 2. Position closes on Delta Exchange (empty list returned)
    tenant.rest_client.get_open_positions = AsyncMock(return_value=[])
    await tenant.sync_live_metrics()

    # 3. Must be completely cleared in both strategy and state_manager
    assert strat.active_position is None
    tracked = tenant.state_manager.get_active_intent_for_product(14823)
    assert tracked is None, "Stale intent must NOT linger in state_manager after position close!"


@pytest.mark.asyncio
async def test_sync_live_metrics_clears_pending_order_cancelled_on_exchange(btc_spec: ContractSpec) -> None:
    """Verifies strategy.pending_order and state_manager intent are cleared when order is cancelled on Delta."""
    cfg = TenantConfig(
        tenant_id="rupdave-22",
        name="RupDave Live",
        mode="REAL",
        delta_api_key="key",
        delta_api_secret="secret",
        symbols=["BTCUSD"],
    )
    tenant = TenantInstance(config=cfg, contract_specs={"BTCUSD": btc_spec})

    intent = OrderIntent(
        symbol="BTCUSD",
        product_id=27,
        trigger_price=69127.5,
        initial_stop_loss=69300.0,
        current_stop_loss=69300.0,
        take_profit=68700.0,
        contract_lots=55,
        target_asset_qty=0.055,
        capital_at_risk_usd=9.5,
        signal_candle_high=69300.0,
        signal_candle_low=69120.0,
        signal_candle_close=69150.0,
        signal_bar_time=1000.0,
        created_at_epoch_s=1000.0,
        signal_type=SignalType.BEARISH_BREAKDOWN,
        status=OrderStatus.PENDING,
        exchange_order_id=999888,
    )
    tenant.state_manager.register_intent(intent)
    tenant.strategies["BTCUSD"].pending_order = intent

    assert tenant.state_manager.get_active_intent_for_product(27) is not None
    assert tenant.strategies["BTCUSD"].pending_order is not None

    # Exchange reports 0 open orders and 0 positions
    tenant.rest_client.get_wallet_balances = AsyncMock(return_value=[])
    tenant.rest_client.get_open_positions = AsyncMock(return_value=[])
    tenant.rest_client.get_open_orders = AsyncMock(return_value=[])

    await tenant.sync_live_metrics()

    assert tenant.strategies["BTCUSD"].pending_order is None
    assert tenant.state_manager.get_active_intent_for_product(27) is None
    assert intent.is_active is False


@pytest.mark.asyncio
async def test_execute_tenant_intent_unblocks_trade_when_exchange_is_flat(btc_spec: ContractSpec) -> None:
    """Verifies _execute_tenant_intent purges stale in-memory intents and executes new valid signals when exchange is flat."""
    bot_cfg = BotConfig(
        target_symbols="BTCUSD",
        delta_base_url="https://api.india.delta.exchange",
        delta_api_key="valid_api_key_test_12345",
        delta_api_secret="valid_api_secret_test_12345",
    )
    orchestrator = TradingBotOrchestrator(config=bot_cfg, mode="real")
    orchestrator.contract_specs["BTCUSD"] = btc_spec

    tenant_cfg = TenantConfig(
        tenant_id="rupdave-22",
        name="RupDave Live",
        mode="REAL",
        delta_api_key="key",
        delta_api_secret="secret",
        symbols=["BTCUSD"],
    )
    tenant = TenantInstance(config=tenant_cfg, contract_specs={"BTCUSD": btc_spec})
    tenant.live_balance = 190.47
    tenant.live_equity = 190.47
    tenant.live_positions = []  # Exchange is completely flat
    tenant.live_orders = []     # Exchange has no open orders
    orchestrator.tenant_manager.tenants["rupdave-22"] = tenant

    # Artificially inject a stale in-memory intent (simulating the bug that blocked trades)
    stale_intent = OrderIntent(
        symbol="BTCUSD",
        product_id=27,
        trigger_price=70000.0,
        initial_stop_loss=71000.0,
        current_stop_loss=71000.0,
        take_profit=68000.0,
        contract_lots=10,
        target_asset_qty=0.01,
        capital_at_risk_usd=10.0,
        signal_candle_high=70500.0,
        signal_candle_low=69500.0,
        signal_candle_close=69800.0,
        signal_bar_time=100.0,
        created_at_epoch_s=100.0,
        signal_type=SignalType.BEARISH_BREAKDOWN,
        status=OrderStatus.PENDING,
    )
    tenant.state_manager.register_intent(stale_intent)
    assert tenant.state_manager.get_active_intent_for_product(27) is not None

    # New fresh intent arrives
    fresh_intent = OrderIntent(
        symbol="BTCUSD",
        product_id=27,
        trigger_price=69127.5,
        initial_stop_loss=69300.0,
        current_stop_loss=69300.0,
        take_profit=68781.5,
        contract_lots=55,
        target_asset_qty=0.055,
        capital_at_risk_usd=9.52,
        signal_candle_high=69300.0,
        signal_candle_low=69120.0,
        signal_candle_close=69150.0,
        signal_bar_time=time.time(),
        created_at_epoch_s=time.time(),
        signal_type=SignalType.BEARISH_BREAKDOWN,
        status=OrderStatus.PENDING,
    )

    # Mock execution manager
    mock_order = ExchangeOrder(
        id=777666,
        client_order_id="det_123",
        symbol="BTCUSD",
        side="sell",
        order_type="stop_order",
        size=55,
        unfilled_size=55,
        stop_price=69127.5,
        state="open",
        product_id=27,
    )
    async def fake_execute(order_intent: OrderIntent, account_equity: float) -> ExchangeOrder:
        order_intent.exchange_order_id = mock_order.id
        return mock_order

    tenant.get_account_equity = AsyncMock(return_value=190.47)
    tenant.execution_manager = MagicMock()
    tenant.execution_manager.execute_signal = AsyncMock(side_effect=fake_execute)

    # Execute intent
    await orchestrator._execute_tenant_intent(tenant, "BTCUSD", btc_spec, fresh_intent)

    # Assert that execution was NOT skipped! Stale intent was purged and fresh signal executed!
    assert tenant.execution_manager.execute_signal.called
    assert stale_intent.is_active is False
    assert fresh_intent.exchange_order_id == 777666


@pytest.mark.asyncio
async def test_fanout_order_intent_unblocks_trade_when_exchange_is_flat(btc_spec: ContractSpec) -> None:
    """Verifies fanout_order_intent purges stale in-memory intents and fans out when exchange is flat."""
    tm = TenantManager()
    cfg = TenantConfig(
        tenant_id="rupdave-22",
        name="RupDave Live",
        mode="REAL",
        delta_api_key="key",
        delta_api_secret="secret",
        symbols=["BTCUSD"],
    )
    tenant = TenantInstance(config=cfg, contract_specs={"BTCUSD": btc_spec})
    tenant.live_balance = 190.47
    tenant.live_equity = 190.47
    tenant.live_positions = []
    tenant.live_orders = []
    tm.tenants["rupdave-22"] = tenant

    # Inject stale intent
    stale_intent = OrderIntent(
        symbol="BTCUSD",
        product_id=27,
        trigger_price=70000.0,
        initial_stop_loss=71000.0,
        current_stop_loss=71000.0,
        take_profit=68000.0,
        contract_lots=10,
        target_asset_qty=0.01,
        capital_at_risk_usd=10.0,
        signal_candle_high=70500.0,
        signal_candle_low=69500.0,
        signal_candle_close=69800.0,
        signal_bar_time=100.0,
        created_at_epoch_s=100.0,
        signal_type=SignalType.BEARISH_BREAKDOWN,
        status=OrderStatus.PENDING,
    )
    tenant.state_manager.register_intent(stale_intent)

    master_intent = OrderIntent(
        symbol="BTCUSD",
        product_id=27,
        trigger_price=69127.5,
        initial_stop_loss=69300.0,
        current_stop_loss=69300.0,
        take_profit=68781.5,
        contract_lots=55,
        target_asset_qty=0.055,
        capital_at_risk_usd=9.52,
        signal_candle_high=69300.0,
        signal_candle_low=69120.0,
        signal_candle_close=69150.0,
        signal_bar_time=time.time(),
        created_at_epoch_s=time.time(),
        signal_type=SignalType.BEARISH_BREAKDOWN,
        status=OrderStatus.PENDING,
    )

    mock_order = ExchangeOrder(
        id=888777,
        client_order_id="fanout_det",
        symbol="BTCUSD",
        side="sell",
        order_type="stop_order",
        size=55,
        unfilled_size=55,
        stop_price=69127.5,
        state="open",
        product_id=27,
    )
    async def fake_execute(order_intent: OrderIntent, account_equity: float) -> ExchangeOrder:
        order_intent.exchange_order_id = mock_order.id
        return mock_order

    tenant.get_account_equity = AsyncMock(return_value=190.47)
    tenant.execution_manager = MagicMock()
    tenant.execution_manager.execute_signal = AsyncMock(side_effect=fake_execute)

    executed = await tm.fanout_order_intent(
        base_signal_id="SIG_BTC_12345",
        master_intent=master_intent,
        spec=btc_spec,
    )

    assert "rupdave-22" in executed
    assert tenant.execution_manager.execute_signal.called
    assert stale_intent.is_active is False


@pytest.mark.asyncio
async def test_emergency_flatten_cleans_all_state_manager_intents(sol_spec: ContractSpec) -> None:
    """Verifies emergency_flatten completely flushes state_manager intents and strategy active/pending states."""
    cfg = TenantConfig(
        tenant_id="rupdave-22",
        name="RupDave Live",
        mode="REAL",
        delta_api_key="key",
        delta_api_secret="secret",
        symbols=["SOLUSD"],
    )
    tenant = TenantInstance(config=cfg, contract_specs={"SOLUSD": sol_spec})

    intent = OrderIntent(
        symbol="SOLUSD",
        product_id=14823,
        trigger_price=80.0,
        initial_stop_loss=79.0,
        current_stop_loss=79.0,
        take_profit=82.0,
        contract_lots=11,
        target_asset_qty=11.0,
        capital_at_risk_usd=11.0,
        signal_candle_high=80.5,
        signal_candle_low=79.5,
        signal_candle_close=80.0,
        signal_bar_time=100.0,
        created_at_epoch_s=100.0,
        signal_type=SignalType.BULLISH_BREAKOUT,
        status=OrderStatus.TRIGGERED,
    )
    tenant.state_manager.register_intent(intent)
    tenant.strategies["SOLUSD"].active_position = intent

    tenant.execution_manager = MagicMock()
    tenant.execution_manager.emergency_flatten = AsyncMock(return_value={"cancelled_orders": 0, "closed_positions": 1})

    res = await tenant.emergency_flatten()

    assert res["status"] == "success"
    assert tenant.strategies["SOLUSD"].active_position is None
    assert tenant.strategies["SOLUSD"].pending_order is None
    assert tenant.state_manager.get_active_intent_for_product(14823) is None
