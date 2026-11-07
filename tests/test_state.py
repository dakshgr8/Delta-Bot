"""
tests.test_state
~~~~~~~~~~~~~~~~

Unit and integration tests for Phase 5:
- Private WebSocket HMAC-SHA256 authentication payload generation ("GET" + timestamp + "/live")
- Entry STOP order fill transitions and automatic child bracket ID binding
- Real-time UserTrade fill execution telemetry
- Hybrid Push/Pull dual-source REST reconciliation & dropped state correction
- Ghost Position discovery and isolation without unintended automated closes
- Liquidation Proximity Guard alerting and auto-emergency flatten triggering
- Zero-latency trailing stop ratchets wired into ExecutionManager
"""

import asyncio
import hashlib
import hmac
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from delta_bot.auth import DeltaAuthEngine
from delta_bot.client import DeltaRestClient
from delta_bot.config import BotConfig
from delta_bot.execution import ExecutionConfig, ExecutionManager, ExecutionMode
from delta_bot.models import (
    ContractSpec,
    ExchangeOrder,
    GhostPosition,
    LiquidationAlert,
    OrderUpdate,
    PositionInfo,
    UserTrade,
)
from delta_bot.private_ws import PrivateWebSocketClient
from delta_bot.state import TradingStateManager
from delta_bot.strategy import OrderIntent, OrderStatus, SignalType


@pytest.fixture
def test_config() -> BotConfig:
    return BotConfig(
        DELTA_API_KEY="test_api_key_12345",
        DELTA_API_SECRET="test_secret_67890",
        DELTA_WS_URL="wss://socket.delta.exchange",
    )


@pytest.fixture
def btc_spec() -> ContractSpec:
    return ContractSpec(
        product_id=139,
        symbol="BTCUSDT",
        contract_type="perpetual_futures",
        contract_value=0.001,
        tick_size=0.5,
        maker_commission_rate=0.0001,
        taker_commission_rate=0.0001,
        trading_status="operational",
    )


@pytest.fixture
def pending_long_intent() -> OrderIntent:
    return OrderIntent(
        symbol="BTCUSDT",
        signal_type=SignalType.BULLISH_BREAKOUT,
        status=OrderStatus.PENDING,
        trigger_price=51000.0,
        initial_stop_loss=50500.0,
        current_stop_loss=50500.0,
        take_profit=52000.0,
        contract_lots=20,
        target_asset_qty=0.02,
        capital_at_risk_usd=100.0,
        signal_candle_high=50950.0,
        signal_candle_low=50450.0,
        signal_candle_close=50900.0,
        signal_bar_time=1700000000.0,
        created_at_epoch_s=1700000000.0,
        exchange_order_id=881122,
        product_id=139,
    )


# -----------------------------------------------------------------------------
# 1. Private WebSocket Authentication Tests
# -----------------------------------------------------------------------------

def test_private_ws_auth_payload_generation(test_config: BotConfig) -> None:
    """
    Verifies that private WebSocket HMAC authentication strictly matches Delta's specification:
    Signature data: "GET" + timestamp + "/live"
    """
    auth_engine = DeltaAuthEngine(
        api_key=test_config.delta_api_key,
        api_secret=test_config.delta_api_secret,
    )

    fixed_timestamp = "1712750400"
    payload = auth_engine.create_websocket_auth_payload(timestamp=fixed_timestamp)

    assert payload["type"] == "auth"
    assert payload["payload"]["api-key"] == "test_api_key_12345"
    assert payload["payload"]["timestamp"] == fixed_timestamp

    # Independently compute expected HMAC-SHA256 over "GET" + ts + "/live"
    pre_hash_str = f"GET{fixed_timestamp}/live"
    expected_sig = hmac.new(
        key="test_secret_67890".encode("utf-8"),
        msg=pre_hash_str.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()

    assert payload["payload"]["signature"] == expected_sig


@pytest.mark.asyncio
async def test_private_ws_client_auth_and_subscriptions(test_config: BotConfig) -> None:
    """Verifies that PrivateWebSocketClient builds correct auth frame and subscriptions."""
    client = PrivateWebSocketClient(test_config, symbols=["BTCUSDT", "ETHUSDT"])

    auth_payload = client.build_auth_payload()
    assert auth_payload["type"] == "auth"
    assert auth_payload["payload"]["api-key"] == "test_api_key_12345"

    sub_payload = client.build_subscription_payload()
    assert sub_payload["type"] == "subscribe"
    channels = sub_payload["payload"]["channels"]
    assert len(channels) == 3

    channel_names = [c["name"] for c in channels]
    assert "orders" in channel_names
    assert "positions" in channel_names
    assert "user_trades" in channel_names

    # Test simulation of auth response
    auth_events: list[bool] = []
    client.on_auth_state(lambda state: auth_events.append(state))

    await client.simulate_auth_response(success=True)
    assert client.is_authenticated is True
    assert auth_events == [True]


# -----------------------------------------------------------------------------
# 2. Fill Detection & Automatic Child Bracket Binding Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_order_fill_transition_and_child_bracket_binding(
    pending_long_intent: OrderIntent,
) -> None:
    """
    Verifies that when an entry STOP order transitions to 'closed' (filled):
    1. OrderIntent transitions to TRIGGERED with fill_price and filled_size.
    2. Spawned child Stop Loss order from Delta private feed binds child_stop_loss_order_id.
    3. get_child_stop_loss_order_id() returns the bound ID immediately.
    """
    mock_client = AsyncMock(spec=DeltaRestClient)
    state_mgr = TradingStateManager(mock_client)

    state_mgr.register_intent(pending_long_intent)
    assert pending_long_intent.is_pending is True

    fill_records: list[tuple[OrderIntent, float, int]] = []
    state_mgr.on_fill(lambda intent, px, sz: fill_records.append((intent, px, sz)))

    # 1. Entry STOP order executes and fills on Delta
    entry_fill_update = OrderUpdate(
        id=881122,  # Parent entry order ID
        product_id=139,
        symbol="BTCUSDT",
        side="buy",
        order_type="market_order",
        state="closed",
        size=20,
        stop_price=51000.5,
    )
    await state_mgr.handle_order_update(entry_fill_update)

    assert pending_long_intent.is_triggered is True
    assert pending_long_intent.status == OrderStatus.TRIGGERED
    assert pending_long_intent.fill_price == 51000.5
    assert pending_long_intent.filled_size == 20
    assert len(fill_records) == 1

    # 2. Delta Exchange spawns child Stop Loss order (exit side: sell)
    child_sl_id = 992233
    child_sl_update = OrderUpdate(
        id=child_sl_id,
        product_id=139,
        symbol="BTCUSDT",
        side="sell",
        order_type="market_order",
        stop_order_type="stop_loss_order",
        stop_price=50500.0,
        state="open",
        size=20,
    )
    await state_mgr.handle_order_update(child_sl_update)

    # 3. Verify immediate binding for zero-latency ratchets
    assert pending_long_intent.child_stop_loss_order_id == child_sl_id
    assert state_mgr.get_child_stop_loss_order_id(product_id=139) == child_sl_id


@pytest.mark.asyncio
async def test_user_trade_fill_execution(pending_long_intent: OrderIntent) -> None:
    """Verifies that execution fill report from 'user_trades' channel transitions OrderIntent."""
    mock_client = AsyncMock(spec=DeltaRestClient)
    state_mgr = TradingStateManager(mock_client)
    state_mgr.register_intent(pending_long_intent)

    trade = UserTrade(
        trade_id="TR_98765",
        order_id=881122,
        product_id=139,
        symbol="BTCUSDT",
        side="buy",
        size=20,
        price=51002.0,
        commission=0.102,
        role="taker",
    )

    await state_mgr.handle_user_trade(trade)

    assert pending_long_intent.is_triggered is True
    assert pending_long_intent.fill_price == 51002.0
    assert pending_long_intent.filled_size == 20


# -----------------------------------------------------------------------------
# 3. Dual-Source REST Reconciliation (Dropped Packet Correction)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rest_reconciliation_corrects_missed_fill_and_binds_child_sl(
    pending_long_intent: OrderIntent,
) -> None:
    """
    Verifies that if a WebSocket packet was dropped:
    1. REST reconciliation identifies that the entry order closed and a position is open.
    2. Transitions the pending OrderIntent to TRIGGERED.
    3. Discovers resting child Stop Loss order from open orders on Delta and binds it.
    """
    mock_client = AsyncMock(spec=DeltaRestClient)

    # Mock REST responses: position is open, entry order is gone, child SL order is resting
    mock_client.get_open_positions.return_value = [
        PositionInfo(
            product_id=139,
            symbol="BTCUSDT",
            size=20,
            entry_price=51005.0,
            mark_price=51100.0,
            liquidation_price=42000.0,
        )
    ]
    mock_client.get_open_orders.return_value = [
        ExchangeOrder(
            id=554433,  # Child Stop Loss order resting on book
            product_id=139,
            symbol="BTCUSDT",
            side="sell",
            order_type="market_order",
            size=20,
            stop_price=50500.0,
            state="open",
        )
    ]

    state_mgr = TradingStateManager(mock_client)
    state_mgr.register_intent(pending_long_intent)
    assert pending_long_intent.is_pending is True

    # Execute REST fallback reconciliation
    report = await state_mgr.reconcile_with_rest()

    assert report["synced_positions"] == 1
    assert report["child_orders_bound"] == 1

    # Verify intent was reconciled to TRIGGERED and child SL bound
    assert pending_long_intent.is_triggered is True
    assert pending_long_intent.fill_price == 51005.0
    assert pending_long_intent.child_stop_loss_order_id == 554433
    assert state_mgr.get_child_stop_loss_order_id(product_id=139) == 554433


# -----------------------------------------------------------------------------
# 4. Ghost Position Recovery & Protection
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ghost_position_detection_without_blind_closure() -> None:
    """
    Verifies that if REST reconciliation discovers an open position on Delta Exchange
    that is not tracked in local bot memory:
    1. Tags it as a GhostPosition (EXTERNAL_UNTRACKED).
    2. Emits critical alert.
    3. Does NOT blindly close it.
    """
    mock_client = AsyncMock(spec=DeltaRestClient)
    mock_client.get_open_positions.return_value = [
        PositionInfo(
            product_id=176,
            symbol="ETHUSDT",
            size=50,
            entry_price=2200.0,
            mark_price=2210.0,
            liquidation_price=1800.0,
        )
    ]
    mock_client.get_open_orders.return_value = []

    state_mgr = TradingStateManager(mock_client)

    ghost_alerts: list[GhostPosition] = []
    state_mgr.on_ghost_position(lambda ghost: ghost_alerts.append(ghost))

    report = await state_mgr.reconcile_with_rest()

    assert report["ghost_positions_found"] == 1
    assert len(ghost_alerts) == 1

    ghost = ghost_alerts[0]
    assert ghost.symbol == "ETHUSDT"
    assert ghost.product_id == 176
    assert ghost.size == 50
    assert ghost.tag == "EXTERNAL_UNTRACKED"

    # Confirm ghost is isolated in state manager
    ghosts = state_mgr.get_ghost_positions()
    assert len(ghosts) == 1
    assert ghosts[0].product_id == 176

    # Verify that close/cancel was NEVER called
    mock_client.place_order.assert_not_awaited()
    mock_client.cancel_all_orders.assert_not_awaited()


# -----------------------------------------------------------------------------
# 5. Liquidation Watcher & Proximity Guard
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_liquidation_proximity_guard_triggering() -> None:
    """
    Verifies that when mark price comes dangerously close (<= 1.5 * ATR) to liquidation:
    1. Emits critical LiquidationAlert.
    2. Automatically executes emergency_flatten() on ExecutionManager.
    """
    mock_client = AsyncMock(spec=DeltaRestClient)
    mock_exec_mgr = AsyncMock(spec=ExecutionManager)

    state_mgr = TradingStateManager(
        client=mock_client,
        execution_manager=mock_exec_mgr,
        proximity_atr_mult=1.5,
    )

    alerts: list[LiquidationAlert] = []
    state_mgr.on_liquidation_alert(lambda alert: alerts.append(alert))

    # Long Position: entry=50000, liquidation=48000, ATR=1000
    # Threshold distance = 1.5 * 1000 = 1500
    # Safe distance: Mark=49800 -> distance=1800 (> 1500) -> NO ALERT
    pos = PositionInfo(
        product_id=139,
        symbol="BTCUSDT",
        size=10,
        entry_price=50000.0,
        mark_price=49800.0,
        liquidation_price=48000.0,
    )
    await state_mgr.handle_position_update(pos)
    await state_mgr.update_market_context("BTCUSDT", mark_price=49800.0, atr=1000.0, high=50100.0, low=49700.0)

    assert len(alerts) == 0
    mock_exec_mgr.emergency_flatten.assert_not_awaited()

    # Dangerous price shock: Mark=49400 -> distance = 49400 - 48000 = 1400 <= 1500!
    # MUST TRIGGER ALERT & EMERGENCY FLATTEN!
    await state_mgr.update_market_context("BTCUSDT", mark_price=49400.0, atr=1000.0)

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.symbol == "BTCUSDT"
    assert alert.mark_price == 49400.0
    assert alert.liquidation_price == 48000.0
    assert alert.distance == 1400.0
    assert alert.threshold_distance == 1500.0

    # Verify auto-execution of emergency_flatten
    mock_exec_mgr.emergency_flatten.assert_awaited_once_with(symbol="BTCUSDT")


# -----------------------------------------------------------------------------
# 6. ExecutionManager Zero-Latency Ratchet Integration
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execution_manager_uses_cached_child_sl_from_state_manager(
    btc_spec: ContractSpec,
    pending_long_intent: OrderIntent,
) -> None:
    """
    Verifies that ExecutionManager queries TradingStateManager first for the child stop ID,
    achieving zero-latency trailing stop ratchets without REST round-trips.
    """
    mock_client = AsyncMock(spec=DeltaRestClient)
    mock_client.get_cached_contract_spec.return_value = btc_spec
    mock_client.config = BotConfig(DELTA_API_KEY="test_key", DELTA_API_SECRET="test_secret")

    # Wire state manager into execution manager
    state_mgr = TradingStateManager(mock_client)
    exec_config = ExecutionConfig(execution_mode=ExecutionMode.LIVE)
    exec_mgr = ExecutionManager(mock_client, exec_config, state_manager=state_mgr)

    # Position is triggered; WebSocket already cached child SL order ID 667788
    pending_long_intent.status = OrderStatus.TRIGGERED
    pending_long_intent.child_stop_loss_order_id = None  # Not yet on intent object!
    state_mgr.register_intent(pending_long_intent)

    # Simulate child stop order arrival from private WebSocket
    await state_mgr.handle_order_update(
        OrderUpdate(
            id=667788,
            product_id=139,
            symbol="BTCUSDT",
            side="sell",
            order_type="market_order",
            stop_order_type="stop_loss_order",
            stop_price=50500.0,
            state="open",
            size=20,
        )
    )

    # Ratchet stop loss
    new_sl = 50800.0
    success = await exec_mgr.ratchet_exchange_stop_loss(pending_long_intent, new_sl)

    assert success is True
    # Assert child SL ID was resolved from state manager
    assert pending_long_intent.child_stop_loss_order_id == 667788

    # Assert edit_order was sent directly with child ID 667788 and stop_price
    mock_client.edit_order.assert_awaited_once_with(
        product_id=139,
        order_id=667788,
        stop_price=50800.0,
    )
    # REST discovery was bypassed completely!
    mock_client.find_active_stop_loss_order.assert_not_awaited()
