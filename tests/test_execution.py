"""
tests/test_execution.py
~~~~~~~~~~~~~~~~~~~~~~~

Unit test suite for Order Execution & Margin Management Engine (Phase 4).
Tests:
- Trap 1: Exchange-native stop order placement with attached brackets (zero client latency slippage)
- Trap 1: Resting order cancellation on intermediate EMA invalidation & TTL expiration
- Trap 1: Exchange-native trailing stop loss ratchet
- Trap 2: Leverage configuration & caching (POST /v2/products/{id}/orders/leverage)
- Trap 2: Dual-constraint margin sizing (risk-limited vs leverage-limited)
- Trap 2: Insufficient margin rejection before order submission
- Trap 3: Dynamic Product ID resolution across Delta Global vs Delta India
- ExecutionMode: DRY_RUN vs LIVE mock payload verification
"""

from unittest.mock import AsyncMock, MagicMock
import pytest

from delta_bot.client import DeltaRestClient
from delta_bot.config import BotConfig
from delta_bot.exceptions import DeltaAPIError, InsufficientMarginError
from delta_bot.execution import ExecutionConfig, OrderExecutionManager
from delta_bot.models import (
    ContractSpec,
    ExchangeOrder,
    ExecutionMode,
    MarginCheckResult,
    PositionInfo,
    WalletBalance,
)
from delta_bot.strategy import OrderIntent, OrderStatus, SignalType


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def btc_global_spec() -> ContractSpec:
    """Delta Exchange Global BTCUSDT perpetual."""
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
def btc_india_spec() -> ContractSpec:
    """Delta Exchange India BTCUSDT perpetual (different product_id)."""
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
def sample_bullish_intent() -> OrderIntent:
    return OrderIntent(
        symbol="BTCUSDT",
        signal_type=SignalType.BULLISH_BREAKOUT,
        status=OrderStatus.PENDING,
        trigger_price=51000.5,
        initial_stop_loss=50600.0,
        current_stop_loss=50600.0,
        take_profit=51801.5,
        rr_ratio=2.0,
        contract_lots=100,
        target_asset_qty=0.1,
        capital_at_risk_usd=40.05,
        signal_candle_high=51000.0,
        signal_candle_low=50600.0,
        signal_candle_close=50950.0,
        signal_bar_time=1700000000.0,
        created_at_epoch_s=1700000000.0,
        ttl_bars=3,
    )


# -----------------------------------------------------------------------------
# 1. Trap 1: Exchange-Native Stop Orders with Brackets
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_native_stop_order_with_brackets_dry_run(
    btc_global_spec: ContractSpec,
    sample_bullish_intent: OrderIntent,
) -> None:
    """Verifies that dry-run execution generates simulated exchange stop orders with brackets."""
    mock_client = AsyncMock(spec=DeltaRestClient)
    mock_client.get_cached_contract_spec.return_value = btc_global_spec
    mock_client.config = BotConfig()

    exec_config = ExecutionConfig(
        execution_mode=ExecutionMode.DRY_RUN,
        leverage=10,
        stop_trigger_method="mark_price",
    )
    manager = OrderExecutionManager(mock_client, exec_config)

    order = await manager.execute_stop_order_with_brackets(
        order_intent=sample_bullish_intent,
        account_equity=10000.0,
        available_margin_override=5000.0,
    )

    assert isinstance(order, ExchangeOrder)
    assert order.product_id == 139
    assert order.side == "buy"
    assert order.order_type == "market_order"
    assert order.stop_price == 51000.5
    assert order.bracket_stop_loss_price == 50600.0
    assert order.bracket_take_profit_price == 51801.5
    assert order.state == "open"

    # Verify intent tracking
    assert sample_bullish_intent.exchange_order_id == order.id
    assert sample_bullish_intent.product_id == 139
    assert sample_bullish_intent.leverage == 10


@pytest.mark.asyncio
async def test_native_stop_order_with_brackets_live_submission(
    btc_global_spec: ContractSpec,
    sample_bullish_intent: OrderIntent,
) -> None:
    """Verifies that live mode passes all bracket parameters and integer product_id to POST /v2/orders."""
    mock_client = AsyncMock(spec=DeltaRestClient)
    mock_client.get_cached_contract_spec.return_value = btc_global_spec
    mock_client.config = BotConfig()

    mock_placed_order = ExchangeOrder(
        id=98765432,
        client_order_id=sample_bullish_intent.intent_id,
        product_id=139,
        symbol="BTCUSDT",
        side="buy",
        order_type="market_order",
        size=50,
        stop_price=51000.5,
        bracket_stop_loss_price=50600.0,
        bracket_take_profit_price=51801.5,
        state="open",
        created_at=1700000000.0,
    )
    mock_client.place_order.return_value = mock_placed_order

    exec_config = ExecutionConfig(execution_mode=ExecutionMode.LIVE, leverage=5)
    manager = OrderExecutionManager(mock_client, exec_config)

    order = await manager.execute_stop_order_with_brackets(
        order_intent=sample_bullish_intent,
        account_equity=10000.0,
        available_margin_override=5000.0,
    )

    assert order.id == 98765432
    assert sample_bullish_intent.exchange_order_id == 98765432

    # Verify parameters sent to client.place_order
    mock_client.place_order.assert_awaited_once()
    call_kwargs = mock_client.place_order.await_args.kwargs
    assert call_kwargs["product_id"] == 139
    assert call_kwargs["size"] == sample_bullish_intent.contract_lots
    assert call_kwargs["side"] == "buy"
    assert call_kwargs["order_type"] == "market_order"
    assert call_kwargs["stop_price"] == 51000.5
    assert call_kwargs["stop_order_type"] == "stop_loss_order"
    assert call_kwargs["stop_trigger_method"] == "mark_price"
    assert call_kwargs["bracket_stop_loss_price"] == 50600.0
    assert call_kwargs["bracket_take_profit_price"] == 51801.5
    assert call_kwargs["time_in_force"] == "gtc"
    assert call_kwargs["symbol"] == "BTCUSDT"
    assert call_kwargs["client_order_id"].startswith("DBOT_BTCUSDT_")
    assert len(call_kwargs["client_order_id"]) <= 32


# -----------------------------------------------------------------------------
# 2. Trap 1: Resting Order Cancellation (EMA Invalidation & TTL)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resting_order_cancellation(
    btc_global_spec: ContractSpec,
    sample_bullish_intent: OrderIntent,
) -> None:
    """Verifies that cancelling an invalidated/expired intent purges the resting order on Delta."""
    mock_client = AsyncMock(spec=DeltaRestClient)
    mock_client.get_cached_contract_spec.return_value = btc_global_spec
    mock_client.config = BotConfig()

    sample_bullish_intent.exchange_order_id = 998877
    sample_bullish_intent.product_id = 139

    manager = OrderExecutionManager(mock_client, ExecutionConfig(execution_mode=ExecutionMode.LIVE))

    # Invalidation trigger
    success = await manager.cancel_resting_order(
        order_intent=sample_bullish_intent,
        reason="EMA 20 cross invalidation",
    )

    assert success is True
    mock_client.cancel_order.assert_awaited_once_with(
        product_id=139,
        order_id=998877,
    )


@pytest.mark.asyncio
async def test_trailing_stop_ratchet_pending_order(
    btc_global_spec: ContractSpec,
    sample_bullish_intent: OrderIntent,
) -> None:
    """
    Verifies that ratcheting SL for an unfilled PENDING entry order targets
    Delta Exchange's pending bracket endpoint (PUT /v2/orders/bracket) using the parent order ID.
    """
    mock_client = AsyncMock(spec=DeltaRestClient)
    mock_client.get_cached_contract_spec.return_value = btc_global_spec
    mock_client.config = BotConfig()

    sample_bullish_intent.status = OrderStatus.PENDING
    sample_bullish_intent.exchange_order_id = 112233
    sample_bullish_intent.product_id = 139

    manager = OrderExecutionManager(mock_client, ExecutionConfig(execution_mode=ExecutionMode.LIVE))

    new_sl = 50850.0
    success = await manager.ratchet_exchange_stop_loss(sample_bullish_intent, new_sl)

    assert success is True
    assert sample_bullish_intent.current_stop_loss == 50850.0
    mock_client.edit_bracket_order.assert_awaited_once_with(
        product_id=139,
        order_id=112233,
        bracket_stop_loss_price=50850.0,
        bracket_take_profit_price=None,
    )


@pytest.mark.asyncio
async def test_trailing_stop_ratchet_filled_position_with_child_order_id(
    btc_global_spec: ContractSpec,
    sample_bullish_intent: OrderIntent,
) -> None:
    """
    Critical Edge Case:
    Verifies that once an order triggers and fills (TRIGGERED), the parent order ID is closed.
    Ratcheting MUST target the spawned child Stop Loss order ID via PUT /v2/orders (stop_price),
    NOT the parent entry order ID, preventing 'open_order_not_found' error.
    """
    mock_client = AsyncMock(spec=DeltaRestClient)
    mock_client.get_cached_contract_spec.return_value = btc_global_spec
    mock_client.config = BotConfig()

    sample_bullish_intent.status = OrderStatus.TRIGGERED
    sample_bullish_intent.exchange_order_id = 112233  # Parent order ID (now closed on Delta)
    sample_bullish_intent.child_stop_loss_order_id = 445566  # Spawned child Stop Loss order ID
    sample_bullish_intent.product_id = 139

    manager = OrderExecutionManager(mock_client, ExecutionConfig(execution_mode=ExecutionMode.LIVE))

    new_sl = 50850.0
    success = await manager.ratchet_exchange_stop_loss(sample_bullish_intent, new_sl)

    assert success is True
    assert sample_bullish_intent.current_stop_loss == 50850.0

    # Must edit the CHILD stop order, NOT the parent entry order!
    mock_client.edit_order.assert_awaited_once_with(
        product_id=139,
        order_id=445566,
        stop_price=50850.0,
    )
    # Must NOT call edit_bracket_order with the parent ID
    mock_client.edit_bracket_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_trailing_stop_ratchet_filled_position_discovers_child_order(
    btc_global_spec: ContractSpec,
    sample_bullish_intent: OrderIntent,
) -> None:
    """
    Verifies that if child_stop_loss_order_id is not yet cached on an open position,
    the execution engine queries Delta Exchange (find_active_stop_loss_order), caches the
    child order ID, and modifies it via PUT /v2/orders.
    """
    mock_client = AsyncMock(spec=DeltaRestClient)
    mock_client.get_cached_contract_spec.return_value = btc_global_spec
    mock_client.config = BotConfig(DELTA_API_KEY="test_key", DELTA_API_SECRET="test_secret")

    # Mock discovered active child Stop Loss order on exchange
    mock_child_order = ExchangeOrder(
        id=778899,
        product_id=139,
        side="sell",
        order_type="market_order",
        size=50,
        stop_price=50600.0,
        state="open",
    )
    mock_client.find_active_stop_loss_order.return_value = mock_child_order

    sample_bullish_intent.status = OrderStatus.TRIGGERED
    sample_bullish_intent.exchange_order_id = 112233
    sample_bullish_intent.child_stop_loss_order_id = None  # Not yet cached!
    sample_bullish_intent.product_id = 139

    manager = OrderExecutionManager(mock_client, ExecutionConfig(execution_mode=ExecutionMode.LIVE))

    new_sl = 50850.0
    success = await manager.ratchet_exchange_stop_loss(sample_bullish_intent, new_sl)

    assert success is True
    # Verify dynamic discovery was invoked
    mock_client.find_active_stop_loss_order.assert_awaited_once_with(
        product_id=139,
        side="sell",
    )
    # Verify child_stop_loss_order_id was captured and stored
    assert sample_bullish_intent.child_stop_loss_order_id == 778899

    # Verify edit_order targeted the discovered child ID with stop_price
    mock_client.edit_order.assert_awaited_once_with(
        product_id=139,
        order_id=778899,
        stop_price=50850.0,
    )


@pytest.mark.asyncio
async def test_trailing_stop_ratchet_filled_position_fallback_to_position_bracket(
    btc_global_spec: ContractSpec,
    sample_bullish_intent: OrderIntent,
) -> None:
    """
    Verifies fallback: If position is open and no child stop order is discovered,
    the execution engine updates the position-level bracket via POST /v2/orders/bracket.
    """
    mock_client = AsyncMock(spec=DeltaRestClient)
    mock_client.get_cached_contract_spec.return_value = btc_global_spec
    mock_client.config = BotConfig(DELTA_API_KEY="test_key", DELTA_API_SECRET="test_secret")
    mock_client.find_active_stop_loss_order.return_value = None  # No child order on book

    sample_bullish_intent.status = OrderStatus.TRIGGERED
    sample_bullish_intent.exchange_order_id = 112233
    sample_bullish_intent.child_stop_loss_order_id = None
    sample_bullish_intent.product_id = 139

    manager = OrderExecutionManager(mock_client, ExecutionConfig(execution_mode=ExecutionMode.LIVE))

    new_sl = 50850.0
    success = await manager.ratchet_exchange_stop_loss(sample_bullish_intent, new_sl)

    assert success is True
    # Verify fallback to set_position_bracket (POST /v2/orders/bracket)
    mock_client.set_position_bracket.assert_awaited_once_with(
        product_id=139,
        bracket_stop_loss_price=50850.0,
        bracket_take_profit_price=None,
    )


@pytest.mark.asyncio
async def test_trailing_stop_ratchet_prevents_open_order_not_found_on_parent_id(
    btc_global_spec: ContractSpec,
    sample_bullish_intent: OrderIntent,
) -> None:
    """
    Direct Trap Guard:
    Verifies that calling ratchet on an open position NEVER sends the parent entry order ID
    to edit_order, preventing the critical Delta error:
    {'error': {'code': 'open_order_not_found'}, 'success': False}.
    """
    mock_client = AsyncMock(spec=DeltaRestClient)
    mock_client.get_cached_contract_spec.return_value = btc_global_spec
    mock_client.config = BotConfig(DELTA_API_KEY="test_key", DELTA_API_SECRET="test_secret")

    parent_order_id = 99881122
    child_stop_id = 44332211

    sample_bullish_intent.status = OrderStatus.TRIGGERED
    sample_bullish_intent.exchange_order_id = parent_order_id
    sample_bullish_intent.child_stop_loss_order_id = child_stop_id
    sample_bullish_intent.product_id = 139

    manager = OrderExecutionManager(mock_client, ExecutionConfig(execution_mode=ExecutionMode.LIVE))

    new_sl = 50900.0
    success = await manager.ratchet_exchange_stop_loss(sample_bullish_intent, new_sl)

    assert success is True

    # Assert that parent_order_id was NEVER passed to edit_order or edit_bracket_order
    for call in mock_client.edit_order.await_args_list:
        assert call.kwargs.get("order_id") != parent_order_id
        assert call.kwargs.get("order_id") == child_stop_id
        assert "bracket_stop_loss_price" not in call.kwargs  # Must use stop_price on active child order!
        assert call.kwargs.get("stop_price") == new_sl

    mock_client.edit_bracket_order.assert_not_awaited()


# -----------------------------------------------------------------------------
# 3. Trap 2: Dual-Constraint Margin & Leverage Sizing
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_margin_and_leverage_sizing_constraints(btc_global_spec: ContractSpec) -> None:
    """
    Verifies dual-constraint sizing:
    1. Ample margin: Sized by risk budget
    2. Restricted margin: Sized down by leverage constraint
    3. Severe margin deficit: Rejected with InsufficientMarginError
    """
    mock_client = AsyncMock(spec=DeltaRestClient)
    mock_client.config = BotConfig()

    manager = OrderExecutionManager(mock_client, ExecutionConfig(leverage=5, margin_safety_buffer=0.90))

    # Scenario A: Ample margin ($10,000 equity, $5,000 free margin)
    # Trigger=50000, SL=49000 (risk = $1000 per BTC).
    # Risk budget (1% of 10,000) = $100 -> Target BTC = 0.1 BTC -> 100 contracts (0.001 value)
    # Margin req at 5x = (100 * 0.001 * 50000) / 5 = $1,000 <= $4,500 ($5000 * 0.90)
    res_a = await manager.check_and_size_margin(
        spec=btc_global_spec,
        trigger_price=50000.0,
        stop_loss_price=49000.0,
        account_equity=10000.0,
        risk_pct=1.0,
        available_margin_override=5000.0,
    )
    assert res_a.is_margin_sufficient is True
    assert res_a.final_lots == 100
    assert res_a.required_margin == pytest.approx(1000.0)

    # Scenario B: Leverage constraint binds
    # Free margin is only $300. Usable margin = 300 * 0.90 = $270.
    # At 5x leverage, max notional = $270 * 5 = $1,350 -> 0.027 BTC -> 27 contracts.
    # Risk allows 100 contracts, but margin only allows 27 contracts -> Capped at 27 contracts!
    res_b = await manager.check_and_size_margin(
        spec=btc_global_spec,
        trigger_price=50000.0,
        stop_loss_price=49000.0,
        account_equity=10000.0,
        risk_pct=1.0,
        available_margin_override=300.0,
    )
    assert res_b.is_margin_sufficient is True
    assert res_b.final_lots == 27
    assert res_b.final_lots < res_b.risk_based_lots
    assert res_b.required_margin <= 270.0

    # Scenario C: Severe margin deficit (< 1 contract affordance)
    # Free margin is only $5.0. 1 contract (0.001 BTC = $50 notional) at 5x requires $10.0 initial margin.
    res_c = await manager.check_and_size_margin(
        spec=btc_global_spec,
        trigger_price=50000.0,
        stop_loss_price=49000.0,
        account_equity=10000.0,
        risk_pct=1.0,
        available_margin_override=5.0,
    )
    assert res_c.is_margin_sufficient is False
    assert res_c.final_lots == 0
    assert "insufficient for 1 BTCUSDT contract" in str(res_c.rejection_reason)


@pytest.mark.asyncio
async def test_insufficient_margin_rejection_prevents_api_call(
    btc_global_spec: ContractSpec,
    sample_bullish_intent: OrderIntent,
) -> None:
    """Verifies that an order with insufficient margin raises InsufficientMarginError and halts."""
    mock_client = AsyncMock(spec=DeltaRestClient)
    mock_client.get_cached_contract_spec.return_value = btc_global_spec
    mock_client.config = BotConfig()

    manager = OrderExecutionManager(mock_client, ExecutionConfig(leverage=5))

    with pytest.raises(InsufficientMarginError) as exc_info:
        await manager.execute_stop_order_with_brackets(
            order_intent=sample_bullish_intent,
            account_equity=1000.0,
            available_margin_override=2.0,  # Far too low
        )

    assert "insufficient" in str(exc_info.value).lower()
    # Ensure no order was sent to the exchange
    mock_client.place_order.assert_not_called()


# -----------------------------------------------------------------------------
# 4. Trap 3: Delta Exchange India vs Global Product IDs
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dynamic_product_id_resolution(
    btc_global_spec: ContractSpec,
    btc_india_spec: ContractSpec,
    sample_bullish_intent: OrderIntent,
) -> None:
    """
    Verifies that the execution manager correctly differentiates between
    Delta Global (product_id: 139) and Delta India (product_id: 27).
    """
    mock_client = AsyncMock(spec=DeltaRestClient)
    mock_client.config = BotConfig()

    # 1. Delta Global Execution
    mock_client.get_cached_contract_spec.return_value = btc_global_spec
    manager_global = OrderExecutionManager(mock_client, ExecutionConfig(execution_mode=ExecutionMode.DRY_RUN))
    order_global = await manager_global.execute_stop_order_with_brackets(
        order_intent=sample_bullish_intent,
        available_margin_override=5000.0,
    )
    assert order_global.product_id == 139

    # 2. Delta India Execution
    mock_client.get_cached_contract_spec.return_value = btc_india_spec
    intent_india = sample_bullish_intent.model_copy(update={"symbol": "BTCUSD"})
    manager_india = OrderExecutionManager(mock_client, ExecutionConfig(execution_mode=ExecutionMode.DRY_RUN))
    order_india = await manager_india.execute_stop_order_with_brackets(
        order_intent=intent_india,
        available_margin_override=5000.0,
    )
    assert order_india.product_id == 27


# -----------------------------------------------------------------------------
# 5. OrderRouter & LocalRegistry Tests
# -----------------------------------------------------------------------------

def test_client_order_id_generation() -> None:
    """Verifies client_order_id is deterministic, starts with DBOT_, and is <= 32 chars."""
    from delta_bot.execution import OrderRouter

    cid_btc = OrderRouter.generate_client_order_id("BTCUSDT")
    assert cid_btc.startswith("DBOT_BTCUSDT_")
    assert len(cid_btc) <= 32

    cid_eth = OrderRouter.generate_client_order_id("ETHUSDT")
    assert cid_eth.startswith("DBOT_ETHUSDT_")
    assert len(cid_eth) <= 32


def test_local_registry_indexing() -> None:
    """Verifies LocalRegistry indexes orders across client_id, exchange_id, and intent_id."""
    from delta_bot.execution import LocalRegistry

    registry = LocalRegistry()
    order = ExchangeOrder(
        id=776655,
        client_order_id="DBOT_BTCUSDT_12345",
        product_id=139,
        symbol="BTCUSDT",
        side="buy",
        order_type="market_order",
        size=10,
    )

    registry.register(order, intent_id="intent_abc")

    # Lookups
    assert registry.get_by_exchange_id(776655) == order
    assert registry.get_by_client_id("DBOT_BTCUSDT_12345") == order
    assert registry.get_by_intent_id("intent_abc") == order
    assert len(registry.active_orders) == 1

    # Removal
    registry.remove(776655)
    assert registry.get_by_exchange_id(776655) is None
    assert registry.get_by_client_id("DBOT_BTCUSDT_12345") is None
    assert registry.get_by_intent_id("intent_abc") is None
    assert len(registry.active_orders) == 0


@pytest.mark.asyncio
async def test_trailing_stop_ratchet_loosening_rejected(
    btc_global_spec: ContractSpec,
    sample_bullish_intent: OrderIntent,
) -> None:
    """Verifies that attempting to loosen a stop loss is strictly rejected."""
    mock_client = AsyncMock(spec=DeltaRestClient)
    mock_client.get_cached_contract_spec.return_value = btc_global_spec
    mock_client.config = BotConfig()

    sample_bullish_intent.exchange_order_id = 12345
    sample_bullish_intent.current_stop_loss = 50600.0  # Current Long SL

    manager = OrderExecutionManager(mock_client, ExecutionConfig(execution_mode=ExecutionMode.LIVE))

    # Attempt to loosen SL down to 50500.0 (worse risk for Long) -> MUST REJECT!
    loosening_res = await manager.handle_trailing_stop_ratchet(sample_bullish_intent, 50500.0)
    assert loosening_res is False
    assert sample_bullish_intent.current_stop_loss == 50600.0  # Unchanged!

    # Favorable ratchet up to 50750.0 -> MUST ACCEPT!
    favorable_res = await manager.handle_trailing_stop_ratchet(sample_bullish_intent, 50750.0)
    assert favorable_res is True
    assert sample_bullish_intent.current_stop_loss == 50750.0


@pytest.mark.asyncio
async def test_emergency_flatten_kill_switch(btc_global_spec: ContractSpec) -> None:
    """Verifies that emergency_flatten purges open orders, closes positions, and trips circuit breaker."""
    mock_client = AsyncMock(spec=DeltaRestClient)
    mock_client.get_cached_contract_spec.return_value = btc_global_spec
    mock_client.cancel_all_orders.return_value = [{"id": 111}, {"id": 222}]
    mock_client.config = BotConfig()

    manager = OrderExecutionManager(mock_client, ExecutionConfig(execution_mode=ExecutionMode.LIVE))

    report = await manager.emergency_flatten("BTCUSDT")

    assert report["cancelled_orders"] == 2
    assert manager.risk_manager.is_circuit_breaker_tripped is True
    assert len(manager.registry.active_orders) == 0


@pytest.mark.asyncio
async def test_anti_flood_idempotency_shield(
    btc_global_spec: ContractSpec,
    sample_bullish_intent: OrderIntent,
) -> None:
    """Verifies that duplicate order submissions within 5 seconds are blocked by Anti-Flood Shield."""
    mock_client = AsyncMock(spec=DeltaRestClient)
    mock_client.get_cached_contract_spec.return_value = btc_global_spec
    mock_client.config = BotConfig()
    mock_client.place_order.return_value = ExchangeOrder(
        id=777,
        product_id=btc_global_spec.product_id,
        symbol=btc_global_spec.symbol,
        side="buy",
        order_type="market_order",
        size=10,
        stop_price=51000.0,
        state="open",
    )

    manager = OrderExecutionManager(mock_client, ExecutionConfig(execution_mode=ExecutionMode.LIVE))

    # First submission -> succeeds
    order1 = await manager.execute_signal(sample_bullish_intent, account_equity=10000.0)
    assert order1 is not None
    assert order1.id == 777

    # Immediate duplicate submission (same symbol, side, trigger_price) -> blocked!
    duplicate_intent = sample_bullish_intent.model_copy()
    duplicate_intent.intent_id = "dup_intent_999"
    order2 = await manager.execute_signal(duplicate_intent, account_equity=10000.0)
    assert order2 is None
    assert duplicate_intent.status == OrderStatus.CANCELLED_MANUAL
    assert "Anti-flood" in (duplicate_intent.cancellation_reason or "")


@pytest.mark.asyncio
async def test_stale_feed_shield(
    btc_global_spec: ContractSpec,
    sample_bullish_intent: OrderIntent,
) -> None:
    """Verifies that live order dispatching aborts if market feed quotes are older than 15s."""
    mock_client = AsyncMock(spec=DeltaRestClient)
    mock_client.get_cached_contract_spec.return_value = btc_global_spec
    mock_client.config = BotConfig()

    manager = OrderExecutionManager(mock_client, ExecutionConfig(execution_mode=ExecutionMode.LIVE))

    # Mock state_manager with stale feed (>15s)
    mock_sm = MagicMock()
    mock_sm.is_reconciling = False
    mock_sm.get_market_context_age.return_value = 22.5  # 22.5 seconds old
    manager.set_state_manager(mock_sm)

    res = await manager.execute_signal(sample_bullish_intent, account_equity=10000.0)
    assert res is None
    assert sample_bullish_intent.status == OrderStatus.CANCELLED_MANUAL
    assert "Stale feed" in (sample_bullish_intent.cancellation_reason or "")

    # Now simulate fresh feed (<15s)
    mock_sm.get_market_context_age.return_value = 1.2
    mock_client.place_order.return_value = ExchangeOrder(
        id=888,
        product_id=btc_global_spec.product_id,
        symbol=btc_global_spec.symbol,
        side="buy",
        order_type="market_order",
        size=10,
        stop_price=51000.0,
        state="open",
    )

    fresh_intent = sample_bullish_intent.model_copy()
    fresh_intent.intent_id = "fresh_intent_1"
    fresh_intent.trigger_price = 51200.0  # distinct trigger to pass anti-flood
    res2 = await manager.execute_signal(fresh_intent, account_equity=10000.0)
    assert res2 is not None
    assert res2.id == 888


@pytest.mark.asyncio
async def test_split_brain_reconnect_shield(
    btc_global_spec: ContractSpec,
    sample_bullish_intent: OrderIntent,
) -> None:
    """Verifies that orders are locked and blocked when state reconciliation is in progress."""
    mock_client = AsyncMock(spec=DeltaRestClient)
    mock_client.get_cached_contract_spec.return_value = btc_global_spec
    mock_client.config = BotConfig()

    manager = OrderExecutionManager(mock_client, ExecutionConfig(execution_mode=ExecutionMode.LIVE))

    # Case 1: Manual execution lock active
    manager.is_execution_locked = True
    intent1 = sample_bullish_intent.model_copy()
    intent1.intent_id = "lock_test_1"
    res1 = await manager.execute_signal(intent1, account_equity=10000.0)
    assert res1 is None
    assert intent1.status == OrderStatus.CANCELLED_MANUAL

    # Case 2: State manager reconciliation in progress
    manager.is_execution_locked = False
    mock_sm = MagicMock()
    mock_sm.is_reconciling = True
    manager.set_state_manager(mock_sm)

    intent2 = sample_bullish_intent.model_copy()
    intent2.intent_id = "lock_test_2"
    intent2.trigger_price = 51500.0
    res2 = await manager.execute_signal(intent2, account_equity=10000.0)
    assert res2 is None
    assert intent2.status == OrderStatus.CANCELLED_MANUAL
    assert "Split-brain" in (intent2.cancellation_reason or "")

