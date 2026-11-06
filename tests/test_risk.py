"""
tests/test_risk.py
~~~~~~~~~~~~~~~~~~

Unit tests for Risk Management & Pre-Trade Safety Engine (Phase 4).
Tests:
- Dual-constraint lot sizing (Risk-Based vs Margin-Based)
- Margin safety buffer (90% usable free collateral)
- Insufficient margin rejection
- Daily maximum drawdown circuit breaker (4.0% loss halt)
- Maximum concurrent position limit (strictly 1 active position)
- Session rollover circuit breaker reset
"""

import pytest

from delta_bot.models import ContractSpec, MarginCheckResult
from delta_bot.risk import CircuitBreakerState, RiskConfig, RiskManager


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
def risk_manager() -> RiskManager:
    config = RiskConfig(
        max_account_risk_pct=1.0,
        max_leverage=5,
        margin_safety_buffer=0.90,
        daily_max_drawdown_pct=4.0,
        max_concurrent_positions=1,
    )
    return RiskManager(config=config, initial_equity=10000.0)


# -----------------------------------------------------------------------------
# 1. Dual-Constraint Sizing & Pre-Trade Checks
# -----------------------------------------------------------------------------

def test_risk_manager_pre_trade_sizing_sufficient(
    risk_manager: RiskManager,
    btc_spec: ContractSpec,
) -> None:
    """Verifies standard risk-based sizing when free margin is abundant."""
    # Equity = $10,000, 1% risk = $100.
    # Trigger = 50,000, SL = 49,000 -> Risk per BTC = $1,000.
    # Target BTC = 0.1 BTC -> 100 contracts (0.001 BTC lot value).
    # Free margin = $5,000. Required margin at 5x = (100 * 0.001 * 50,000) / 5 = $1,000.
    res = risk_manager.size_position(
        spec=btc_spec,
        trigger_price=50000.0,
        stop_loss_price=49000.0,
        account_equity=10000.0,
        available_margin=5000.0,
    )

    assert res.is_margin_sufficient is True
    assert res.final_lots == 100
    assert res.required_margin == 1000.0
    assert res.risk_based_lots == 100
    assert res.margin_based_lots > 100  # Margin is not the bottleneck
    assert res.rejection_reason is None


def test_risk_manager_margin_constraint_binding(
    risk_manager: RiskManager,
    btc_spec: ContractSpec,
) -> None:
    """Verifies that margin limits throttle position size down when collateral is tight."""
    # Free margin = $200. Usable margin = 200 * 0.90 = $180.
    # Max affordable notional at 5x = $180 * 5 = $900 -> 0.018 BTC -> 18 contracts.
    # Risk budget would allow 100 contracts, but margin strictly caps it at 18 contracts!
    res = risk_manager.size_position(
        spec=btc_spec,
        trigger_price=50000.0,
        stop_loss_price=49000.0,
        account_equity=10000.0,
        available_margin=200.0,
    )

    assert res.is_margin_sufficient is True
    assert res.final_lots == 18
    assert res.final_lots < res.risk_based_lots
    assert res.required_margin <= 180.0


def test_risk_manager_insufficient_margin_rejection(
    risk_manager: RiskManager,
    btc_spec: ContractSpec,
) -> None:
    """Verifies order rejection when free margin cannot even afford 1 minimum contract."""
    # Free margin = $5.0. 1 contract (0.001 BTC at $50,000 = $50 notional) at 5x requires $10.0 initial margin.
    res = risk_manager.size_position(
        spec=btc_spec,
        trigger_price=50000.0,
        stop_loss_price=49000.0,
        account_equity=10000.0,
        available_margin=5.0,
    )

    assert res.is_margin_sufficient is False
    assert res.final_lots == 0
    assert "insufficient for 1 BTCUSDT contract" in str(res.rejection_reason)


# -----------------------------------------------------------------------------
# 2. Circuit Breaker & Daily Max Drawdown
# -----------------------------------------------------------------------------

def test_daily_max_drawdown_circuit_breaker(risk_manager: RiskManager) -> None:
    """Verifies that exceeding the 4.0% daily drawdown limit trips the circuit breaker and halts trading."""
    # Initial session equity = $10,000. Drawdown limit = 4.0% ($400 loss).
    assert risk_manager.is_circuit_breaker_tripped is False

    # Equity dips to $9,700 (3.0% loss) -> Allowed
    ok_1 = risk_manager.update_daily_equity(9700.0)
    assert ok_1 is True
    assert risk_manager.is_circuit_breaker_tripped is False
    assert risk_manager.circuit_breaker.current_drawdown_pct == 3.0

    # Equity dips to $9,580 (4.2% loss) -> BREACH! Circuit breaker trips!
    ok_2 = risk_manager.update_daily_equity(9580.0)
    assert ok_2 is False
    assert risk_manager.is_circuit_breaker_tripped is True
    assert "Daily max drawdown breached" in str(risk_manager.circuit_breaker.trip_reason)

    # Pre-trade check should now strictly reject any new order intent
    can_open, reason = risk_manager.can_open_new_position()
    assert can_open is False
    assert "Circuit breaker is TRIPPED" in str(reason)


def test_circuit_breaker_reset_on_session_rollover(risk_manager: RiskManager) -> None:
    """Verifies that circuit breaker can be reset on daily session rollover (00:00 UTC)."""
    # Trip circuit breaker
    risk_manager.update_daily_equity(9500.0)
    assert risk_manager.is_circuit_breaker_tripped is True

    # 00:00 UTC rollover with new starting equity of $9,500
    risk_manager.reset_daily_session(current_equity=9500.0)
    assert risk_manager.is_circuit_breaker_tripped is False
    assert risk_manager.circuit_breaker.session_start_equity == 9500.0
    assert risk_manager.circuit_breaker.current_drawdown_pct == 0.0

    can_open, reason = risk_manager.can_open_new_position()
    assert can_open is True
    assert reason is None


# -----------------------------------------------------------------------------
# 3. Max Concurrent Positions Guard
# -----------------------------------------------------------------------------

def test_max_concurrent_positions_guard(risk_manager: RiskManager) -> None:
    """Verifies strict enforcement of max 1 concurrent position across the bot."""
    assert risk_manager.active_positions_count == 0

    # Can open first position
    can_open_1, reason_1 = risk_manager.can_open_new_position()
    assert can_open_1 is True
    assert reason_1 is None

    # Simulate trade entry
    risk_manager.record_trade_entry()
    assert risk_manager.active_positions_count == 1

    # Attempt to open second position -> REJECTED
    can_open_2, reason_2 = risk_manager.can_open_new_position()
    assert can_open_2 is False
    assert "Concurrent position limit reached" in str(reason_2)

    # Simulate trade exit
    risk_manager.record_trade_exit(pnl=+50.0, current_equity=10050.0)
    assert risk_manager.active_positions_count == 0

    # Can open position again
    can_open_3, reason_3 = risk_manager.can_open_new_position()
    assert can_open_3 is True
    assert reason_3 is None
