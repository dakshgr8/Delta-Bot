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
        enable_drawdown_circuit_breaker=True,
        daily_max_drawdown_pct=4.0,
        max_concurrent_positions=1,
        enable_revenge_breaker=True,
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


def test_risk_manager_zero_equity_or_margin_returns_valid_model(
    risk_manager: RiskManager,
    btc_spec: ContractSpec,
) -> None:
    """Verifies that 0 equity or 0 margin cleanly returns valid MarginCheckResult without Pydantic ValidationError."""
    res_zero_eq = risk_manager.size_position(
        spec=btc_spec,
        trigger_price=50000.0,
        stop_loss_price=49000.0,
        account_equity=0.0,
        available_margin=0.0,
    )
    assert isinstance(res_zero_eq, MarginCheckResult)
    assert res_zero_eq.is_margin_sufficient is False
    assert res_zero_eq.final_lots == 0
    assert res_zero_eq.symbol == "BTCUSDT"
    assert res_zero_eq.product_id == 139
    assert res_zero_eq.leverage == 5.0
    assert res_zero_eq.risk_budget_usd == 0.0
    assert "Insufficient collateral" in str(res_zero_eq.rejection_reason)


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


# -----------------------------------------------------------------------------
# 4. Prop Firm & Risk Protection Safeguards (Feature D)
# -----------------------------------------------------------------------------

def test_minimum_lot_risk_gatekeeper(risk_manager: RiskManager, btc_spec: ContractSpec) -> None:
    """Verifies that trades are rejected if 1 minimum lot exceeds the allowed risk allocation."""
    # Equity = $200, 1% risk budget = $2.00
    # Trigger = 50,000, SL = 45,000 -> SL distance = $5,000
    # 1 contract of BTCUSDT (0.001 BTC) risks 1 * 0.001 * 5000 = $5.00
    # $5.00 > $2.00 * 1.05 tolerance -> Must reject!
    res = risk_manager.size_position(
        spec=btc_spec,
        trigger_price=50000.0,
        stop_loss_price=45000.0,
        account_equity=200.0,
        available_margin=500.0,
    )
    assert res.is_margin_sufficient is False
    assert "exceeds risk allocation" in str(res.rejection_reason)


def test_revenge_trading_breaker_quarantine(risk_manager: RiskManager) -> None:
    """Verifies that 2 rapid stop-outs trigger quarantine for that symbol."""
    sym = "BTCUSDT"

    # Loss 1: First stop out
    risk_manager.record_trade_exit(pnl=-40.0, current_equity=9960.0, symbol=sym)
    can_open, _ = risk_manager.can_open_new_position(symbol=sym)
    assert can_open is True  # 1 loss is tolerated

    # Loss 2: Second stop out within 15 minutes
    risk_manager.record_trade_exit(pnl=-50.0, current_equity=9910.0, symbol=sym)

    # Symbol BTCUSDT must now be quarantined
    can_open_btc, reason_btc = risk_manager.can_open_new_position(symbol=sym)
    assert can_open_btc is False
    assert "[REVENGE BREAKER]" in str(reason_btc)
    assert "quarantined" in str(reason_btc)

    # Other symbols (e.g. ETHUSDT) remain unblocked
    can_open_eth, _ = risk_manager.can_open_new_position(symbol="ETHUSDT")
    assert can_open_eth is True


def test_daily_rollover_spread_spike_shield(risk_manager: RiskManager) -> None:
    """Verifies that new orders are blocked during daily rollover window (21:58 - 22:10 UTC)."""
    from datetime import datetime, timezone

    # 1. During 21:58 UTC -> Blocked
    dt_rollover_1 = datetime(2026, 11, 7, 21, 59, 0, tzinfo=timezone.utc)
    can_open_1, reason_1 = risk_manager.can_open_new_position(now_utc=dt_rollover_1)
    assert can_open_1 is False
    assert "[ROLLOVER SHIELD]" in str(reason_1)

    # 2. During 22:05 UTC -> Blocked
    dt_rollover_2 = datetime(2026, 11, 7, 22, 5, 0, tzinfo=timezone.utc)
    can_open_2, reason_2 = risk_manager.can_open_new_position(now_utc=dt_rollover_2)
    assert can_open_2 is False
    assert "[ROLLOVER SHIELD]" in str(reason_2)

    # 3. Normal trading hours (14:30 UTC) -> Allowed
    dt_normal = datetime(2026, 11, 7, 14, 30, 0, tzinfo=timezone.utc)
    can_open_3, reason_3 = risk_manager.can_open_new_position(now_utc=dt_normal)
    assert can_open_3 is True
    assert reason_3 is None


def test_margin_sizing_never_exceeds_max_usable_margin(risk_manager: RiskManager) -> None:
    """Verifies that margin-based lot sizing never overshoots usable margin due to rounding."""
    xrp_spec = ContractSpec(
        product_id=14969,
        symbol="XRPUSD",
        contract_type="perpetual_futures",
        contract_value=1.0,
        tick_size=0.0001,
        maker_commission_rate=0.0002,
        taker_commission_rate=0.0005,
        quoting_symbol="USD",
    )
    # Available margin = $10,000, 90% buffer -> max usable margin = $9,000.00
    # At trigger 1.1572 and 5x lev, notional capacity = 9000 * 5 / 1.1572 = 38886.9685
    # Sizing must floor to 38,886 lots and NEVER round up to 38,887 (which would cost $9000.01)
    res = risk_manager.size_position(
        spec=xrp_spec,
        trigger_price=1.1572,
        stop_loss_price=1.1586,
        account_equity=10000.0,
        available_margin=10000.0,
    )
    assert res.is_margin_sufficient is True
    assert res.final_lots == 38886
    assert res.required_margin <= 9000.00
    assert res.rejection_reason is None


def test_unfunded_live_account_does_not_trip_circuit_breaker() -> None:
    """Verifies that an unfunded live account (equity=0.0) does not trip a 100% drawdown circuit breaker."""
    config = RiskConfig(
        max_account_risk_pct=1.0,
        max_leverage=5,
        margin_safety_buffer=0.90,
        daily_max_drawdown_pct=4.0,
        max_concurrent_positions=1,
    )
    rm = RiskManager(config=config, initial_equity=0.0)
    assert rm.circuit_breaker.session_start_equity == 0.0

    # Updating equity with 0.0 must not trip circuit breaker
    is_ok = rm.update_daily_equity(0.0)
    assert is_ok is True
    assert rm.circuit_breaker.is_tripped is False

    can_open, reason = rm.can_open_new_position()
    # Circuit breaker is not tripped
    assert "Circuit breaker is TRIPPED" not in str(reason)


def test_default_risk_manager_has_no_drawdown_or_concurrent_or_revenge_restrictions() -> None:
    """Verifies that by default, concurrent positions are unlimited, daily drawdown breaker is disabled, and revenge breaker is disabled."""
    rm = RiskManager(initial_equity=10000.0)
    assert rm.config.enable_drawdown_circuit_breaker is False
    assert rm.config.daily_max_drawdown_pct is None
    assert rm.config.max_concurrent_positions is None
    assert rm.config.enable_revenge_breaker is False

    # 1. Multiple concurrent positions allowed across symbols
    can_open, reason = rm.can_open_new_position(active_positions=10)
    assert can_open is True
    assert reason is None

    # 2. Large drawdown does NOT trip circuit breaker
    is_safe = rm.update_daily_equity(5000.0)  # 50% loss
    assert is_safe is True
    assert rm.is_circuit_breaker_tripped is False

    # 3. Repeated stop-outs do NOT quarantine symbol
    sym = "BTCUSDT"
    rm.record_trade_exit(pnl=-100.0, current_equity=4900.0, symbol=sym)
    rm.record_trade_exit(pnl=-100.0, current_equity=4800.0, symbol=sym)
    can_open_sym, reason_sym = rm.can_open_new_position(symbol=sym)
    assert can_open_sym is True
    assert reason_sym is None


