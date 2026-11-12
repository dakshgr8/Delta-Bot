"""
tests/test_circuit_breaker_and_tenant_drawdown_fixes.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regression test suite verifying fixes for:
1. Circuit breaker trip does not flood/spam critical logs on subsequent calls.
2. RiskManager reset_circuit_breaker and reset_daily_circuit_breaker work properly.
3. Increasing max_drawdown_pct auto-resets circuit breaker when current DD is within new threshold.
4. RingBufferLogHandler tenant isolation filters out other tenants' logs and default paper breaker logs.
5. REST control endpoints for resetting circuit breaker (/api/control/reset-circuit-breaker and /api/tenants/{id}/control/reset-circuit-breaker).
6. Starting/resuming trading automatically clears tripped circuit breaker state.
7. min_stop_loss_pct default is 0.05% across configs.
8. ExecutionMode.REAL tenants are recognized by orchestrator ticker/candle context updates.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import time
import logging

from delta_bot.config import BotConfig
from delta_bot.models import ExecutionMode, ContractSpec
from delta_bot.risk import RiskConfig, RiskManager
from delta_bot.strategy import StrategyConfig
from delta_bot.tenant import TenantInstance, TenantConfig
from delta_bot.server import RingBufferLogHandler, create_app
from fastapi.testclient import TestClient


def test_circuit_breaker_no_log_spam_on_repeated_checks(caplog) -> None:
    """Verifies that circuit breaker only logs critical warning once when transitioning to tripped state."""
    caplog.set_level(logging.CRITICAL)
    rm = RiskManager(
        RiskConfig(
            daily_max_drawdown_pct=5.0,
            enable_drawdown_circuit_breaker=True,
        ),
        initial_equity=100.0,
        name="test_tenant",
    )
    # Start equity = 100, equity drops to 90 -> 10% loss > 5% threshold
    assert rm.update_daily_equity(90.0) is False
    assert rm.is_circuit_breaker_tripped is True

    # Repeated calls on closed candles while already tripped
    for _ in range(5):
        assert rm.update_daily_equity(89.0) is False

    # Should only log critical message once
    critical_logs = [r for r in caplog.records if r.levelno == logging.CRITICAL and "CIRCUIT BREAKER TRIPPED" in r.message]
    assert len(critical_logs) == 1
    assert "[TENANT test_tenant]" in critical_logs[0].message


def test_risk_manager_reset_circuit_breaker() -> None:
    """Verifies that reset_circuit_breaker and reset_daily_circuit_breaker clear tripped status and re-anchor equity."""
    rm = RiskManager(
        RiskConfig(daily_max_drawdown_pct=5.0, enable_drawdown_circuit_breaker=True),
        initial_equity=100.0,
    )
    rm.update_daily_equity(90.0)
    assert rm.is_circuit_breaker_tripped is True

    # Reset with current equity
    rm.reset_circuit_breaker(current_equity=92.0)
    assert rm.is_circuit_breaker_tripped is False
    assert rm.circuit_breaker.current_drawdown_pct == 0.0
    assert rm.circuit_breaker.session_start_equity == 92.0

    # Trip again and reset via alias
    rm.update_daily_equity(80.0)
    assert rm.is_circuit_breaker_tripped is True
    rm.reset_daily_circuit_breaker(current_equity=85.0)
    assert rm.is_circuit_breaker_tripped is False
    assert rm.circuit_breaker.session_start_equity == 85.0


def test_tenant_apply_config_updates_resets_circuit_breaker_on_increased_threshold() -> None:
    """Verifies that increasing max_drawdown_pct automatically resets a tripped circuit breaker if new threshold > current DD."""
    cfg = TenantConfig(
        tenant_id="test-drawdown-res",
        name="Drawdown Test",
        max_drawdown_pct=5.0,
        initial_balance=100.0,
    )
    tenant = TenantInstance(cfg)
    tenant.risk_manager.update_daily_equity(94.0)  # 6.0% loss -> tripped against 5.0%
    assert tenant.risk_manager.is_circuit_breaker_tripped is True
    assert tenant.risk_manager.circuit_breaker.current_drawdown_pct == 6.0

    # User adjusts max_drawdown_pct to 8.0%
    tenant.apply_config_updates({"max_drawdown_pct": 8.0})
    assert tenant.config.max_drawdown_pct == 8.0
    assert tenant.risk_manager.config.daily_max_drawdown_pct == 8.0
    # Current DD is 6.0% < 8.0%, circuit breaker should auto-reset!
    assert tenant.risk_manager.is_circuit_breaker_tripped is False


def test_ring_buffer_log_isolation() -> None:
    """Verifies that RingBufferLogHandler filters logs so custom tenants don't see other tenant or paper breaker logs."""
    handler = RingBufferLogHandler(capacity=50)

    logger = logging.getLogger("test_isolation")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # 1. Global paper breaker log
    logger.critical("*** CIRCUIT BREAKER TRIPPED *** [TENANT default] Reason: Daily max drawdown breached: 5.74% loss")
    # 2. General market log
    logger.info("[TICKER] BTCUSD Mark: $73500.00")
    # 3. Log for tenant A
    logger.info("[TENANT tenant-a:BTCUSD] Breakout signal detected")
    # 4. Log for rupdave-22
    logger.info("[TENANT rupdave-22:BTCUSD] Breakout order dispatched: 10 lots")

    # Fetch logs for rupdave-22
    rup_logs = handler.get_logs(tenant_id="rupdave-22")
    rup_msgs = [l["message"] for l in rup_logs]

    # rupdave-22 should see its own log and general market log
    assert any("[TENANT rupdave-22:BTCUSD]" in m for m in rup_msgs)
    assert any("[TICKER]" in m for m in rup_msgs)
    # rupdave-22 must NOT see tenant-a's log or default paper circuit breaker log
    assert not any("[TENANT tenant-a" in m for m in rup_msgs)
    assert not any("CIRCUIT BREAKER TRIPPED" in m for m in rup_msgs)


def test_default_min_stop_loss_pct_is_safe_for_high_priced_assets() -> None:
    """Verifies that default min_stop_loss_pct is 0.05% across BotConfig, RiskConfig, and StrategyConfig."""
    b_cfg = BotConfig()
    r_cfg = RiskConfig()
    s_cfg = StrategyConfig(symbol="BTCUSD")

    assert b_cfg.min_stop_loss_pct == 0.05
    assert r_cfg.min_stop_loss_pct == 0.05
    assert s_cfg.min_stop_loss_pct == 0.05


@pytest.mark.asyncio
async def test_control_endpoints_reset_circuit_breaker() -> None:
    """Verifies that /api/control/reset-circuit-breaker and /api/tenants/{id}/control/reset-circuit-breaker work cleanly."""
    mock_orch = MagicMock()
    mock_orch.config = BotConfig(dashboard_auth_token_str="")
    mock_orch.mode = ExecutionMode.PAPER
    mock_orch.is_running = True
    mock_orch.paper_simulator = MagicMock(current_equity=10000.0)
    mock_orch.risk_manager = RiskManager(
        RiskConfig(daily_max_drawdown_pct=5.0, enable_drawdown_circuit_breaker=True),
        initial_equity=10000.0,
        name="default",
    )
    # Trip breaker
    mock_orch.risk_manager.update_daily_equity(9000.0)
    assert mock_orch.risk_manager.is_circuit_breaker_tripped is True

    # Setup tenant
    t_cfg = TenantConfig(tenant_id="user-t1", name="User Tenant 1", initial_balance=500.0)
    user_tenant = TenantInstance(t_cfg)
    user_tenant.risk_manager.update_daily_equity(400.0)  # Tripped
    assert user_tenant.risk_manager.is_circuit_breaker_tripped is True

    mock_orch.tenant_manager = MagicMock()
    mock_orch.tenant_manager.get_tenant.side_effect = lambda tid: user_tenant if tid == "user-t1" else None

    app = create_app(mock_orch)
    client = TestClient(app)

    # 1. Reset tenant circuit breaker
    res_t = client.post("/api/tenants/user-t1/control/reset-circuit-breaker")
    assert res_t.status_code == 200
    assert res_t.json()["status"] == "success"
    assert user_tenant.risk_manager.is_circuit_breaker_tripped is False

    # 2. Reset master orchestrator circuit breaker
    res_m = client.post("/api/control/reset-circuit-breaker")
    assert res_m.status_code == 200
    assert res_m.json()["status"] == "success"
    assert mock_orch.risk_manager.is_circuit_breaker_tripped is False


@pytest.mark.asyncio
async def test_control_start_automatically_resets_tripped_circuit_breaker() -> None:
    """Verifies that starting or resuming an engine automatically clears tripped circuit breaker."""
    mock_orch = MagicMock()
    mock_orch.config = BotConfig(dashboard_auth_token_str="")
    mock_orch.mode = ExecutionMode.PAPER
    mock_orch.is_running = False
    mock_orch.paper_simulator = MagicMock(current_equity=10000.0)
    mock_orch.risk_manager = RiskManager(
        RiskConfig(daily_max_drawdown_pct=5.0, enable_drawdown_circuit_breaker=True),
        initial_equity=10000.0,
        name="default",
    )
    mock_orch.risk_manager.update_daily_equity(9000.0)
    assert mock_orch.risk_manager.is_circuit_breaker_tripped is True

    t_cfg = TenantConfig(tenant_id="user-t2", name="User Tenant 2", initial_balance=500.0)
    user_tenant = TenantInstance(t_cfg)
    user_tenant.risk_manager.update_daily_equity(400.0)
    assert user_tenant.risk_manager.is_circuit_breaker_tripped is True

    mock_orch.tenant_manager = MagicMock()
    mock_orch.tenant_manager.get_tenant.side_effect = lambda tid: user_tenant if tid == "user-t2" else None

    app = create_app(mock_orch)
    client = TestClient(app)

    # Start tenant
    res = client.post("/api/tenants/user-t2/control/start")
    assert res.status_code == 200
    assert user_tenant.is_running is True
    assert user_tenant.risk_manager.is_circuit_breaker_tripped is False

    # Start orchestrator
    res_o = client.post("/api/control/start")
    assert res_o.status_code == 200
    assert mock_orch.is_running is True
    assert mock_orch.risk_manager.is_circuit_breaker_tripped is False
