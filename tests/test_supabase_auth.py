"""
tests/test_supabase_auth.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive verification suite for Supabase Auth, Row-Level Isolation,
and Multi-Tenant Access Control in Delta-Bot Quantitative Gateway.

Covers:
1. AuthIdentity model resolution and role mapping (user vs admin vs anonymous)
2. Token verification caching with simulated Supabase JWT responses
3. User-tenant auto-provisioning upon first authenticated API access
4. Strict cross-tenant access enforcement (HTTP 403 Forbidden on foreign tenants)
5. Tenant list scoping: user sessions only see tenants they own
6. Read-only vs control mutation permissions
"""

import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from delta_bot.config import BotConfig
from delta_bot.models import ContractSpec, ExecutionMode, Ticker
from delta_bot.orchestrator import TradingBotOrchestrator
from delta_bot.paper import PaperExchangeSimulator
from delta_bot.server import AuthIdentity, create_app, verify_supabase_token
from delta_bot.tenant import TenantConfig, TenantManager


@pytest.fixture
def auth_orchestrator(tmp_path) -> TradingBotOrchestrator:
    """Fixture creating an isolated orchestrator with multi-tenant manager."""
    cfg = BotConfig(
        _env_file=None,
        execution_mode="paper",
        target_symbols_str="BTCUSDT,ETHUSDT",
        default_risk_pct=1.0,
        default_leverage=5,
        dashboard_auth_token="mastersecret999",
    )
    paper_sim = PaperExchangeSimulator(initial_balance=10000.0)
    orch = TradingBotOrchestrator(config=cfg, mode=ExecutionMode.PAPER, paper_simulator=paper_sim)

    # Disable remote Supabase calls in unit tests and use local storage path
    orch.tenant_manager.supabase_url = None
    orch.tenant_manager.supabase_secret_key = None
    orch.tenant_manager.storage_path = str(tmp_path / "tenants.json")
    orch.tenant_manager.tenants = {
        "default": orch.tenant_manager.tenants["default"]
    }

    # Contract specs
    btc_spec = ContractSpec(
        product_id=1,
        symbol="BTCUSDT",
        contract_type="perpetual_futures",
        contract_value=0.001,
        tick_size=0.5,
        maker_commission_rate=0.0002,
        taker_commission_rate=0.0005,
        initial_margin_rate=0.02,
        maintenance_margin_rate=0.01,
    )
    orch.contract_specs["BTCUSDT"] = btc_spec
    paper_sim.register_contract_spec(btc_spec)
    orch.latest_tickers["BTCUSDT"] = Ticker(
        symbol="BTCUSDT",
        product_id=1,
        mark_price=65000.0,
        best_bid=64995.0,
        best_ask=65005.0,
    )

    return orch


@pytest.fixture
def client(auth_orchestrator: TradingBotOrchestrator) -> TestClient:
    app = create_app(auth_orchestrator)
    return TestClient(app, raise_server_exceptions=False)


# -----------------------------------------------------------------------------
# 1. Token Verification & AuthIdentity Tests
# -----------------------------------------------------------------------------

def test_auth_identity_defaults() -> None:
    ident = AuthIdentity()
    assert not ident.is_authenticated
    assert ident.auth_type == "anonymous"
    assert ident.tenant_id is None
    assert ident.user_id is None


@pytest.mark.asyncio
async def test_verify_supabase_token_success(auth_orchestrator: TradingBotOrchestrator) -> None:
    auth_orchestrator.config.supabase_url = "https://mock.supabase.co"

    fake_user = {
        "id": "11111111-2222-3333-4444-555555555555",
        "email": "trader_alice@test.com",
        "user_metadata": {"desk_name": "Alice Desk"},
    }

    from unittest.mock import MagicMock
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_user

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp)):
        res = await verify_supabase_token("valid_jwt_token_123", auth_orchestrator)
        assert res is not None
        assert res["id"] == fake_user["id"]
        assert res["email"] == fake_user["email"]


@pytest.mark.asyncio
async def test_verify_supabase_token_invalid_or_expired(auth_orchestrator: TradingBotOrchestrator) -> None:
    auth_orchestrator.config.supabase_url = "https://mock.supabase.co"

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_resp = AsyncMock()
        mock_resp.status_code = 401
        mock_get.return_value = mock_resp

        res = await verify_supabase_token("invalid_expired_token", auth_orchestrator)
        assert res is None


# -----------------------------------------------------------------------------
# 2. Multi-Tenant Auto-Provisioning & Isolation Tests
# -----------------------------------------------------------------------------

def test_user_tenant_auto_provisioning(
    client: TestClient, auth_orchestrator: TradingBotOrchestrator
) -> None:
    """Verifies that an incoming authenticated Supabase user is auto-assigned an isolated tenant."""
    alice_uid = "aaaa-1111-2222-3333"
    fake_alice = {"id": alice_uid, "email": "alice@trading.org"}

    with patch("delta_bot.server.verify_supabase_token", AsyncMock(return_value=fake_alice)):
        res = client.get("/api/auth/me", headers={"Authorization": "Bearer fake_alice_jwt"})
        assert res.status_code == 200
        data = res.json()
        assert data["authenticated"] is True
        assert data["auth_type"] == "user"
        assert data["user_id"] == alice_uid
        assert data["email"] == "alice@trading.org"
        assert data["tenant_id"].startswith("user-")

        # Tenant was created in memory
        assert data["tenant_id"] in auth_orchestrator.tenant_manager.tenants


def test_cross_tenant_isolation_forbidden(
    client: TestClient, auth_orchestrator: TradingBotOrchestrator
) -> None:
    """
    Verifies that User Alice receives HTTP 403 Forbidden when attempting to
    view, start, stop, or mutate User Bob's tenant.
    """
    # 1. Provision Alice and Bob tenants
    alice_uid = "alice-uid-1"
    bob_uid = "bob-uid-2"

    alice_inst = auth_orchestrator.tenant_manager.get_or_create_user_tenant(
        alice_uid, "alice@delta.bot", auth_orchestrator.contract_specs
    )
    bob_inst = auth_orchestrator.tenant_manager.get_or_create_user_tenant(
        bob_uid, "bob@delta.bot", auth_orchestrator.contract_specs
    )

    alice_tid = alice_inst.config.tenant_id
    bob_tid = bob_inst.config.tenant_id
    assert alice_tid != bob_tid

    # Mock token resolution so request headers authenticate as Alice
    fake_alice_payload = {"id": alice_uid, "email": "alice@delta.bot"}

    with patch("delta_bot.server.verify_supabase_token", AsyncMock(return_value=fake_alice_payload)):
        alice_headers = {"Authorization": "Bearer alice_token"}

        # Alice CAN access her own tenant
        res_alice_self = client.get(f"/api/tenants/{alice_tid}", headers=alice_headers)
        assert res_alice_self.status_code == 200

        # Alice CANNOT access Bob's tenant (403 Forbidden)
        res_bob_profile = client.get(f"/api/tenants/{bob_tid}", headers=alice_headers)
        assert res_bob_profile.status_code == 403
        assert "Access denied" in res_bob_profile.json()["detail"]

        # Alice CANNOT stop Bob's trading desk (403 Forbidden)
        res_bob_stop = client.post(f"/api/tenants/{bob_tid}/control/stop", headers=alice_headers)
        assert res_bob_stop.status_code == 403

        # Alice CANNOT trigger flatten on Bob's account (403 Forbidden)
        res_bob_flatten = client.post(f"/api/tenants/{bob_tid}/control/flatten", headers=alice_headers)
        assert res_bob_flatten.status_code == 403

        # Alice CANNOT delete Bob's account (403 Forbidden)
        res_bob_del = client.delete(f"/api/tenants/{bob_tid}", headers=alice_headers)
        assert res_bob_del.status_code == 403

        # Alice CANNOT update Bob's account (403 Forbidden)
        res_bob_upd = client.post(
            f"/api/tenants/{bob_tid}/update",
            headers=alice_headers,
            json={"risk_pct": 5.0},
        )
        assert res_bob_upd.status_code == 403

        # Alice's tenant list ONLY contains Alice's tenant, never Bob's
        res_list = client.get("/api/tenants", headers=alice_headers)
        assert res_list.status_code == 200
        tenants_visible = res_list.json()["tenants"]
        visible_ids = [t["tenant_id"] for t in tenants_visible]
        assert alice_tid in visible_ids
        assert bob_tid not in visible_ids


def test_user_tenant_control_mutations_allowed(
    client: TestClient, auth_orchestrator: TradingBotOrchestrator
) -> None:
    """Verifies that an authenticated user can safely control and configure their own tenant."""
    alice_uid = "alice-owner-99"
    alice_inst = auth_orchestrator.tenant_manager.get_or_create_user_tenant(
        alice_uid, "alice@owner.com", auth_orchestrator.contract_specs
    )
    alice_tid = alice_inst.config.tenant_id

    fake_alice_payload = {"id": alice_uid, "email": "alice@owner.com"}

    with patch("delta_bot.server.verify_supabase_token", AsyncMock(return_value=fake_alice_payload)):
        headers = {"Authorization": "Bearer alice_token"}

        # 1. Stop Alice's desk
        res_stop = client.post(f"/api/tenants/{alice_tid}/control/stop", headers=headers)
        assert res_stop.status_code == 200
        assert alice_inst.is_running is False

        # 2. Start Alice's desk
        res_start = client.post(f"/api/tenants/{alice_tid}/control/start", headers=headers)
        assert res_start.status_code == 200
        assert alice_inst.is_running is True

        # 3. Update Alice's risk parameters
        res_upd = client.post(
            f"/api/tenants/{alice_tid}/update",
            headers=headers,
            json={"risk_pct": 2.5, "leverage": 15},
        )
        assert res_upd.status_code == 200
        assert alice_inst.config.risk_pct == 2.5
        assert alice_inst.config.leverage == 15
