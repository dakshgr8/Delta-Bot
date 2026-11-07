"""
tests/test_tenant.py
~~~~~~~~~~~~~~~~~~~~

Unit & Integration Test Suite for Delta-Bot Multi-Tenant Architecture (Phase 9).
Covers:
1. Tenant configuration & credential masking (safe_dict)
2. TenantManager lifecycle: Create, List, Retrieve, Update, and Delete
3. Safety rules: Default tenant deletion rejection & duplicate ID prevention
4. Portfolio & order isolation across concurrent tenants
5. Multi-Tenant REST API endpoints (/api/tenants, /api/tenants/{id})
6. Real-time WebSocket dynamic tenant switching
"""

import pytest
from fastapi.testclient import TestClient

from delta_bot.config import BotConfig
from delta_bot.models import ContractSpec, ExecutionMode, Ticker
from delta_bot.orchestrator import TradingBotOrchestrator
from delta_bot.paper import PaperExchangeSimulator
from delta_bot.server import create_app
from delta_bot.tenant import (
    TenantConfig,
    TenantInstance,
    TenantManager,
    decrypt_secret,
    encrypt_secret,
    generate_deterministic_client_order_id,
    generate_deterministic_tenant_order_id,
)


@pytest.fixture
def multi_tenant_orchestrator(tmp_path) -> TradingBotOrchestrator:
    """Fixture creating an orchestrator with custom storage file."""
    cfg = BotConfig(
        _env_file=None,
        execution_mode="paper",
        target_symbols_str="BTCUSDT,ETHUSDT",
        default_risk_pct=1.0,
        default_leverage=5,
    )
    paper_sim = PaperExchangeSimulator(initial_balance=10000.0)
    orch = TradingBotOrchestrator(config=cfg, mode=ExecutionMode.PAPER, paper_simulator=paper_sim)

    # Point tenant storage to isolated temp dir and disable external sync for unit tests
    orch.tenant_manager.supabase_url = None
    orch.tenant_manager.supabase_secret_key = None
    orch.tenant_manager.storage_path = str(tmp_path / "tenants.json")
    orch.tenant_manager.tenants = {
        "default": orch.tenant_manager.tenants["default"]
    }

    # Seed contract specs
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
    for t in orch.tenant_manager.tenants.values():
        t.register_contract_spec(btc_spec)

    orch.latest_tickers["BTCUSDT"] = Ticker(
        symbol="BTCUSDT",
        product_id=1,
        mark_price=68000.0,
        best_bid=67995.0,
        best_ask=68000.0,
    )
    return orch


@pytest.fixture
def mt_client(multi_tenant_orchestrator: TradingBotOrchestrator):
    """FastAPI TestClient fixture bound to multi-tenant orchestrator."""
    app = create_app(multi_tenant_orchestrator)
    with TestClient(app) as test_client:
        yield test_client


def test_tenant_config_safe_dict() -> None:
    """Verifies that TenantConfig masks sensitive API keys in safe_dict."""
    cfg = TenantConfig(
        tenant_id="alpha_desk",
        name="Alpha Desk",
        delta_api_key="my_secret_api_key_12345",
        delta_api_secret="my_private_secret_999",
    )
    safe = cfg.safe_dict()
    assert safe["tenant_id"] == "alpha_desk"
    assert safe["has_api_key"] is True
    assert "my_secret_api_key_12345" not in str(safe)
    assert "my_private_secret_999" not in str(safe)
    assert safe["delta_api_key_masked"].startswith("my_s")


def test_tenant_manager_lifecycle(multi_tenant_orchestrator: TradingBotOrchestrator) -> None:
    """Verifies tenant creation, retrieval, updates, and deletion."""
    mgr = multi_tenant_orchestrator.tenant_manager

    # 1. Default tenant is automatically present
    assert "default" in mgr.tenants
    default_tenant = mgr.get_tenant("default")
    assert default_tenant is not None
    assert default_tenant.config.name == "Default Account"

    # 2. Create custom tenant
    t_cfg = TenantConfig(
        tenant_id="tenant_alice",
        name="Alice Fund",
        initial_balance=25000.0,
        leverage=10,
        risk_pct=1.5,
    )
    alice = mgr.create_tenant(t_cfg, contract_specs=multi_tenant_orchestrator.contract_specs)
    assert alice.config.tenant_id == "tenant_alice"
    assert "tenant_alice" in mgr.tenants
    assert alice.paper_simulator.wallet.balance == 25000.0

    # 3. Retrieve tenant
    retrieved = mgr.get_tenant("tenant_alice")
    assert retrieved is not None
    assert retrieved.config.name == "Alice Fund"

    # 4. Reject duplicate tenant ID
    with pytest.raises(ValueError, match="already exists"):
        mgr.create_tenant(t_cfg)

    # 5. Update tenant parameters
    updated = mgr.update_tenant("tenant_alice", {"name": "Alice Quantitative", "leverage": 20})
    assert updated.config.name == "Alice Quantitative"
    assert updated.config.leverage == 20

    # 6. Reject deletion of default tenant
    import asyncio
    with pytest.raises(ValueError, match="cannot be deleted"):
        asyncio.run(mgr.delete_tenant("default"))


def test_tenant_portfolio_isolation(multi_tenant_orchestrator: TradingBotOrchestrator) -> None:
    """Verifies that positions and balances in Tenant A do not bleed into Tenant B."""
    mgr = multi_tenant_orchestrator.tenant_manager

    # Create Alice (20,000 USD) and Bob (5,000 USD)
    alice = mgr.create_tenant(
        TenantConfig(tenant_id="alice", name="Alice", initial_balance=20000.0),
        contract_specs=multi_tenant_orchestrator.contract_specs,
    )
    bob = mgr.create_tenant(
        TenantConfig(tenant_id="bob", name="Bob", initial_balance=5000.0),
        contract_specs=multi_tenant_orchestrator.contract_specs,
    )

    # Submit stop order exclusively for Alice
    order_alice = alice.paper_simulator.submit_stop_order(
        symbol="BTCUSDT",
        product_id=1,
        side="buy",
        size=3,
        stop_price=70000.0,
        bracket_stop_loss_price=69000.0,
        bracket_take_profit_price=72000.0,
        leverage=5,
    )

    # Alice has 1 resting order, Bob and Default have 0
    assert len(alice.paper_simulator._pending_orders) == 1
    assert len(bob.paper_simulator._pending_orders) == 0
    default_tenant = mgr.get_tenant("default")
    assert len(default_tenant.paper_simulator._pending_orders) == 0

    # Alice balance and Bob balance remain independent
    assert alice.paper_simulator.wallet.balance == 20000.0
    assert bob.paper_simulator.wallet.balance == 5000.0


def test_server_multi_tenant_rest_endpoints(mt_client: TestClient, multi_tenant_orchestrator: TradingBotOrchestrator) -> None:
    """Tests /api/tenants CRUD REST endpoints."""
    # 1. GET /api/tenants -> contains default tenant
    res_list = mt_client.get("/api/tenants")
    assert res_list.status_code == 200
    tenants = res_list.json()["tenants"]
    assert any(t["tenant_id"] == "default" for t in tenants)

    # 2. POST /api/tenants -> create new tenant
    res_create = mt_client.post(
        "/api/tenants",
        json={
            "tenant_id": "quant_fund",
            "name": "Quant Fund LP",
            "initial_balance": 50000.0,
            "mode": "paper",
            "risk_pct": 2.0,
            "leverage": 10,
        },
    )
    assert res_create.status_code == 201
    assert res_create.json()["status"] == "success"
    assert res_create.json()["tenant"]["tenant_id"] == "quant_fund"

    # 3. GET /api/tenants/quant_fund -> returns full state packet
    res_get = mt_client.get("/api/tenants/quant_fund")
    assert res_get.status_code == 200
    data = res_get.json()
    assert data["collateral"]["balance"] == 50000.0
    assert data["current_tenant"]["tenant_id"] == "quant_fund"

    # 4. POST /api/tenants/quant_fund/control/stop & /start
    res_stop = mt_client.post("/api/tenants/quant_fund/control/stop")
    assert res_stop.status_code == 200
    assert res_stop.json()["status"] == "success"
    assert multi_tenant_orchestrator.tenant_manager.get_tenant("quant_fund").is_running is False

    res_start = mt_client.post("/api/tenants/quant_fund/control/start")
    assert res_start.status_code == 200
    assert multi_tenant_orchestrator.tenant_manager.get_tenant("quant_fund").is_running is True

    # 5. GET /api/positions with X-Tenant-ID header
    res_pos = mt_client.get("/api/positions", headers={"X-Tenant-ID": "quant_fund"})
    assert res_pos.status_code == 200
    assert res_pos.json()["count"] == 0

    # 6. DELETE /api/tenants/quant_fund
    res_del = mt_client.delete("/api/tenants/quant_fund")
    assert res_del.status_code == 200
    assert "deleted" in res_del.json()["message"]
    assert "quant_fund" not in multi_tenant_orchestrator.tenant_manager.tenants


def test_websocket_tenant_switching(mt_client: TestClient, multi_tenant_orchestrator: TradingBotOrchestrator) -> None:
    """Tests switching tenant stream over real-time WebSocket connection."""
    # Create tenant Charlie
    multi_tenant_orchestrator.tenant_manager.create_tenant(
        TenantConfig(tenant_id="charlie", name="Charlie Account", initial_balance=12345.0),
        contract_specs=multi_tenant_orchestrator.contract_specs,
    )

    with mt_client.websocket_connect("/ws/live?tenant_id=default") as ws:
        # Initial packet is default tenant
        packet_def = ws.receive_json()
        assert packet_def["type"] == "state_update"
        assert packet_def["current_tenant"]["tenant_id"] == "default"

        # Switch to Charlie
        ws.send_json({"action": "switch_tenant", "tenant_id": "charlie"})
        packet_charlie = ws.receive_json()
        assert packet_charlie["type"] == "state_update"
        assert packet_charlie["current_tenant"]["tenant_id"] == "charlie"
        assert packet_charlie["collateral"]["balance"] == 12345.0


def test_encryption_and_decryption_credentials() -> None:
    """Verifies AES/Fernet encryption for API credentials at rest."""
    raw_key = "delta_api_key_test_12345"
    raw_secret = "delta_api_secret_test_67890"

    # Encrypt
    enc_k = encrypt_secret(raw_key)
    enc_s = encrypt_secret(raw_secret)
    assert enc_k.startswith("enc:")
    assert enc_s.startswith("enc:")
    assert enc_k != raw_key
    assert enc_s != raw_secret

    # Decrypt
    dec_k = decrypt_secret(enc_k)
    dec_s = decrypt_secret(enc_s)
    assert dec_k == raw_key
    assert dec_s == raw_secret

    # Idempotent encrypt
    assert encrypt_secret(enc_k) == enc_k

    # Backward compatibility with unencrypted legacy tokens
    assert decrypt_secret("legacy_plain_key") == "legacy_plain_key"
    assert decrypt_secret("") == ""
    assert encrypt_secret("") == ""


def test_tenant_storage_credential_encryption(tmp_path) -> None:
    """Verifies that API credentials written to disk are encrypted, and decrypted upon reload."""
    import json
    storage_file = str(tmp_path / "tenants_enc_test.json")
    mgr = TenantManager(storage_path=storage_file)

    t_cfg = TenantConfig(
        tenant_id="secure_trader",
        name="Secure Trader",
        delta_api_key="my_super_secret_api_key_xyz",
        delta_api_secret="my_super_secret_api_secret_abc",
    )
    mgr.create_tenant(t_cfg)

    # Inspect file directly on disk to verify raw keys are NOT exposed
    with open(storage_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "secure_trader" in data
    stored_k = data["secure_trader"]["delta_api_key"]
    stored_s = data["secure_trader"]["delta_api_secret"]
    assert stored_k.startswith("enc:")
    assert stored_s.startswith("enc:")
    assert "my_super_secret_api_key_xyz" not in stored_k
    assert "my_super_secret_api_secret_abc" not in stored_s

    # Bootstrap fresh TenantManager and load from disk
    new_mgr = TenantManager(storage_path=storage_file)
    new_mgr.load_from_disk(contract_specs={})
    reloaded = new_mgr.get_tenant("secure_trader")
    assert reloaded is not None
    assert reloaded.config.delta_api_key == "my_super_secret_api_key_xyz"
    assert reloaded.config.delta_api_secret == "my_super_secret_api_secret_abc"


@pytest.mark.asyncio
async def test_live_metrics_synchronization(multi_tenant_orchestrator: TradingBotOrchestrator) -> None:
    """Verifies that live tenants pull real wallet balances and positions from Delta Exchange."""
    from unittest.mock import AsyncMock
    from delta_bot.models import PositionInfo, WalletBalance
    from delta_bot.server import get_full_state_packet

    mgr = multi_tenant_orchestrator.tenant_manager
    live_cfg = TenantConfig(
        tenant_id="live_desk",
        name="Live Desk",
        mode="live",
        delta_api_key="key_123",
        delta_api_secret="secret_456",
    )
    tenant = mgr.create_tenant(live_cfg, contract_specs=multi_tenant_orchestrator.contract_specs)
    assert tenant.mode == ExecutionMode.LIVE

    # Mock DeltaRestClient responses
    mock_balances = [
        WalletBalance(
            asset_id=1,
            asset_symbol="USDT",
            balance=5432.10,
            available_balance=4800.00,
            order_margin=100.0,
            position_margin=532.10,
            blocked_margin=632.10,
            unrealized_pnl=150.75,
        )
    ]
    mock_positions = [
        PositionInfo(
            product_id=1,
            symbol="BTCUSDT",
            size=2,
            entry_price=67500.0,
            mark_price=68000.0,
            liquidation_price=55000.0,
            margin=532.10,
            unrealized_pnl=150.75,
        )
    ]

    tenant.rest_client.get_wallet_balances = AsyncMock(return_value=mock_balances)
    tenant.rest_client.get_open_positions = AsyncMock(return_value=mock_positions)

    # Sync live metrics
    await tenant.sync_live_metrics()

    # Verify tenant cached metrics
    assert tenant.live_balance == 5432.10
    assert tenant.live_available_margin == 4800.00
    assert tenant.live_unrealized_pnl == 150.75
    assert tenant.live_equity == 5582.85
    assert len(tenant.live_positions) == 1
    assert tenant.live_positions[0]["symbol"] == "BTCUSDT"
    assert tenant.live_positions[0]["size"] == 2
    assert tenant.live_positions[0]["unrealized_pnl"] == 150.75

    # Verify tenant summary telemetry
    summary = tenant.get_summary()
    assert summary["balance"] == 5432.10
    assert summary["equity"] == 5582.85
    assert summary["unrealized_pnl"] == 150.75
    assert summary["available_margin"] == 4800.00
    assert summary["open_positions_count"] == 1

    # Verify full state packet broadcast
    packet = get_full_state_packet(multi_tenant_orchestrator, tenant_id="live_desk")
    col = packet["collateral"]
    assert col["balance"] == 5432.10
    assert col["equity"] == 5582.85
    assert col["unrealized_pnl"] == 150.75
    assert col["available_margin"] == 4800.00
    assert len(packet["positions"]) == 1
    assert packet["positions"][0]["symbol"] == "BTCUSDT"


def test_deterministic_tenant_order_id_generation() -> None:
    """Verifies Feature A: Deterministic UUIDv5 Multi-Tenant Order ID Fan-Out."""
    base_signal_id = "SIG_BTCUSDT_1712750400"
    tenant_alice = "tenant_alice_123"
    tenant_bob = "tenant_bob_456"

    # Determinism: same input produces identical output
    alice_order_id1 = generate_deterministic_tenant_order_id(base_signal_id, tenant_alice)
    alice_order_id2 = generate_deterministic_tenant_order_id(base_signal_id, tenant_alice)
    assert alice_order_id1 == alice_order_id2

    # Isolation: different tenants produce different order IDs
    bob_order_id = generate_deterministic_tenant_order_id(base_signal_id, tenant_bob)
    assert alice_order_id1 != bob_order_id

    # Client order ID constraints (<= 32 characters for Delta Exchange)
    alice_cid = generate_deterministic_client_order_id(base_signal_id, tenant_alice)
    bob_cid = generate_deterministic_client_order_id(base_signal_id, tenant_bob)
    assert len(alice_cid) <= 32
    assert len(bob_cid) <= 32
    assert alice_cid.startswith("D_")
    assert bob_cid.startswith("D_")
    assert alice_cid != bob_cid


@pytest.mark.asyncio
async def test_multi_tenant_candle_close_fanout(multi_tenant_orchestrator: TradingBotOrchestrator) -> None:
    """Verifies that candle close breakouts fan out with deterministic per-tenant order IDs."""
    from delta_bot.strategy import OrderIntent, SignalType

    mgr = multi_tenant_orchestrator.tenant_manager
    alice = mgr.create_tenant(
        TenantConfig(tenant_id="fanout_alice", name="Alice", initial_balance=20000.0),
        contract_specs=multi_tenant_orchestrator.contract_specs,
    )
    bob = mgr.create_tenant(
        TenantConfig(tenant_id="fanout_bob", name="Bob", initial_balance=10000.0),
        contract_specs=multi_tenant_orchestrator.contract_specs,
    )

    btc_spec = multi_tenant_orchestrator.contract_specs["BTCUSDT"]

    master_intent = OrderIntent(
        symbol="BTCUSDT",
        signal_type=SignalType.BULLISH_BREAKOUT,
        trigger_price=68000.0,
        initial_stop_loss=67500.0,
        current_stop_loss=67500.0,
        take_profit=69000.0,
        contract_lots=2,
        target_asset_qty=0.002,
        capital_at_risk_usd=100.0,
        signal_candle_high=67900.0,
        signal_candle_low=67400.0,
        signal_candle_close=67850.0,
        signal_bar_time=2000.0,
        created_at_epoch_s=2000.0,
    )

    base_sig = "SIG_BTCUSDT_2000"
    executed = await mgr.fanout_order_intent(base_sig, master_intent, btc_spec)

    assert "fanout_alice" in executed
    assert "fanout_bob" in executed

    alice_orders = list(alice.paper_simulator._pending_orders.values())
    bob_orders = list(bob.paper_simulator._pending_orders.values())

    assert len(alice_orders) == 1
    assert len(bob_orders) == 1

    a_ord = alice_orders[0]
    b_ord = bob_orders[0]

    # Both orders derive from the same base signal
    assert a_ord.client_order_id in alice.order_to_base_id
    assert b_ord.client_order_id in bob.order_to_base_id
    assert alice.order_to_base_id[a_ord.client_order_id] == base_sig
    assert bob.order_to_base_id[b_ord.client_order_id] == base_sig

    # Zero collision: Alice and Bob have unique client_order_ids
    assert a_ord.client_order_id != b_ord.client_order_id
    assert a_ord.client_order_id.startswith("D_")
    assert b_ord.client_order_id.startswith("D_")
    assert len(a_ord.client_order_id) <= 32
    assert len(b_ord.client_order_id) <= 32


def test_wal_failover_buffering_and_replay_recovery(tmp_path: Path) -> None:
    """Verifies that Supabase network outages trigger local WAL buffering and recovery replays seamlessly."""
    from unittest.mock import MagicMock, patch
    import json
    from delta_bot.tenant import TenantManager, TenantConfig

    storage_file = tmp_path / "tenants.json"
    wal_file = tmp_path / "wal_buffer.json"

    mgr = TenantManager(
        storage_path=str(storage_file),
        wal_path=str(wal_file),
        supabase_url="https://mock.supabase.co",
        supabase_secret_key="mock_key",
    )

    t_cfg = TenantConfig(tenant_id="wal_tenant_1", name="WAL Tenant", mode="paper", initial_balance=5000.0)
    instance = mgr.create_tenant(t_cfg)

    # 1. Simulate Supabase HTTP 503 Outage during sync
    mock_err_resp = MagicMock()
    mock_err_resp.status_code = 503
    mock_err_resp.text = "Service Unavailable"

    with patch("httpx.Client.post", return_value=mock_err_resp):
        success = mgr.sync_tenant_to_supabase(instance)
        assert success is False

    # 2. Verify transaction safely written to WAL buffer
    assert wal_file.exists()
    with open(wal_file, "r", encoding="utf-8") as f:
        wal_data = json.load(f)
    assert len(wal_data) == 1
    assert wal_data[0]["action"] == "upsert_tenant"
    assert wal_data[0]["payload"]["tenant_id"] == "wal_tenant_1"

    # 3. Simulate recovery: flush WAL buffer with HTTP 201 Created
    mock_ok_resp = MagicMock()
    mock_ok_resp.status_code = 201

    with patch("httpx.Client.post", return_value=mock_ok_resp):
        replayed = mgr.flush_wal_buffer()
        assert replayed == 1

    # 4. Verify WAL file is flushed clean
    if wal_file.exists():
        with open(wal_file, "r", encoding="utf-8") as f:
            clean_data = json.load(f)
        assert len(clean_data) == 0


def test_wal_deduplication_on_23505_cloud_conflict(tmp_path: Path) -> None:
    """Verifies that duplicate key collisions (PostgreSQL code 23505) during WAL replay are gracefully reconciled."""
    from unittest.mock import MagicMock, patch
    import json
    from delta_bot.tenant import TenantManager, TenantConfig

    wal_file = tmp_path / "wal_buffer.json"
    mgr = TenantManager(
        storage_path=str(tmp_path / "tenants.json"),
        wal_path=str(wal_file),
        supabase_url="https://mock.supabase.co",
        supabase_secret_key="mock_key",
    )

    # Pre-populate WAL with a record
    mgr._write_to_wal("upsert_tenant", {"tenant_id": "existing_tenant", "name": "Existing"})
    assert wal_file.exists()

    # Simulate Postgres 23505 duplicate key error
    mock_dup_resp = MagicMock()
    mock_dup_resp.status_code = 409
    mock_dup_resp.text = '{"code": "23505", "message": "duplicate key value violates unique constraint"}'

    with patch("httpx.Client.post", return_value=mock_dup_resp):
        replayed = mgr.flush_wal_buffer()
        assert replayed == 1  # Deduplicated and treated as success!

    # WAL buffer should now be clean
    if wal_file.exists():
        with open(wal_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 0


def test_tenant_leverage_100x_and_risk_updates(tmp_path: Path) -> None:
    """Verifies that leverage up to 100x, risk per trade, and daily drawdown propagate dynamically."""
    storage_file = tmp_path / "tenants.json"
    mgr = TenantManager(storage_path=str(storage_file))

    t_cfg = TenantConfig(
        tenant_id="desk_100x",
        name="100x Desk",
        mode="paper",
        leverage=5,
        risk_pct=1.0,
        max_drawdown_pct=4.0,
        symbols=["BTCUSDT", "ETHUSDT", "XAUTUSD"],
    )
    instance = mgr.create_tenant(t_cfg)

    # 1. Update leverage to 100x
    updated = mgr.update_tenant("desk_100x", {"leverage": 100})
    assert updated.config.leverage == 100
    assert updated.risk_manager.config.default_leverage == 100
    assert updated.risk_manager.config.max_leverage == 100

    # 2. Update risk_pct to 2.5%
    updated = mgr.update_tenant("desk_100x", {"risk_pct": 2.5})
    assert updated.config.risk_pct == 2.5
    assert updated.risk_manager.config.max_account_risk_pct == 2.5
    for strat in updated.strategies.values():
        assert strat.config.risk_per_trade_pct == 2.5

    # 3. Update max_drawdown_pct to 7.5%
    updated = mgr.update_tenant("desk_100x", {"max_drawdown_pct": 7.5})
    assert updated.config.max_drawdown_pct == 7.5
    assert updated.risk_manager.config.daily_max_drawdown_pct == 7.5


def test_perpetual_futures_strict_enforcement(tmp_path: Path) -> None:
    """Verifies that non-perpetual contracts (spot/options) are strictly rejected from tenant specs."""
    storage_file = tmp_path / "tenants.json"
    mgr = TenantManager(storage_path=str(storage_file))

    instance = mgr.create_tenant(TenantConfig(tenant_id="perp_desk", name="Perpetual Desk", mode="paper"))

    # Valid perpetual contract
    perp_spec = ContractSpec(
        product_id=131253,
        symbol="XAUTUSD",
        contract_type="perpetual_futures",
        contract_value=0.001,
        tick_size=0.01,
        maker_commission_rate=0.0001,
        taker_commission_rate=0.0001,
    )
    instance.register_contract_spec(perp_spec)
    assert "XAUTUSD" in instance.contract_specs

    # Invalid spot contract
    spot_spec = ContractSpec(
        product_id=999,
        symbol="BTCUSD_SPOT",
        contract_type="spot",
        contract_value=1.0,
        tick_size=0.01,
        maker_commission_rate=0.0001,
        taker_commission_rate=0.0001,
    )
    instance.register_contract_spec(spot_spec)
    assert "BTCUSD_SPOT" not in instance.contract_specs

    # Invalid options contract
    option_spec = ContractSpec(
        product_id=888,
        symbol="C-BTC-70000",
        contract_type="call_options",
        contract_value=1.0,
        tick_size=0.5,
        maker_commission_rate=0.0001,
        taker_commission_rate=0.0001,
    )
    instance.register_contract_spec(option_spec)
    assert "C-BTC-70000" not in instance.contract_specs


def test_xautusd_stop_loss_and_risk_sizing(tmp_path: Path) -> None:
    """Verifies Stop Loss per-unit risk distance and lot sizing for XAUTUSD under 100x leverage."""
    storage_file = tmp_path / "tenants.json"
    mgr = TenantManager(storage_path=str(storage_file))

    instance = mgr.create_tenant(TenantConfig(
        tenant_id="gold_desk",
        name="Gold Desk",
        mode="paper",
        leverage=100,
        risk_pct=1.0,
        initial_balance=10000.0,
    ))

    xaut_spec = ContractSpec(
        product_id=131253,
        symbol="XAUTUSD",
        contract_type="perpetual_futures",
        contract_value=0.001,
        tick_size=0.01,
        quoting_symbol="USD",
        maker_commission_rate=0.0001,
        taker_commission_rate=0.0001,
    )
    instance.register_contract_spec(xaut_spec)

    # Trigger = $4000.00, SL = $3960.00 ($40 per oz risk)
    # Per-unit risk = $40.00
    # Risk budget (1% of $10,000) = $100.00
    # Target asset qty = $100 / $40 = 2.5 oz
    # Lots = 2.5 / 0.001 = 2500 contracts
    margin_res = instance.risk_manager.size_position(
        spec=xaut_spec,
        trigger_price=4000.00,
        stop_loss_price=3960.00,
        account_equity=10000.0,
        available_margin=9500.0,
        risk_pct_override=1.0,
    )

    assert margin_res.is_margin_sufficient is True
    assert margin_res.final_lots == 2500
    # Required initial margin at 100x: (2500 * 0.001 * 4000) / 100 = $100.00 USD
    assert margin_res.required_margin == pytest.approx(100.0, abs=0.1)




