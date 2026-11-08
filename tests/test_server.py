"""
tests/test_server.py
~~~~~~~~~~~~~~~~~~~~

Unit & Integration Test Suite for Delta-Bot Phase 7 Web Gateway & Control API.
Covers:
1. Dashboard HTML serving (GET / and GET /dashboard)
2. High-level status & collateral telemetry (GET /api/status)
3. Active positions query (GET /api/positions)
4. Resting breakout orders query (GET /api/orders)
5. Technical indicators query (GET /api/indicators/{symbol})
6. System logs ringbuffer query (GET /api/logs)
7. Trading loop start/stop control (POST /api/control/start and /stop)
8. Emergency flatten kill-switch (POST /api/control/flatten)
9. Runtime configuration hot-reloading (POST /api/config)
10. Configuration validation constraints (out-of-bound rejections)
11. Real-time WebSocket state streaming & ping/pong (WebSocket /ws/live)
"""

import logging
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.websockets import WebSocketDisconnect

from delta_bot.config import BotConfig
from delta_bot.models import ContractSpec, ExecutionMode, Ticker
from delta_bot.orchestrator import TradingBotOrchestrator
from delta_bot.paper import PaperExchangeSimulator, VirtualPosition
from delta_bot.server import create_app


@pytest.fixture
def test_orchestrator() -> TradingBotOrchestrator:
    """Creates a configured Paper-mode TradingBotOrchestrator test fixture."""
    cfg = BotConfig(
        execution_mode="paper",
        target_symbols_str="BTCUSDT,ETHUSDT",
        default_risk_pct=1.0,
        default_leverage=5,
    )
    paper_sim = PaperExchangeSimulator(initial_balance=10000.0)
    orch = TradingBotOrchestrator(config=cfg, mode=ExecutionMode.PAPER, paper_simulator=paper_sim)

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

    # Seed latest ticker
    orch.latest_tickers["BTCUSDT"] = Ticker(
        symbol="BTCUSDT",
        product_id=1,
        mark_price=68000.0,
        best_bid=67995.0,
        best_ask=68000.0,
    )
    return orch


@pytest.fixture
def client(test_orchestrator: TradingBotOrchestrator):
    """Creates a FastAPI TestClient bound to the test orchestrator within lifespan context."""
    app = create_app(test_orchestrator)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_orchestrator() -> TradingBotOrchestrator:
    """Creates a configured Paper-mode TradingBotOrchestrator test fixture with DASHBOARD_AUTH_TOKEN."""
    cfg = BotConfig(
        execution_mode="paper",
        target_symbols_str="BTCUSDT,ETHUSDT",
        default_risk_pct=1.0,
        default_leverage=5,
        dashboard_auth_token=SecretStr("supersecrettoken123"),
    )
    paper_sim = PaperExchangeSimulator(initial_balance=10000.0)
    orch = TradingBotOrchestrator(config=cfg, mode=ExecutionMode.PAPER, paper_simulator=paper_sim)

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
        mark_price=68000.0,
        best_bid=67995.0,
        best_ask=68000.0,
    )
    return orch


@pytest.fixture
def auth_client(auth_orchestrator: TradingBotOrchestrator):
    """Creates a FastAPI TestClient bound to the authenticated orchestrator within lifespan context."""
    app = create_app(auth_orchestrator)
    with TestClient(app) as test_client:
        yield test_client


# -----------------------------------------------------------------------------
# 1. Dashboard HTML Endpoints
# -----------------------------------------------------------------------------

def test_server_dashboard_html_endpoints(client: TestClient) -> None:
    """Verifies that GET / and GET /dashboard serve the single-page dashboard HTML."""
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "text/html" in res_root.headers["content-type"]
    assert "DELTA-BOT" in res_root.text
    assert "Quantitative Terminal" in res_root.text

    res_dash = client.get("/dashboard")
    assert res_dash.status_code == 200
    assert "DELTA-BOT" in res_dash.text
    # Win rate and Max Drawdown tabs/cards must be removed from metrics summary
    assert "id=\"card-winrate\"" not in res_dash.text
    assert "id=\"card-maxdd\"" not in res_dash.text
    assert "id=\"card-equity\"" in res_dash.text
    assert "id=\"card-upnl\"" in res_dash.text


# -----------------------------------------------------------------------------
# 2. Status & Telemetry REST Endpoints
# -----------------------------------------------------------------------------

def test_server_api_status(client: TestClient, test_orchestrator: TradingBotOrchestrator) -> None:
    """Verifies that GET /api/status returns daemon state, operational mode, and collateral."""
    test_orchestrator.is_running = True
    res = client.get("/api/status")

    assert res.status_code == 200
    data = res.json()

    assert "status" in data
    assert "collateral" in data

    status_data = data["status"]
    assert status_data["is_running"] is True
    assert status_data["mode"] == "PAPER"
    assert "memory_mb" in status_data
    assert status_data["circuit_breaker_tripped"] is False
    assert status_data["target_symbols"] == ["BTCUSDT", "ETHUSDT"]

    collateral = data["collateral"]
    assert collateral["equity"] == 10000.0
    assert collateral["balance"] == 10000.0
    assert collateral["unrealized_pnl"] == 0.0


# -----------------------------------------------------------------------------
# 3. Positions & Orders Queries
# -----------------------------------------------------------------------------

def test_server_api_positions(client: TestClient, test_orchestrator: TradingBotOrchestrator) -> None:
    """Verifies that GET /api/positions returns active positions."""
    # Initially 0 positions
    res_empty = client.get("/api/positions")
    assert res_empty.status_code == 200
    assert res_empty.json()["count"] == 0

    # Seed 1 position in paper sim
    test_orchestrator.paper_simulator._open_positions[1] = VirtualPosition(
        product_id=1,
        symbol="BTCUSDT",
        side="buy",
        size=15,
        entry_price=67500.0,
        mark_price=68000.0,
        unrealized_pnl=7.50,
        stop_loss_price=66500.0,
        take_profit_price=70000.0,
        child_stop_loss_order_id=999901,
        allocated_margin=202.50,
    )

    res_pos = client.get("/api/positions")
    assert res_pos.status_code == 200
    pos_data = res_pos.json()
    assert pos_data["count"] == 1

    first_pos = pos_data["positions"][0]
    assert first_pos["symbol"] == "BTCUSDT"
    assert first_pos["side"] == "BUY"
    assert first_pos["size"] == 15
    assert first_pos["entry_price"] == 67500.0
    assert first_pos["unrealized_pnl"] == 7.50
    assert first_pos["child_stop_loss_order_id"] == 999901


def test_server_api_orders(client: TestClient, test_orchestrator: TradingBotOrchestrator) -> None:
    """Verifies that GET /api/orders returns resting pending stop breakout orders."""
    res_empty = client.get("/api/orders")
    assert res_empty.status_code == 200
    assert res_empty.json()["count"] == 0

    # Seed 1 resting stop order
    v_order = test_orchestrator.paper_simulator.submit_stop_order(
        product_id=1,
        symbol="BTCUSDT",
        side="buy",
        size=10,
        stop_price=68500.0,
        bracket_stop_loss_price=67500.0,
        bracket_take_profit_price=70500.0,
    )

    res_orders = client.get("/api/orders")
    assert res_orders.status_code == 200
    orders_data = res_orders.json()
    assert orders_data["count"] == 1

    first_order = orders_data["orders"][0]
    assert first_order["order_id"] == v_order.id
    assert first_order["symbol"] == "BTCUSDT"
    assert first_order["stop_price"] == 68500.0
    assert first_order["status"] == "OPEN"


# -----------------------------------------------------------------------------
# 4. Indicators & Technical Metrics
# -----------------------------------------------------------------------------

def test_server_api_indicators(client: TestClient) -> None:
    """Verifies that GET /api/indicators/{symbol} returns technical metrics or 404 for unknown asset."""
    # Valid target asset
    res_btc = client.get("/api/indicators/BTCUSDT")
    assert res_btc.status_code == 200
    btc_metrics = res_btc.json()
    assert btc_metrics["symbol"] == "BTCUSDT"
    assert "session_vwap" in btc_metrics
    assert "trend_regime" in btc_metrics

    # Invalid / untracked asset returns 404
    res_invalid = client.get("/api/indicators/DOGEUSDT")
    assert res_invalid.status_code == 404
    assert "is not being tracked" in res_invalid.json()["detail"]


# -----------------------------------------------------------------------------
# 5. System Logs RingBuffer Query
# -----------------------------------------------------------------------------

def test_server_api_logs(client: TestClient) -> None:
    """Verifies that GET /api/logs retrieves formatted log records."""
    test_logger = logging.getLogger("delta_bot.test_runner")
    test_logger.setLevel(logging.INFO)
    test_logger.info("Unique verification log entry token 987654")

    res = client.get("/api/logs?limit=50")
    assert res.status_code == 200
    data = res.json()
    assert "logs" in data
    assert any("Unique verification log entry token 987654" in log["message"] for log in data["logs"])


# -----------------------------------------------------------------------------
# 6. Daemon Controls: Start, Stop & Emergency Flatten
# -----------------------------------------------------------------------------

def test_server_control_start_and_stop(client: TestClient, test_orchestrator: TradingBotOrchestrator) -> None:
    """Verifies that POST /api/control/start and /stop toggle trading loop execution."""
    test_orchestrator.is_running = True

    # Stop trading loop
    res_stop = client.post("/api/control/stop")
    assert res_stop.status_code == 200
    assert res_stop.json()["status"] == "success"
    assert test_orchestrator.is_running is False

    # Start trading loop
    res_start = client.post("/api/control/start")
    assert res_start.status_code == 200
    assert res_start.json()["status"] == "success"
    assert test_orchestrator.is_running is True


def test_server_control_emergency_flatten(client: TestClient, test_orchestrator: TradingBotOrchestrator) -> None:
    """Verifies that POST /api/control/flatten closes active positions and releases margin."""
    # Seed 1 open position
    test_orchestrator.paper_simulator._open_positions[1] = VirtualPosition(
        product_id=1,
        symbol="BTCUSDT",
        side="buy",
        size=10,
        entry_price=67000.0,
        mark_price=67500.0,
        unrealized_pnl=5.00,
        allocated_margin=135.00,
    )
    test_orchestrator.paper_simulator.wallet.blocked_margin = 135.00

    assert len(test_orchestrator.paper_simulator._open_positions) == 1

    # Trigger emergency flatten
    res_flatten = client.post("/api/control/flatten")
    assert res_flatten.status_code == 200
    assert res_flatten.json()["status"] == "success"

    # Positions should now be completely cleared
    assert len(test_orchestrator.paper_simulator._open_positions) == 0
    assert test_orchestrator.paper_simulator.wallet.blocked_margin == 0.0


# -----------------------------------------------------------------------------
# 7. Hot-Reload Configuration Updates
# -----------------------------------------------------------------------------

def test_server_config_hot_reload_success(client: TestClient, test_orchestrator: TradingBotOrchestrator) -> None:
    """Verifies that POST /api/config hot-reloads risk and strategy parameters."""
    payload = {
        "risk_pct": 2.5,
        "leverage": 10,
        "max_pending_bars": 5,
        "trailing_sl_enabled": True,
        "trailing_stop_atr_mult": 2.0,
    }
    res = client.post("/api/config", json=payload)
    assert res.status_code == 200
    assert res.json()["status"] == "success"

    # Verify orchestrator config updated
    assert test_orchestrator.config.default_risk_pct == 2.5
    assert test_orchestrator.config.default_leverage == 10

    # Verify strategy configs updated
    for strat in test_orchestrator.strategies.values():
        assert strat.config.risk_per_trade_pct == 2.5
        assert strat.config.max_pending_bars == 5
        assert strat.config.trailing_stop is True
        assert strat.config.trailing_stop_atr_mult == 2.0


def test_server_config_hot_reload_validation_error(client: TestClient) -> None:
    """Verifies that POST /api/config rejects out-of-range parameters with 422 Unprocessable Entity."""
    # risk_pct > 10.0 is invalid
    res = client.post("/api/config", json={"risk_pct": 50.0})
    assert res.status_code == 422

    # leverage < 1 is invalid
    res_lev = client.post("/api/config", json={"leverage": 0})
    assert res_lev.status_code == 422


# -----------------------------------------------------------------------------
# 8. Real-Time WebSocket Streaming & Ping/Pong
# -----------------------------------------------------------------------------

def test_server_websocket_live_stream(client: TestClient) -> None:
    """Verifies that WebSocket /ws/live streams initial state, responds to pings, handles pause/resume, and disconnects cleanly."""
    with client.websocket_connect("/ws/live") as ws:
        # Receive immediate initial state update
        data = ws.receive_json()
        assert data["type"] == "state_update"
        assert "status" in data
        assert "collateral" in data
        assert "tickers" in data
        assert "positions" in data
        assert "orders" in data

        # Send ping frame and receive pong
        ws.send_text("ping")
        pong = ws.receive_json()
        assert pong["type"] == "pong"
        assert "timestamp" in pong

        # Test pause and resume actions
        import json
        ws.send_text(json.dumps({"action": "pause"}))
        ws.send_text(json.dumps({"action": "resume"}))
        resume_pkt = ws.receive_json()
        assert resume_pkt["type"] == "state_update"


# -----------------------------------------------------------------------------
# 9. Position Close By Product ID (Phase 8)
# -----------------------------------------------------------------------------

def test_server_close_position_by_product_id(
    client: TestClient, test_orchestrator: TradingBotOrchestrator
) -> None:
    """Verifies that POST /api/positions/{product_id}/close terminates target position."""
    paper_sim = test_orchestrator.paper_simulator
    assert paper_sim is not None

    # Seed an open position for product_id=1
    paper_sim._open_positions[1] = VirtualPosition(
        product_id=1,
        symbol="BTCUSDT",
        side="buy",
        size=10,
        entry_price=68000.0,
        mark_price=68100.0,
        unrealized_pnl=1.0,
        allocated_margin=13.6,
    )

    res = client.post("/api/positions/1/close")
    assert res.status_code == 200
    assert res.json()["status"] == "success"
    assert "closed" in res.json()["message"].lower()
    assert 1 not in paper_sim._open_positions

    # Closing again should return 404
    res_404 = client.post("/api/positions/1/close")
    assert res_404.status_code == 404


# -----------------------------------------------------------------------------
# 10. Dashboard Authentication Guard (Phase 8)
# -----------------------------------------------------------------------------

def test_server_auth_token_guard_rejects_unauthorized(auth_client: TestClient) -> None:
    """Verifies that POST/PUT control endpoints reject requests without valid credentials with 401."""
    # 1. Read-only endpoints remain accessible
    res_status = auth_client.get("/api/status")
    assert res_status.status_code == 200

    res_positions = auth_client.get("/api/positions")
    assert res_positions.status_code == 200

    # 2. Control endpoints without token return 401 Unauthorized
    res_start = auth_client.post("/api/control/start")
    assert res_start.status_code == 401
    assert "unauthorized" in res_start.json()["detail"].lower()

    res_stop = auth_client.post("/api/control/stop")
    assert res_stop.status_code == 401

    res_flatten = auth_client.post("/api/control/flatten")
    assert res_flatten.status_code == 401

    res_close = auth_client.post("/api/positions/1/close")
    assert res_close.status_code == 401

    res_config = auth_client.post("/api/config", json={"risk_pct": 2.0})
    assert res_config.status_code == 401

    # 3. Wrong token via X-Dashboard-Token returns 401
    res_wrong_header = auth_client.post(
        "/api/control/start", headers={"X-Dashboard-Token": "invalid_token_xyz"}
    )
    assert res_wrong_header.status_code == 401

    # 4. Wrong token via Authorization: Bearer returns 401
    res_wrong_bearer = auth_client.post(
        "/api/control/start", headers={"Authorization": "Bearer invalid_bearer_token"}
    )
    assert res_wrong_bearer.status_code == 401


def test_server_auth_token_guard_accepts_authorized_tokens(auth_client: TestClient) -> None:
    """Verifies that requests with matching token via header or Bearer auth succeed with 200."""
    valid_token = "supersecrettoken123"

    # 1. Authorization using X-Dashboard-Token header
    res_start = auth_client.post(
        "/api/control/start", headers={"X-Dashboard-Token": valid_token}
    )
    assert res_start.status_code == 200
    assert res_start.json()["status"] == "success"

    # 2. Authorization using Authorization: Bearer <TOKEN>
    res_stop = auth_client.post(
        "/api/control/stop", headers={"Authorization": f"Bearer {valid_token}"}
    )
    assert res_stop.status_code == 200
    assert res_stop.json()["status"] == "success"

    # 3. Config hot-reload using Bearer token
    res_config = auth_client.post(
        "/api/config",
        headers={"Authorization": f"Bearer {valid_token}"},
        json={"risk_pct": 2.2},
    )
    assert res_config.status_code == 200
    assert res_config.json()["status"] == "success"


def test_server_websocket_auth_guard(auth_client: TestClient) -> None:
    """Verifies WebSocket handshake authentication enforcement on /ws/live."""
    # 1. Connecting without token raises WebSocketDisconnect with policy violation code (1008)
    with pytest.raises(WebSocketDisconnect) as exc_info_no_token:
        with auth_client.websocket_connect("/ws/live"):
            _ = False
    assert exc_info_no_token.value.code == 1008

    # 2. Connecting with wrong query param token fails with 1008
    with pytest.raises(WebSocketDisconnect) as exc_info_wrong_token:
        with auth_client.websocket_connect("/ws/live?token=bogus_token"):
            _ = False
    assert exc_info_wrong_token.value.code == 1008

    # 3. Connecting with correct query param token succeeds and streams state
    with auth_client.websocket_connect("/ws/live?token=supersecrettoken123") as ws:
        packet = ws.receive_json()
        assert packet["type"] == "state_update"
        assert "status" in packet
        assert "tickers" in packet


# -----------------------------------------------------------------------------
# 12. Dynamic Execution Mode & Symbol Toggle Endpoints
# -----------------------------------------------------------------------------

def test_server_control_mode_switch(client: TestClient, test_orchestrator: TradingBotOrchestrator) -> None:
    """Verifies that POST /api/control/mode validates and executes mode transitions."""
    # 1. Switching to paper succeeds
    res_paper = client.post("/api/control/mode", json={"mode": "paper"})
    assert res_paper.status_code == 200
    assert res_paper.json()["status"] == "success"
    assert test_orchestrator.mode == ExecutionMode.PAPER

    # 2. Switching to live without valid credentials raises 400
    res_live = client.post("/api/control/mode", json={"mode": "live"})
    assert res_live.status_code == 400
    assert "live mode" in res_live.json()["detail"].lower()

    # 3. Invalid mode string raises 400
    res_invalid = client.post("/api/control/mode", json={"mode": "sandbox"})
    assert res_invalid.status_code == 400
    assert "invalid execution mode" in res_invalid.json()["detail"].lower()


def test_server_demo_and_real_mode_switching_with_keys(client: TestClient, test_orchestrator: TradingBotOrchestrator) -> None:
    """Verifies configuring demo testnet keys and switching between REAL and DEMO environments."""
    # 1. Configure Demo credentials and switch to DEMO mode
    res_cfg_demo = client.post(
        "/api/config",
        json={
            "demo_api_key": "demokey12345",
            "demo_api_secret": "demosecret67890",
            "mode": "demo",
        },
    )
    assert res_cfg_demo.status_code == 200
    assert test_orchestrator.mode == ExecutionMode.DEMO
    assert test_orchestrator.config.is_demo_mode is True

    # 2. Check /api/status telemetry reflects DEMO mode & testnet URL
    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    status = res_status.json()["status"]
    assert status["mode"] == "DEMO"
    assert status["is_demo"] is True
    assert status["has_demo_key"] is True
    assert "cdn-ind.testnet.deltaex.org" in status["active_base_url"]

    # 3. Configure Real Delta India credentials and switch to REAL mode
    res_cfg_real = client.post(
        "/api/config",
        json={
            "delta_api_key": "realkey12345",
            "delta_api_secret": "realsecret67890",
            "mode": "real",
        },
    )
    assert res_cfg_real.status_code == 200
    assert test_orchestrator.mode == ExecutionMode.REAL
    assert test_orchestrator.config.is_demo_mode is False

    # 4. Check /api/status telemetry reflects REAL mode & Delta India production URL
    res_status_real = client.get("/api/status")
    assert res_status_real.status_code == 200
    status_real = res_status_real.json()["status"]
    assert status_real["mode"] == "REAL"
    assert status_real["is_demo"] is False
    assert status_real["has_real_key"] is True
    assert "api.india.delta.exchange" in status_real["active_base_url"]


def test_server_control_symbol_toggle(client: TestClient, test_orchestrator: TradingBotOrchestrator) -> None:
    """Verifies that POST /api/control/symbol_toggle enables/disables symbol breakout evaluation."""
    # 1. Disable BTCUSDT
    res_disable = client.post("/api/control/symbol_toggle", json={"symbol": "BTCUSDT", "enabled": False})
    assert res_disable.status_code == 200
    assert "BTCUSDT" not in test_orchestrator.enabled_symbols

    # 2. Re-enable BTCUSDT
    res_enable = client.post("/api/control/symbol_toggle", json={"symbol": "BTCUSDT", "enabled": True})
    assert res_enable.status_code == 200
    assert "BTCUSDT" in test_orchestrator.enabled_symbols

    # 3. Non-existent symbol returns 404
    res_unknown = client.post("/api/control/symbol_toggle", json={"symbol": "UNKNOWNCOIN", "enabled": True})
    assert res_unknown.status_code == 404


def test_server_order_cancel_endpoint(client: TestClient, test_orchestrator: TradingBotOrchestrator) -> None:
    """Verifies that POST /api/orders/{order_id}/cancel cancels resting orders."""
    # 1. Place a resting stop order in paper simulator
    assert test_orchestrator.paper_simulator is not None
    order = test_orchestrator.paper_simulator.submit_stop_order(
        symbol="BTCUSDT",
        product_id=1,
        side="buy",
        size=5,
        stop_price=70000.0,
        bracket_stop_loss_price=69000.0,
        bracket_take_profit_price=72000.0,
        leverage=5,
    )
    assert order.id in test_orchestrator.paper_simulator._pending_orders

    # 2. Cancel order via API
    res_cancel = client.post(f"/api/orders/{order.id}/cancel")
    assert res_cancel.status_code == 200
    assert res_cancel.json()["status"] == "success"
    assert order.id not in test_orchestrator.paper_simulator._pending_orders

    # 3. Cancelling non-existent order returns 404
    res_not_found = client.post("/api/orders/9999999/cancel")
    assert res_not_found.status_code == 404


def test_server_config_expanded_fields(client: TestClient, test_orchestrator: TradingBotOrchestrator) -> None:
    """Verifies hot-reloading of all 9 strategy and risk parameters."""
    res = client.post(
        "/api/config",
        json={
            "timeframe": "5m",
            "breakout_entry_buffer": 2.5,
            "min_adx_threshold": 25.0,
            "max_drawdown_pct": 8.0,
            "risk_pct": 2.0,
            "leverage": 10,
            "atr_multiplier": 2.0,
            "stop_loss_mode": "trailing",
            "order_ttl_bars": 4,
        },
    )
    assert res.status_code == 200
    assert res.json()["status"] == "success"
    assert test_orchestrator.config.timeframe == "5m"
    assert test_orchestrator.config.breakout_entry_buffer == 2.5
    assert test_orchestrator.config.min_adx_threshold == 25.0
    assert test_orchestrator.config.max_drawdown_pct == 8.0
    assert test_orchestrator.config.default_risk_pct == 2.0
    assert test_orchestrator.config.default_leverage == 10
    assert test_orchestrator.config.trailing_atr_multiplier == 2.0
    assert test_orchestrator.config.order_ttl_bars == 4

    # Invalid timeframe rejected with 422
    res_bad_tf = client.post("/api/config", json={"timeframe": "2h"})
    assert res_bad_tf.status_code == 422

    # Invalid max_drawdown_pct rejected with 422
    res_bad_dd = client.post("/api/config", json={"max_drawdown_pct": 99.0})
    assert res_bad_dd.status_code == 422


def test_server_static_spa_files(client: TestClient) -> None:
    """Verifies that static index.html and assets are correctly served."""
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "DELTA-BOT" in res_root.text

    res_static = client.get("/static/index.html")
    assert res_static.status_code == 200
    assert "DELTA-BOT" in res_static.text


def test_server_control_alias_endpoints(client: TestClient, test_orchestrator: TradingBotOrchestrator) -> None:
    """Verifies alias routes for stop/pause, delete order, close position, and config."""
    # 1. Test /api/control/pause alias
    test_orchestrator.is_running = True
    res_pause = client.post("/api/control/pause")
    assert res_pause.status_code == 200
    assert test_orchestrator.is_running is False

    # 2. Test /api/control/config alias
    res_cfg = client.post("/api/control/config", json={"risk_pct": 1.5})
    assert res_cfg.status_code == 200
    assert test_orchestrator.config.default_risk_pct == 1.5

    # 3. Test DELETE /api/orders/{order_id} alias
    assert test_orchestrator.paper_simulator is not None
    order = test_orchestrator.paper_simulator.submit_stop_order(
        symbol="BTCUSDT",
        product_id=1,
        side="buy",
        size=2,
        stop_price=70000.0,
        bracket_stop_loss_price=69000.0,
        bracket_take_profit_price=72000.0,
        leverage=5,
    )
    res_del = client.delete(f"/api/orders/{order.id}")
    assert res_del.status_code == 200
    assert res_del.json()["status"] == "success"

    # 4. Test /api/control/close_position alias
    test_orchestrator.paper_simulator._open_positions[1] = VirtualPosition(
        product_id=1,
        symbol="BTCUSDT",
        side="buy",
        size=2,
        entry_price=68000.0,
        mark_price=68100.0,
        unrealized_pnl=1.0,
        allocated_margin=13.6,
    )
    res_close = client.post("/api/control/close_position", json={"product_id": 1, "symbol": "BTCUSDT"})
    assert res_close.status_code == 200
    assert res_close.json()["status"] == "success"
    assert 1 not in test_orchestrator.paper_simulator._open_positions


def test_server_api_keys_configure_and_reset(
    client: TestClient, test_orchestrator: TradingBotOrchestrator
) -> None:
    """Verifies that Delta API credentials can be dynamically configured and reset via REST."""
    from pydantic import SecretStr

    # 1. Initially no keys
    test_orchestrator.config.delta_api_key = SecretStr("")
    test_orchestrator.config.delta_api_secret = SecretStr("")
    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    assert res_status.json()["status"]["has_api_key"] is False

    # 2. Configure keys via /api/config
    payload = {
        "delta_api_key": "my_test_delta_key_12345",
        "delta_api_secret": "my_test_delta_secret_67890",
        "delta_base_url": "https://api.india.delta.exchange",
    }
    res_cfg = client.post("/api/config", json=payload)
    assert res_cfg.status_code == 200
    assert test_orchestrator.config.api_key_str == "my_test_delta_key_12345"
    assert test_orchestrator.config.api_secret_str == "my_test_delta_secret_67890"
    assert test_orchestrator.config.delta_base_url == "https://api.india.delta.exchange"

    # Status reflects configured keys
    res_status2 = client.get("/api/status")
    assert res_status2.json()["status"]["has_api_key"] is True
    assert "my_t...2345" in res_status2.json()["status"]["delta_api_key_masked"]

    # 3. Reset keys via /api/config/reset-keys
    res_reset = client.post("/api/config/reset-keys")
    assert res_reset.status_code == 200
    assert res_reset.json()["status"] == "success"
    assert test_orchestrator.config.api_key_str == ""
    assert test_orchestrator.config.api_secret_str == ""

    # Status reflects reset keys
    res_status3 = client.get("/api/status")
    assert res_status3.json()["status"]["has_api_key"] is False


def test_server_tenant_control_syncs_to_supabase(client: TestClient, test_orchestrator: TradingBotOrchestrator) -> None:
    """Verifies that tenant start/stop/flatten endpoints sync state to Supabase in real-time."""
    from unittest.mock import patch
    from delta_bot.tenant import TenantConfig

    # Retrieve or register tenant
    t_id = "server_sync_t"
    if t_id in test_orchestrator.tenant_manager.tenants:
        tenant = test_orchestrator.tenant_manager.tenants[t_id]
    else:
        t_cfg = TenantConfig(
            tenant_id=t_id,
            name="Server Sync Tenant",
            mode="demo",
            is_active=True,
        )
        tenant = test_orchestrator.tenant_manager.create_tenant(t_cfg)
    tenant.is_running = True
    tenant.config.is_active = True

    try:
        sync_calls = []
        with patch.object(test_orchestrator.tenant_manager, "sync_tenant_to_supabase", side_effect=lambda inst: sync_calls.append(inst)):
            # 1. Stop tenant
            res_stop = client.post(f"/api/tenants/{t_id}/control/stop")
            assert res_stop.status_code == 200
            assert tenant.is_running is False
            assert tenant.config.is_active is False
            assert len(sync_calls) >= 1
            assert sync_calls[-1].config.tenant_id == t_id

            # 2. Start tenant
            sync_calls.clear()
            res_start = client.post(f"/api/tenants/{t_id}/control/start")
            assert res_start.status_code == 200
            assert tenant.is_running is True
            assert tenant.config.is_active is True
            assert len(sync_calls) >= 1
            assert sync_calls[-1].config.tenant_id == t_id
    finally:
        test_orchestrator.tenant_manager.tenants.pop(t_id, None)
        test_orchestrator.tenant_manager.save_to_disk()


def test_live_positions_and_resting_orders_api(client: TestClient, test_orchestrator: TradingBotOrchestrator) -> None:
    """Verifies that in Live/Demo mode, GET /api/positions and /api/orders return ONLY live Delta API data without db history."""
    import uuid
    from unittest.mock import AsyncMock
    from delta_bot.tenant import TenantConfig

    t_id = f"live_desk_test_{uuid.uuid4().hex[:6]}"
    # Add a live-mode tenant
    cfg = TenantConfig(
        tenant_id=t_id,
        name="Live Desk",
        user_id="user_live_1",
        mode="real",
        delta_api_key="mock_key",
        delta_api_secret="mock_secret",
        is_active=True,
    )
    t = test_orchestrator.tenant_manager.create_tenant(cfg, contract_specs=test_orchestrator.contract_specs)
    t.mode = ExecutionMode.REAL
    t.live_positions = [
        {
            "product_id": 1,
            "symbol": "BTCUSDT",
            "side": "BUY",
            "size": 10,
            "entry_price": 68000.0,
            "mark_price": 68500.0,
            "unrealized_pnl": 5.0,
        }
    ]
    t.live_orders = [
        {
            "order_id": 123456,
            "id": 123456,
            "symbol": "BTCUSDT",
            "side": "BUY",
            "product_id": 1,
            "size": 10,
            "stop_price": 69000.0,
            "trigger_price": 69000.0,
            "status": "OPEN",
        }
    ]
    t.rest_client.cancel_order = AsyncMock(return_value={"status": "success"})

    try:
        # GET /api/positions for this tenant
        res_pos = client.get(f"/api/positions?tenant_id={t_id}")
        assert res_pos.status_code == 200
        p_data = res_pos.json()
        assert p_data["count"] == 1
        assert p_data["positions"][0]["symbol"] == "BTCUSDT"
        assert p_data["positions"][0]["size"] == 10

        # When live positions are empty, returns 0 and does NOT fall back to state_manager history
        t.live_positions = []
        t.state_manager._open_positions[1] = VirtualPosition(product_id=1, symbol="BTCUSDT", side="buy", size=999, entry_price=50000.0, mark_price=50000.0)
        res_empty = client.get(f"/api/positions?tenant_id={t_id}")
        assert res_empty.status_code == 200
        assert res_empty.json()["count"] == 0
        assert len(res_empty.json()["positions"]) == 0

        # GET /api/orders for this tenant returns live resting orders
        res_ord = client.get(f"/api/orders?tenant_id={t_id}")
        assert res_ord.status_code == 200
        o_data = res_ord.json()
        assert o_data["count"] == 1
        assert o_data["orders"][0]["order_id"] == 123456

        # Cancel live resting order via POST /api/orders/{order_id}/cancel
        res_cancel = client.post(f"/api/orders/123456/cancel?tenant_id={t_id}")
        assert res_cancel.status_code == 200
        assert res_cancel.json()["status"] == "success"
        t.rest_client.cancel_order.assert_awaited_once_with(product_id=1, order_id=123456)
    finally:
        test_orchestrator.tenant_manager.tenants.pop(t_id, None)
        test_orchestrator.tenant_manager.save_to_disk()



