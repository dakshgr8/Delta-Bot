# Delta-Bot: High-Performance Crypto Trading Bot (Production Ready - Phases 1 through 8 Complete)

A quantitative-grade asynchronous Python trading daemon, REST client, real-time WebSocket market data streaming engine, systematic strategy engine, state reconciliation manager, paper/live deployment daemon, real-time Web Dashboard & Control API, and hardened Linux production infrastructure for **Delta Exchange**, engineered specifically for high-throughput algorithmic trading across perpetual futures markets (**BTC**, **ETH**, **SOL**, **XRP**).

---

## 1. Architectural Highlights

### Phase 1: Foundation & REST Client
- **Modern Async REST Engine**: Built on `httpx.AsyncClient` with HTTP/2 multiplexing, persistent connection pooling, and connection keep-alive.
- **HMAC-SHA256 Cryptographic Engine**: Deterministic request signing adhering strictly to Delta's canonical specification:
  $$\text{Signature Data} = \text{METHOD} + \text{timestamp} + \text{path} + \text{query\_string} + \text{payload}$$
- **Dynamic Clock Synchronization**: Delta enforces a strict **5-second validity window** for API signatures. The client measures network transit round-trip time (RTT) and continuously tracks server clock offsets from response headers (`Request-In-Time` / `Date`) without requiring dedicated polling overhead.
- **Resilient Retry & Backoff**: Exponential backoff with **Full Jitter** on transient failures (`HTTP 429`, `500`, `502`, `503`, `504`, socket disconnects, and timeouts).
- **Exchange Telemetry & Rate-Limit Tracking**: Dynamically parses and updates `X-RATE-LIMIT` quotas (`limit`, `remaining`, `reset_ms`) to prevent burst throttling.
- **Security & Secret Redaction**: Powered by Pydantic Settings and `SecretStr`, preventing credentials from leaking into logs, tracebacks, or `__repr__`.
- **Zero-Latency Spec Caching**: In-memory caching for perpetual futures specifications (tick sizes, contract values, maker/taker fee schedules).

### Phase 2: Market Data & WebSocket Engine
- **Decoupled Architecture**: Divided into four dedicated subsystems:
  1. **Connection Lifecycle Manager**: 20-second heartbeat ping loop, 35-second watchdog timer, and auto-reconnection with jittered exponential backoff.
  2. **Subscription Dispatcher**: Tracks active channels and symbols, serializes canonical subscribe frames, and transparently re-subscribes upon connection recovery.
  3. **Channel Multiplexer & Sequence Tracker**: Routes incoming feeds (`v2/ticker`, `candlestick_*`, `l2_orderbook`), monitors sequence numbers for packet drop detection.
  4. **Candle Aggregator & Real-Time Bar Builder**:
     - **Confirmed Bar Close Guarantees**: Strict separation between forming ticks and confirmed closed candles. Strategy signals ONLY trigger on closed bars.
     - **In-Memory RingBuffer / CircularBarCache**: Retains the last 500 closed bars per symbol to feed the Strategy Engine.
     - **Daily Session Anchor (00:00 UTC)**: Tracks 00:00:00 UTC session rollover, computes Previous Day High (PDH), Previous Day Low (PDL), Previous Day Close (PDC), and updates cumulative intraday **Session VWAP**.
- **EventEmitter Interface**: High-performance asynchronous callback architecture (`on_candle_close`, `on_ticker`, `on_orderbook`, `on_session_rollover`, `on_disconnect`).

### Phase 3: Strategy & Signal Generation Engine
- **V2 Upgraded Volatility Breakout Strategy**:
  - Implements the *"EMA 20 + Session VWAP + PDH/PDL Volatility Breakout Strategy"*.
  - Strictly evaluated on confirmed closed bars (`on_candle_close`) with **zero lookahead bias**.
- **Pure NumPy Indicator Pipeline**:
  - **EMA (20-period)**: Trend baseline strictly over closed candles.
  - **Session VWAP**: Anchored to 00:00:00 UTC using `SessionAnchorData`.
  - **Wilder's ATR (14-period) & SMA(ATR, 20)**: Volatility expansion filter (`ATR_14 >= SMA(ATR, 20)`).
  - **ADX (14-period) with +DI / -DI**: Consolidation guard (`ADX >= 20.0` required).
  - **Prior Day High/Low (PDH / PDL)**: True daily breakout confirmation avoiding inside-day chop.
- **Signal Rules & Freshness Gate**:
  - **Bullish Breakout**: `Close > EMA 20` and `Close > Session VWAP` and `Close > PDH` and `ADX >= 20` and `ATR >= SMA(ATR, 20)`. Condition must be **Fresh** (met on bar $t$, but was False on bar $t-1$).
  - **Bearish Breakdown**: `Close < EMA 20` and `Close < Session VWAP` and `Close < PDL` and `ADX >= 20` and `ATR >= SMA(ATR, 20)`. Condition must be **Fresh**.

### Phase 4: Order Execution & Risk Management Engine
- **Exchange-Native Stop Orders with Brackets (Trap 1 Guard)**:
  - Submits native stop orders with attached bracket Take Profit and Stop Loss (`POST /v2/orders`) directly to Delta Exchange's matching engine.
  - Eliminates local client latency and execution slippage on fast crypto breakout candles.
- **Resting Order Purging on EMA Invalidation & TTL (Trap 1 Guard)**:
  - If intermediate candle closes across EMA 20 or if `max_pending_bars` elapses without price triggering, the bot sends `DELETE /v2/orders` to immediately purge the resting stop order from Delta's orderbook.
- **Dual-Constraint Margin & Leverage Sizing (Trap 2 Guard)**:
  - Sizes trades based on loss budget: $\text{Lots} = \frac{\text{CapitalAtRisk}}{\text{TickDistance} \times \text{ContractValue}}$.
  - Validates initial margin: $\text{Required Margin} = \frac{\text{Notional}}{\text{Leverage}} \le \text{Available Margin} \times 0.95$. Clamps order size if risk budget exceeds available margin.
- **Dynamic Two-Tier Trailing Stop Ratchet (Trap 3 Guard)**:
  - **Tier 1 (Pending Entry)**: Updates attached bracket on un-triggered entry order (`PUT /v2/orders/bracket`).
  - **Tier 2 (Filled Position)**: Detects parent order fill, identifies spawned child stop-loss order ID, and modifies child stop order directly (`PUT /v2/orders` with `stop_price`). Fallback modifies position bracket (`PUT /v2/positions/change_bracket`).
  - Ratchet monotonicity: strictly tightens protective stops; rejects any widening attempt.
- **Emergency Circuit Breaker & Max Concurrent Positions**:
  - Trips trading halts if daily drawdown exceeds threshold (default: 3.0%).
  - Maximum concurrent position guard (default: 2 active symbols).

### Phase 5: State Reconciliation & Private WebSocket Sync
- **Private WebSocket User Feeds (`delta_bot/private_ws.py`)**:
  - Authenticated WebSocket streaming on private channels (`orders`, `positions`, `user_trades`).
  - Cryptographic handshake using `GET + timestamp + /live` HMAC-SHA256 signature.
  - Auto-reconnect with persistent re-authentication.
- **Unified State Machine (`delta_bot/state.py`)**:
  - Transitions `OrderIntent` states: `PENDING` -> `TRIGGERED` -> `CLOSED`.
  - Zero-latency child bracket binding upon receiving order fill events.
  - Hybrid REST polling reconciliation worker (30-second interval fallback).
  - Liquidation Proximity Guard: Emergency position closure if mark price nears liquidation threshold within $1.5 \times \text{ATR}$.
  - Ghost position isolation without blind closure.

### Phase 6: Final Integration, Orchestrator Daemon & Deployment Engine
- **Unified System Orchestrator (`delta_bot/orchestrator.py`)**:
  - Operates 6 concurrent async tasks:
    - **Task 1**: Public Market Data Engine (`DeltaWebSocketClient` multi-symbol streaming & REST warm-up).
    - **Task 2**: Private WebSocket Engine (`PrivateWebSocketClient` authenticated real-time updates).
    - **Task 3**: Strategy & Signal Dispatcher (`BreakoutStrategyEngine` on confirmed candle closes).
    - **Task 4**: State Reconciliation Worker (`TradingStateManager` periodic REST fallback sync).
    - **Task 5**: Daily Session Rollover Engine (00:00 UTC PDH/PDL update & circuit breaker reset).
    - **Task 6**: Watchdog & Health Telemetry Heartbeat (Terminal dashboard & system health monitoring).
- **Dual Operational Modes (`paper` vs `live`)**:
  - **Paper Trading Simulator (`delta_bot/paper.py`)**:
    - High-fidelity in-memory virtual matching simulator.
    - Simulated orderbook depth slippage and commission fee deductions.
    - Virtual collateral wallet tracking floating equity, realized PnL, win rate %, and high-watermark drawdown.
    - Dispatches synthetic `OrderUpdate`, `UserTrade`, and `PositionInfo` events into `TradingStateManager` so state tracking is identical between modes.
  - **Live Trading Mode**: Real exchange stop and bracket order placement with API credential safety enforcement.
- **Graceful POSIX Signal Trapping & Emergency Flattening**:
  - Traps `SIGINT` (Ctrl+C) and `SIGTERM`.
  - Cancels all resting pending orders.
  - Evaluates `FLATTEN_ON_SHUTDOWN` toggle:
    - `True`: Emergency market-closes all active open positions.
    - `False`: Preserves open positions with exchange-native Stop Loss and Take Profit brackets intact.
- **Production CLI Daemon (`run_bot.py`)**:
  - Standalone executable CLI entry-point with arguments `--mode`, `--symbols`, `--timeframe`, `--leverage`, `--risk-pct`, `--flatten-on-shutdown`, `--web`, `--host`, `--port`, and `--status-interval`.

### Phase 7: Web Dashboard & Real-Time Control API
- **FastAPI Gateway (`delta_bot/server.py`)**:
  - Asynchronous web application embedding directly into `TradingBotOrchestrator` without polling or blocking.
  - Comprehensive REST API endpoints for state introspection, strategy telemetry, configuration hot-reloading, and manual intervention.
  - Real-time WebSocket streaming (`/ws/live`) pushing unified state updates at 500ms intervals to all connected clients.
  - In-memory ring-buffer logging sink (`RingBufferLogHandler`) retaining up to 1,000 structured logs streamed directly to web clients.
- **Dynamic Configuration Hot-Reloading**:
  - `POST /api/config` endpoint with strict Pydantic validation allowing live adjustment of risk per trade, max drawdown circuit breaker, leverage, and breakout parameters without daemon restarts.
- **Lightweight High-Performance Frontend SPA (`web/index.html` & `delta_bot/static/index.html`)**:
  - Clean, dark-mode terminal aesthetic styled with Tailwind CSS CDN and zero runtime JavaScript framework dependencies.
  - Strict XSS immunity: Uses safe DOM construction methods (`textContent`, `setAttribute`, `element.replaceChildren()`) with zero unsanitized `innerHTML` rendering.
  - Top Bar: Real-time Live/Paper badge, engine heartbeat pulse indicator, Global PnL, Start/Stop trading toggle, and Emergency Flatten Kill-Switch with confirmation modal.
  - Target Asset Cards: Live ticker pricing, 24h delta, EMA 20, Session VWAP, ADX Volatility Squeeze Gauge (ADX < 20 consolidation vs $\ge$ 20 breakout expansion), ATR status, and PDH/PDL markers.
  - Active Positions & Pending Breakout Orders Table with direct manual close actions.
  - Live Interactive Terminal Console with auto-scroll and log level filtering.
- **Unified Orchestrator & CLI Integration**:
  - Single CLI command (`python3 run_bot.py --web`) boots the trading daemon and launches the Uvicorn web server concurrently in the same event loop.

### Phase 8: Cyber-Security Hardening & Linux Reliability Infrastructure
- **Constant-Time API & WebSocket Authentication Guard**:
  - Configurable `DASHBOARD_AUTH_TOKEN` stored in `BotConfig` as a protected `SecretStr`.
  - Mutating control endpoints (`/api/control/*`, `/api/config`, `/api/positions/*/close`) enforce header authentication (`X-Dashboard-Token` or `Authorization: Bearer <TOKEN>`) with constant-time cryptographic equality (`secrets.compare_digest`), preventing timing side-channel attacks.
  - WebSocket `/ws/live` enforces handshake token verification via `?token=` query parameter or auth headers, closing unauthorized connections immediately with RFC 6455 policy violation (code `1008`).
  - Frontend SPA automatically persists tokens in `localStorage`, transparently includes auth headers on all mutation requests, auto-prompts on 401/1008 responses, and displays lock badges in the navigation header.
- **Hardened Linux Systemd Service (`deploy/deltabot.service`)**:
  - Unprivileged user execution (`User=deltabot`, `Group=deltabot`).
  - High-resilience lifecycle directives (`Restart=always`, `RestartSec=5s`, `TimeoutStopSec=30s`, `KillMode=mixed`).
  - Comprehensive kernel and filesystem sandboxing: `NoNewPrivileges=true`, `ProtectSystem=full`, `ProtectHome=true`, `PrivateTmp=true`, `PrivateDevices=true`, `MemoryDenyWriteExecute=true`, `RestrictRealtime=true`, `RestrictSUIDSGID=true`, empty capability bounding sets (`CapabilityBoundingSet=`), and `LimitNOFILE=65536`.
- **Production Nginx Reverse-Proxy Template (`deploy/nginx.conf`)**:
  - SSL/TLS termination with modern cipher suites (`TLSv1.2`, `TLSv1.3`).
  - HTTP to HTTPS 301 redirection with ACME certbot challenge pass-through.
  - Strict rate limiting: Dedicated `5r/s` burst zone for `/api/control/` and `/api/config` to eliminate brute force or unauthorized spamming.
  - Full WebSocket protocol proxying with `Upgrade` and `Connection` header handling and extended streaming timeouts.
  - Comprehensive OWASP security headers: HSTS, CSP, X-Frame-Options DENY, X-Content-Type-Options nosniff, Referrer-Policy.

---

## 2. Project Directory Structure

```text
Delta-Bot/
├── .env.example                  # Environment configuration template
├── pytest.ini                    # Pytest configuration with asyncio strict mode
├── requirements.txt              # Production and testing dependencies
├── README.md                     # Complete system documentation
├── run_bot.py                    # Standalone Production CLI Daemon with Web Dashboard (Phase 6 & 7)
├── main.py                       # Async smoke test & demonstration harness
├── delta_bot/                    # Core library package
│   ├── __init__.py               # Public package exports (v8.0.0)
│   ├── auth.py                   # HMAC-SHA256 signer & clock drift sync
│   ├── client.py                 # REST API client with order placement, cancel & edit
│   ├── config.py                 # Pydantic Settings with credential masking, auth token & WS endpoints
│   ├── exceptions.py             # Typed exception hierarchy
│   ├── execution.py              # OrderExecutionManager, Margin Sizing & Native Brackets
│   ├── http_client.py            # Async HTTP engine with retry, pooling & telemetry
│   ├── indicators.py             # Pure NumPy indicator library & IndicatorPipeline
│   ├── models.py                 # Domain models (Candle, OrderBook, Ticker, ExchangeOrder)
│   ├── orchestrator.py           # TradingBotOrchestrator daemon (Phase 6)
│   ├── paper.py                  # PaperExchangeSimulator & VirtualWallet (Phase 6)
│   ├── private_ws.py             # Authenticated Private WebSocket client (Phase 5)
│   ├── risk.py                   # RiskManager, Dual-Constraint Sizing & Circuit Breaker
│   ├── server.py                 # FastAPI Web Gateway, REST Control API, WS Streamer & Auth Guard (Phase 7 & 8)
│   ├── state.py                  # TradingStateManager & REST Reconciliation (Phase 5)
│   ├── static/                   # Static web assets served by FastAPI
│   │   └── index.html            # Embedded Dark Dashboard SPA with Auth support (Phase 7 & 8)
│   ├── strategy.py               # BreakoutStrategyEngine, OrderIntent & Signal Rules
│   └── websocket_client.py       # Public WebSocket Engine, Bar Builder & Lifecycle Manager
├── deploy/                       # Linux Production Deployment Infrastructure (Phase 8)
│   ├── deltabot.service          # Hardened Systemd unit file with security sandboxing
│   └── nginx.conf                # Production Nginx reverse-proxy with SSL, rate limiting & WS upgrade
├── scripts/
│   ├── execute_order_lifecycle.py # Native stop orders, margin sizing & TTL purging (Phase 4)
│   ├── run_strategy_pipeline.py  # Historical warm-up & strategy verification (Phase 3)
│   ├── stream_market_data.py     # Live market data streaming script (Phase 2)
│   └── verify_state_sync.py      # Private WS auth, child bracket binding & state sync (Phase 5)
├── web/                          # Standalone web dashboard root
│   └── index.html                # Zero-dependency Dark Dashboard SPA with Auth (Phase 7 & 8)
└── tests/                        # 102 automated unit and integration tests (100% pass)
    ├── __init__.py
    ├── test_auth.py              # HMAC signing & clock synchronization tests
    ├── test_client.py            # REST client & error classification tests
    ├── test_config.py            # Configuration & URL validation tests
    ├── test_execution.py         # Native stop orders, brackets, margin sizing & product IDs
    ├── test_models.py            # Price/size quantization & model computation tests
    ├── test_orchestrator.py      # Orchestrator daemon, paper trading & signal handling (Phase 6)
    ├── test_risk.py              # Risk limits, margin sizing & circuit breaker tests
    ├── test_server.py            # FastAPI REST endpoints, WebSocket streaming, close, mode switch & Auth (Phase 7 & 8)
    ├── test_state.py             # Private WS auth, child SL discovery & reconciliation tests
    ├── test_strategy.py          # Strategy engine, indicators, triggers & sizing tests
    └── test_websocket.py         # WebSocket engine, bar aggregation & sequence tests
```

---

## 3. Supported Exchange URLs

| Environment | REST API Base URL | WebSocket Base URL |
| :--- | :--- | :--- |
| **Delta Exchange Global** | `https://api.delta.exchange` | `wss://socket.delta.exchange` |
| **Delta Exchange India** | `https://api.india.delta.exchange` | `wss://socket.india.delta.exchange` |
| **Delta Testnet** | `https://testnet-api.delta.exchange` | *REST Testing* |

---

## 4. Configuration & Security

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Configuration parameters:

```ini
DELTA_API_KEY=your_live_api_key_here
DELTA_API_SECRET=your_live_api_secret_here

DELTA_BASE_URL=https://api.india.delta.exchange
DELTA_WS_URL=wss://socket.india.delta.exchange

EXECUTION_MODE=paper
FLATTEN_ON_SHUTDOWN=false
TARGET_SYMBOLS=BTCUSD,ETHUSD,SOLUSD,XRPUSD,XAUTUSD
TIMEFRAME=1m
DEFAULT_LEVERAGE=5
DEFAULT_RISK_PCT=1.0
STATUS_INTERVAL_SEC=60.0
DASHBOARD_AUTH_TOKEN=your_secure_random_dashboard_token_here

REQUEST_TIMEOUT_SEC=10.0
MAX_RETRIES=3
BACKOFF_BASE_SEC=0.5
BACKOFF_MAX_SEC=8.0
```

---

## 5. CLI Daemon & Usage

### Running the Standalone CLI Daemon (`run_bot.py`)

Run in **Paper Trading Mode** with live Web Dashboard:
```bash
python3 run_bot.py --mode paper --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT --timeframe 1m --risk-pct 1.0 --web --host 127.0.0.1 --port 8000
```

Run in **Live Exchange Mode** (submits native stop orders with brackets to Delta Exchange):
```bash
python3 run_bot.py --mode live --symbols BTCUSDT,ETHUSDT --leverage 5 --flatten-on-shutdown --web
```

### CLI Options Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--mode` | string | `paper` | Operational mode: `paper` (virtual simulator) or `live` (Delta Exchange) |
| `--symbols` | string | `BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT` | Target perpetual futures contracts to trade concurrently |
| `--timeframe` | string | `1m` | Candlestick resolution (e.g. `1m`, `3m`, `5m`, `15m`) |
| `--leverage` | integer | `5` | Default leverage tier |
| `--risk-pct` | float | `1.0` | Percentage of account equity risked per breakout trade |
| `--flatten-on-shutdown` | flag | `False` | Emergency market-closes all active positions upon receiving SIGINT/SIGTERM |
| `--web` | flag | `False` | Launch the FastAPI Web Dashboard & Real-Time Control API concurrently |
| `--host` | string | `127.0.0.1` | Host address for FastAPI Web Dashboard binding |
| `--port` | integer | `8000` | Port for FastAPI Web Dashboard server |
| `--status-interval` | float | `60.0` | Watchdog health telemetry heartbeat interval in seconds |
| `--initial-balance` | float | `10000.0` | Starting virtual collateral balance for Paper trading mode |
| `--log-level` | string | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--env-file` | string | `.env` | Path to custom `.env` file |

### Watchdog Health Telemetry Dashboard (Console Output)

Every `STATUS_INTERVAL_SEC` (default: 60s), the orchestrator outputs a structured telemetry dashboard:

```text
====================== DELTA-BOT HEALTH TELEMETRY ======================
Mode: PAPER   | Uptime: 01h 24m 12s  | Status: ACTIVE
Collateral: $10,245.50 Equity | $10,120.00 Balance | UPnL: +$125.50
Performance: Realized PnL: +$120.00 | Win Rate: 66.7% (2/3) | Max DD: 1.15%
Positions (1): BTCUSDT: BUY 10 lots @ 68,050.0000 (UPnL: +$125.50)
Pending Orders (1): ETHUSDT: BUY STOP @ 3,520.0000 (Lots: 50)
Circuit Breaker: NORMAL   | Public WS: CONNECTED  | Private WS: N/A (Paper)
========================================================================
```

---

## 6. Single-Command Launch & Quantitative Web Dashboard (Phases 7 & 8)

Delta-Bot runs the complete backend quantitative engine and serves the full-featured web terminal simultaneously via a single command with **zero external build tools** (no Node.js/npm/Vite needed):

```bash
# Single-command launch running all 4 symbols with real-time web terminal at http://127.0.0.1:8000:
python3 run_bot.py --web --port 8000
```

### Dashboard Layout & Capabilities

1. **Global Header & Control Bar**:
   - **Mode Switch**: Seamless transition between `PAPER` simulator and `LIVE` exchange execution with safety verification.
   - **Master Strategy Control**: Instant Start / Pause button.
   - **Emergency Flatten Kill-Switch**: One-click purge of all resting orders and market-close of active positions.
   - **Auth Guard**: Persistent Bearer / Token configuration saved securely in browser `localStorage`.
   - **Connection Telemetry**: Live WebSocket round-trip ping/latency measurement, daemon uptime, memory utilization, and circuit breaker status.

2. **Asset Telemetry & Signal Cards (BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT)**:
   - Mark Price, 24h Change ($ and %), Best Bid/Ask, and bid-ask spread in basis points (bps).
   - Session Indicators: EMA 20, Anchored Intraday Session VWAP (00:00 UTC), Previous Day High (PDH), Previous Day Low (PDL).
   - Breakout Readiness Gauge: Real-time ADX regime indicator (`SQUEEZE < 20` vs `TREND >= 20`) and ATR volatility expansion status.
   - Per-Asset Trading Toggle: Enable or disable signal evaluation on individual symbols without restarting the daemon.

3. **Live Positions & Orders Workspace**:
   - **Active Positions Table**: Entry Price, Size (lots and native coin units), Leverage, Allocated Margin, Floating Unrealized PnL, Ratcheted Trailing Stop, Target Take Profit (2.0R), Liquidation Price, Liquidation Proximity %, and one-click "Close Position".
   - **Resting Breakout Orders Table**: Stop Price, Attached SL/TP brackets, Candle TTL countdown bars remaining (e.g. 2/3 bars), and one-click "Cancel Order".

4. **Strategy & Risk Parameter Hot-Reloading Modal**:
   - Live configuration adjustment from the browser without restarting:
     - **Candle Timeframe**: 1m, 3m, 5m, 15m, 1h
     - **Account Risk %**: 0.1% to 10.0%
     - **Leverage Tier**: 1x to 100x
     - **Stop Loss Mode**: Trailing Ratchet vs Fixed Stop
     - **Trailing Stop ATR Multiplier**: 0.5x to 5.0x
     - **Breakout Entry Buffer**: 0.0 to 20.0 ticks
     - **ADX Filter Threshold**: 5.0 to 50.0
     - **Daily Max Drawdown %**: 0.5% to 50.0% (Circuit Breaker)
     - **Order TTL**: 1 to 20 bars

5. **Live Telemetry & Log Console**:
   - Real-time streaming log feed with auto-scroll lock, log level filters (`ALL`, `INFO`, `WARNING`, `ERROR`, `TRADE`), and clear buffer control.

### REST & WebSocket Endpoint Specifications

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` or `/dashboard` | Serves the zero-dependency, dark-mode Single-Page Application (SPA) |
| `GET` | `/api/status` | Bot status, execution mode, uptime, memory, circuit breaker, wallet equity |
| `GET` | `/api/positions` | Active open positions, entry prices, unrealized PnL, liquidation thresholds |
| `GET` | `/api/orders` | Active pending breakout stop orders and historical executions |
| `GET` | `/api/indicators/{symbol}` | Technical indicators (EMA 20, VWAP, ATR 14, SMA ATR 20, ADX 14, PDH, PDL) |
| `GET` | `/api/logs` | Buffered in-memory logs (up to 200 entries) with `?limit=100` support |
| `POST` | `/api/control/start` | Starts / resumes the trading loop and order placement |
| `POST` | `/api/control/stop` | Gracefully suspends the trading loop (halts new order submissions) |
| `POST` | `/api/control/flatten` | Emergency kill-switch: purges resting orders and market-closes all open positions |
| `POST` | `/api/control/mode` | Safely switches execution mode between `paper` and `live` |
| `POST` | `/api/control/symbol_toggle` | Enables or disables breakout signal generation for an individual asset |
| `POST` | `/api/orders/{order_id}/cancel` | Manually cancels an individual resting breakout stop order |
| `POST` | `/api/config` | Hot-reloads all 9 strategy and risk parameters simultaneously |
| `WS` | `/ws/live` | Real-time WebSocket channel pushing full unified state snapshots every 500ms |

### Real-Time Live Stream Packet (`/ws/live`)

Connected WebSocket clients receive synchronized JSON telemetry frames every 500ms:

```json
{
  "type": "state_update",
  "timestamp": 1730887200.123,
  "status": {
    "is_running": true,
    "mode": "PAPER",
    "uptime_seconds": 3600.0,
    "circuit_breaker_tripped": false
  },
  "collateral": {
    "equity": 10245.5,
    "balance": 10120.0,
    "unrealized_pnl": 125.5,
    "realized_pnl": 120.0,
    "win_rate_pct": 66.7,
    "max_drawdown_pct": 1.15
  },
  "tickers": { ... },
  "positions": [ ... ],
  "orders": [ ... ],
  "indicators": { ... },
  "logs": [ ... ]
}
```

---

## 7. Programmatic Python Usage

### Running TradingBotOrchestrator in Python

```python
import asyncio
from delta_bot import BotConfig, ExecutionMode, TradingBotOrchestrator

async def main():
    config = BotConfig(
        execution_mode="paper",
        target_symbols_str="BTCUSDT,ETHUSDT",
        timeframe="1m",
        default_risk_pct=1.0,
        flatten_on_shutdown=False,
    )

    orchestrator = TradingBotOrchestrator(config=config, mode=ExecutionMode.PAPER)
    orchestrator.setup_signal_handlers()

    # Runs until SIGINT/SIGTERM is received
    await orchestrator.run()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 8. Automated Test Suite

Run the full pytest suite covering all 8 phases:

```bash
pytest -v
```

```text
============================= test session starts ==============================
rootdir: /home/daksh/Delta-Bot
collected 97 items

tests/test_auth.py (5 tests) ............................................ PASSED
tests/test_client.py (3 tests) .......................................... PASSED
tests/test_config.py (7 tests) .......................................... PASSED
tests/test_execution.py (15 tests) ...................................... PASSED
tests/test_models.py (6 tests) .......................................... PASSED
tests/test_orchestrator.py (12 tests) ................................... PASSED
tests/test_risk.py (6 tests) ............................................ PASSED
tests/test_server.py (20 tests) ......................................... PASSED
tests/test_state.py (8 tests) ........................................... PASSED
tests/test_strategy.py (10 tests) ....................................... PASSED
tests/test_websocket.py (10 tests) ...................................... PASSED
======================= 102 passed in 3.69s ========================
```

---

## 9. Production Linux Deployment (systemd & Nginx)

Delta-Bot includes production-ready deployment configurations in `deploy/`.

### 9.1 Systemd Service Installation (`deploy/deltabot.service`)

1. Create dedicated unprivileged service user and application directory:
   ```bash
   sudo useradd --system --shell /usr/sbin/nologin --comment "Delta-Bot Daemon User" deltabot
   sudo mkdir -p /opt/deltabot /etc/deltabot
   sudo chown -R deltabot:deltabot /opt/deltabot /etc/deltabot
   ```

2. Copy source code and create virtual environment:
   ```bash
   sudo cp -r . /opt/deltabot/
   cd /opt/deltabot
   python3 -m venv venv
   ./venv/bin/pip install --upgrade pip
   ./venv/bin/pip install -r requirements.txt
   sudo chown -R deltabot:deltabot /opt/deltabot
   ```

3. Configure environment secrets:
   ```bash
   sudo cp .env.example /etc/deltabot/deltabot.env
   sudo chmod 600 /etc/deltabot/deltabot.env
   sudo chown deltabot:deltabot /etc/deltabot/deltabot.env
   # Edit /etc/deltabot/deltabot.env with your live API keys and DASHBOARD_AUTH_TOKEN
   ```

4. Enable and start systemd service:
   ```bash
   sudo cp deploy/deltabot.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now deltabot
   sudo systemctl status deltabot
   ```

5. View live daemon journal logs:
   ```bash
   journalctl -u deltabot -f
   ```

### 9.2 Hardened Nginx Reverse Proxy (`deploy/nginx.conf`)

1. Install Nginx and Certbot for automated TLS certificates:
   ```bash
   sudo apt update && sudo apt install -y nginx certbot python3-certbot-nginx
   ```

2. Obtain SSL/TLS certificate:
   ```bash
   sudo certbot certonly --standalone -d deltabot.yourdomain.com
   ```

3. Deploy configuration template:
   ```bash
   sudo cp deploy/nginx.conf /etc/nginx/nginx.conf
   sudo nginx -t
   sudo systemctl reload nginx
   ```

### 9.3 Security Features & Controls

| Feature | Implementation | Benefit |
| :--- | :--- | :--- |
| **Sandboxing** | `ProtectSystem=full`, `ProtectHome=true`, `PrivateTmp=true` | Daemon cannot alter system binaries or access user home directories |
| **Privilege Separation** | `User=deltabot`, `CapabilityBoundingSet=` | Drops all root capabilities; prevents privilege escalation |
| **Constant-Time Auth** | `secrets.compare_digest` in FastAPI guard | Protects mutating control endpoints against timing side-channel attacks |
| **WebSocket Guard** | RFC 6455 Code 1008 on missing/invalid token | Prevents unauthorized live market data and order state sniffing |
| **Strict Rate-Limiting** | Nginx `limit_req_zone` (5r/s burst=10) | Eliminates brute-force attacks on kill-switch and config endpoints |
| **OWASP Headers** | HSTS, CSP, X-Frame-Options DENY, nosniff | Hardens browser frontend against clickjacking, XSS, and MIME-sniffing |

---

## 10. Cloud Deployment (Vercel Frontend + Render Backend)

Delta-Bot is pre-configured for split cloud deployment: **Render** for the Python daemon & FastAPI WebSocket backend, and **Vercel** for the high-speed edge frontend dashboard.

### 10.1 Backend Deployment on Render

1. Log into [Render](https://dashboard.render.com).
2. Click **New +** → **Blueprint**.
3. Select your repository `dakshgr8/Delta-Bot`.
4. Render will automatically read `render.yaml` and configure:
   - **Service Name**: `delta-bot-backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python run_bot.py --web --host 0.0.0.0 --port $PORT --mode paper`
   - **Health Check Path**: `/api/status`
5. Set your environment variables in the Render Dashboard:
   - `EXECUTION_MODE`: `paper` (or `live` for real trading)
   - `DASHBOARD_AUTH_TOKEN`: A strong shared secret token (auto-generated by Blueprint)
   - `DELTA_API_KEY`: Your Delta Exchange API Key (required for `live`)
   - `DELTA_API_SECRET`: Your Delta Exchange API Secret (required for `live`)
6. Click **Apply**. Once built, note your backend URL:
   `https://<your-service-name>.onrender.com`

### 10.2 Frontend Deployment on Vercel

1. Log into [Vercel](https://vercel.com).
2. Click **Add New...** → **Project**, and import `dakshgr8/Delta-Bot`.
3. In **Project Settings**:
   - **Framework Preset**: `Other`
   - **Root Directory**: `web`
4. Click **Deploy**. Your frontend is live at:
   `https://delta-bot-frontend.vercel.app`

### 10.3 Connecting Frontend & Backend

1. Open your Vercel URL: `https://delta-bot-frontend.vercel.app`.
2. Click the **🌐 SERVER** button in the top navigation bar.
3. Paste your Render backend URL (e.g. `https://delta-bot-backend.onrender.com`).
4. Click **Test Connection** to verify `/api/status` connectivity, then click **Save & Connect**.
5. If `DASHBOARD_AUTH_TOKEN` is configured on Render, click the **🔑 AUTH** button and enter your token to enable trading controls and kill-switch commands.


