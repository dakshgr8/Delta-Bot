"""
delta_bot.server
~~~~~~~~~~~~~~~~

FastAPI Gateway & Real-Time Control WebSocket API for Delta-Bot (Phase 7).
Provides:
- Embedded Asynchronous FastAPI Backend hooked directly into TradingBotOrchestrator.
- REST API: Status, Positions, Resting Orders, Indicators, Logs, Control & Hot-Reload.
- Real-Time WebSocket Streaming (/ws/live) at 500ms frequency.
- In-Memory RingBuffer Log Handler capturing formatted system output.
- Zero-Dependency Embedded Single-Page Dashboard (Vanilla HTML5 / Tailwind CSS).
"""

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import os
import resource
import secrets
import time
from typing import Any, AsyncGenerator

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AliasChoices, BaseModel, Field

from delta_bot.models import ExecutionMode
from delta_bot.orchestrator import TradingBotOrchestrator

logger = logging.getLogger("delta_bot.server")
STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))


# -----------------------------------------------------------------------------
# 1. Pydantic Models for REST API Payloads
# -----------------------------------------------------------------------------

class ConfigUpdateRequest(BaseModel):
    """Payload schema for runtime hot-reloading of bot configuration parameters."""

    timeframe: str | None = Field(
        default=None,
        pattern=r"^(1m|3m|5m|15m|1h)$",
        description="Candle timeframe resolution (e.g. 1m, 3m, 5m, 15m, 1h)",
    )
    risk_pct: float | None = Field(
        default=None,
        ge=0.1,
        le=10.0,
        description="Percentage of equity risked per breakout trade (0.1% to 10.0%)",
    )
    leverage: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Default leverage tier across target perpetual contracts",
    )
    trailing_sl_enabled: bool | None = Field(
        default=None,
        description="Toggle dynamic trailing stop loss ratchet feature (True=Trailing SL, False=Fixed SL)",
    )
    stop_loss_mode: str | None = Field(
        default=None,
        description="Stop loss mode: 'trailing' or 'fixed'",
    )
    trailing_stop_atr_mult: float | None = Field(
        default=None,
        validation_alias=AliasChoices("trailing_stop_atr_mult", "atr_multiplier"),
        ge=0.5,
        le=5.0,
        description="Multiplier for ATR-based dynamic trailing stop ratcheting",
    )
    breakout_entry_buffer: float | None = Field(
        default=None,
        ge=0.0,
        le=20.0,
        description="Buffer in tick sizes added above High or below Low",
    )
    min_adx_threshold: float | None = Field(
        default=None,
        ge=5.0,
        le=50.0,
        description="Minimum ADX required to confirm non-consolidation (breakout filter)",
    )
    max_drawdown_pct: float | None = Field(
        default=None,
        ge=0.5,
        le=50.0,
        description="Circuit breaker daily maximum drawdown limit percentage",
    )
    max_pending_bars: int | None = Field(
        default=None,
        validation_alias=AliasChoices("max_pending_bars", "order_ttl_bars"),
        ge=1,
        le=20,
        description="TTL expiration bars for pending resting stop breakout orders",
    )


class ModeSwitchRequest(BaseModel):
    """Payload schema for switching execution mode between paper and live."""

    mode: str = Field(description="Target operational mode: 'paper' or 'live'")


class SymbolToggleRequest(BaseModel):
    """Payload schema for enabling or disabling trading on a specific asset symbol."""

    symbol: str = Field(description="Target contract symbol (e.g. BTCUSDT)")
    enabled: bool = Field(description="Whether breakout trading evaluation is enabled")


class ControlActionResponse(BaseModel):
    """Standard response envelope for control API operations."""

    status: str
    action: str
    message: str
    timestamp: float = Field(default_factory=time.time)


# -----------------------------------------------------------------------------
# 2. In-Memory RingBuffer Log Handler
# -----------------------------------------------------------------------------

class RingBufferLogHandler(logging.Handler):
    """
    In-memory FIFO RingBuffer log handler retaining the last N log entries (default: 200).
    Allows live terminal logs to be served via REST API and broadcast over WebSocket.
    """

    def __init__(self, capacity: int = 200) -> None:
        super().__init__()
        self.capacity: int = capacity
        self.buffer: deque[dict[str, Any]] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            entry = {
                "timestamp": record.created,
                "time_str": datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
                    "%H:%M:%S"
                ),
                "level": record.levelname,
                "name": record.name,
                "message": msg,
            }
            self.buffer.append(entry)
        except Exception as err:
            logger.debug("Failed to record log in RingBufferLogHandler: %s", err)

    def get_logs(
        self, limit: int = 50, level: str | None = None
    ) -> list[dict[str, Any]]:
        """Returns recent formatted log entries with optional level filtering."""
        logs = list(self.buffer)
        if level:
            lvl_upper = level.upper()
            logs = [e for e in logs if e["level"] == lvl_upper]
        return logs[-limit:] if limit > 0 else logs


# Global singleton log handler attached to root/bot loggers
log_handler = RingBufferLogHandler(capacity=200)
formatter = logging.Formatter("%(message)s")
log_handler.setFormatter(formatter)
log_handler.setLevel(logging.DEBUG)

bot_logger = logging.getLogger("delta_bot")
bot_logger.setLevel(logging.DEBUG)
bot_logger.addHandler(log_handler)


# -----------------------------------------------------------------------------
# 3. WebSocket Connection & Broadcast Manager
# -----------------------------------------------------------------------------

class WebSocketManager:
    """Manages active WebSocket subscribers streaming real-time daemon state."""

    def __init__(self) -> None:
        self.active_connections: set[WebSocket] = set()
        self._lock: asyncio.Lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Accepts and registers incoming client WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.debug("WebSocket client connected. Active subscribers: %d", len(self.active_connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        """Unregisters disconnected WebSocket client."""
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.debug("WebSocket client disconnected. Active subscribers: %d", len(self.active_connections))

    async def broadcast(self, payload: dict[str, Any]) -> None:
        """Broadcasts JSON payload to all active subscribers, discarding dead sockets."""
        if not self.active_connections:
            return

        dead_connections: list[WebSocket] = []
        text_data = json.dumps(payload, default=str)

        async with self._lock:
            for ws in self.active_connections:
                try:
                    await ws.send_text(text_data)
                except Exception as send_err:
                    logger.debug("Error sending WebSocket frame to client: %s", send_err)
                    dead_connections.append(ws)

            for dead in dead_connections:
                self.active_connections.discard(dead)


# -----------------------------------------------------------------------------
# 4. State Aggregation & Telemetry Helpers
# -----------------------------------------------------------------------------

def get_process_memory_mb() -> float:
    """Calculates current process RSS memory in MB using standard resource module."""
    try:
        # On Linux, ru_maxrss is in Kilobytes
        usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(usage_kb / 1024.0, 2)
    except Exception as mem_err:
        logger.debug("Failed to calculate process memory: %s", mem_err)
        return 0.0


def format_uptime_human(seconds: float) -> str:
    """Converts seconds into human-readable DDd HHh MMm SSs format."""
    sec_int = int(seconds)
    days, rem = divmod(sec_int, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours:02d}h {minutes:02d}m {secs:02d}s"
    return f"{hours:02d}h {minutes:02d}m {secs:02d}s"


def get_all_positions_summary(orchestrator: TradingBotOrchestrator) -> list[dict[str, Any]]:
    """Extracts standardized position list across Paper and Live environments with leverage, margin & liquidation metrics."""
    positions_list: list[dict[str, Any]] = []
    lev = orchestrator.config.default_leverage

    if orchestrator.mode == ExecutionMode.PAPER and orchestrator.paper_simulator is not None:
        for pid, pos in orchestrator.paper_simulator._open_positions.items():
            spec = orchestrator.contract_specs.get(pos.symbol)
            cv = spec.contract_value if spec else 1.0
            native_val = round(pos.size * cv, 4)

            # Estimate liquidation price and distance
            if pos.side.lower() == "buy":
                liq_px = round(pos.entry_price * (1.0 - (1.0 / max(1, lev)) + 0.01), 4)
                dist_pct = (
                    round(max(0.0, (pos.mark_price - liq_px) / pos.mark_price * 100.0), 2)
                    if pos.mark_price > 0
                    else 0.0
                )
            else:
                liq_px = round(pos.entry_price * (1.0 + (1.0 / max(1, lev)) - 0.01), 4)
                dist_pct = (
                    round(max(0.0, (liq_px - pos.mark_price) / pos.mark_price * 100.0), 2)
                    if pos.mark_price > 0
                    else 0.0
                )

            positions_list.append({
                "product_id": pid,
                "symbol": pos.symbol,
                "side": pos.side.upper(),
                "size": pos.size,
                "contract_value": cv,
                "native_value": native_val,
                "leverage": lev,
                "entry_price": round(pos.entry_price, 4),
                "mark_price": round(pos.mark_price, 4),
                "unrealized_pnl": round(pos.unrealized_pnl, 2),
                "stop_loss_price": pos.stop_loss_price,
                "take_profit_price": pos.take_profit_price,
                "child_stop_loss_order_id": pos.child_stop_loss_order_id,
                "child_take_profit_order_id": pos.child_take_profit_order_id,
                "allocated_margin": round(pos.allocated_margin, 2),
                "liquidation_price": liq_px,
                "liquidation_distance_pct": dist_pct,
                "mode": "PAPER",
            })
    else:
        for p in orchestrator.state_manager.get_all_positions():
            child_sl = orchestrator.state_manager.get_child_stop_loss_order_id(p.product_id)
            spec = orchestrator.contract_specs.get(p.symbol)
            cv = spec.contract_value if spec else 1.0
            native_val = round(abs(p.size) * cv, 4)
            alloc_margin = (
                round((abs(p.size) * cv * p.entry_price) / max(1, lev), 2) if p.entry_price > 0 else 0.0
            )

            liq_px = round(p.liquidation_price, 4) if p.liquidation_price else None
            dist_pct = (
                round(abs(p.mark_price - liq_px) / p.mark_price * 100.0, 2)
                if (liq_px and p.mark_price > 0)
                else 0.0
            )

            positions_list.append({
                "product_id": p.product_id,
                "symbol": p.symbol,
                "side": "BUY" if p.size > 0 else "SELL",
                "size": abs(p.size),
                "contract_value": cv,
                "native_value": native_val,
                "leverage": lev,
                "entry_price": round(p.entry_price, 4),
                "mark_price": round(p.mark_price, 4),
                "unrealized_pnl": round(p.unrealized_pnl, 2),
                "stop_loss_price": None,
                "take_profit_price": None,
                "child_stop_loss_order_id": child_sl,
                "child_take_profit_order_id": None,
                "allocated_margin": alloc_margin,
                "liquidation_price": liq_px,
                "liquidation_distance_pct": dist_pct,
                "mode": "LIVE",
            })

    return positions_list


def get_all_orders_summary(orchestrator: TradingBotOrchestrator) -> list[dict[str, Any]]:
    """Extracts standardized resting pending stop breakout orders with TTL countdown."""
    orders_list: list[dict[str, Any]] = []

    def get_order_ttl(sym: str) -> tuple[int, int, int]:
        strategy = orchestrator.strategies.get(sym)
        intent = strategy.pending_order if strategy else None
        bars_elapsed = intent.bars_elapsed if intent else 0
        ttl_bars = intent.ttl_bars if intent else (strategy.config.max_pending_bars if strategy else 3)
        ttl_rem = max(0, ttl_bars - bars_elapsed)
        return bars_elapsed, ttl_bars, ttl_rem

    if orchestrator.mode == ExecutionMode.PAPER and orchestrator.paper_simulator is not None:
        for oid, o in orchestrator.paper_simulator._pending_orders.items():
            b_el, ttl_b, ttl_rem = get_order_ttl(o.symbol)
            orders_list.append({
                "order_id": oid,
                "client_order_id": o.client_order_id,
                "product_id": o.product_id,
                "symbol": o.symbol,
                "side": o.side.upper(),
                "size": o.size,
                "stop_price": round(o.stop_price, 4),
                "bracket_stop_loss_price": o.bracket_stop_loss_price,
                "bracket_take_profit_price": o.bracket_take_profit_price,
                "status": o.status.upper(),
                "bars_elapsed": b_el,
                "ttl_bars": ttl_b,
                "ttl_remaining": ttl_rem,
                "created_at": o.created_at,
                "mode": "PAPER",
            })
    elif orchestrator.execution_manager is not None:
        for ao in orchestrator.execution_manager.registry.active_orders():
            b_el, ttl_b, ttl_rem = get_order_ttl(ao.symbol)
            orders_list.append({
                "order_id": ao.id,
                "client_order_id": ao.client_order_id,
                "product_id": ao.product_id,
                "symbol": ao.symbol,
                "side": ao.side.upper(),
                "size": ao.size,
                "stop_price": ao.stop_price,
                "bracket_stop_loss_price": ao.bracket_stop_loss_price,
                "bracket_take_profit_price": ao.bracket_take_profit_price,
                "status": ao.state.upper(),
                "bars_elapsed": b_el,
                "ttl_bars": ttl_b,
                "ttl_remaining": ttl_rem,
                "created_at": ao.created_at,
                "mode": "LIVE",
            })

    return orders_list


def get_symbol_indicators_summary(
    orchestrator: TradingBotOrchestrator, symbol: str
) -> dict[str, Any] | None:
    """Extracts latest technical indicators and regime classifications for a symbol."""
    clean = symbol.strip().upper()
    strategy = orchestrator.strategies.get(clean)
    if not strategy:
        return None

    snap = strategy.indicator_pipeline.latest_snapshot
    session = orchestrator.public_ws.get_session_anchor(clean)

    if snap is None:
        return {
            "symbol": clean,
            "ema_20": None,
            "session_vwap": session.session_vwap if session else None,
            "atr_14": None,
            "atr_sma_20": None,
            "adx_14": None,
            "plus_di": None,
            "minus_di": None,
            "pdh": session.prior_day_high if session else None,
            "pdl": session.prior_day_low if session else None,
            "pdc": session.prior_day_close if session else None,
            "is_volatility_expanded": False,
            "trend_regime": "WARMING_UP",
        }

    # Determine regime string
    if snap.adx_14 >= strategy.config.adx_threshold and snap.is_volatility_expanded:
        regime = "TRENDING_BREAKOUT"
    elif snap.adx_14 < strategy.config.adx_threshold:
        regime = "CONSOLIDATION_SQUEEZE"
    else:
        regime = "NEUTRAL"

    return {
        "symbol": clean,
        "ema_20": round(snap.ema_20, 4),
        "session_vwap": round(snap.session_vwap, 4),
        "atr_14": round(snap.atr_14, 4),
        "atr_sma_20": round(snap.atr_sma_20, 4),
        "adx_14": round(snap.adx_14, 2),
        "plus_di": round(getattr(snap, "plus_di_14", getattr(snap, "plus_di", 0.0)), 2),
        "minus_di": round(getattr(snap, "minus_di_14", getattr(snap, "minus_di", 0.0)), 2),
        "pdh": round(snap.pdh, 4) if snap.pdh else None,
        "pdl": round(snap.pdl, 4) if snap.pdl else None,
        "pdc": round(snap.pdc, 4) if snap.pdc else None,
        "is_volatility_expanded": snap.is_volatility_expanded,
        "trend_regime": regime,
    }


def get_tickers_summary(orchestrator: TradingBotOrchestrator) -> dict[str, dict[str, Any]]:
    """Serializes latest mark price, bid, ask, spread, 24h delta, and status across target assets."""
    tickers: dict[str, dict[str, Any]] = {}
    enabled_set = getattr(orchestrator, "enabled_symbols", set(orchestrator.target_symbols))

    for sym in orchestrator.target_symbols:
        t = orchestrator.latest_tickers.get(sym)
        session = orchestrator.public_ws.get_session_anchor(sym)
        pdc = session.prior_day_close if session and session.prior_day_close else None

        if t:
            bid = t.best_bid if t.best_bid is not None else t.mark_price
            ask = t.best_ask if t.best_ask is not None else t.mark_price
            ts = (t.timestamp_usec / 1_000_000.0) if t.timestamp_usec else time.time()
            mark = t.mark_price
            vol = t.volume_24h or 0.0
            spread = t.spread_bps if t.spread_bps else 0.0
        else:
            bid = 0.0
            ask = 0.0
            ts = time.time()
            mark = 0.0
            vol = 0.0
            spread = 0.0

        if pdc and pdc > 0 and mark > 0:
            chg_pct = round(((mark - pdc) / pdc) * 100.0, 2)
            chg_usd = round(mark - pdc, 4)
        else:
            chg_pct = 0.0
            chg_usd = 0.0

        tickers[sym] = {
            "symbol": sym,
            "mark_price": round(mark, 4),
            "bid_price": round(bid, 4),
            "ask_price": round(ask, 4),
            "spread_bps": round(spread, 2),
            "change_24h_pct": chg_pct,
            "change_24h_usd": chg_usd,
            "volume_24h": round(vol, 2),
            "trading_enabled": sym in enabled_set,
            "timestamp": ts,
        }
    return tickers


def get_full_state_packet(orchestrator: TradingBotOrchestrator) -> dict[str, Any]:
    """Generates the unified 500ms broadcast state packet for clients."""
    uptime = max(0.0, time.time() - orchestrator.start_time) if orchestrator.start_time > 0 else 0.0

    if orchestrator.mode == ExecutionMode.PAPER and orchestrator.paper_simulator is not None:
        p_stats = orchestrator.paper_simulator.get_summary()
        balance = p_stats["balance"]
        equity = p_stats["equity"]
        realized_pnl = p_stats["realized_pnl"]
        unrealized_pnl = p_stats["unrealized_pnl"]
        win_rate = p_stats["win_rate_pct"]
        drawdown = p_stats["max_drawdown_pct"]
        total_trades = p_stats["total_trades"]
        winning_trades = p_stats["winning_trades"]
        available_margin = p_stats["available_margin"]
    else:
        balance = 10000.0
        equity = 10000.0
        realized_pnl = 0.0
        unrealized_pnl = 0.0
        win_rate = 0.0
        drawdown = 0.0
        total_trades = 0
        winning_trades = 0
        available_margin = 10000.0

    indicators = {}
    for sym in orchestrator.target_symbols:
        ind = get_symbol_indicators_summary(orchestrator, sym)
        if ind:
            indicators[sym] = ind

    return {
        "type": "state_update",
        "timestamp": time.time(),
        "status": {
            "is_running": orchestrator.is_running,
            "mode": orchestrator.mode.value.upper(),
            "uptime_seconds": round(uptime, 1),
            "uptime_human": format_uptime_human(uptime),
            "memory_mb": get_process_memory_mb(),
            "circuit_breaker_tripped": bool(orchestrator.risk_manager.is_circuit_breaker_tripped),
            "public_ws_connected": bool(orchestrator.public_ws.is_connected),
            "private_ws_connected": bool(orchestrator.private_ws.is_connected) if orchestrator.private_ws else False,
            "target_symbols": orchestrator.target_symbols,
            "enabled_symbols": list(getattr(orchestrator, "enabled_symbols", set(orchestrator.target_symbols))),
            "timeframe": orchestrator.timeframe,
            "has_auth_token": bool(orchestrator.config.dashboard_auth_token_str),
            "config": {
                "risk_pct": orchestrator.config.default_risk_pct,
                "leverage": orchestrator.config.default_leverage,
                "timeframe": orchestrator.timeframe,
                "max_drawdown_pct": orchestrator.risk_manager.config.daily_max_drawdown_pct,
                "max_pending_bars": next((s.config.max_pending_bars for s in orchestrator.strategies.values()), 3),
                "trailing_sl_enabled": next((s.config.trailing_stop for s in orchestrator.strategies.values()), True),
                "trailing_stop_atr_mult": next((s.config.trailing_stop_atr_mult for s in orchestrator.strategies.values()), 2.0),
                "adx_threshold": next((s.config.adx_threshold for s in orchestrator.strategies.values()), 20.0),
                "breakout_buffer_ticks": next((s.config.price_buffer_ticks for s in orchestrator.strategies.values()), 1.0),
            },
        },
        "collateral": {
            "balance": round(balance, 2),
            "equity": round(equity, 2),
            "realized_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "win_rate_pct": round(win_rate, 2),
            "max_drawdown_pct": round(drawdown, 2),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "available_margin": round(available_margin, 2),
        },
        "tickers": get_tickers_summary(orchestrator),
        "positions": get_all_positions_summary(orchestrator),
        "orders": get_all_orders_summary(orchestrator),
        "indicators": indicators,
        "logs": log_handler.get_logs(limit=25),
    }


# -----------------------------------------------------------------------------
# 5. FastAPI Application Factory
# -----------------------------------------------------------------------------

def create_app(orchestrator: TradingBotOrchestrator) -> FastAPI:
    """
    Constructs and configures the FastAPI Gateway application hooked into the
    running TradingBotOrchestrator instance.
    """
    ws_manager = WebSocketManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # Background WebSocket Broadcaster task (streams every 500ms)
        broadcaster_task = asyncio.create_task(
            _ws_broadcaster_worker(orchestrator, ws_manager),
            name="FastAPI-WebSocket-Broadcaster",
        )
        logger.info("FastAPI Web Gateway initialized and WebSocket broadcaster running")
        yield
        # Teardown
        broadcaster_task.cancel()
        try:
            await broadcaster_task
        except asyncio.CancelledError:
            logger.debug("FastAPI WebSocket broadcaster terminated cleanly")

    app = FastAPI(
        title="Delta-Bot Quantitative Gateway",
        description="REST & Real-Time WebSocket API for Delta-Bot Multi-Asset Trading Daemon",
        version="7.0.0",
        lifespan=lifespan,
    )

    # CORS configuration for secure access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -------------------------------------------------------------------------
    # Authentication Dependency (Phase 8 Guard)
    # -------------------------------------------------------------------------

    async def verify_dashboard_auth(
        x_dashboard_token: str | None = Header(default=None, alias="X-Dashboard-Token"),
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> bool:
        """
        Enforces token authentication using constant-time comparison.
        If DASHBOARD_AUTH_TOKEN is configured in BotConfig, rejects unauthorized requests with 401.
        If DASHBOARD_AUTH_TOKEN is not configured, allows unrestricted access (e.g. dev / testing).
        """
        expected_token = orchestrator.config.dashboard_auth_token_str
        if not expected_token:
            return True

        provided_token: str | None = None
        if x_dashboard_token:
            provided_token = x_dashboard_token.strip()
        elif authorization:
            parts = authorization.strip().split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                provided_token = parts[1].strip()
            elif len(parts) == 1:
                provided_token = parts[0].strip()

        if not provided_token or not secrets.compare_digest(provided_token, expected_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Invalid or missing dashboard authentication token.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return True

    # -------------------------------------------------------------------------
    # REST API Endpoints
    # -------------------------------------------------------------------------

    @app.get("/", response_class=FileResponse)
    @app.get("/dashboard", response_class=FileResponse)
    async def get_dashboard() -> Any:
        """Serves the single-page quantitative dashboard UI."""
        index_file = os.path.join(STATIC_DIR, "index.html")
        if os.path.isfile(index_file):
            return FileResponse(index_file, media_type="text/html")
        return HTMLResponse(content=get_embedded_dashboard_html(), status_code=status.HTTP_200_OK)

    @app.get("/api/status")
    async def get_bot_status() -> JSONResponse:
        """Returns high-level daemon telemetry, operational mode, and collateral."""
        packet = get_full_state_packet(orchestrator)
        return JSONResponse(
            content={
                "status": packet["status"],
                "collateral": packet["collateral"],
            }
        )

    @app.get("/api/positions")
    async def get_positions() -> JSONResponse:
        """Returns live active positions with entry, mark, and liquidation levels."""
        positions = get_all_positions_summary(orchestrator)
        return JSONResponse(content={"count": len(positions), "positions": positions})

    @app.get("/api/orders")
    async def get_orders() -> JSONResponse:
        """Returns resting pending stop breakout orders waiting for price trigger."""
        orders = get_all_orders_summary(orchestrator)
        return JSONResponse(content={"count": len(orders), "orders": orders})

    @app.get("/api/indicators/{symbol}")
    async def get_symbol_indicators(symbol: str) -> JSONResponse:
        """Returns technical metrics (EMA 20, VWAP, ATR, ADX, PDH, PDL) for specified asset."""
        clean = symbol.strip().upper()
        metrics = get_symbol_indicators_summary(orchestrator, clean)
        if metrics is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Symbol '{clean}' is not being tracked by the active orchestrator.",
            )
        return JSONResponse(content=metrics)

    @app.get("/api/logs")
    async def get_system_logs(
        limit: int = Query(default=50, ge=1, le=200),
        level: str | None = Query(default=None),
    ) -> JSONResponse:
        """Retrieves formatted entries from the in-memory RingBuffer log handler."""
        logs = log_handler.get_logs(limit=limit, level=level)
        return JSONResponse(content={"count": len(logs), "logs": logs})

    @app.post(
        "/api/control/start",
        response_model=ControlActionResponse,
        dependencies=[Depends(verify_dashboard_auth)],
    )
    async def control_start() -> ControlActionResponse:
        """Resumes trade signal evaluation and breakout order dispatching."""
        orchestrator.is_running = True
        logger.info("[CONTROL API] Trading loop STARTED via web interface")
        return ControlActionResponse(
            status="success",
            action="start",
            message="Trading signal evaluation resumed.",
        )

    @app.post(
        "/api/control/stop",
        response_model=ControlActionResponse,
        dependencies=[Depends(verify_dashboard_auth)],
    )
    async def control_stop() -> ControlActionResponse:
        """Gracefully pauses new order generation while preserving resting brackets."""
        orchestrator.is_running = False
        logger.warning("[CONTROL API] Trading loop STOPPED/PAUSED via web interface")
        return ControlActionResponse(
            status="success",
            action="stop",
            message="Trading loop paused. New orders will not be dispatched.",
        )

    @app.post(
        "/api/control/flatten",
        response_model=ControlActionResponse,
        dependencies=[Depends(verify_dashboard_auth)],
    )
    async def control_emergency_flatten(
        symbol: str | None = Query(default=None, description="Optional single asset symbol")
    ) -> ControlActionResponse:
        """Triggers emergency market-close of active positions and purges resting orders."""
        logger.warning(
            "[CONTROL API] Emergency Flatten requested via web interface (Target: %s)",
            symbol or "ALL",
        )
        if orchestrator.mode == ExecutionMode.LIVE and orchestrator.execution_manager is not None:
            try:
                res = await orchestrator.execution_manager.emergency_flatten(symbol=symbol)
                count = len(res.get("flattened_positions", []))
                return ControlActionResponse(
                    status="success",
                    action="flatten",
                    message=f"Live emergency flatten executed. Closed {count} positions.",
                )
            except Exception as e:
                logger.error("[CONTROL API] Live emergency flatten failed: %s", e)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Live emergency flatten failed: {e}",
                ) from e
        elif orchestrator.mode == ExecutionMode.PAPER and orchestrator.paper_simulator is not None:
            if symbol:
                clean_sym = symbol.strip().upper()
                spec = orchestrator.contract_specs.get(clean_sym)
                if spec:
                    closed = orchestrator.paper_simulator.close_position(spec.product_id)
                    count = 1 if closed else 0
                else:
                    count = 0
            else:
                flattened = orchestrator.paper_simulator.flatten_all()
                count = len(flattened)
            return ControlActionResponse(
                status="success",
                action="flatten",
                message=f"Paper positions flattened. Closed {count} positions.",
            )

        return ControlActionResponse(
            status="success",
            action="flatten",
            message="No active execution simulator found to flatten.",
        )

    @app.post(
        "/api/positions/{product_id}/close",
        response_model=ControlActionResponse,
        dependencies=[Depends(verify_dashboard_auth)],
    )
    async def close_position_by_product_id(product_id: int) -> ControlActionResponse:
        """Manually closes active position for a specific product ID."""
        logger.warning("[CONTROL API] Close position requested for product_id=%d", product_id)
        if orchestrator.mode == ExecutionMode.LIVE and orchestrator.execution_manager is not None:
            target_sym = None
            for sym, spec in orchestrator.contract_specs.items():
                if spec.product_id == product_id:
                    target_sym = sym
                    break
            if not target_sym:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Product ID {product_id} is not mapped to any active contract symbol.",
                )
            try:
                res = await orchestrator.execution_manager.emergency_flatten(symbol=target_sym)
                count = len(res.get("flattened_positions", []))
                return ControlActionResponse(
                    status="success",
                    action="close_position",
                    message=f"Live position for {target_sym} (product {product_id}) closed.",
                )
            except Exception as e:
                logger.error("[CONTROL API] Live position close failed: %s", e)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Live position close failed: {e}",
                ) from e
        elif orchestrator.mode == ExecutionMode.PAPER and orchestrator.paper_simulator is not None:
            closed = orchestrator.paper_simulator.close_position(product_id)
            if not closed:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No active paper position found for product_id={product_id}",
                )
            return ControlActionResponse(
                status="success",
                action="close_position",
                message=f"Paper position for product {product_id} closed.",
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Execution engine is not active or initialized.",
        )

    @app.post(
        "/api/control/mode",
        response_model=ControlActionResponse,
        dependencies=[Depends(verify_dashboard_auth)],
    )
    async def control_switch_mode(payload: ModeSwitchRequest) -> ControlActionResponse:
        """Switches execution mode between PAPER and LIVE with safety validation."""
        target_mode = payload.mode.strip().lower()
        if target_mode not in ("paper", "live"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid execution mode '{payload.mode}'. Must be 'paper' or 'live'.",
            )
        try:
            from delta_bot.exceptions import ConfigurationError
            new_mode = orchestrator.set_execution_mode(target_mode)
            logger.warning("[CONTROL API] Switched execution mode to %s", new_mode.value.upper())
            return ControlActionResponse(
                status="success",
                action="mode_switch",
                message=f"Switched execution mode to {new_mode.value.upper()}.",
            )
        except ConfigurationError as ce:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(ce),
            ) from ce
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to switch execution mode: {e}",
            ) from e

    @app.post(
        "/api/control/symbol_toggle",
        response_model=ControlActionResponse,
        dependencies=[Depends(verify_dashboard_auth)],
    )
    async def control_symbol_toggle(payload: SymbolToggleRequest) -> ControlActionResponse:
        """Enables or disables breakout signal evaluation for a specific symbol."""
        clean = payload.symbol.strip().upper()
        if clean not in orchestrator.target_symbols:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Symbol '{clean}' is not in active target contracts ({', '.join(orchestrator.target_symbols)}).",
            )
        is_active = orchestrator.set_symbol_trading_enabled(clean, payload.enabled)
        state_str = "ENABLED" if is_active else "PAUSED"
        logger.info("[CONTROL API] Symbol %s trading %s", clean, state_str)
        return ControlActionResponse(
            status="success",
            action="symbol_toggle",
            message=f"{clean} trading {state_str}.",
        )

    @app.post(
        "/api/orders/{order_id}/cancel",
        response_model=ControlActionResponse,
        dependencies=[Depends(verify_dashboard_auth)],
    )
    async def cancel_order_by_id(order_id: str) -> ControlActionResponse:
        """Manually cancels a resting pending breakout order."""
        logger.warning("[CONTROL API] Order cancellation requested for #%s", order_id)
        cancelled = False

        # 1. Check strategy pending orders
        for strat in orchestrator.strategies.values():
            if strat.pending_order:
                if (
                    strat.pending_order.intent_id == order_id
                    or str(strat.pending_order.exchange_order_id) == str(order_id)
                ):
                    strat.pending_order.cancel(reason="Manual cancellation via Web UI")
                    await orchestrator._handle_order_cancel(
                        strat.pending_order, reason="Manual Web UI Cancel"
                    )
                    strat.pending_order = None
                    cancelled = True

        # 2. Check paper simulator
        if orchestrator.mode == ExecutionMode.PAPER and orchestrator.paper_simulator is not None:
            try:
                numeric_id = int(order_id)
                for pid, o in list(orchestrator.paper_simulator._pending_orders.items()):
                    if o.order_id == numeric_id or str(o.order_id) == str(order_id):
                        orchestrator.paper_simulator.cancel_order(
                            product_id=o.product_id, order_id=numeric_id
                        )
                        cancelled = True
            except ValueError:
                logger.debug("Non-numeric order_id %s for paper cancellation", order_id)

        # 3. Check live execution manager
        if orchestrator.mode == ExecutionMode.LIVE and orchestrator.execution_manager is not None:
            try:
                numeric_id = int(order_id)
                for ao in orchestrator.execution_manager.registry.active_orders():
                    if ao.id == numeric_id:
                        await orchestrator.execution_manager.order_router.cancel_order(
                            product_id=ao.product_id, order_id=numeric_id
                        )
                        cancelled = True
            except ValueError:
                logger.debug("Non-numeric order_id %s for paper cancellation", order_id)

        if not cancelled:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pending order #{order_id} not found in active order registry.",
            )

        return ControlActionResponse(
            status="success",
            action="order_cancel",
            message=f"Pending order #{order_id} cancelled.",
        )

    @app.post(
        "/api/config",
        response_model=ControlActionResponse,
        dependencies=[Depends(verify_dashboard_auth)],
    )
    async def update_runtime_config(payload: ConfigUpdateRequest) -> ControlActionResponse:
        """Hot-reloads risk and strategy parameters across all active engines."""
        updated: list[str] = []

        if payload.timeframe is not None:
            orchestrator.timeframe = payload.timeframe
            orchestrator.config.timeframe = payload.timeframe
            for s in orchestrator.strategies.values():
                s.config = s.config.model_copy(update={"resolution": payload.timeframe})
            updated.append(f"timeframe={payload.timeframe}")

        if payload.risk_pct is not None:
            orchestrator.config.default_risk_pct = payload.risk_pct
            orchestrator.risk_manager.config = orchestrator.risk_manager.config.model_copy(
                update={"max_account_risk_pct": payload.risk_pct}
            )
            for s in orchestrator.strategies.values():
                s.config = s.config.model_copy(update={"risk_per_trade_pct": payload.risk_pct})
            updated.append(f"risk_pct={payload.risk_pct}%")

        if payload.leverage is not None:
            orchestrator.config.default_leverage = payload.leverage
            orchestrator.risk_manager.config = orchestrator.risk_manager.config.model_copy(
                update={"max_leverage": payload.leverage}
            )
            if orchestrator.execution_manager is not None:
                orchestrator.execution_manager.config = orchestrator.execution_manager.config.model_copy(
                    update={"leverage": payload.leverage}
                )
            updated.append(f"leverage={payload.leverage}x")

        if payload.trailing_sl_enabled is not None:
            for s in orchestrator.strategies.values():
                s.config = s.config.model_copy(update={"trailing_stop": payload.trailing_sl_enabled})
            updated.append(f"trailing_stop={payload.trailing_sl_enabled}")

        if payload.stop_loss_mode is not None:
            is_trailing = payload.stop_loss_mode.strip().lower() == "trailing"
            for s in orchestrator.strategies.values():
                s.config = s.config.model_copy(update={"trailing_stop": is_trailing})
            updated.append(f"stop_loss_mode={payload.stop_loss_mode}")

        if payload.trailing_stop_atr_mult is not None:
            orchestrator.config.trailing_atr_multiplier = payload.trailing_stop_atr_mult
            for s in orchestrator.strategies.values():
                s.config = s.config.model_copy(update={"trailing_stop_atr_mult": payload.trailing_stop_atr_mult})
            updated.append(f"trailing_stop_atr_mult={payload.trailing_stop_atr_mult}")

        if payload.breakout_entry_buffer is not None:
            orchestrator.config.breakout_entry_buffer = payload.breakout_entry_buffer
            for s in orchestrator.strategies.values():
                s.config = s.config.model_copy(update={"price_buffer_ticks": payload.breakout_entry_buffer})
            updated.append(f"buffer={payload.breakout_entry_buffer} ticks")

        if payload.min_adx_threshold is not None:
            orchestrator.config.min_adx_threshold = payload.min_adx_threshold
            for s in orchestrator.strategies.values():
                s.config = s.config.model_copy(update={"adx_threshold": payload.min_adx_threshold})
            updated.append(f"adx_threshold={payload.min_adx_threshold}")

        if payload.max_drawdown_pct is not None:
            orchestrator.config.max_drawdown_pct = payload.max_drawdown_pct
            orchestrator.risk_manager.config = orchestrator.risk_manager.config.model_copy(
                update={"daily_max_drawdown_pct": payload.max_drawdown_pct}
            )
            updated.append(f"max_drawdown_pct={payload.max_drawdown_pct}%")

        if payload.max_pending_bars is not None:
            orchestrator.config.order_ttl_bars = payload.max_pending_bars
            for s in orchestrator.strategies.values():
                s.config = s.config.model_copy(update={"max_pending_bars": payload.max_pending_bars})
            updated.append(f"max_pending_bars={payload.max_pending_bars}")

        logger.info("[CONTROL API] Hot-reloaded configuration: %s", ", ".join(updated))
        return ControlActionResponse(
            status="success",
            action="config_update",
            message=f"Configuration hot-reloaded: {', '.join(updated)}",
        )

    # -------------------------------------------------------------------------
    # WebSocket Broadcast Streamer (/ws/live)
    # -------------------------------------------------------------------------

    @app.websocket("/ws/live")
    async def websocket_live_stream(
        websocket: WebSocket,
        token: str | None = Query(default=None),
    ) -> None:
        """Streams real-time state packets every 500ms to connected dashboard clients."""
        expected_token = orchestrator.config.dashboard_auth_token_str
        if expected_token:
            provided_token = token.strip() if token else None
            if not provided_token:
                h_token = websocket.headers.get("x-dashboard-token")
                if h_token:
                    provided_token = h_token.strip()
                else:
                    auth_header = websocket.headers.get("authorization")
                    if auth_header:
                        parts = auth_header.strip().split()
                        if len(parts) == 2 and parts[0].lower() == "bearer":
                            provided_token = parts[1].strip()
                        elif len(parts) == 1:
                            provided_token = parts[0].strip()

            if not provided_token or not secrets.compare_digest(provided_token, expected_token):
                logger.warning("[WS AUTH] WebSocket connection rejected: invalid or missing authentication token")
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized")
                return

        await ws_manager.connect(websocket)
        try:
            # Send immediate initial state on connect
            initial_packet = get_full_state_packet(orchestrator)
            await websocket.send_text(json.dumps(initial_packet, default=str))

            # Maintain connection and listen for optional client ping messages
            while True:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "timestamp": time.time()}))
        except WebSocketDisconnect:
            await ws_manager.disconnect(websocket)
        except Exception as ws_err:
            logger.debug("WebSocket connection error: %s", ws_err)
            await ws_manager.disconnect(websocket)

    # Mount static assets directory
    if os.path.isdir(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


# -----------------------------------------------------------------------------
# 6. WebSocket Background Broadcaster Loop (500ms Interval)
# -----------------------------------------------------------------------------

async def _ws_broadcaster_worker(
    orchestrator: TradingBotOrchestrator, ws_manager: WebSocketManager
) -> None:
    """Dispatches full state packets to active WebSocket subscribers every 500ms."""
    while True:
        try:
            await asyncio.sleep(0.5)
            if ws_manager.active_connections:
                packet = get_full_state_packet(orchestrator)
                await ws_manager.broadcast(packet)
        except asyncio.CancelledError:
            break
        except Exception as b_err:
            logger.debug("Error in WebSocket broadcast loop: %s", b_err)


# -----------------------------------------------------------------------------
# 7. Embedded Quantitative Dashboard Single-Page HTML
# -----------------------------------------------------------------------------

def get_embedded_dashboard_html() -> str:
    """
    Returns high-performance responsive dark-themed dashboard SPA.
    Loads from delta_bot/static/index.html with fallback.
    """
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<title>Delta-Bot | Quantitative Terminal</title></head>"
        "<body><h1>DELTA-BOT Quantitative Terminal</h1></body></html>"
    )
