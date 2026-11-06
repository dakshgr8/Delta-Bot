"""
delta_bot.private_ws
~~~~~~~~~~~~~~~~~~~~

Asynchronous Private WebSocket User Feed Engine for Delta Exchange.
Manages:
- Authenticated WebSocket Connection with HMAC-SHA256 signing ("GET" + timestamp + "/live")
- Immediate authentication frame dispatch and confirmation handling
- Automatic re-authentication & channel re-subscription upon reconnect
- Private channels multiplexing: 'orders', 'positions', 'user_trades'
- Heartbeat ping/pong management and watchdog inactivity monitor
- Async event dispatching to TradingStateManager and ExecutionManager
"""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
import json
import logging
import random
import time
from typing import Any

import websockets
from websockets.asyncio.client import ClientConnection

from delta_bot.auth import DeltaAuthEngine
from delta_bot.config import BotConfig
from delta_bot.models import OrderUpdate, PositionInfo, UserTrade

logger = logging.getLogger("delta_bot.private_ws")


class PrivateWebSocketClient:
    """
    High-performance authenticated WebSocket client for Delta Exchange private user feeds.
    Connects to wss://socket.delta.exchange or wss://socket.india.delta.exchange, authenticates
    via HMAC-SHA256 handshake frame, and multiplexes real-time account telemetry:
    - 'orders': Order creation, amendments, resting status, and fill transitions
    - 'positions': Real-time position changes, entry price, liquidation price, margin
    - 'user_trades': Trade executions, fee charges, maker/taker classifications
    """

    def __init__(
        self,
        config: BotConfig,
        auth_engine: DeltaAuthEngine | None = None,
        symbols: list[str] | None = None,
    ) -> None:
        self.config: BotConfig = config
        self._auth_engine: DeltaAuthEngine = auth_engine or DeltaAuthEngine(
            api_key=config.delta_api_key,
            api_secret=config.delta_api_secret,
            user_agent=config.user_agent,
        )
        self.symbols: list[str] = [s.strip().upper() for s in (symbols or ["all"]) if s.strip()]

        # Connection & State
        self._ws_url: str = config.delta_ws_url
        self._ws: ClientConnection | None = None
        self._is_connected: bool = False
        self._is_authenticated: bool = False
        self._running: bool = False
        self._last_msg_epoch: float = 0.0

        # Background tasks
        self._conn_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._watchdog_task: asyncio.Task[None] | None = None

        # Event Callbacks
        self._order_callbacks: list[Callable[[OrderUpdate], Any]] = []
        self._position_callbacks: list[Callable[[PositionInfo], Any]] = []
        self._trade_callbacks: list[Callable[[UserTrade], Any]] = []
        self._auth_callbacks: list[Callable[[bool], Any]] = []

    # -------------------------------------------------------------------------
    # Public Properties & Event Registration
    # -------------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """True if the underlying WebSocket connection is open."""
        return self._is_connected and self._ws is not None

    @property
    def is_authenticated(self) -> bool:
        """True if Delta Exchange has validated credentials and authorized private feeds."""
        return self._is_authenticated

    def on_order(self, callback: Callable[[OrderUpdate], Any]) -> None:
        """Registers a listener for real-time order state updates."""
        self._order_callbacks.append(callback)

    def on_position(self, callback: Callable[[PositionInfo], Any]) -> None:
        """Registers a listener for real-time position updates."""
        self._position_callbacks.append(callback)

    def on_trade(self, callback: Callable[[UserTrade], Any]) -> None:
        """Registers a listener for real-time execution fills and fee reports."""
        self._trade_callbacks.append(callback)

    def on_auth_state(self, callback: Callable[[bool], Any]) -> None:
        """Registers a listener for authentication state transitions."""
        self._auth_callbacks.append(callback)

    # -------------------------------------------------------------------------
    # Authentication & Subscription Wire Frames
    # -------------------------------------------------------------------------

    def build_auth_payload(self) -> dict[str, Any]:
        """
        Constructs the Delta Exchange WebSocket authentication frame.
        Signature message format: METHOD + timestamp + path -> "GET" + timestamp + "/live"
        """
        return self._auth_engine.create_websocket_auth_payload()

    def build_subscription_payload(self) -> dict[str, Any]:
        """
        Constructs the subscription frame for private channels:
        - orders
        - positions
        - user_trades
        """
        target_symbols = self.symbols if self.symbols else ["all"]
        return {
            "type": "subscribe",
            "payload": {
                "channels": [
                    {"name": "orders", "symbols": target_symbols},
                    {"name": "positions", "symbols": target_symbols},
                    {"name": "user_trades", "symbols": target_symbols},
                ]
            },
        }

    # -------------------------------------------------------------------------
    # Lifecycle Management (Start / Stop)
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Starts the persistent private WebSocket connection worker."""
        if self._running:
            return
        self._running = True
        logger.info("Starting Private WebSocket Engine targeting [%s]...", self._ws_url)
        self._conn_task = asyncio.create_task(self._connection_loop(), name="delta_private_ws_conn")

    async def stop(self) -> None:
        """Gracefully disconnects and cancels all background workers."""
        self._running = False
        self._is_connected = False
        self._is_authenticated = False

        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception as exc:
                logger.debug("Exception closing private WebSocket: %s", exc)
            finally:
                self._ws = None

        if self._conn_task and not self._conn_task.done():
            self._conn_task.cancel()
            try:
                await self._conn_task
            except asyncio.CancelledError:
                logger.debug("Private WS connection task cancelled successfully")

        logger.info("Private WebSocket Engine stopped.")

    async def __aenter__(self) -> "PrivateWebSocketClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()

    # -------------------------------------------------------------------------
    # Internal Connection & Event Loops
    # -------------------------------------------------------------------------

    async def _send_json(self, data: dict[str, Any]) -> None:
        """Dispatches JSON message across active socket."""
        if self._ws is not None:
            await self._ws.send(json.dumps(data))

    async def _heartbeat_loop(self) -> None:
        """Periodically sends ping frames every 15-20 seconds to keep link alive."""
        interval = self.config.ws_heartbeat_interval_sec
        try:
            while self._running and self._is_connected:
                await asyncio.sleep(interval)
                if self._ws is not None:
                    try:
                        await self._send_json({"type": "ping"})
                        logger.debug("Dispatched private WebSocket heartbeat ping")
                    except Exception as exc:
                        logger.warning("Failed to send private WS ping: %s", exc)
                        break
        except asyncio.CancelledError:
            logger.debug("Private WS heartbeat task cancelled")

    async def _watchdog_loop(self) -> None:
        """Monitors inactivity and triggers reconnect if no frames arrive for > 35s."""
        timeout_threshold = 35.0
        try:
            while self._running and self._is_connected:
                await asyncio.sleep(5.0)
                elapsed = time.time() - self._last_msg_epoch
                if elapsed > timeout_threshold and self._last_msg_epoch > 0:
                    logger.warning(
                        "Private WS Watchdog alert: no frames received for %.1fs (threshold: %.1fs). Reconnecting...",
                        elapsed,
                        timeout_threshold,
                    )
                    if self._ws is not None:
                        await self._ws.close()
                    break
        except asyncio.CancelledError:
            logger.debug("Private WS watchdog task cancelled")

    async def _connection_loop(self) -> None:
        """Continuous connection loop with exponential backoff and auto-reauth."""
        attempt = 0
        max_attempts = self.config.ws_reconnect_max_attempts

        while self._running:
            attempt += 1
            if max_attempts > 0 and attempt > max_attempts:
                logger.error("Max private WebSocket reconnection attempts (%d) reached.", max_attempts)
                break

            try:
                logger.info("Connecting to Delta Private WebSocket [%s] (Attempt %d)...", self._ws_url, attempt)
                async with websockets.connect(
                    self._ws_url,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=5.0,
                    open_timeout=self.config.request_timeout_sec,
                ) as ws:
                    self._ws = ws
                    self._is_connected = True
                    self._last_msg_epoch = time.time()
                    attempt = 0

                    logger.info("✓ Private WebSocket transport connected. Dispatching auth handshake...")

                    # 1. Dispatch HMAC authentication frame
                    auth_frame = self.build_auth_payload()
                    await ws.send(json.dumps(auth_frame))
                    logger.info("Dispatched HMAC auth frame: API-Key=%s... Timestamp=%s",
                                auth_frame["payload"]["api-key"][:4], auth_frame["payload"]["timestamp"])

                    # 2. Launch heartbeat & watchdog tasks
                    self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="priv_ws_heartbeat")
                    self._watchdog_task = asyncio.create_task(self._watchdog_loop(), name="priv_ws_watchdog")

                    # 3. Message reception loop
                    async for raw_message in ws:
                        self._last_msg_epoch = time.time()
                        text = raw_message.decode("utf-8") if isinstance(raw_message, bytes) else raw_message
                        await self.handle_raw_message(text)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._is_connected = False
                self._is_authenticated = False
                self._ws = None

                if self._heartbeat_task and not self._heartbeat_task.done():
                    self._heartbeat_task.cancel()
                if self._watchdog_task and not self._watchdog_task.done():
                    self._watchdog_task.cancel()

                if not self._running:
                    break

                backoff_delay = min(
                    self.config.backoff_base_sec * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5),
                    self.config.backoff_max_sec,
                )
                logger.warning(
                    "Private WebSocket error: %s. Reconnecting in %.2fs (Attempt %d)...",
                    exc,
                    backoff_delay,
                    attempt,
                )
                await asyncio.sleep(backoff_delay)

    # -------------------------------------------------------------------------
    # Message Dispatching & Multiplexing
    # -------------------------------------------------------------------------

    async def handle_raw_message(self, text: str) -> None:
        """Parses and routes incoming WebSocket JSON messages."""
        try:
            msg = json.loads(text)
        except Exception as exc:
            logger.warning("Could not parse incoming private WS JSON frame: %s | text=%r", exc, text[:100])
            return

        msg_type = str(msg.get("type", "")).lower()
        channel = str(msg.get("channel", "") or msg_type).lower()

        # 1. Handle Ping / Pong Heartbeats
        if msg_type == "ping":
            await self._send_json({"type": "pong"})
            return
        elif msg_type == "pong":
            logger.debug("Received private WS pong heartbeat response")
            return

        # 2. Handle Authentication Handshake Confirmation
        if msg_type in ("auth", "key-auth"):
            success = bool(msg.get("success", False))
            if success or msg.get("status") in ("success", "authenticated"):
                self._is_authenticated = True
                logger.info("✓ Private WebSocket authenticated successfully with Delta Exchange.")

                # Subscribe to private user channels
                sub_payload = self.build_subscription_payload()
                await self._send_json(sub_payload)
                logger.info("Dispatched private channel subscriptions: %s", sub_payload["payload"]["channels"])

                await self._emit_all(self._auth_callbacks, True)
            else:
                self._is_authenticated = False
                err_msg = msg.get("message") or msg.get("error", "Unknown auth rejection")
                logger.error("Delta Private WebSocket authentication failed: %s", err_msg)
                await self._emit_all(self._auth_callbacks, False)
            return

        # 3. Route Channel Telemetry
        data_payload = msg.get("data") if "data" in msg else msg

        if channel == "orders" or msg_type == "orders":
            await self._handle_orders_channel(data_payload)
        elif channel == "positions" or msg_type == "positions":
            await self._handle_positions_channel(data_payload)
        elif channel == "user_trades" or msg_type == "user_trades":
            await self._handle_user_trades_channel(data_payload)
        elif msg_type == "subscriptions":
            logger.info("Private subscription confirmed by Delta: %s", msg)
        elif msg_type == "error":
            logger.error("Delta Private WS Error response: %s", msg)
        else:
            logger.debug("Unrouted private WS frame: %s", msg)

    async def _handle_orders_channel(self, payload: Any) -> None:
        """Parses order events and invokes registered listeners."""
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if isinstance(item, dict):
                order_update = OrderUpdate.from_api_dict(item)
                logger.debug(
                    "Private WS OrderUpdate: ID=%d | %s %s | Size=%d | State=%s | SL=%s | TP=%s",
                    order_update.id,
                    order_update.symbol,
                    order_update.side.upper(),
                    order_update.size,
                    order_update.state.upper(),
                    order_update.bracket_stop_loss_price,
                    order_update.bracket_take_profit_price,
                )
                await self._emit_all(self._order_callbacks, order_update)

    async def _handle_positions_channel(self, payload: Any) -> None:
        """Parses position events and invokes registered listeners."""
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if isinstance(item, dict):
                pos_info = PositionInfo.from_api_dict(item)
                logger.debug(
                    "Private WS PositionUpdate: %s (id=%d) | Size=%d | Entry=%.4f | Liq=%s | uPnL=%.4f",
                    pos_info.symbol,
                    pos_info.product_id,
                    pos_info.size,
                    pos_info.entry_price,
                    pos_info.liquidation_price,
                    pos_info.unrealized_pnl,
                )
                await self._emit_all(self._position_callbacks, pos_info)

    async def _handle_user_trades_channel(self, payload: Any) -> None:
        """Parses user fill reports and invokes registered listeners."""
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if isinstance(item, dict):
                trade = UserTrade.from_api_dict(item)
                logger.debug(
                    "Private WS UserTrade Fill: TradeID=%s | OrderID=%d | %s %s | Lots=%d | Price=%.4f | Fee=%.4f",
                    trade.trade_id,
                    trade.order_id,
                    trade.symbol,
                    trade.side.upper(),
                    trade.size,
                    trade.price,
                    trade.commission,
                )
                await self._emit_all(self._trade_callbacks, trade)

    async def _emit_all(self, callbacks: list[Callable[[Any], Any]], *args: Any) -> None:
        """Dispatches an event across registered callbacks (sync or async)."""
        for cb in callbacks:
            try:
                res = cb(*args)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as exc:
                logger.error("Error executing private WS callback: %s", exc)

    # -------------------------------------------------------------------------
    # Simulation & Testing Helpers
    # -------------------------------------------------------------------------

    async def simulate_auth_response(self, success: bool = True) -> None:
        """Simulates an authentication response frame for tests."""
        payload = {"type": "auth", "success": success}
        await self.handle_raw_message(json.dumps(payload))

    async def simulate_order_update(self, data: dict[str, Any] | OrderUpdate) -> None:
        """Directly injects an OrderUpdate into the client dispatcher."""
        if isinstance(data, OrderUpdate):
            await self._emit_all(self._order_callbacks, data)
        else:
            await self._handle_orders_channel(data)

    async def simulate_position_update(self, data: dict[str, Any] | PositionInfo) -> None:
        """Directly injects a PositionInfo into the client dispatcher."""
        if isinstance(data, PositionInfo):
            await self._emit_all(self._position_callbacks, data)
        else:
            await self._handle_positions_channel(data)

    async def simulate_user_trade(self, data: dict[str, Any] | UserTrade) -> None:
        """Directly injects a UserTrade fill into the client dispatcher."""
        if isinstance(data, UserTrade):
            await self._emit_all(self._trade_callbacks, data)
        else:
            await self._handle_user_trades_channel(data)
