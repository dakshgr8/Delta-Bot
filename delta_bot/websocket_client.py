"""
delta_bot.websocket_client
~~~~~~~~~~~~~~~~~~~~~~~~~~

High-performance asynchronous Market Data and WebSocket Engine for Delta Exchange.
Orchestrates:
- Connection Lifecycle Manager: Auto-reconnect with jittered backoff & 20s heartbeats
- Subscription Dispatcher: Dynamic channels/symbols subscriptions & state restoration
- Channel Multiplexer: Route messages (v2/ticker, candlestick_*, l2_orderbook) & sequence tracking
- Real-time Bar Builder & Candle Aggregator: Forming vs closed candle guarantees, 500-bar RingBuffer,
  and 00:00 UTC daily session anchor (PDH/PDL and Session VWAP)
- Async EventEmitter: Callback dispatching for strategy and execution modules
"""

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import json
import logging
import random
import time
from typing import Any, Awaitable, Callable

import websockets
from websockets.asyncio.client import ClientConnection

from delta_bot.config import BotConfig
from delta_bot.models import (
    Candle,
    CircularBarCache,
    OrderBook,
    SessionAnchorData,
    Ticker,
)

logger = logging.getLogger("delta_bot.websocket")


class SequenceTracker:
    """
    Detects packet drops and gaps by tracking monotonically increasing sequence numbers.
    Critical for Level 2 orderbook integrity and data reliability.
    """

    def __init__(self, strict: bool = True) -> None:
        self.strict: bool = strict
        self._last_seq: dict[str, int] = {}
        self.total_gaps_detected: int = 0
        self.total_packets_dropped: int = 0

    def track_sequence(self, stream_key: str, seq_no: int) -> tuple[bool, int]:
        """
        Records sequence number and identifies gaps.
        Returns: (has_gap, dropped_packets_count)
        """
        prev_seq = self._last_seq.get(stream_key)
        self._last_seq[stream_key] = seq_no

        if prev_seq is None:
            return False, 0

        if not self.strict:
            # In non-strict mode (default for live throttled L2 orderbook feeds),
            # matching engine sequence jumps are normal between periodic snapshots.
            # Only trigger gap/invalidation on sequence regressions (stale/out-of-order frames).
            if seq_no < prev_seq:
                logger.warning(
                    "Out-of-order sequence regression on [%s]: expected >= %d, got %d",
                    stream_key,
                    prev_seq,
                    seq_no,
                )
                self.total_gaps_detected += 1
                return True, 1
            return False, 0

        expected_seq = prev_seq + 1
        if seq_no > expected_seq:
            dropped = seq_no - expected_seq
            self.total_gaps_detected += 1
            self.total_packets_dropped += dropped
            logger.warning(
                "Sequence gap on [%s]: expected %d, got %d (Dropped %d packets)",
                stream_key,
                expected_seq,
                seq_no,
                dropped,
            )
            return True, dropped

        return False, 0

    def reset(self, stream_key: str | None = None) -> None:
        """Clears sequence history for a stream or all streams."""
        if stream_key:
            self._last_seq.pop(stream_key, None)
        else:
            self._last_seq.clear()


class SubscriptionDispatcher:
    """
    Manages channel and symbol subscriptions.
    Builds canonical Delta subscription payloads and handles re-subscription on reconnect.
    """

    def __init__(self) -> None:
        # Map channel_name -> set of symbols
        self._subscriptions: dict[str, set[str]] = defaultdict(set)

    def add_subscription(self, channel: str, symbols: list[str]) -> bool:
        """Registers symbols for a channel. Returns True if any new symbols added."""
        clean_symbols = [s.strip().upper() for s in symbols if s.strip()]
        before_count = len(self._subscriptions[channel])
        self._subscriptions[channel].update(clean_symbols)
        return len(self._subscriptions[channel]) > before_count

    def remove_subscription(self, channel: str, symbols: list[str]) -> bool:
        """Unregisters symbols from a channel."""
        clean_symbols = [s.strip().upper() for s in symbols if s.strip()]
        before_count = len(self._subscriptions[channel])
        self._subscriptions[channel].difference_update(clean_symbols)
        if not self._subscriptions[channel]:
            self._subscriptions.pop(channel, None)
        return len(self._subscriptions.get(channel, set())) < before_count

    def build_subscription_payload(
        self,
        channels: dict[str, set[str]] | None = None,
        action: str = "subscribe",
    ) -> str:
        """
        Builds standard Delta Exchange subscription frame:
        {
            "type": "subscribe",
            "payload": {
                "channels": [
                    {"name": "v2/ticker", "symbols": ["BTCUSDT"]},
                    {"name": "candlestick_1m", "symbols": ["BTCUSDT"]}
                ]
            }
        }
        """
        target = channels if channels is not None else self._subscriptions
        channel_items: list[dict[str, Any]] = []

        for ch_name, sym_set in target.items():
            if sym_set:
                channel_items.append({
                    "name": ch_name,
                    "symbols": sorted(list(sym_set)),
                })

        return json.dumps({
            "type": action,
            "payload": {
                "channels": channel_items,
            },
        })

    def get_all_subscriptions(self) -> dict[str, set[str]]:
        """Returns deep copy of current active subscriptions."""
        return {k: set(v) for k, v in self._subscriptions.items()}

    @property
    def has_subscriptions(self) -> bool:
        """True if at least one active channel subscription exists."""
        return any(bool(v) for v in self._subscriptions.values())


class CandleAggregator:
    """
    Real-time Bar Builder & Indicator Data Structures:
    1. In-memory RingBuffer / Circular Bar Cache (retaining the last 500 closed bars per symbol).
    2. Strict state separation: Strategy signals MUST only trigger on confirmed closed bars.
    3. Daily Session Anchor: Tracks 00:00 UTC session rollover, PDH/PDL, and Intraday Session VWAP.
    """

    def __init__(
        self,
        capacity: int = 500,
        on_candle_close: Callable[[str, Candle], Awaitable[None]] | None = None,
        on_candle_update: Callable[[str, Candle], Awaitable[None]] | None = None,
        on_session_rollover: Callable[[str, SessionAnchorData], Awaitable[None]] | None = None,
    ) -> None:
        self.capacity: int = capacity
        self._bar_caches: dict[tuple[str, str], CircularBarCache] = {}
        self._session_anchors: dict[str, SessionAnchorData] = {}
        self._on_candle_close_cb = on_candle_close
        self._on_candle_update_cb = on_candle_update
        self._on_session_rollover_cb = on_session_rollover

    def get_bar_cache(self, symbol: str, resolution: str = "1m") -> CircularBarCache:
        """Retrieves or creates circular bar cache for a symbol and resolution."""
        sym = symbol.strip().upper()
        key = (sym, resolution)
        if key in self._bar_caches:
            return self._bar_caches[key]

        # Check USD/USDT alias
        if sym.endswith("USD") and not sym.endswith("USDT"):
            alt_key = (sym[:-3] + "USDT", resolution)
            if alt_key in self._bar_caches:
                return self._bar_caches[alt_key]
        elif sym.endswith("USDT"):
            alt_key = (sym[:-4] + "USD", resolution)
            if alt_key in self._bar_caches:
                return self._bar_caches[alt_key]

        self._bar_caches[key] = CircularBarCache(
            symbol=sym,
            resolution=resolution,
            capacity=self.capacity,
        )
        return self._bar_caches[key]

    def seed_historical_bars(self, symbol: str, resolution: str, candles: list[Candle]) -> int:
        """
        Seeds the CircularBarCache and initializes Daily Session Anchor (00:00 UTC) with historical bars.
        Guarantees that 00:00 UTC rollover, PDH/PDL, and Session VWAP are populated before streaming.
        """
        clean_symbol = symbol.strip().upper()
        cache = self.get_bar_cache(clean_symbol, resolution)
        count = cache.seed_historical_bars(candles)
        anchor = self.get_session_anchor(clean_symbol)
        for c in candles:
            anchor = self._check_session_rollover(clean_symbol, c.candle_start_epoch_s, c.open)
            anchor.update(c)

        # Cross-register alias cache and anchor to ensure synchronized access
        alt_sym = None
        if clean_symbol.endswith("USD") and not clean_symbol.endswith("USDT"):
            alt_sym = clean_symbol[:-3] + "USDT"
        elif clean_symbol.endswith("USDT"):
            alt_sym = clean_symbol[:-4] + "USD"

        if alt_sym:
            alt_key = (alt_sym, resolution)
            if alt_key not in self._bar_caches:
                self._bar_caches[alt_key] = cache
            if alt_sym not in self._session_anchors:
                self._session_anchors[alt_sym] = anchor

        return count

    def get_session_anchor(self, symbol: str) -> SessionAnchorData:
        """Retrieves or initializes Daily Session Anchor (00:00 UTC) for a symbol."""
        clean_symbol = symbol.strip().upper()
        if clean_symbol in self._session_anchors:
            return self._session_anchors[clean_symbol]

        # Check USD/USDT alias
        if clean_symbol.endswith("USD") and not clean_symbol.endswith("USDT"):
            alt = clean_symbol[:-3] + "USDT"
            if alt in self._session_anchors:
                return self._session_anchors[alt]
        elif clean_symbol.endswith("USDT"):
            alt = clean_symbol[:-4] + "USD"
            if alt in self._session_anchors:
                return self._session_anchors[alt]

        now_utc = datetime.now(timezone.utc)
        session_date = now_utc.strftime("%Y-%m-%d")
        # 00:00:00 UTC epoch seconds
        midnight = datetime(now_utc.year, now_utc.month, now_utc.day, tzinfo=timezone.utc)
        start_epoch_s = midnight.timestamp()

        self._session_anchors[clean_symbol] = SessionAnchorData(
            symbol=clean_symbol,
            session_date=session_date,
            session_start_epoch_s=start_epoch_s,
        )
        return self._session_anchors[clean_symbol]

    def _check_session_rollover(
        self,
        symbol: str,
        candle_start_epoch_s: float,
        current_price: float,
    ) -> SessionAnchorData:
        """
        Detects if candle has crossed 00:00:00 UTC daily session boundary.
        If rollover detected: captures PDH/PDL/PDC, resets intraday accumulators, and triggers callback.
        """
        anchor = self.get_session_anchor(symbol)
        candle_dt = datetime.fromtimestamp(candle_start_epoch_s, tz=timezone.utc)
        candle_date_str = candle_dt.strftime("%Y-%m-%d")

        if candle_date_str != anchor.session_date:
            midnight = datetime(candle_dt.year, candle_dt.month, candle_dt.day, tzinfo=timezone.utc)
            next_start_s = midnight.timestamp()

            logger.info(
                "Daily session rollover triggered on [%s] at 00:00 UTC: %s -> %s (PDH=%.2f, PDL=%.2f, PDC=%.2f)",
                symbol,
                anchor.session_date,
                candle_date_str,
                anchor.session_high,
                anchor.session_low,
                anchor.session_close,
            )
            new_anchor = anchor.rollover(
                next_session_date=candle_date_str,
                next_session_start_s=next_start_s,
                next_open=current_price,
            )
            self._session_anchors[symbol] = new_anchor
            return new_anchor

        return anchor

    async def process_candle_update(self, raw_data: dict[str, Any]) -> None:
        """
        Core Bar Building engine:
        - Ingests streaming candle ticks.
        - Detects when prior bar has completed and transitioned to a new bar start time.
        - Confirms closed candles into CircularBarCache and fires on_candle_close.
        - Updates real-time Session VWAP and detects 00:00 UTC session rollover.
        """
        incoming_candle = Candle.from_api_dict(raw_data, is_closed=False)
        symbol = incoming_candle.symbol.upper()
        resolution = incoming_candle.resolution
        cache = self.get_bar_cache(symbol, resolution)

        current_open = cache.open_candle

        # Check if incoming candle has advanced past the current open candle
        if current_open is not None and incoming_candle.candle_start_time_usec > current_open.candle_start_time_usec:
            # PREVIOUS BAR IS CONFIRMED CLOSED!
            closed_bar = Candle(
                symbol=current_open.symbol,
                resolution=current_open.resolution,
                open=current_open.open,
                high=current_open.high,
                low=current_open.low,
                close=current_open.close,
                volume=current_open.volume,
                candle_start_time_usec=current_open.candle_start_time_usec,
                is_closed=True,
                timestamp_usec=raw_data.get("timestamp"),
            )

            # Store in ring buffer
            cache.append(closed_bar)
            logger.debug(
                "Confirmed closed bar [%s %s]: O=%.2f H=%.2f L=%.2f C=%.2f V=%.2f",
                symbol,
                resolution,
                closed_bar.open,
                closed_bar.high,
                closed_bar.low,
                closed_bar.close,
                closed_bar.volume,
            )

            # Session VWAP and Rollover logic
            anchor = self._check_session_rollover(
                symbol=symbol,
                candle_start_epoch_s=incoming_candle.candle_start_epoch_s,
                current_price=incoming_candle.open,
            )
            anchor.update(closed_bar)

            # Dispatch session rollover event if date transitioned
            if self._on_session_rollover_cb and anchor.session_date != anchor.session_date:
                await self._on_session_rollover_cb(symbol, anchor)

            # Fire on_candle_close callback
            if self._on_candle_close_cb:
                await self._on_candle_close_cb(symbol, closed_bar)

            # Start new forming bar
            cache.set_open_bar(incoming_candle)
            if self._on_candle_update_cb:
                await self._on_candle_update_cb(symbol, incoming_candle)

        else:
            # Still in current forming bar interval
            cache.set_open_bar(incoming_candle)
            anchor = self.get_session_anchor(symbol)
            anchor.update(incoming_candle)

            if self._on_candle_update_cb:
                await self._on_candle_update_cb(symbol, incoming_candle)


class ChannelMultiplexer:
    """
    Demultiplexes incoming raw WebSocket messages and delegates them to dedicated handlers:
    - Heartbeat / Pong
    - Subscriptions acknowledgment
    - v2/ticker -> on_ticker
    - candlestick_* -> CandleAggregator
    - l2_orderbook -> SequenceTracker & on_orderbook
    """

    def __init__(
        self,
        aggregator: CandleAggregator,
        sequence_tracker: SequenceTracker,
        on_ticker: Callable[[str, Ticker], Awaitable[None]] | None = None,
        on_orderbook: Callable[[str, OrderBook], Awaitable[None]] | None = None,
        on_orderbook_resync: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.aggregator: CandleAggregator = aggregator
        self.sequence_tracker: SequenceTracker = sequence_tracker
        self.on_ticker_cb = on_ticker
        self.on_orderbook_cb = on_orderbook
        self.on_orderbook_resync_cb = on_orderbook_resync

    async def handle_message(self, raw_text: str) -> None:
        """Parses and multiplexes single incoming WebSocket JSON frame."""
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as ex:
            logger.warning("Failed to decode WebSocket JSON message: %s (Raw: %r)", ex, raw_text[:100])
            return

        msg_type = data.get("type")

        # 1. Heartbeat or Pong keep-alive frames
        if msg_type in ("heartbeat", "pong"):
            logger.debug("Received keep-alive message [%s]", msg_type)
            return

        # 2. Subscription confirmation
        if msg_type == "subscriptions":
            logger.info("Exchange confirmed active channel subscriptions: %s", data.get("channels"))
            return

        # 3. Real-time Tickers (/v2/ticker)
        if msg_type == "v2/ticker":
            ticker = Ticker.from_api_dict(data)
            if self.on_ticker_cb:
                await self.on_ticker_cb(ticker.symbol, ticker)
            return

        # 4. Candlesticks (e.g., candlestick_1m, candlestick_5m)
        if isinstance(msg_type, str) and msg_type.startswith("candlestick_"):
            await self.aggregator.process_candle_update(data)
            return

        # 5. Level 2 Orderbook (l2_orderbook)
        if msg_type == "l2_orderbook":
            symbol = str(data.get("symbol", "")).upper()
            action = data.get("action")
            seq_no = data.get("last_sequence_no")
            has_gap = False
            dropped = 0
            if action == "snapshot":
                self.sequence_tracker.reset(f"ob:{symbol}")
                if seq_no is not None:
                    self.sequence_tracker.track_sequence(f"ob:{symbol}", int(seq_no))
            elif seq_no is not None:
                has_gap, dropped = self.sequence_tracker.track_sequence(f"ob:{symbol}", int(seq_no))

            orderbook = OrderBook.from_api_dict(data)
            if has_gap:
                orderbook = orderbook.copy_invalidated()
                logger.warning(
                    "Orderbook invalidated for %s due to sequence gap (Dropped %d packets). Requesting fresh snapshot...",
                    symbol,
                    dropped,
                )
                if self.on_orderbook_resync_cb:
                    await self.on_orderbook_resync_cb(symbol)

            if self.on_orderbook_cb:
                await self.on_orderbook_cb(symbol, orderbook)
            return

        # 6. Exchange error or notification
        if msg_type == "error":
            logger.error("Delta Exchange WebSocket error frame: %s", data)
            return

        logger.debug("Unhandled WebSocket message type [%s]", msg_type)


class ConnectionLifecycleManager:
    """
    Manages WebSocket connection lifecycle, keep-alive heartbeats, and auto-reconnection.
    Features:
    - 20s heartbeat task sending ping frames and enabling exchange heartbeats
    - Watchdog timer: resets connection if no message received within 35 seconds
    - Seamless exponential backoff with full jitter on disconnection
    - Automatic channel re-subscription after reconnecting
    """

    def __init__(
        self,
        config: BotConfig,
        dispatcher: SubscriptionDispatcher,
        multiplexer: ChannelMultiplexer,
        on_disconnect: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._config: BotConfig = config
        self._dispatcher: SubscriptionDispatcher = dispatcher
        self._multiplexer: ChannelMultiplexer = multiplexer
        self._on_disconnect_cb = on_disconnect

        self._ws_url: str = config.delta_ws_url
        self._ws: ClientConnection | None = None
        self._running: bool = False
        self._is_connected: bool = False
        self._last_msg_epoch: float = 0.0

        self._read_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._watchdog_task: asyncio.Task[None] | None = None

    @property
    def is_connected(self) -> bool:
        """Current WebSocket connection state."""
        return self._is_connected and self._ws is not None

    async def start(self) -> None:
        """Starts connection management loop in background task."""
        if self._running:
            return
        self._running = True
        self._read_task = asyncio.create_task(self._connection_loop(), name="ws_connection_loop")
        logger.info("Started WebSocket connection lifecycle manager targeting %s", self._ws_url)

    async def stop(self) -> None:
        """Gracefully disconnects and stops background tasks."""
        self._running = False
        self._is_connected = False

        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception as close_err:
                logger.debug("Socket close exception ignored during shutdown: %s", close_err)
            self._ws = None

        if self._read_task and not self._read_task.done():
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                logger.debug("Read loop cancelled during shutdown")

        logger.info("Stopped WebSocket connection lifecycle manager.")

    async def _send_json(self, data: dict[str, Any]) -> None:
        """Sends raw JSON payload over active socket."""
        if self._ws is not None:
            await self._ws.send(json.dumps(data))

    async def _heartbeat_loop(self) -> None:
        """Periodically sends ping frames to Delta Exchange to keep link alive."""
        interval = self._config.ws_heartbeat_interval_sec
        logger.debug("Started WebSocket heartbeat task (interval: %.1fs)", interval)
        try:
            while self._running and self._is_connected:
                await asyncio.sleep(interval)
                if self._ws is not None:
                    try:
                        await self._send_json({"type": "ping"})
                        logger.debug("Dispatched heartbeat ping frame")
                    except Exception as ex:
                        logger.warning("Failed to send ping heartbeat: %s", ex)
                        break
        except asyncio.CancelledError:
            logger.debug("Heartbeat task cancelled")

    async def _watchdog_loop(self) -> None:
        """
        Monitors socket inactivity.
        If no message has been received for > 35 seconds, triggers socket reset.
        """
        timeout_threshold = 35.0
        try:
            while self._running and self._is_connected:
                await asyncio.sleep(5.0)
                elapsed = time.time() - self._last_msg_epoch
                if elapsed > timeout_threshold and self._last_msg_epoch > 0:
                    logger.warning(
                        "Watchdog alert: no WebSocket frames received for %.1fs (Threshold: %.1fs). Reconnecting...",
                        elapsed,
                        timeout_threshold,
                    )
                    if self._ws is not None:
                        await self._ws.close()
                    break
        except asyncio.CancelledError:
            logger.debug("Watchdog task cancelled")

    async def _connection_loop(self) -> None:
        """Continuous connection and auto-reconnect worker."""
        attempt = 0
        max_attempts = self._config.ws_reconnect_max_attempts

        while self._running:
            attempt += 1
            if max_attempts > 0 and attempt > max_attempts:
                logger.error("Max WebSocket reconnection attempts (%d) reached. Aborting.", max_attempts)
                break

            try:
                logger.info("Connecting to Delta WebSocket [%s] (Attempt %d)...", self._ws_url, attempt)
                async with websockets.connect(
                    self._ws_url,
                    ping_interval=None,  # Handled via explicit application-layer heartbeat
                    ping_timeout=None,
                    close_timeout=5.0,
                    open_timeout=self._config.request_timeout_sec,
                ) as ws:
                    self._ws = ws
                    self._is_connected = True
                    self._last_msg_epoch = time.time()
                    attempt = 0  # Reset backoff on successful handshake
                    logger.info("✓ Connected to Delta WebSocket successfully.")

                    # 1. Enable server-side heartbeats
                    await self._send_json({"type": "enable_heartbeat"})

                    # 2. Re-subscribe all active channels
                    if self._dispatcher.has_subscriptions:
                        sub_frame = self._dispatcher.build_subscription_payload()
                        await ws.send(sub_frame)
                        logger.info("Dispatched active channel subscriptions on reconnect.")

                    # 3. Launch background heartbeat and watchdog tasks
                    self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="ws_heartbeat")
                    self._watchdog_task = asyncio.create_task(self._watchdog_loop(), name="ws_watchdog")

                    # 4. Message reception loop
                    async for raw_message in ws:
                        self._last_msg_epoch = time.time()
                        if isinstance(raw_message, bytes):
                            text = raw_message.decode("utf-8")
                        else:
                            text = raw_message
                        await self._multiplexer.handle_message(text)

            except asyncio.CancelledError:
                break
            except Exception as ex:
                reason = f"{type(ex).__name__}: {ex}"
                logger.warning("WebSocket link dropped: %s", reason)
                if self._on_disconnect_cb:
                    try:
                        await self._on_disconnect_cb(reason)
                    except Exception as cb_err:
                        logger.warning("Error in on_disconnect callback: %s", cb_err)

            finally:
                self._is_connected = False
                if self._heartbeat_task and not self._heartbeat_task.done():
                    self._heartbeat_task.cancel()
                if self._watchdog_task and not self._watchdog_task.done():
                    self._watchdog_task.cancel()

            if self._running:
                # Exponential backoff with full jitter
                base_delay = min(30.0, self._config.backoff_base_sec * (2**attempt))
                jittered_delay = base_delay * random.uniform(0.7, 1.3)
                logger.info("Reconnecting WebSocket in %.2fs (Attempt %d)...", jittered_delay, attempt)
                await asyncio.sleep(jittered_delay)


class DeltaWebSocketClient:
    """
    High-Performance Unified Market Data & WebSocket Client.
    Provides async event callbacks, channel subscriptions, and access to the
    500-bar CircularBarCache and Daily Session Anchor.
    """

    TARGET_BASE_SYMBOLS = ("BTC", "ETH", "SOL", "XRP")

    def __init__(self, config: BotConfig | None = None) -> None:
        self._config: BotConfig = config or BotConfig()
        self._sequence_tracker: SequenceTracker = SequenceTracker(
            strict=self._config.strict_orderbook_sequence_check
        )
        self._dispatcher: SubscriptionDispatcher = SubscriptionDispatcher()

        # Callbacks
        self._on_candle_close_cb: Callable[[str, Candle], Awaitable[None]] | None = None
        self._on_candle_update_cb: Callable[[str, Candle], Awaitable[None]] | None = None
        self._on_ticker_cb: Callable[[str, Ticker], Awaitable[None]] | None = None
        self._on_orderbook_cb: Callable[[str, OrderBook], Awaitable[None]] | None = None
        self._on_disconnect_cb: Callable[[str], Awaitable[None]] | None = None
        self._on_session_rollover_cb: Callable[[str, SessionAnchorData], Awaitable[None]] | None = None

        self._aggregator: CandleAggregator = CandleAggregator(
            capacity=self._config.bar_cache_capacity,
            on_candle_close=self._dispatch_candle_close,
            on_candle_update=self._dispatch_candle_update,
            on_session_rollover=self._dispatch_session_rollover,
        )

        self._multiplexer: ChannelMultiplexer = ChannelMultiplexer(
            aggregator=self._aggregator,
            sequence_tracker=self._sequence_tracker,
            on_ticker=self._dispatch_ticker,
            on_orderbook=self._dispatch_orderbook,
            on_orderbook_resync=self._handle_orderbook_resync,
        )

        self._lifecycle: ConnectionLifecycleManager = ConnectionLifecycleManager(
            config=self._config,
            dispatcher=self._dispatcher,
            multiplexer=self._multiplexer,
            on_disconnect=self._dispatch_disconnect,
        )

    # Callback Registration
    def on_candle_close(self, callback: Callable[[str, Candle], Awaitable[None]]) -> None:
        """Registers async callback: on_candle_close(symbol: str, candle: Candle)."""
        self._on_candle_close_cb = callback

    def on_candle_update(self, callback: Callable[[str, Candle], Awaitable[None]]) -> None:
        """Registers async callback: on_candle_update(symbol: str, candle: Candle)."""
        self._on_candle_update_cb = callback

    def on_ticker(self, callback: Callable[[str, Ticker], Awaitable[None]]) -> None:
        """Registers async callback: on_ticker(symbol: str, ticker: Ticker)."""
        self._on_ticker_cb = callback

    def on_orderbook(self, callback: Callable[[str, OrderBook], Awaitable[None]]) -> None:
        """Registers async callback: on_orderbook(symbol: str, orderbook: OrderBook)."""
        self._on_orderbook_cb = callback

    def on_disconnect(self, callback: Callable[[str], Awaitable[None]]) -> None:
        """Registers async callback: on_disconnect(reason: str)."""
        self._on_disconnect_cb = callback

    def on_session_rollover(self, callback: Callable[[str, SessionAnchorData], Awaitable[None]]) -> None:
        """Registers async callback: on_session_rollover(symbol: str, session: SessionAnchorData)."""
        self._on_session_rollover_cb = callback

    # Internal Dispatchers
    async def _dispatch_candle_close(self, symbol: str, candle: Candle) -> None:
        if self._on_candle_close_cb:
            try:
                await self._on_candle_close_cb(symbol, candle)
            except Exception as ex:
                logger.error("Exception in on_candle_close callback: %s", ex, exc_info=True)

    async def _dispatch_candle_update(self, symbol: str, candle: Candle) -> None:
        if self._on_candle_update_cb:
            try:
                await self._on_candle_update_cb(symbol, candle)
            except Exception as ex:
                logger.error("Exception in on_candle_update callback: %s", ex, exc_info=True)

    async def _dispatch_ticker(self, symbol: str, ticker: Ticker) -> None:
        if self._on_ticker_cb:
            try:
                await self._on_ticker_cb(symbol, ticker)
            except Exception as ex:
                logger.error("Exception in on_ticker callback: %s", ex, exc_info=True)

    async def _dispatch_orderbook(self, symbol: str, orderbook: OrderBook) -> None:
        if self._on_orderbook_cb:
            try:
                await self._on_orderbook_cb(symbol, orderbook)
            except Exception as ex:
                logger.error("Exception in on_orderbook callback: %s", ex, exc_info=True)

    async def _dispatch_disconnect(self, reason: str) -> None:
        if self._on_disconnect_cb:
            try:
                await self._on_disconnect_cb(reason)
            except Exception as ex:
                logger.error("Exception in on_disconnect callback: %s", ex, exc_info=True)

    async def _dispatch_session_rollover(self, symbol: str, session: SessionAnchorData) -> None:
        if self._on_session_rollover_cb:
            try:
                await self._on_session_rollover_cb(symbol, session)
            except Exception as ex:
                logger.error("Exception in on_session_rollover callback: %s", ex, exc_info=True)

    # Lifecycle & Subscription Operations
    async def connect(self) -> None:
        """Connects to the WebSocket feed and starts lifecycle tasks."""
        await self._lifecycle.start()
        # Wait up to 5 seconds for connection to establish
        for _ in range(50):
            if self._lifecycle.is_connected:
                break
            await asyncio.sleep(0.1)

    async def disconnect(self) -> None:
        """Disconnects and releases resources."""
        await self._lifecycle.stop()

    @property
    def is_connected(self) -> bool:
        """Returns True if the underlying socket is active."""
        return self._lifecycle.is_connected

    async def subscribe_tickers(self, symbols: list[str]) -> None:
        """
        Subscribes to 'v2/ticker' for specified symbols.
        Streams mark price, index price, best bid/ask, and 24h stats.
        """
        channel = "v2/ticker"
        self._dispatcher.add_subscription(channel, symbols)
        if self.is_connected:
            payload = self._dispatcher.build_subscription_payload({channel: set(symbols)})
            await self._lifecycle._send_json(json.loads(payload))

    async def subscribe_candles(self, symbols: list[str], resolutions: list[str] | None = None) -> None:
        """
        Subscribes to candle feeds ('candlestick_1m', 'candlestick_5m', etc.) for symbols.
        Feeds the real-time Bar Builder and RingBuffer.
        """
        resolutions = resolutions or ["1m"]
        for res in resolutions:
            channel = f"candlestick_{res}"
            self._dispatcher.add_subscription(channel, symbols)
            if self.is_connected:
                payload = self._dispatcher.build_subscription_payload({channel: set(symbols)})
                await self._lifecycle._send_json(json.loads(payload))

    async def subscribe_orderbook(self, symbols: list[str]) -> None:
        """
        Subscribes to 'l2_orderbook' channel for top-of-book depth and slippage estimation.
        """
        channel = "l2_orderbook"
        self._dispatcher.add_subscription(channel, symbols)
        if self.is_connected:
            payload = self._dispatcher.build_subscription_payload({channel: set(symbols)})
            await self._lifecycle._send_json(json.loads(payload))

    async def _handle_orderbook_resync(self, symbol: str) -> None:
        """
        Triggered when a sequence gap corrupts the Level 2 orderbook.
        Requests a fresh snapshot from Delta Exchange by re-issuing subscription with rate limiting.
        """
        clean_symbol = symbol.strip().upper()
        now = time.time()
        if not hasattr(self, "_last_resync_time"):
            self._last_resync_time: dict[str, float] = {}
        if now - self._last_resync_time.get(clean_symbol, 0.0) < 5.0:
            return
        self._last_resync_time[clean_symbol] = now

        logger.warning("Orderbook sequence gap detected on [%s]. Re-synchronizing ladder...", clean_symbol)
        if self.is_connected:
            self._sequence_tracker.reset(f"ob:{clean_symbol}")
            await self._lifecycle._send_json({
                "type": "subscribe",
                "payload": {
                    "channels": [
                        {"name": "l2_orderbook", "symbols": [clean_symbol]}
                    ]
                }
            })
            logger.info("Dispatched orderbook re-synchronization request for [%s].", clean_sym := clean_symbol)

    async def warmup_cache(
        self,
        rest_client: Any,
        symbols: list[str],
        resolutions: list[str] | None = None,
        limit: int = 500,
    ) -> dict[str, int]:
        """
        REST Warm-up / Historical Backfill Routine.
        Calls Delta REST API (GET /v2/history/candles) to prime the 500-bar RingBuffer
        and Daily Session Anchor for all specified symbols before WebSocket trading starts.
        Completely eliminates the 'Cold Start' buffer problem.
        """
        resolutions = resolutions or ["1m"]
        results: dict[str, int] = {}
        for sym in symbols:
            clean_sym = sym.strip().upper()
            for res in resolutions:
                try:
                    candles = await rest_client.get_historical_candles(
                        symbol=clean_sym,
                        resolution=res,
                        limit=limit,
                    )
                    seeded_count = self._aggregator.seed_historical_bars(clean_sym, res, candles)
                    results[f"{clean_sym}:{res}"] = seeded_count
                    logger.info(
                        "Cold-Start Warm-up: Primed %d historical %s bars for %s (RingBuffer count: %d).",
                        seeded_count,
                        res,
                        clean_sym,
                        self._aggregator.get_bar_cache(clean_sym, res).count,
                    )
                except Exception as ex:
                    logger.warning("Cold-Start Warm-up failed for %s (%s): %s", clean_sym, res, ex)
                    results[f"{clean_sym}:{res}"] = 0
        return results

    def get_bar_cache(self, symbol: str, resolution: str = "1m") -> CircularBarCache:
        """Retrieves in-memory 500-bar RingBuffer for specified symbol and resolution."""
        return self._aggregator.get_bar_cache(symbol, resolution)

    def get_session_anchor(self, symbol: str) -> SessionAnchorData:
        """Retrieves Daily Session Anchor (00:00 UTC) data for specified symbol."""
        return self._aggregator.get_session_anchor(symbol)

    async def __aenter__(self) -> "DeltaWebSocketClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.disconnect()
