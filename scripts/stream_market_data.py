"""
scripts/stream_market_data.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standalone live verification script for Phase 2 Market Data & WebSocket Engine.
Streams real-time ticker quotes and candlestick bars (1m) from Delta Exchange
for target perpetual futures: BTC, ETH, SOL, XRP.

Usage:
    python3 scripts/stream_market_data.py [--duration SECONDS] [--symbols SYM1 SYM2 ...]
Example:
    python3 scripts/stream_market_data.py --duration 30
"""

import argparse
import asyncio
from datetime import datetime, timezone
import logging
import os
import signal
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from delta_bot.config import BotConfig
from delta_bot.models import Candle, SessionAnchorData, Ticker
from delta_bot.websocket_client import DeltaWebSocketClient

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("stream_market_data")


class MarketDataStreamer:
    """Manages console formatting and live telemetry for market data streams."""

    def __init__(self, target_symbols: list[str]) -> None:
        self.target_symbols: list[str] = [s.upper() for s in target_symbols]
        self._last_ticker_print: dict[str, float] = {}
        self._ticker_count: int = 0
        self._candle_update_count: int = 0
        self._candle_close_count: int = 0

    async def handle_ticker(self, symbol: str, ticker: Ticker) -> None:
        """Processes and displays real-time ticker stream."""
        self._ticker_count += 1
        now = time.time()
        last_t = self._last_ticker_print.get(symbol, 0.0)

        # Rate-limit ticker terminal printing to at most 1 update per 2 seconds per symbol
        if now - last_t >= 2.0:
            self._last_ticker_print[symbol] = now
            bid_str = f"{ticker.best_bid:.4f}" if ticker.best_bid else "N/A"
            ask_str = f"{ticker.best_ask:.4f}" if ticker.best_ask else "N/A"
            spread_bps_str = f"{ticker.spread_bps:.2f} bps" if ticker.spread_bps is not None else "N/A"
            funding_pct = (ticker.funding_rate * 100.0) if ticker.funding_rate is not None else 0.0

            logger.info(
                "TICKER [%-7s] Mark: %10.4f | Bid: %-9s | Ask: %-9s | Spread: %-10s | OI: %9.2f | Funding: %+.4f%%",
                symbol,
                ticker.mark_price,
                bid_str,
                ask_str,
                spread_bps_str,
                ticker.open_interest,
                funding_pct,
            )

    async def handle_candle_update(self, symbol: str, candle: Candle) -> None:
        """Processes real-time forming candle ticks."""
        self._candle_update_count += 1
        # Debug level to avoid terminal flooding while forming
        logger.debug(
            "FORMING BAR [%s %s] O=%.2f H=%.2f L=%.2f C=%.2f V=%.2f",
            symbol,
            candle.resolution,
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
        )

    async def handle_candle_close(self, symbol: str, candle: Candle) -> None:
        """Highlights confirmed closed candlestick bars."""
        self._candle_close_count += 1
        dt = datetime.fromtimestamp(candle.candle_start_epoch_s, tz=timezone.utc)
        time_str = dt.strftime("%H:%M:%S UTC")

        border = "=" * 90
        print(f"\n{border}")
        logger.info(
            "*** CONFIRMED BAR CLOSE *** [%s %s @ %s]",
            symbol,
            candle.resolution,
            time_str,
        )
        logger.info(
            "    Open: %10.4f | High: %10.4f | Low: %10.4f | Close: %10.4f | Vol: %10.2f",
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
        )
        logger.info(
            "    Range: %8.4f | Body: %8.4f | Direction: %s",
            candle.range,
            candle.body,
            "BULLISH ▲" if candle.is_bullish else "BEARISH ▼",
        )
        print(f"{border}\n")

    async def handle_session_rollover(self, symbol: str, session: SessionAnchorData) -> None:
        """Logs daily 00:00 UTC session rollover events."""
        logger.info(
            "SESSION ROLLOVER [%s]: Session Date=%s | PDH=%.2f | PDL=%.2f | PDC=%.2f | VWAP=%.2f",
            symbol,
            session.session_date,
            session.prior_day_high or 0.0,
            session.prior_day_low or 0.0,
            session.prior_day_close or 0.0,
            session.session_vwap,
        )

    async def handle_disconnect(self, reason: str) -> None:
        """Logs connection drops."""
        logger.warning("WebSocket disconnected: %s. Auto-reconnection in progress...", reason)

    def print_summary(self) -> None:
        """Prints cumulative stream statistics on termination."""
        print("\n" + "=" * 70)
        logger.info("MARKET DATA STREAM SESSION SUMMARY")
        print("=" * 70)
        logger.info("Total Ticker updates ingested:       %d", self._ticker_count)
        logger.info("Total Forming Candle ticks:          %d", self._candle_update_count)
        logger.info("Total Confirmed Closed Bars:         %d", self._candle_close_count)
        print("=" * 70 + "\n")


async def main() -> None:
    """Entry point for live market data streaming script."""
    parser = argparse.ArgumentParser(description="Delta Exchange Live Market Data WebSocket Streamer")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Run duration in seconds (0 for indefinite until Ctrl+C)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
        help="Target perpetual contract symbols (default: BTCUSDT ETHUSDT SOLUSDT XRPUSDT)",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        default=True,
        help="Execute REST historical backfill before streaming (default: True)",
    )
    args = parser.parse_args()

    config = BotConfig()
    target_symbols = [s.upper() for s in args.symbols]
    streamer = MarketDataStreamer(target_symbols)

    logger.info("Initializing Delta-Bot Market Data WebSocket Engine (Phase 2)...")
    logger.info("WebSocket Endpoint: %s", config.delta_ws_url)
    logger.info("Target Symbols:     %s", target_symbols)
    if args.duration > 0:
        logger.info("Execution Duration: %.1f seconds", args.duration)
    else:
        logger.info("Execution Duration: INDEFINITE (Press Ctrl+C to terminate)")

    client = DeltaWebSocketClient(config=config)

    # Optional REST historical warm-up to seed the 500-bar buffer on minute zero
    if args.warmup:
        logger.info("Executing REST historical warm-up routine to eliminate Cold-Start delay...")
        from delta_bot.client import DeltaRestClient
        async with DeltaRestClient(config) as rest_client:
            warmup_stats = await client.warmup_cache(
                rest_client=rest_client,
                symbols=target_symbols,
                resolutions=["1m"],
                limit=50,
            )
            for sym_res, count in warmup_stats.items():
                logger.info("  ✓ Seeded %d closed bars for %s (Zero cold-start delay)", count, sym_res)

    # Register callbacks
    client.on_ticker(streamer.handle_ticker)
    client.on_candle_update(streamer.handle_candle_update)
    client.on_candle_close(streamer.handle_candle_close)
    client.on_session_rollover(streamer.handle_session_rollover)
    client.on_disconnect(streamer.handle_disconnect)

    stop_event = asyncio.Event()

    def _sig_handler(*_: Any) -> None:
        logger.info("Termination signal received. Shutting down...")
        stop_event.set()

    # Register signal handlers for clean SIGINT / SIGTERM termination
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, _sig_handler)
        except (NotImplementedError, RuntimeError) as sig_err:
            logger.debug("Signal handler skipped: %s", sig_err)

    async with client:
        # Subscribe to channels
        logger.info("Subscribing to 'v2/ticker' for %s...", target_symbols)
        await client.subscribe_tickers(target_symbols)

        logger.info("Subscribing to 'candlestick_1m' for %s...", target_symbols)
        await client.subscribe_candles(target_symbols, resolutions=["1m"])

        # Also subscribe to 'l2_orderbook' for BTCUSDT to verify depth stream
        logger.info("Subscribing to 'l2_orderbook' for BTCUSDT...")
        await client.subscribe_orderbook(["BTCUSDT"])

        logger.info("Streaming active. Listening to market feeds...")

        if args.duration > 0:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=args.duration)
            except asyncio.TimeoutError:
                logger.info("Specified duration of %.1fs completed.", args.duration)
        else:
            await stop_event.wait()

    streamer.print_summary()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application closed by keyboard interrupt.")
