#!/usr/bin/env python3
"""
run_bot.py
~~~~~~~~~~

Standalone Production CLI Daemon for Delta-Bot (Phase 6).
Provides command-line configuration, ASCII telemetry banners, signal trapping,
and lifecycle orchestration across Paper and Live environments.

Usage:
    # Run Paper Trading with default settings (10,000 USD virtual collateral):
    python3 run_bot.py --mode paper

    # Run Multi-Asset Paper Trading with custom symbols and timeframe:
    python3 run_bot.py --mode paper --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT --timeframe 5m --risk-pct 1.5

    # Run Live Trading on Delta Exchange (Requires DELTA_API_KEY and DELTA_API_SECRET in .env):
    python3 run_bot.py --mode live --leverage 5 --flatten-on-shutdown
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Any

# Ensure repository root is on Python module search path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from delta_bot.config import BotConfig
from delta_bot.exceptions import ConfigurationError
from delta_bot.models import ExecutionMode
from delta_bot.orchestrator import TradingBotOrchestrator
from delta_bot.paper import PaperExchangeSimulator


def setup_daemon_logging(log_level: str) -> None:
    """Configures structured terminal logging format for daemon operation."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    log_format = "%(asctime)s [%(levelname)-7s] [%(name)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    # Silence noisy transport loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)


def print_startup_banner(
    mode: str,
    symbols: list[str],
    timeframe: str,
    leverage: int,
    risk_pct: float,
    flatten_on_shutdown: bool,
    initial_balance: float,
    web_enabled: bool = False,
    web_url: str = "",
) -> None:
    """Prints the Delta-Bot launch banner with active operational parameters."""
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    DELTA-BOT QUANTITATIVE TRADING DAEMON                     ║
║              High-Performance Multi-Asset Breakout Architecture              ║
╚══════════════════════════════════════════════════════════════════════════════╝
 Operational Mode        : {mode.upper()}
 Target Contracts        : {', '.join(symbols)}
 Candle Timeframe        : {timeframe}
 Default Leverage        : {leverage}x
 Risk Per Trade          : {risk_pct:.2f}% of Account Equity
 Collateral Initializer  : ${initial_balance:,.2f} USD
 Flatten On Shutdown     : {'ENABLED (Market Close)' if flatten_on_shutdown else 'DISABLED (Preserve Brackets)'}
 Web Dashboard (Phase 7) : {web_url if web_enabled else 'DISABLED (Use --web)'}
 Python Runtime          : {sys.version.split()[0]} ({sys.platform})
════════════════════════════════════════════════════════════════════════════════
"""
    print(banner, flush=True)


def parse_cli_args() -> argparse.Namespace:
    """Parses command line arguments for the orchestrator daemon."""
    parser = argparse.ArgumentParser(
        description="Delta-Bot Quantitative Trading Orchestrator Daemon (Phase 6 & 7)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["paper", "live"],
        default=os.environ.get("EXECUTION_MODE", "paper").lower(),
        help="Operational execution mode: 'paper' (virtual simulator) or 'live' (Delta Exchange)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,XAUTUSD",
        help="Comma-separated target perpetual contracts to trade concurrently",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1m",
        help="Candle resolution timeframe (e.g. 1m, 3m, 5m, 15m)",
    )
    parser.add_argument(
        "--leverage",
        type=int,
        default=5,
        help="Default leverage tier to set on contract specification",
    )
    parser.add_argument(
        "--risk-pct",
        type=float,
        default=1.0,
        help="Percentage of account equity risked per breakout trade",
    )
    parser.add_argument(
        "--flatten-on-shutdown",
        action="store_true",
        default=False,
        help="Emergency market-close all active positions upon receiving SIGINT/SIGTERM",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        default=os.environ.get("WEB_ENABLED", "false").lower() in ("true", "1", "yes"),
        help="Launch the FastAPI Web Dashboard & Real-Time Control API concurrently",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("HOST", "0.0.0.0"),
        help="Host address for FastAPI Web Dashboard binding",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="Port for FastAPI Web Dashboard server",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=60.0,
        help="Watchdog health telemetry heartbeat interval in seconds",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=10000.0,
        help="Starting virtual collateral balance for Paper trading mode",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity level",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="Path to custom .env file containing API credentials",
    )
    return parser.parse_args()


async def main() -> None:
    """Daemon execution entry-point."""
    args = parse_cli_args()
    setup_daemon_logging(args.log_level)
    logger = logging.getLogger("delta_bot.daemon")

    # Load and override configuration with CLI options
    env_file = args.env_file if os.path.exists(args.env_file) else None
    config = BotConfig(
        _env_file=env_file,
        execution_mode=args.mode,
        flatten_on_shutdown=args.flatten_on_shutdown,
        target_symbols_str=args.symbols,
        timeframe=args.timeframe,
        default_leverage=args.leverage,
        default_risk_pct=args.risk_pct,
        status_interval_sec=args.status_interval,
    )

    clean_symbols = config.target_symbols
    web_url = f"http://{args.host}:{args.port}"

    print_startup_banner(
        mode=args.mode,
        symbols=clean_symbols,
        timeframe=args.timeframe,
        leverage=args.leverage,
        risk_pct=args.risk_pct,
        flatten_on_shutdown=args.flatten_on_shutdown,
        initial_balance=args.initial_balance,
        web_enabled=args.web,
        web_url=web_url,
    )

    # Prepare Paper Simulator if in Paper mode
    paper_sim: PaperExchangeSimulator | None = None
    if args.mode.lower() == "paper":
        paper_sim = PaperExchangeSimulator(
            initial_balance=args.initial_balance,
        )

    try:
        orchestrator = TradingBotOrchestrator(
            config=config,
            mode=args.mode,
            paper_simulator=paper_sim,
        )
    except ConfigurationError as ce:
        logger.critical("Daemon startup failed due to configuration error: %s", ce)
        sys.exit(1)

    # Attach POSIX signal traps
    orchestrator.setup_signal_handlers()

    # Launch FastAPI Web Server if --web is enabled
    web_server_task: asyncio.Task[Any] | None = None
    uvicorn_server: Any | None = None
    if args.web:
        try:
            import uvicorn
            from delta_bot.server import create_app
            app = create_app(orchestrator)
            server_config = uvicorn.Config(
                app=app,
                host=args.host,
                port=args.port,
                log_level="warning",
                access_log=False,
            )
            uvicorn_server = uvicorn.Server(server_config)
            web_server_task = asyncio.create_task(
                uvicorn_server.serve(),
                name="FastAPI-Web-Dashboard",
            )
            logger.info("Web Dashboard active at %s (WebSocket: ws://%s:%d/ws/live)", web_url, args.host, args.port)
        except Exception as web_err:
            logger.error("Failed to start FastAPI Web Gateway: %s", web_err)

    try:
        await orchestrator.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.warning("Trapped manual termination request from user")
    finally:
        if uvicorn_server is not None:
            uvicorn_server.should_exit = True
            if web_server_task is not None and not web_server_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(web_server_task), timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    pass
        if web_server_task is not None and not web_server_task.done():
            web_server_task.cancel()
            try:
                await web_server_task
            except (asyncio.CancelledError, Exception):
                logger.debug("Web server task stopped cleanly")

        if not orchestrator._shutdown_completed:
            await orchestrator.shutdown(reason="CLI termination exit")
        logger.info("Delta-Bot daemon process terminated cleanly")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting cleanly.", flush=True)
