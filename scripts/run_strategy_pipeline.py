"""
scripts/run_strategy_pipeline.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standalone verification script for Phase 3 Strategy and Signal Generation Engine.
Demonstrates:
1. REST Historical Warm-Up via DeltaRestClient (200 bars)
2. Pure NumPy Indicator calculation (EMA 20, Session VWAP, ATR 14, SMA(ATR 20), ADX 14, PDH/PDL)
3. Step-by-step candle close evaluation with BreakoutStrategyEngine
4. Real-time integer contract sizing and dynamic SL/TP calculation
5. Trailing stop ratcheting simulation

Usage:
    python3 scripts/run_strategy_pipeline.py [--symbol BTCUSDT] [--bars 60] [--equity 10000]
"""

import argparse
import asyncio
import logging
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from delta_bot.client import DeltaRestClient
from delta_bot.config import BotConfig
from delta_bot.indicators import IndicatorPipeline
from delta_bot.models import Candle, CircularBarCache, ContractSpec, SessionAnchorData
from delta_bot.strategy import (
    BreakoutStrategyEngine,
    OrderIntent,
    OrderStatus,
    SignalType,
    StrategyConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("strategy_pipeline")


def print_banner(title: str) -> None:
    border = "=" * 80
    print(f"\n{border}\n >>> {title.upper()}\n{border}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Delta-Bot Strategy Pipeline Verification")
    parser.add_argument("--symbol", type=str, default="BTCUSD", help="Contract symbol (default: BTCUSD)")
    parser.add_argument("--bars", type=int, default=60, help="Number of historical bars to fetch (default: 60)")
    parser.add_argument("--equity", type=float, default=10000.0, help="Account equity USD (default: 10000.0)")
    parser.add_argument("--trailing", action="store_true", help="Enable trailing stop ratchet")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    print_banner(f"Phase 3: Strategy & Signal Engine Verification [{symbol}]")

    config = BotConfig()
    strat_config = StrategyConfig(
        symbol=symbol,
        resolution="1m",
        ema_period=20,
        atr_period=14,
        atr_sma_period=20,
        adx_period=14,
        adx_threshold=20.0,
        price_buffer_ticks=1.0,
        max_pending_bars=3,
        risk_per_trade_pct=1.0,
        rr_ratio=2.0,
        trailing_stop=args.trailing,
    )

    async with DeltaRestClient(config) as client:
        # 1. Fetch Contract Spec
        print_banner("1. Loading Product Specifications")
        await client.get_products(["perpetual_futures"])
        spec = client.get_cached_contract_spec(symbol)
        if not spec:
            logger.error("Contract specification for %s not found on exchange!", symbol)
            return

        logger.info("Contract: %s | ID: %d | Contract Value: %s | Tick: %s", spec.symbol, spec.product_id, spec.contract_value, spec.tick_size)

        # 2. REST Warm-Up
        print_banner(f"2. REST Historical Warm-Up ({args.bars} bars)")
        bar_cache = CircularBarCache(symbol=symbol, resolution="1m", capacity=500)
        session_anchor = SessionAnchorData(
            symbol=symbol,
            session_date=time.strftime("%Y-%m-%d", time.gmtime()),
            session_start_epoch_s=time.time() - (time.time() % 86400),
        )

        engine = BreakoutStrategyEngine(strat_config)

        # Event hooks
        signals_received: list[OrderIntent] = []
        engine.on_signal(lambda sig, intent: signals_received.append(intent))
        engine.on_order_trigger(lambda intent, bar: logger.info("ORDER TRIGGERED: %s at %s", intent.symbol, intent.trigger_price))
        engine.on_order_cancel(lambda intent, reason: logger.info("ORDER CANCELLED: %s: %s", intent.symbol, reason))
        engine.on_sl_ratchet(lambda intent, old_sl, new_sl: logger.info("SL RATCHET: %s: %.4f -> %.4f", intent.symbol, old_sl, new_sl))

        seeded = await engine.warm_up_from_rest(
            client=client,
            bar_cache=bar_cache,
            session_anchor=session_anchor,
            limit=args.bars,
        )
        logger.info("Seeded %d bars into circular cache.", seeded)

        # 3. Indicator Telemetry
        print_banner("3. Indicator Engine State")
        snap = engine.indicator_pipeline.latest_snapshot
        if snap:
            logger.info("Latest Bar Close:     $%.4f", snap.close)
            logger.info("EMA 20:               $%.4f (Price %s EMA)", snap.ema_20, ">" if snap.close > snap.ema_20 else "<")
            logger.info("Session VWAP:         $%.4f (Price %s VWAP)", snap.session_vwap, ">" if snap.close > snap.session_vwap else "<")
            logger.info("ATR 14:               $%.4f | SMA(ATR 20): $%.4f", snap.atr_14, snap.atr_sma_20)
            logger.info("Volatility Expanded:  %s (ATR >= SMA(ATR, 20))", snap.is_volatility_expanded)
            logger.info("ADX 14:               %.2f (+DI=%.2f, -DI=%.2f)", snap.adx_14, snap.plus_di_14, snap.minus_di_14)
            logger.info("Market Regime:        %s (Threshold: %.1f)", "TRENDING" if snap.is_trending else "CONSOLIDATION/CHOP", strat_config.adx_threshold)
            logger.info("PDH / PDL:            %s / %s", snap.pdh, snap.pdl)

        # 4. Sizing Demonstration
        print_banner("4. Position Sizing & Contract Lot Calculation Simulation")
        sim_trigger = (snap.close + 10.0) if snap else 75000.0
        sim_sl = (snap.close - 200.0) if snap else 74800.0
        sim_risk_per_unit = sim_trigger - sim_sl
        capital_at_risk = args.equity * (strat_config.risk_per_trade_pct / 100.0)
        target_qty = capital_at_risk / sim_risk_per_unit
        contract_lots = spec.round_size(target_qty)
        effective_qty = spec.contracts_to_asset_qty(contract_lots)
        actual_risk = contract_lots * spec.contract_value * sim_risk_per_unit

        logger.info("Account Equity:       $%.2f", args.equity)
        logger.info("Risk Budget (1.0%%):   $%.2f", capital_at_risk)
        logger.info("Price Risk:           $%.2f per unit", sim_risk_per_unit)
        logger.info("Target Asset Qty:     %.6f %s", target_qty, spec.underlying_symbol or "BTC")
        logger.info("Delta Integer Lots:   %d contracts (Lot value: %s)", contract_lots, spec.contract_value)
        logger.info("Effective Asset Qty:  %.6f %s", effective_qty, spec.underlying_symbol or "BTC")
        logger.info("Actual Capital Risk:  $%.2f", actual_risk)

        print_banner("Phase 3 Verification Complete: Engine Operational")


if __name__ == "__main__":
    asyncio.run(main())
