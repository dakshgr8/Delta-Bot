"""
scripts/execute_order_lifecycle.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standalone verification script for Phase 4 Order Execution & Risk Management.
Demonstrates:
1. Dynamic Product ID Resolution across Delta Exchange (Global, India, Testnet)
2. Dual-Constraint Margin Sizing Engine (Risk budget vs Available margin at 5x/10x leverage)
3. Exchange-Native Stop Order placement with attached Brackets (zero slippage)
4. Local Signal Invalidation & TTL Purging of resting orders on Delta Exchange
5. Exchange-native Trailing Stop Ratchet synchronization

Usage:
    python3 scripts/execute_order_lifecycle.py [--symbol BTCUSDT] [--leverage 5] [--equity 10000]
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
from delta_bot.execution import ExecutionConfig, OrderExecutionManager
from delta_bot.models import ExecutionMode
from delta_bot.strategy import OrderIntent, OrderStatus, SignalType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("order_lifecycle")


def print_banner(title: str) -> None:
    border = "=" * 80
    print(f"\n{border}\n >>> {title.upper()}\n{border}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Delta-Bot Order Execution & Lifecycle Verification")
    parser.add_argument("--symbol", type=str, default="BTCUSD", help="Contract symbol (default: BTCUSD)")
    parser.add_argument("--leverage", type=int, default=5, help="Leverage tier (default: 5)")
    parser.add_argument("--equity", type=float, default=10000.0, help="Account equity USD (default: 10000.0)")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Run in DRY_RUN simulated mode")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    print_banner(f"Phase 4: Order Execution & Risk Engine [{symbol}]")

    config = BotConfig()
    exec_config = ExecutionConfig(
        execution_mode=ExecutionMode.DRY_RUN if args.dry_run else ExecutionMode.LIVE,
        leverage=args.leverage,
        margin_safety_buffer=0.90,
        stop_trigger_method="mark_price",
    )

    async with DeltaRestClient(config) as client:
        manager = OrderExecutionManager(client, exec_config)

        # ---------------------------------------------------------------------
        # 1. Product ID Resolution (Trap 3 Guard)
        # ---------------------------------------------------------------------
        print_banner("1. Dynamic Product ID Resolution")
        spec = await manager.resolve_product_spec(symbol)
        logger.info(
            "Resolved Contract: %s | Integer Product ID: %d | Contract Value: %s %s | Tick: %s",
            spec.symbol,
            spec.product_id,
            spec.contract_value,
            spec.underlying_symbol or "BTC",
            spec.tick_size,
        )

        # ---------------------------------------------------------------------
        # 2. Leverage & Dual-Constraint Margin Sizing (Trap 2 Guard)
        # ---------------------------------------------------------------------
        print_banner(f"2. Dual-Constraint Margin Sizing ({args.leverage}x Leverage)")
        ticker = await client.get_ticker(spec.symbol)
        current_px = ticker.mark_price

        # Simulated breakout levels
        trigger_px = spec.round_price(current_px + 100.0)
        sl_px = spec.round_price(current_px - 400.0)
        tp_px = spec.round_price(trigger_px + (2.0 * (trigger_px - sl_px)))

        margin_telemetry = await manager.check_and_size_margin(
            spec=spec,
            trigger_price=trigger_px,
            stop_loss_price=sl_px,
            account_equity=args.equity,
            risk_pct=1.0,
            available_margin_override=1000.0,  # $1,000 available margin
        )

        logger.info("Account Equity:         $%.2f", args.equity)
        logger.info("Available Margin:       $%.2f", margin_telemetry.available_margin)
        logger.info("Configured Leverage:    %.0fx", margin_telemetry.leverage)
        logger.info("Per-Unit Risk:          $%.2f", abs(trigger_px - sl_px))
        logger.info("Risk-Based Lots:        %d contracts (budget: $%.2f)", margin_telemetry.risk_based_lots, margin_telemetry.risk_budget_usd)
        logger.info("Margin-Based Lots:      %d contracts (max affordable at %.0fx leverage)", margin_telemetry.margin_based_lots, margin_telemetry.leverage)
        logger.info("Final Constrained Lots: %d contracts", margin_telemetry.final_lots)
        logger.info("Required Initial Margin:$%.2f (<= Usable Margin)", margin_telemetry.required_margin)
        logger.info("Margin Check Status:    %s", "PASSED" if margin_telemetry.is_margin_sufficient else "FAILED")

        # ---------------------------------------------------------------------
        # 3. Exchange-Native Stop Order Placement with Brackets (Trap 1 Guard)
        # ---------------------------------------------------------------------
        print_banner("3. Exchange-Native Stop Order with Attached Brackets")
        intent = OrderIntent(
            symbol=spec.symbol,
            signal_type=SignalType.BULLISH_BREAKOUT,
            status=OrderStatus.PENDING,
            trigger_price=trigger_px,
            initial_stop_loss=sl_px,
            current_stop_loss=sl_px,
            take_profit=tp_px,
            rr_ratio=2.0,
            contract_lots=margin_telemetry.final_lots,
            target_asset_qty=spec.contracts_to_asset_qty(margin_telemetry.final_lots),
            capital_at_risk_usd=margin_telemetry.final_lots * spec.contract_value * abs(trigger_px - sl_px),
            signal_candle_high=current_px + 80.0,
            signal_candle_low=current_px - 380.0,
            signal_candle_close=current_px + 50.0,
            signal_bar_time=time.time(),
            created_at_epoch_s=time.time(),
            ttl_bars=3,
        )

        order = await manager.execute_stop_order_with_brackets(
            order_intent=intent,
            account_equity=args.equity,
            available_margin_override=1000.0,
        )

        logger.info("Exchange Order ID:      %d", order.id)
        logger.info("Product ID:             %d", order.product_id)
        logger.info("Side / Type:            %s / %s", order.side.upper(), order.order_type)
        logger.info("Stop Trigger Price:     $%.2f (%s)", order.stop_price, order.stop_trigger_method)
        logger.info("Bracket Stop Loss:      $%.2f", order.bracket_stop_loss_price)
        logger.info("Bracket Take Profit:    $%.2f", order.bracket_take_profit_price)
        logger.info("Order State:            %s", order.state.upper())

        # ---------------------------------------------------------------------
        # 4. Trailing Stop Ratchet Synchronization (Two-Tier Delta Architecture)
        # ---------------------------------------------------------------------
        print_banner("4. Trailing Stop Ratchet Synchronization (Two-Tier Routing)")
        ratchet_sl = spec.round_price(sl_px + 200.0)
        logger.info("[Tier 1: Pending Order] Ratcheting pending bracket on entry order ID=%d: $%.2f -> $%.2f", order.id, sl_px, ratchet_sl)
        ratchet_pending_ok = await manager.ratchet_exchange_stop_loss(intent, ratchet_sl)
        logger.info("Pending Bracket Ratchet: %s (via PUT /v2/orders/bracket)", "SUCCESS" if ratchet_pending_ok else "FAILED")

        # Simulate breakout fill -> Position is now TRIGGERED & child Stop Loss order is spawned
        logger.info("[Tier 2: Filled Position] Simulating price trigger breach & breakout execution...")
        intent.status = OrderStatus.TRIGGERED
        intent.child_stop_loss_order_id = order.id + 1000  # Spawned child Stop Loss order
        ratchet_filled_sl = spec.round_price(ratchet_sl + 150.0)
        logger.info(
            "Position open! Ratcheting active child stop order ID=%d (parent ID=%d closed): $%.2f -> $%.2f",
            intent.child_stop_loss_order_id,
            intent.exchange_order_id,
            ratchet_sl,
            ratchet_filled_sl,
        )
        ratchet_filled_ok = await manager.ratchet_exchange_stop_loss(intent, ratchet_filled_sl)
        logger.info(
            "Filled Position SL Ratchet: %s (Targeted child order %d via PUT /v2/orders stop_price, avoided open_order_not_found)",
            "SUCCESS" if ratchet_filled_ok else "FAILED",
            intent.child_stop_loss_order_id,
        )

        # ---------------------------------------------------------------------
        # 5. Local Invalidation & TTL Resting Order Purging
        # ---------------------------------------------------------------------
        print_banner("5. Local Invalidation & Resting Order Purging")
        # Reset to pending for cancellation demonstration
        intent.status = OrderStatus.PENDING
        cancel_ok = await manager.cancel_resting_order(
            order_intent=intent,
            reason="Intermediate candle closed below EMA 20 (Signal Invalidated)",
        )
        logger.info("Resting Order Cancel:   %s (Removed from Delta Exchange book)", "SUCCESS" if cancel_ok else "FAILED")

        print_banner("Phase 4 Execution Lifecycle Verification: COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
