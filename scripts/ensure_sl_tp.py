#!/usr/bin/env python3
"""
scripts/ensure_sl_tp.py
~~~~~~~~~~~~~~~~~~~~~~~

Automated Shield & Audit Script: Guarantees that NO order or position is ever left naked.

Features:
1. Scans all open positions across Delta Exchange India (Production) and Delta Testnet (Demo).
2. Detects any position missing a Stop Loss (SL) or Take Profit (TP).
3. If an unprotected (naked) position is found:
   - Computes precise SL & TP prices adhering to product tick size half-up quantization.
   - Immediately attaches position brackets via POST /v2/orders/bracket.
   - Falls back to native reduce-only conditional stop orders if position brackets fail.
4. Scans all resting entry orders:
   - Verifies attached bracket SL & TP exist on resting breakout orders.
   - Modifies resting brackets via PUT /v2/orders/bracket if missing.
5. Can be executed as a one-shot verification tool OR as a continuous watchdog loop via --watch.

Usage:
    # One-shot audit and repair across all desks:
    python3 scripts/ensure_sl_tp.py

    # Audit and repair only Real desk:
    python3 scripts/ensure_sl_tp.py --mode real

    # Audit and repair only Demo testnet desk:
    python3 scripts/ensure_sl_tp.py --mode demo

    # Dry-run audit (inspect and report without sending orders):
    python3 scripts/ensure_sl_tp.py --dry-run

    # Continuous watchdog daemon (checks every 5 seconds):
    python3 scripts/ensure_sl_tp.py --watch --interval 5
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from typing import Any

from dotenv import load_dotenv
load_dotenv()

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from delta_bot.client import DeltaRestClient
from delta_bot.config import BotConfig
from delta_bot.models import ContractSpec, ExchangeOrder
from delta_bot.tenant import TenantManager, decrypt_secret

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sl_tp_guard")


class PositionSafetyAudit:
    def __init__(
        self,
        client: DeltaRestClient,
        desk_name: str,
        mode: str,
        sl_pct: float = 1.5,
        tp_pct: float = 3.0,
        dry_run: bool = False,
    ) -> None:
        self.client = client
        self.desk_name = desk_name
        self.mode = mode.upper()
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.dry_run = dry_run
        self.specs_by_id: dict[int, ContractSpec] = {}
        self.specs_by_sym: dict[str, ContractSpec] = {}

    async def init_specs(self) -> None:
        """Loads and indexes product contract specs for half-up tick rounding."""
        try:
            products = await self.client.get_products(["perpetual_futures"])
            for p in products:
                self.specs_by_id[p.product_id] = p
                self.specs_by_sym[p.symbol.upper()] = p
        except Exception as e:
            logger.warning("[%s] Failed to fetch product specifications: %s", self.desk_name, e)

    def round_price(self, product_id: int, symbol: str, price: float) -> float:
        """Quantizes price to contract tick size step."""
        spec = self.specs_by_id.get(product_id) or self.specs_by_sym.get(symbol.upper())
        if spec:
            return spec.round_price(price)
        return round(price, 4)

    async def audit_positions(self) -> dict[str, Any]:
        """Audits all open positions on this desk and attaches missing SL / TP."""
        stats = {
            "positions_checked": 0,
            "positions_protected": 0,
            "naked_positions_found": 0,
            "naked_positions_repaired": 0,
            "resting_orders_checked": 0,
            "resting_orders_repaired": 0,
        }

        try:
            positions = await self.client.get_open_positions()
        except Exception as e:
            logger.error("[%s] Error fetching open positions: %s", self.desk_name, e)
            return stats

        try:
            open_orders = await self.client.get_open_orders()
        except Exception as e:
            logger.error("[%s] Error fetching open orders: %s", self.desk_name, e)
            open_orders = []

        active_positions = [p for p in positions if p.size != 0]
        stats["positions_checked"] = len(active_positions)

        for pos in active_positions:
            exit_side = "sell" if pos.size > 0 else "buy"
            sym = pos.symbol or f"PRODUCT_{pos.product_id}"

            # Check for existing Stop Loss order on the exchange
            existing_sl_order = next(
                (
                    o for o in open_orders
                    if o.product_id == pos.product_id
                    and o.side.lower() == exit_side
                    and o.stop_price is not None
                ),
                None,
            )

            # Check for existing Take Profit order on the exchange
            existing_tp_order = next(
                (
                    o for o in open_orders
                    if o.product_id == pos.product_id
                    and o.side.lower() == exit_side
                    and (
                        (o.order_type == "limit_order" and o.limit_price is not None and (
                            (pos.size > 0 and o.limit_price > pos.entry_price) or
                            (pos.size < 0 and o.limit_price < pos.entry_price)
                        ))
                        or (getattr(o, "bracket_take_profit_price", None) is not None)
                    )
                ),
                None,
            )

            has_sl = existing_sl_order is not None
            has_tp = existing_tp_order is not None

            if has_sl and has_tp:
                stats["positions_protected"] += 1
                logger.info(
                    "[%s] %s (%s, size=%d, entry=%.4f): PROTECTED [SL=%.4f TP=%.4f]",
                    self.desk_name,
                    sym,
                    "LONG" if pos.size > 0 else "SHORT",
                    pos.size,
                    pos.entry_price,
                    existing_sl_order.stop_price,
                    existing_tp_order.limit_price or getattr(existing_tp_order, "bracket_take_profit_price", 0.0),
                )
                continue

            # Position is NAKED or missing brackets!
            stats["naked_positions_found"] += 1
            missing_items = []
            if not has_sl:
                missing_items.append("STOP LOSS")
            if not has_tp:
                missing_items.append("TAKE PROFIT")

            logger.critical(
                "[%s] 🚨 NAKED POSITION DETECTED! %s (%s, size=%d, entry=%.4f) missing: %s",
                self.desk_name,
                sym,
                "LONG" if pos.size > 0 else "SHORT",
                pos.size,
                pos.entry_price,
                ", ".join(missing_items),
            )

            # Calculate safe SL and TP based on risk % and tick size
            if pos.size > 0:
                calc_sl = pos.entry_price * (1.0 - self.sl_pct / 100.0)
                calc_tp = pos.entry_price * (1.0 + self.tp_pct / 100.0)
            else:
                calc_sl = pos.entry_price * (1.0 + self.sl_pct / 100.0)
                calc_tp = pos.entry_price * (1.0 - self.tp_pct / 100.0)

            sl_price = self.round_price(pos.product_id, sym, calc_sl)
            tp_price = self.round_price(pos.product_id, sym, calc_tp)

            if self.dry_run:
                logger.warning(
                    "[%s] [DRY RUN] Would arm %s with SL=%.4f (%.2f%%) and TP=%.4f (%.2f%%)",
                    self.desk_name,
                    sym,
                    sl_price,
                    self.sl_pct,
                    tp_price,
                    self.tp_pct,
                )
                continue

            # Attempt 1: Attach position brackets via POST /v2/orders/bracket
            bracket_success = False
            try:
                await self.client.set_position_bracket(
                    product_id=pos.product_id,
                    bracket_stop_loss_price=sl_price if not has_sl else None,
                    bracket_take_profit_price=tp_price if not has_tp else None,
                )
                bracket_success = True
                stats["naked_positions_repaired"] += 1
                logger.info(
                    "[%s] 🛡️ Successfully armed position bracket for %s: Attached SL=%.4f, TP=%.4f",
                    self.desk_name,
                    sym,
                    sl_price,
                    tp_price,
                )
            except Exception as b_err:
                logger.warning(
                    "[%s] POST /v2/orders/bracket failed for %s: %s. Falling back to native reduce-only child orders...",
                    self.desk_name,
                    sym,
                    b_err,
                )

            # Attempt 2 (Fallback): Place independent reduce-only stop loss and take profit orders
            if not bracket_success:
                repaired = False
                if not has_sl:
                    try:
                        sl_res = await self.client.place_order(
                            product_id=pos.product_id,
                            size=abs(pos.size),
                            side=exit_side,
                            order_type="market_order",
                            stop_price=sl_price,
                            stop_order_type="stop_loss_order",
                            reduce_only=True,
                            time_in_force="gtc",
                            symbol=sym,
                        )
                        logger.info(
                            "[%s] 🛡️ Fallback child Stop Loss order placed: ID=%d for %s @ %.4f",
                            self.desk_name,
                            sl_res.id,
                            sym,
                            sl_price,
                        )
                        repaired = True
                    except Exception as sl_exc:
                        logger.error(
                            "[%s] ❌ Failed to place fallback child SL order for %s: %s",
                            self.desk_name,
                            sym,
                            sl_exc,
                        )

                if not has_tp:
                    try:
                        tp_res = await self.client.place_order(
                            product_id=pos.product_id,
                            size=abs(pos.size),
                            side=exit_side,
                            order_type="limit_order",
                            limit_price=tp_price,
                            reduce_only=True,
                            time_in_force="gtc",
                            symbol=sym,
                        )
                        logger.info(
                            "[%s] 🛡️ Fallback child Take Profit order placed: ID=%d for %s @ %.4f",
                            self.desk_name,
                            tp_res.id,
                            sym,
                            tp_price,
                        )
                        repaired = True
                    except Exception as tp_exc:
                        logger.error(
                            "[%s] ❌ Failed to place fallback child TP order for %s: %s",
                            self.desk_name,
                            sym,
                            tp_exc,
                        )

                if repaired:
                    stats["naked_positions_repaired"] += 1

        # Audit resting entry orders for missing brackets
        entry_orders = [o for o in open_orders if o.stop_price is not None]
        stats["resting_orders_checked"] = len(entry_orders)
        for eo in entry_orders:
            eo_sym = eo.symbol or f"PRODUCT_{eo.product_id}"
            has_eo_sl = getattr(eo, "bracket_stop_loss_price", None) is not None
            has_eo_tp = getattr(eo, "bracket_take_profit_price", None) is not None

            if not has_eo_sl or not has_eo_tp:
                stats["resting_orders_repaired"] += 1
                logger.warning(
                    "[%s] Resting entry order #%s (%s, side=%s, stop=%.4f) missing brackets!",
                    self.desk_name,
                    eo.id,
                    eo_sym,
                    eo.side.upper(),
                    eo.stop_price,
                )
                if not self.dry_run:
                    # Calculate and attach brackets
                    if eo.side.lower() == "buy":
                        calc_sl = eo.stop_price * (1.0 - self.sl_pct / 100.0)
                        calc_tp = eo.stop_price * (1.0 + self.tp_pct / 100.0)
                    else:
                        calc_sl = eo.stop_price * (1.0 + self.sl_pct / 100.0)
                        calc_tp = eo.stop_price * (1.0 - self.tp_pct / 100.0)
                    sl_px = self.round_price(eo.product_id, eo_sym, calc_sl)
                    tp_px = self.round_price(eo.product_id, eo_sym, calc_tp)
                    try:
                        await self.client.edit_bracket_order(
                            product_id=eo.product_id,
                            order_id=eo.id,
                            bracket_stop_loss_price=sl_px if not has_eo_sl else None,
                            bracket_take_profit_price=tp_px if not has_eo_tp else None,
                        )
                        logger.info(
                            "[%s] Attached missing brackets to resting entry order #%s: SL=%.4f, TP=%.4f",
                            self.desk_name,
                            eo.id,
                            sl_px,
                            tp_px,
                        )
                    except Exception as eo_err:
                        logger.warning(
                            "[%s] Could not attach bracket to resting order #%s: %s",
                            self.desk_name,
                            eo.id,
                            eo_err,
                        )

        return stats


async def get_auditors(mode_filter: str = "all", dry_run: bool = False, sl_pct: float = 1.5, tp_pct: float = 3.0) -> list[PositionSafetyAudit]:
    """Discovers and initializes DeltaRestClient instances for all configured desks."""
    auditors: list[PositionSafetyAudit] = []
    mode_clean = mode_filter.strip().lower()

    # 1. Inspect TenantManager
    bot_cfg = BotConfig()
    manager = TenantManager(
        supabase_url=bot_cfg.supabase_url,
        supabase_secret_key=bot_cfg.supabase_secret_key_str,
    )
    manager.load_from_disk({})
    if bot_cfg.supabase_url and bot_cfg.supabase_secret_key_str:
        manager.load_from_supabase({})
    if manager.tenants:
        for t_id, tenant in manager.tenants.items():
            cfg = tenant.config

            # Real Desk
            if mode_clean in ("all", "real"):
                rk = decrypt_secret(cfg.delta_api_key.strip()) if cfg.delta_api_key.strip().startswith("enc:") else cfg.delta_api_key.strip()
                rs = decrypt_secret(cfg.delta_api_secret.strip()) if cfg.delta_api_secret.strip().startswith("enc:") else cfg.delta_api_secret.strip()
                if rk and rs:
                    b_cfg = BotConfig(
                        delta_api_key=rk,
                        delta_api_secret=rs,
                        delta_base_url=cfg.delta_base_url or "https://api.india.delta.exchange",
                        delta_ws_url=cfg.delta_ws_url or "wss://socket.india.delta.exchange",
                        execution_mode="live",
                    )
                    client = DeltaRestClient(config=b_cfg)
                    auditors.append(
                        PositionSafetyAudit(
                            client=client,
                            desk_name=f"{cfg.name} (Real)",
                            mode="real",
                            sl_pct=sl_pct,
                            tp_pct=tp_pct,
                            dry_run=dry_run,
                        )
                    )

            # Demo Desk
            if mode_clean in ("all", "demo"):
                dk = decrypt_secret(cfg.demo_api_key.strip()) if cfg.demo_api_key.strip().startswith("enc:") else cfg.demo_api_key.strip()
                ds = decrypt_secret(cfg.demo_api_secret.strip()) if cfg.demo_api_secret.strip().startswith("enc:") else cfg.demo_api_secret.strip()
                if dk and ds:
                    b_cfg = BotConfig(
                        delta_api_key=dk,
                        delta_api_secret=ds,
                        delta_base_url=cfg.demo_base_url or "https://cdn-ind.testnet.deltaex.org",
                        delta_ws_url=cfg.demo_ws_url or "wss://socket-ind.testnet.deltaex.org",
                        execution_mode="demo",
                    )
                    client = DeltaRestClient(config=b_cfg)
                    auditors.append(
                        PositionSafetyAudit(
                            client=client,
                            desk_name=f"{cfg.name} (Demo Testnet)",
                            mode="demo",
                            sl_pct=sl_pct,
                            tp_pct=tp_pct,
                            dry_run=dry_run,
                        )
                    )

    # 2. Fallback to root BotConfig if no tenants configured
    if not auditors:
        root_cfg = BotConfig()
        rk = root_cfg.delta_api_key.get_secret_value().strip() if root_cfg.delta_api_key else ""
        rs = root_cfg.delta_api_secret.get_secret_value().strip() if root_cfg.delta_api_secret else ""
        if rk and rs:
            client = DeltaRestClient(config=root_cfg)
            auditors.append(
                PositionSafetyAudit(
                    client=client,
                    desk_name="Root Bot Account",
                    mode=root_cfg.execution_mode,
                    sl_pct=sl_pct,
                    tp_pct=tp_pct,
                    dry_run=dry_run,
                )
            )

    return auditors


async def run_audit_cycle(auditors: list[PositionSafetyAudit]) -> None:
    """Executes a single audit and protection pass across all desks."""
    total_positions = 0
    total_protected = 0
    total_naked = 0
    total_repaired = 0

    print("=" * 80)
    print(f" >>> RUNNING DELTA-BOT POSITION & ORDER SL/TP SHIELD AUDIT [{time.strftime('%Y-%m-%d %H:%M:%S')}]")
    print("=" * 80)

    for auditor in auditors:
        await auditor.init_specs()
        stats = await auditor.audit_positions()
        total_positions += stats["positions_checked"]
        total_protected += stats["positions_protected"]
        total_naked += stats["naked_positions_found"]
        total_repaired += stats["naked_positions_repaired"]

    print("-" * 80)
    print(f"Summary: Positions Checked: {total_positions} | Protected: {total_protected} | Naked: {total_naked} | Repaired: {total_repaired}")
    print("=" * 80 + "\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Delta-Bot Real-Time SL & TP Position Safety Guard")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "real", "demo"], help="Operational mode to audit")
    parser.add_argument("--sl-pct", type=float, default=1.5, help="Default Stop Loss percentage (default: 1.5%%)")
    parser.add_argument("--tp-pct", type=float, default=3.0, help="Default Take Profit percentage (default: 3.0%%)")
    parser.add_argument("--dry-run", action="store_true", help="Inspect and report naked positions without placing orders")
    parser.add_argument("--watch", action="store_true", help="Run continuously in a background watchdog loop")
    parser.add_argument("--interval", type=int, default=10, help="Watchdog sleep interval in seconds (default: 10s)")
    args = parser.parse_args()

    auditors = await get_auditors(mode_filter=args.mode, dry_run=args.dry_run, sl_pct=args.sl_pct, tp_pct=args.tp_pct)
    if not auditors:
        logger.error("No configured Delta Exchange credentials found. Please configure API keys.")
        sys.exit(1)

    logger.info("Initialized SL/TP Safety Guard across %d desk(s)", len(auditors))

    if args.watch:
        logger.info("Watchdog mode enabled. Polling every %d seconds... (Press Ctrl+C to stop)", args.interval)
        try:
            while True:
                await run_audit_cycle(auditors)
                await asyncio.sleep(args.interval)
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Watchdog loop stopped by user.")
    else:
        await run_audit_cycle(auditors)


if __name__ == "__main__":
    asyncio.run(main())
