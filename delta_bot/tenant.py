"""
delta_bot.tenant
~~~~~~~~~~~~~~~~

Multi-Tenant Architecture & Account Isolation Engine for Delta-Bot.
Supports concurrent isolated tenants, each with:
- Dedicated Virtual or Live Collateral Wallet
- Isolated RiskManager & Position/Order State
- Custom Strategy & Asset Configuration
- Dedicated Paper Simulator or Authenticated Live Exchange Credentials
- Real-Time WebSocket & Telemetry Scoping
"""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any
import uuid

from pydantic import BaseModel, Field

from delta_bot.client import DeltaRestClient
from delta_bot.config import BotConfig
from delta_bot.exceptions import AuthenticationError, ConfigurationError
from delta_bot.execution import ExecutionConfig, ExecutionManager, OrderRouter
from delta_bot.models import (
    Candle,
    ContractSpec,
    ExecutionMode,
    OrderBook,
    SessionAnchorData,
    Ticker,
)
from delta_bot.paper import PaperExchangeSimulator
from delta_bot.risk import RiskConfig, RiskManager
from delta_bot.state import TradingStateManager
from delta_bot.strategy import (
    BreakoutStrategyEngine,
    OrderIntent,
    OrderStatus,
    SignalType,
    StrategyConfig as BreakoutStrategyConfig,
)

logger = logging.getLogger("delta_bot.tenant")


# -----------------------------------------------------------------------------
# Cryptographic Storage Security for API Credentials (AES-GCM / Fernet)
# -----------------------------------------------------------------------------

def get_encryption_fernet() -> Any:
    """Instantiates Fernet cipher derived from server-side master key."""
    try:
        from cryptography.fernet import Fernet
        secret = (
            os.environ.get("TENANT_ENCRYPTION_KEY")
            or os.environ.get("DASHBOARD_AUTH_TOKEN")
            or os.environ.get("DELTA_API_SECRET")
            or "delta-bot-master-secret-vault-2026"
        )
        key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())
        return Fernet(key)
    except Exception as e:
        logger.error("Failed to initialize cryptographic cipher: %s", e)
        return None


def encrypt_secret(val: str | None) -> str:
    """Encrypts raw API credentials at rest before writing to database or disk."""
    if not val or not str(val).strip():
        return ""
    v_str = str(val).strip()
    if v_str.startswith("enc:"):
        return v_str
    try:
        f = get_encryption_fernet()
        if f is None:
            return v_str
        token = f.encrypt(v_str.encode("utf-8")).decode("utf-8")
        return f"enc:{token}"
    except Exception as e:
        logger.error("Encryption failed: %s", e)
        return v_str


def decrypt_secret(val: str | None) -> str:
    """Decrypts ciphertext from database or disk back into raw credential string."""
    if not val or not str(val).strip():
        return ""
    v_str = str(val).strip()
    if not v_str.startswith("enc:"):
        return v_str
    try:
        f = get_encryption_fernet()
        if f is None:
            return v_str
        raw_token = v_str[4:]
        return f.decrypt(raw_token.encode("utf-8")).decode("utf-8")
    except Exception as e:
        logger.error("Decryption failed: %s", e)
        return ""


# -----------------------------------------------------------------------------
# Deterministic Multi-Tenant Order ID Fan-Out & Traceability (UUIDv5)
# -----------------------------------------------------------------------------

def generate_deterministic_tenant_order_id(base_signal_id: str, tenant_id: str) -> str:
    """
    Generates deterministic UUIDv5 mapping master signal to an isolated tenant order.
    Ensures zero primary key collisions in cloud ledger across concurrent tenants.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_OID, f"{base_signal_id}:{tenant_id}"))


def generate_deterministic_client_order_id(base_signal_id: str, tenant_id: str, max_len: int = 32) -> str:
    """
    Generates deterministic exchange client_order_id (<= 32 chars) for Delta Exchange.
    Format: 'D_' + 30 hex characters of uuid5 digest.
    """
    hex_digest = uuid.uuid5(uuid.NAMESPACE_OID, f"{base_signal_id}:{tenant_id}").hex
    return f"D_{hex_digest[:max_len - 2]}"


class TenantConfig(BaseModel):
    """Data specification and serialized configuration for an isolated tenant."""

    tenant_id: str
    user_id: str | None = None
    name: str
    mode: str = "live"  # Exclusively "live" for real trading
    initial_balance: float = 10000.0
    delta_api_key: str = ""
    delta_api_secret: str = ""
    delta_base_url: str = "https://api.india.delta.exchange"
    delta_ws_url: str = "wss://socket.india.delta.exchange"
    risk_pct: float = 1.0
    leverage: int = 5
    timeframe: str = "1m"
    max_drawdown_pct: float = 4.0
    symbols: list[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"])
    is_active: bool = True
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def safe_dict(self) -> dict[str, Any]:
        """Provides a sanitized dictionary view masking private credentials."""
        d = self.model_dump()
        k = d.get("delta_api_key")
        if k:
            k_str = k.get_secret_value() if hasattr(k, "get_secret_value") else str(k)
            k_clean = k_str.strip()
            if k_clean.startswith("enc:"):
                k_clean = decrypt_secret(k_clean)
            if k_clean:
                d["delta_api_key_masked"] = f"{k_clean[:4]}...{k_clean[-4:]}" if len(k_clean) >= 8 else "********"
                d["has_api_key"] = True
            else:
                d["delta_api_key_masked"] = ""
                d["has_api_key"] = False
        else:
            d["delta_api_key_masked"] = ""
            d["has_api_key"] = False
        d.pop("delta_api_key", None)
        d.pop("delta_api_secret", None)
        return d


class TenantInstance:
    """Runtime execution container for an isolated tenant."""

    def __init__(
        self,
        config: TenantConfig,
        contract_specs: dict[str, ContractSpec] | None = None,
        state_manager: TradingStateManager | None = None,
        risk_manager: RiskManager | None = None,
        paper_simulator: PaperExchangeSimulator | None = None,
        strategies: dict[str, BreakoutStrategyEngine] | None = None,
    ) -> None:
        self.config = config
        self.mode = ExecutionMode.from_str(config.mode)
        self.is_running: bool = config.is_active
        self.start_time: float = time.time()
        self.enabled_symbols: set[str] = set([s.upper() for s in config.symbols])
        self.contract_specs: dict[str, ContractSpec] = contract_specs or {}

        # Live exchange metrics cache
        self.live_balance: float = 0.0
        self.live_equity: float = 0.0
        self.live_available_margin: float = 0.0
        self.live_unrealized_pnl: float = 0.0
        self.live_realized_pnl: float = 0.0
        self.live_positions: list[dict[str, Any]] = []
        self.last_sync_time: float = 0.0
        self.last_sync_error: str | None = None
        self.order_to_base_id: dict[str, str] = {}

        # BotConfig wrapper for rest client / live credentials
        self.bot_config = BotConfig(
            delta_api_key=config.delta_api_key,
            delta_api_secret=config.delta_api_secret,
            delta_base_url=config.delta_base_url,
            delta_ws_url=config.delta_ws_url,
            execution_mode=config.mode,
            default_leverage=config.leverage,
            default_risk_pct=config.risk_pct,
            timeframe=config.timeframe,
            target_symbols_str=",".join(config.symbols),
        )

        # Clients & State
        self.rest_client = DeltaRestClient(config=self.bot_config)
        self.state_manager = state_manager or TradingStateManager(client=self.rest_client)

        risk_cfg = RiskConfig(
            max_account_risk_pct=config.risk_pct,
            default_leverage=config.leverage,
            max_daily_drawdown_pct=config.max_drawdown_pct,
        )
        self.risk_manager = risk_manager or RiskManager(
            config=risk_cfg,
            initial_equity=config.initial_balance,
        )

        # Simulator or Live Execution Manager
        if self.mode == ExecutionMode.LIVE and config.delta_api_key and config.delta_api_secret:
            self.paper_simulator = None
            exec_cfg = ExecutionConfig(leverage=config.leverage)
            self.execution_manager: ExecutionManager | None = ExecutionManager(
                client=self.rest_client,
                config=exec_cfg,
                state_manager=self.state_manager,
            )
        else:
            self.execution_manager = None
            self.paper_simulator = paper_simulator or PaperExchangeSimulator(
                initial_balance=config.initial_balance,
                state_manager=self.state_manager,
            )
            self.paper_simulator.set_state_manager(self.state_manager)
            for spec in self.contract_specs.values():
                self.paper_simulator.register_contract_spec(spec)

        # Strategy Engines per symbol
        if strategies is not None:
            self.strategies = strategies
        else:
            self.strategies = {}
            for sym in self.enabled_symbols:
                s_cfg = BreakoutStrategyConfig(
                    symbol=sym,
                    resolution=config.timeframe,
                    risk_per_trade_pct=config.risk_pct,
                )
                strat = BreakoutStrategyEngine(config=s_cfg)
                self._register_strategy_callbacks(strat)
                self.strategies[sym] = strat

        # Kick off initial live metrics synchronization if live credentials exist
        if self.mode == ExecutionMode.LIVE and config.delta_api_key and config.delta_api_secret:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.sync_live_metrics())
            except RuntimeError:
                pass

    def _register_strategy_callbacks(self, strategy: BreakoutStrategyEngine) -> None:
        strategy.on_sl_ratchet(self._handle_sl_ratchet)
        strategy.on_order_cancel(self._handle_order_cancel)
        strategy.on_position_close(self._handle_position_close)

    async def _handle_sl_ratchet(self, intent: OrderIntent, old_sl: float, new_sl: float) -> None:
        if self.mode == ExecutionMode.PAPER and self.paper_simulator is not None:
            spec = self.contract_specs.get(intent.symbol)
            if spec:
                self.paper_simulator.edit_order_stop_loss(
                    product_id=spec.product_id,
                    order_id=intent.exchange_order_id or 0,
                    new_sl=new_sl,
                )
        elif self.mode == ExecutionMode.LIVE and self.execution_manager is not None:
            try:
                await self.execution_manager.handle_trailing_stop_ratchet(intent, old_sl, new_sl)
            except Exception as e:
                logger.error("[%s:%s] Error ratcheting SL: %s", self.config.tenant_id, intent.symbol, e)

    async def _handle_order_cancel(self, intent: OrderIntent, reason: str) -> None:
        if self.mode == ExecutionMode.PAPER and self.paper_simulator is not None:
            spec = self.contract_specs.get(intent.symbol)
            if spec and intent.exchange_order_id:
                self.paper_simulator.cancel_order(spec.product_id, intent.exchange_order_id)
        elif self.mode == ExecutionMode.LIVE and self.execution_manager is not None:
            try:
                await self.execution_manager.handle_invalidation(intent)
            except Exception as e:
                logger.error("[%s:%s] Error cancelling order: %s", self.config.tenant_id, intent.symbol, e)

    async def _handle_position_close(self, intent: OrderIntent, status: OrderStatus, exit_price: float) -> None:
        equity = await self.get_account_equity()
        self.risk_manager.record_trade_exit(
            pnl=intent.realized_pnl or 0.0,
            current_equity=equity,
            symbol=intent.symbol,
        )

    def register_contract_spec(self, spec: ContractSpec) -> None:
        """Caches contract specification and registers with paper simulator."""
        self.contract_specs[spec.symbol] = spec
        if self.paper_simulator is not None:
            self.paper_simulator.register_contract_spec(spec)

    async def get_account_equity(self) -> float:
        """Retrieves total floating equity for this tenant."""
        if self.mode == ExecutionMode.PAPER and self.paper_simulator is not None:
            return self.paper_simulator.wallet.equity
        if self.last_sync_time > 0:
            return self.live_equity
        return self.config.initial_balance

    def get_active_positions_count(self) -> int:
        if self.mode == ExecutionMode.PAPER and self.paper_simulator is not None:
            return len(self.paper_simulator._open_positions)
        if self.live_positions:
            return len(self.live_positions)
        return len(self.state_manager.get_all_positions())

    def set_execution_mode(self, mode: ExecutionMode | str) -> ExecutionMode:
        """Dynamically switches tenant between PAPER and LIVE execution modes."""
        new_mode = ExecutionMode.from_str(mode) if isinstance(mode, str) else mode
        if new_mode == ExecutionMode.LIVE:
            if not self.config.delta_api_key or not self.config.delta_api_secret:
                raise ConfigurationError(
                    f"Tenant '{self.config.name}' requires valid DELTA_API_KEY and DELTA_API_SECRET for LIVE mode."
                )
            if self.execution_manager is None:
                exec_cfg = ExecutionConfig(leverage=self.config.leverage)
                self.execution_manager = ExecutionManager(
                    client=self.rest_client,
                    config=exec_cfg,
                    state_manager=self.state_manager,
                )
            self.mode = ExecutionMode.LIVE
            self.config.mode = "live"
            logger.warning("[TENANT %s] Switched to LIVE trading mode", self.config.tenant_id)
        else:
            if self.paper_simulator is None:
                self.paper_simulator = PaperExchangeSimulator(
                    initial_balance=self.config.initial_balance,
                    state_manager=self.state_manager,
                )
                self.paper_simulator.set_state_manager(self.state_manager)
                for spec in self.contract_specs.values():
                    self.paper_simulator.register_contract_spec(spec)
            self.mode = ExecutionMode.PAPER
            self.config.mode = "paper"
            logger.info("[TENANT %s] Switched to PAPER simulation mode", self.config.tenant_id)
        return self.mode

    def reload_credentials(self) -> None:
        """Re-initializes REST client and ExecutionManager when user API keys change."""
        base_u = self.config.delta_base_url or "https://api.india.delta.exchange"
        ws_u = "wss://socket.india.delta.exchange" if "india" in base_u.lower() else self.config.delta_ws_url
        self.config.delta_base_url = base_u
        self.config.delta_ws_url = ws_u

        self.bot_config = BotConfig(
            delta_api_key=self.config.delta_api_key,
            delta_api_secret=self.config.delta_api_secret,
            delta_base_url=self.config.delta_base_url,
            delta_ws_url=self.config.delta_ws_url,
            execution_mode="live",
            default_leverage=self.config.leverage,
            default_risk_pct=self.config.risk_pct,
            timeframe=self.config.timeframe,
            target_symbols_str=",".join(self.config.symbols),
        )
        self.rest_client = DeltaRestClient(config=self.bot_config)
        self.state_manager = TradingStateManager(client=self.rest_client)
        if self.config.delta_api_key and self.config.delta_api_secret:
            exec_cfg = ExecutionConfig(leverage=self.config.leverage)
            self.execution_manager = ExecutionManager(
                client=self.rest_client,
                config=exec_cfg,
                state_manager=self.state_manager,
            )
            self.mode = ExecutionMode.LIVE
            self.config.mode = "live"
            self.paper_simulator = None
            logger.info("[TENANT %s] Live execution manager activated with user API credentials", self.config.tenant_id)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.sync_live_metrics())
            except RuntimeError:
                pass
        else:
            self.execution_manager = None
            self.mode = ExecutionMode.PAPER
            self.config.mode = "paper"
            self.live_balance = 0.0
            self.live_equity = 0.0
            self.live_available_margin = 0.0
            self.live_unrealized_pnl = 0.0
            self.live_realized_pnl = 0.0
            self.live_positions = []
            self.last_sync_time = 0.0
            if self.paper_simulator is None:
                self.paper_simulator = PaperExchangeSimulator(initial_balance=self.config.initial_balance)
                for spec in self.contract_specs.values():
                    self.paper_simulator.register_contract_spec(spec)

    async def sync_live_metrics(self) -> None:
        """Pulls real wallet balances, available margin, and positions from Delta Exchange India."""
        if not self.config.delta_api_key or not self.config.delta_api_secret:
            return
        try:
            balances = await self.rest_client.get_wallet_balances()
            if balances:
                target_bal = next(
                    (b for b in balances if b.asset_symbol.upper() == "USDT"),
                    None,
                )
                if not target_bal:
                    target_bal = max(balances, key=lambda b: b.balance)

                self.live_balance = round(float(target_bal.balance), 4)
                self.live_available_margin = round(float(target_bal.available_balance), 4)
                self.live_unrealized_pnl = round(float(target_bal.unrealized_pnl or 0.0), 4)
                self.live_equity = round(float(self.live_balance + self.live_unrealized_pnl), 4)

            positions = await self.rest_client.get_open_positions()
            formatted_positions: list[dict[str, Any]] = []
            pos_pnl_acc = 0.0

            for pos in positions:
                self.state_manager._open_positions[pos.product_id] = pos
                if pos.size != 0:
                    sym = pos.symbol
                    if not sym:
                        for s_sym, s_spec in self.contract_specs.items():
                            if s_spec.product_id == pos.product_id:
                                sym = s_sym
                                break
                    spec = self.contract_specs.get(sym) if sym else None
                    cv = spec.contract_value if spec else 1.0
                    native_val = round(abs(pos.size) * cv, 4)
                    alloc_margin = round(pos.margin, 2) if pos.margin > 0 else (
                        round((abs(pos.size) * cv * pos.entry_price) / max(1, self.config.leverage), 2)
                        if pos.entry_price > 0 else 0.0
                    )
                    liq_px = round(pos.liquidation_price, 4) if pos.liquidation_price else None
                    dist_pct = (
                        round(abs(pos.mark_price - liq_px) / pos.mark_price * 100.0, 2)
                        if (liq_px and pos.mark_price > 0)
                        else 0.0
                    )
                    child_sl = self.state_manager.get_child_stop_loss_order_id(pos.product_id)
                    formatted_positions.append({
                        "product_id": pos.product_id,
                        "symbol": sym or f"PRODUCT_{pos.product_id}",
                        "side": "BUY" if pos.size > 0 else "SELL",
                        "size": abs(pos.size),
                        "contract_value": cv,
                        "native_value": native_val,
                        "leverage": self.config.leverage,
                        "entry_price": round(pos.entry_price, 4),
                        "mark_price": round(pos.mark_price, 4),
                        "unrealized_pnl": round(pos.unrealized_pnl, 2),
                        "stop_loss_price": None,
                        "take_profit_price": None,
                        "child_stop_loss_order_id": child_sl,
                        "child_take_profit_order_id": None,
                        "allocated_margin": alloc_margin,
                        "liquidation_price": liq_px,
                        "liquidation_distance_pct": dist_pct,
                        "mode": "LIVE",
                    })
                    pos_pnl_acc += pos.unrealized_pnl

            self.live_positions = formatted_positions
            if formatted_positions and self.live_unrealized_pnl == 0.0:
                self.live_unrealized_pnl = round(pos_pnl_acc, 4)
                self.live_equity = round(self.live_balance + self.live_unrealized_pnl, 4)

            self.last_sync_time = time.time()
            self.last_sync_error = None
            logger.info(
                "[TENANT %s] Synced real Delta India metrics: Balance=%.2f, Equity=%.2f, Positions=%d",
                self.config.tenant_id,
                self.live_balance,
                self.live_equity,
                len(self.live_positions),
            )
        except AuthenticationError as auth_err:
            err_code = getattr(auth_err, "error_code", "")
            resp_body = getattr(auth_err, "response_body", {}) or {}
            client_ip = (resp_body.get("context") or {}).get("client_ip")
            if err_code == "ip_not_whitelisted_for_api_key" or "ip_not_whitelisted" in str(auth_err):
                self.last_sync_error = (
                    f"Render IP ({client_ip or '74.220.49.18'}) is not whitelisted. "
                    f"Please remove IP restriction or add {client_ip or '74.220.49.18'} in Delta Exchange API settings."
                )
            elif err_code in ("invalid_api_key", "unauthorized", "signature_mismatch"):
                self.last_sync_error = "Invalid API Key or Secret. Please verify your Delta Exchange India credentials."
            else:
                self.last_sync_error = f"Delta Authentication Failed: {getattr(auth_err, 'message', '') or err_code}"

            logger.error(
                "[TENANT %s] Delta Exchange authentication failed: %s | Hint: %s",
                self.config.tenant_id,
                auth_err,
                self.last_sync_error,
            )
        except Exception as e:
            self.last_sync_error = f"Live Sync Error: {e}"
            logger.warning(
                "[TENANT %s] Failed to sync live wallet metrics from Delta Exchange: %s",
                self.config.tenant_id,
                e,
            )

    async def emergency_flatten(self) -> dict[str, Any]:
        """Emergency market-closes all active positions and cancels resting orders for this tenant."""
        cancelled_orders = 0
        closed_positions = 0

        if self.mode == ExecutionMode.PAPER and self.paper_simulator is not None:
            cancelled_orders = len(self.paper_simulator._pending_orders)
            closed_positions = len(self.paper_simulator._open_positions)
            self.paper_simulator.flatten_all()
        elif self.mode == ExecutionMode.LIVE and self.execution_manager is not None:
            try:
                res = await self.execution_manager.emergency_flatten()
                cancelled_orders = res.get("cancelled_orders", 0)
                closed_positions = res.get("closed_positions", 0)
            except Exception as e:
                logger.error("[TENANT %s] Error during emergency flatten: %s", self.config.tenant_id, e)

        logger.warning(
            "[TENANT %s] EMERGENCY FLATTEN EXECUTED: Cancelled %d orders, Closed %d positions",
            self.config.tenant_id,
            cancelled_orders,
            closed_positions,
        )
        return {
            "tenant_id": self.config.tenant_id,
            "cancelled_orders": cancelled_orders,
            "closed_positions": closed_positions,
            "status": "success",
        }

    def get_summary(self) -> dict[str, Any]:
        """Generates a complete telemetry status dictionary for this tenant."""
        if self.mode == ExecutionMode.PAPER and self.paper_simulator is not None:
            stats = self.paper_simulator.get_summary()
            bal = stats["balance"]
            eq = stats["equity"]
            upnl = stats["unrealized_pnl"]
            rpnl = stats["realized_pnl"]
            win_rate = stats["win_rate_pct"]
            drawdown = stats["max_drawdown_pct"]
            open_pos = len(self.paper_simulator._open_positions)
            resting_orders = len(self.paper_simulator._pending_orders)
            avail_margin = stats["available_margin"]
        else:
            if self.last_sync_time > 0:
                bal = self.live_balance
                eq = self.live_equity
                avail_margin = self.live_available_margin
            else:
                bal = self.config.initial_balance
                eq = bal
                avail_margin = bal
            upnl = self.live_unrealized_pnl
            rpnl = self.live_realized_pnl
            win_rate = 0.0
            drawdown = 0.0
            open_pos = len(self.live_positions) if self.live_positions else len(self.state_manager.get_all_positions())
            act_src = (
                getattr(self.execution_manager.registry, "active_orders", [])
                if self.execution_manager and hasattr(self.execution_manager, "registry")
                else []
            )
            act_list = act_src() if callable(act_src) else (act_src or [])
            resting_orders = len(act_list)

        return {
            "tenant_id": self.config.tenant_id,
            "user_id": self.config.user_id,
            "name": self.config.name,
            "mode": self.mode.value.upper(),
            "is_running": self.is_running,
            "equity": round(eq, 2),
            "balance": round(bal, 2),
            "unrealized_pnl": round(upnl, 2),
            "realized_pnl": round(rpnl, 2),
            "win_rate_pct": round(win_rate, 2),
            "max_drawdown_pct": round(drawdown, 2),
            "available_margin": round(avail_margin, 2),
            "open_positions_count": open_pos,
            "resting_orders_count": resting_orders,
            "leverage": self.config.leverage,
            "risk_pct": self.config.risk_pct,
            "timeframe": self.config.timeframe,
            "symbols": list(self.enabled_symbols),
            "created_at": self.config.created_at,
            "has_api_key": bool(self.config.delta_api_key),
            "last_sync_error": self.last_sync_error,
        }


class TenantManager:
    """Registry and lifecycle coordinator managing all isolated tenant instances."""

    def __init__(
        self,
        storage_path: str = "data/tenants.json",
        default_instance: TenantInstance | None = None,
        supabase_url: str | None = None,
        supabase_secret_key: str | None = None,
        wal_path: str | None = None,
    ) -> None:
        self.storage_path = storage_path
        self.wal_path = wal_path or str(Path(storage_path).parent / "wal_buffer.json")
        self.supabase_url = supabase_url
        self.supabase_secret_key = supabase_secret_key
        self.tenants: dict[str, TenantInstance] = {}
        self.active_tenant_id: str = "default"
        self._inflight_signals: dict[str, float] = {}
        if default_instance:
            self.tenants[default_instance.config.tenant_id] = default_instance
            self.active_tenant_id = default_instance.config.tenant_id

    def get_tenant(self, tenant_id: str | None = None) -> TenantInstance | None:
        """Retrieves tenant instance by ID or user ID, falling back to active or default tenant."""
        if not tenant_id:
            tenant_id = self.active_tenant_id

        clean_id = str(tenant_id).strip()
        if clean_id in self.tenants:
            return self.tenants[clean_id]

        # Check by user_id or slug prefix
        by_user = self.get_tenant_by_user_id(clean_id)
        if by_user:
            return by_user

        if "default" in self.tenants:
            return self.tenants["default"]
        if self.tenants:
            return next(iter(self.tenants.values()))
        return None

    def get_tenant_by_user_id(self, user_id: str) -> TenantInstance | None:
        """Finds active tenant belonging to a Supabase user."""
        if not user_id:
            return None
        clean_uid = str(user_id).strip()
        for t in self.tenants.values():
            if t.config.user_id == clean_uid:
                return t
        prefix = f"user-{clean_uid[:8]}"
        for t_id, t in self.tenants.items():
            if t_id == clean_uid or t_id == prefix:
                t.config.user_id = clean_uid
                return t
        return None

    def get_or_create_user_tenant(
        self,
        user_id: str,
        email: str,
        contract_specs: dict[str, ContractSpec] | None = None,
    ) -> TenantInstance:
        """Retrieves existing tenant for user or provisions a new isolated one."""
        existing = self.get_tenant_by_user_id(user_id)
        if existing:
            return existing

        name_part = email.split("@")[0].replace(".", " ").title() if email else "User Desk"
        slug = f"user-{user_id[:8]}"
        if slug in self.tenants:
            slug = f"user-{user_id[:12]}"

        cfg = TenantConfig(
            tenant_id=slug,
            user_id=user_id,
            name=f"{name_part}'s Trading Desk",
            mode="live",
            initial_balance=10000.0,
            delta_base_url="https://api.india.delta.exchange",
            delta_ws_url="wss://socket.india.delta.exchange",
            leverage=5,
            risk_pct=1.0,
            timeframe="1m",
            max_drawdown_pct=4.0,
            symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
            is_active=True,
        )
        instance = TenantInstance(config=cfg, contract_specs=contract_specs or {})
        self.tenants[slug] = instance
        self.save_to_disk()
        self.sync_tenant_to_supabase(instance)
        logger.info("[USER TENANT PROVISIONED] Created isolated desk '%s' for user %s", slug, email)
        return instance

    def list_tenants(self) -> list[dict[str, Any]]:
        """Returns a list of status summaries across all registered tenants."""
        return [t.get_summary() for t in self.tenants.values()]

    def list_tenants_for_user(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Returns summaries filtered by user, or all if user_id is None (admin)."""
        if user_id is None:
            return [t.get_summary() for t in self.tenants.values()]

        clean_uid = str(user_id).strip()
        user_tenants = [t.get_summary() for t in self.tenants.values() if t.config.user_id == clean_uid]
        if not user_tenants:
            prefix = f"user-{clean_uid[:8]}"
            user_tenants = [
                t.get_summary() for t in self.tenants.values() if t.config.tenant_id in (clean_uid, prefix)
            ]
        return user_tenants

    def create_tenant(
        self,
        config: TenantConfig,
        contract_specs: dict[str, ContractSpec] | None = None,
    ) -> TenantInstance:
        """Creates and registers a new isolated tenant instance."""
        clean_id = self.sanitize_tenant_id(config.tenant_id)
        config.tenant_id = clean_id

        if clean_id in self.tenants:
            raise ValueError(f"Tenant with ID '{clean_id}' already exists.")

        instance = TenantInstance(config=config, contract_specs=contract_specs)
        self.tenants[clean_id] = instance
        self.save_to_disk()
        self.sync_tenant_to_supabase(instance)
        logger.info("[TENANT CREATED] Successfully initialized tenant '%s' (ID: %s)", config.name, clean_id)
        return instance

    def update_tenant(self, tenant_id: str, updates: dict[str, Any]) -> TenantInstance:
        """Updates metadata, credentials, or parameters for an existing tenant."""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            raise KeyError(f"Tenant '{tenant_id}' not found.")

        # Ensure Delta India regional settings
        if "delta_base_url" in updates and updates["delta_base_url"]:
            base_u = str(updates["delta_base_url"]).strip()
            if "india" in base_u.lower():
                updates["delta_ws_url"] = "wss://socket.india.delta.exchange"

        # Update configuration model
        current = tenant.config.model_dump()
        for k, v in updates.items():
            if v is not None and k in current:
                current[k] = v

        new_config = TenantConfig(**current)
        tenant.config = new_config

        # Apply runtime overrides
        if "is_active" in updates:
            tenant.is_running = bool(updates["is_active"])
        if "leverage" in updates:
            tenant.config.leverage = int(updates["leverage"])
        if "risk_pct" in updates:
            tenant.config.risk_pct = float(updates["risk_pct"])
        if "mode" in updates:
            tenant.set_execution_mode(updates["mode"])
        if (
            "delta_api_key" in updates
            or "delta_api_secret" in updates
            or "delta_base_url" in updates
            or "delta_ws_url" in updates
        ):
            tenant.reload_credentials()

        self.save_to_disk()
        self.sync_tenant_to_supabase(tenant)
        logger.info("[TENANT UPDATED] Updated settings for tenant ID '%s'", tenant.config.tenant_id)
        return tenant

    async def delete_tenant(self, tenant_id: str) -> bool:
        """Flattens positions and unregisters a tenant (Default tenant cannot be deleted)."""
        clean_id = tenant_id.strip()
        if clean_id == "default":
            raise ValueError("Root 'default' tenant cannot be deleted.")

        tenant = self.tenants.get(clean_id)
        if not tenant:
            return False

        # Flatten positions and cancel resting orders first
        await tenant.emergency_flatten()

        del self.tenants[clean_id]
        if self.active_tenant_id == clean_id:
            self.active_tenant_id = "default"

        self.save_to_disk()
        self.delete_tenant_from_supabase(clean_id)
        logger.info("[TENANT DELETED] Permanently removed tenant ID '%s'", clean_id)
        return True

    def sanitize_tenant_id(self, raw_id: str) -> str:
        """Converts raw string into a clean URL-friendly identifier."""
        clean = re.sub(r"[^a-zA-Z0-9_\-]", "", raw_id.strip().lower().replace(" ", "-"))
        return clean or f"tenant_{int(time.time())}"

    def save_to_disk(self) -> None:
        """Persists custom tenant configurations to JSON storage file with encrypted credentials."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
            serializable = {}
            for t_id, t in self.tenants.items():
                if t_id == "default":
                    continue  # Default tenant is synthesized from environment
                dump = t.config.model_dump()
                if dump.get("delta_api_key"):
                    dump["delta_api_key"] = encrypt_secret(dump["delta_api_key"])
                if dump.get("delta_api_secret"):
                    dump["delta_api_secret"] = encrypt_secret(dump["delta_api_secret"])
                serializable[t_id] = dump

            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
            logger.debug("Successfully saved %d tenants to %s", len(serializable), self.storage_path)
        except Exception as e:
            logger.error("Failed to persist tenants to %s: %s", self.storage_path, e)

    def load_from_disk(self, contract_specs: dict[str, ContractSpec]) -> None:
        """Loads and bootstraps saved tenant instances from JSON storage, decrypting credentials."""
        if not os.path.isfile(self.storage_path):
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            loaded = 0
            for t_id, c_dict in data.items():
                if t_id in self.tenants:
                    continue
                c_copy = dict(c_dict)
                if c_copy.get("delta_api_key"):
                    c_copy["delta_api_key"] = decrypt_secret(c_copy["delta_api_key"])
                if c_copy.get("delta_api_secret"):
                    c_copy["delta_api_secret"] = decrypt_secret(c_copy["delta_api_secret"])
                cfg = TenantConfig(**c_copy)
                instance = TenantInstance(config=cfg, contract_specs=contract_specs)
                self.tenants[t_id] = instance
                loaded += 1
            logger.info("Loaded %d custom tenants from %s", loaded, self.storage_path)
        except Exception as e:
            logger.error("Error reading tenants from %s: %s", self.storage_path, e)

    def load_from_supabase(self, contract_specs: dict[str, ContractSpec]) -> int:
        """Loads and bootstraps saved tenant instances from Supabase PostgreSQL."""
        if not self.supabase_url or not self.supabase_secret_key:
            return 0

        try:
            import httpx
            headers = {
                "apikey": self.supabase_secret_key,
                "Authorization": f"Bearer {self.supabase_secret_key}",
            }
            url = f"{self.supabase_url.rstrip('/')}/rest/v1/tenants?select=*"
            with httpx.Client(timeout=10.0) as client:
                res = client.get(url, headers=headers)
            if res.status_code != 200:
                logger.warning("Supabase load tenants HTTP status %d: %s", res.status_code, res.text)
                return 0

            data = res.json()
            loaded = 0
            for row in data:
                t_id = row.get("tenant_id")
                if not t_id or t_id == "default":
                    continue
                if t_id in self.tenants:
                    if row.get("user_id"):
                        self.tenants[t_id].config.user_id = str(row["user_id"])
                    continue
                raw_k = decrypt_secret(row.get("delta_api_key") or "")
                raw_s = decrypt_secret(row.get("delta_api_secret") or "")
                cfg = TenantConfig(
                    tenant_id=t_id,
                    user_id=str(row["user_id"]) if row.get("user_id") else None,
                    name=row.get("name", t_id),
                    mode=row.get("mode", "paper"),
                    initial_balance=float(row.get("initial_balance", 10000.0)),
                    delta_api_key=raw_k,
                    delta_api_secret=raw_s,
                    delta_base_url=row.get("delta_base_url") or "https://api.india.delta.exchange",
                    delta_ws_url=row.get("delta_ws_url") or "wss://socket.india.delta.exchange",
                    risk_pct=float(row.get("risk_pct", 1.0)),
                    leverage=int(row.get("leverage", 5)),
                    timeframe=row.get("timeframe", "1m"),
                    max_drawdown_pct=float(row.get("max_drawdown_pct", 4.0)),
                    symbols=row.get("symbols") or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
                    is_active=bool(row.get("is_active", True)),
                    created_at=row.get("created_at") or datetime.now(timezone.utc).isoformat(),
                )
                inst = TenantInstance(config=cfg, contract_specs=contract_specs)
                if inst.paper_simulator and "balance" in row:
                    inst.paper_simulator.wallet.balance = float(row["balance"])
                    if "initial_balance" in row and row["initial_balance"]:
                        inst.paper_simulator.wallet.initial_balance = float(row["initial_balance"])
                self.tenants[t_id] = inst
                loaded += 1
            logger.info("Loaded %d tenants from Supabase", loaded)
            return loaded
        except Exception as e:
            logger.error("Error reading tenants from Supabase: %s", e)
            return 0

    def _write_to_wal(self, action: str, payload: dict[str, Any]) -> None:
        """Appends failed cloud transaction to Write-Ahead Log (WAL) for offline recovery."""
        logger.warning(
            "[WAL FAILOVER] Supabase operation '%s' failed. Safely queueing to local write-ahead log (%s). Zero data lost!",
            action,
            self.wal_path,
        )
        try:
            records: list[dict[str, Any]] = []
            wal_file = Path(self.wal_path)
            if wal_file.exists():
                try:
                    with open(wal_file, "r", encoding="utf-8") as f:
                        records = json.load(f)
                    if not isinstance(records, list):
                        records = []
                except Exception:
                    records = []

            # Deduplicate by action & tenant_id
            tenant_id = payload.get("tenant_id")
            if tenant_id:
                records = [
                    r for r in records
                    if not (r.get("action") == action and r.get("payload", {}).get("tenant_id") == tenant_id)
                ]

            records.append({
                "action": action,
                "payload": payload,
                "timestamp": time.time(),
            })

            wal_file.parent.mkdir(parents=True, exist_ok=True)
            with open(wal_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
        except Exception as e:
            logger.error("[WAL FATAL ERROR] Failed to write to WAL buffer: %s", e)

    def flush_wal_buffer(self) -> int:
        """
        Replays buffered offline transactions to Supabase with deduplication.
        Handles PostgreSQL unique constraint violations (23505) and duplicate keys idempotently.
        Returns number of successfully replayed transactions.
        """
        if not self.supabase_url or not self.supabase_secret_key:
            return 0

        wal_file = Path(self.wal_path)
        if not wal_file.exists():
            return 0

        try:
            with open(wal_file, "r", encoding="utf-8") as f:
                records = json.load(f)
            if not isinstance(records, list) or not records:
                return 0

            logger.info("[WAL RECOVERY] Replaying %d buffered transaction(s) to Supabase...", len(records))
            import httpx

            remaining: list[dict[str, Any]] = []
            replayed_count = 0

            for item in records:
                action = item.get("action")
                payload = item.get("payload", {})
                success = False

                try:
                    if action == "upsert_tenant":
                        headers = {
                            "apikey": self.supabase_secret_key,
                            "Authorization": f"Bearer {self.supabase_secret_key}",
                            "Content-Type": "application/json",
                            "Prefer": "resolution=merge-duplicates",
                        }
                        url = f"{self.supabase_url.rstrip('/')}/rest/v1/tenants"
                        with httpx.Client(timeout=10.0) as client:
                            res = client.post(url, headers=headers, json=payload)
                        if res.status_code in (200, 201):
                            success = True
                        elif "23505" in res.text or "duplicate key" in res.text.lower():
                            logger.info(
                                "[WAL DEDUPLICATION] Tenant %s already present in cloud ledger. Discarding duplicate.",
                                payload.get("tenant_id"),
                            )
                            success = True
                        else:
                            logger.warning(
                                "[WAL REPLAY FAILED] Upsert tenant %s failed with status %d: %s",
                                payload.get("tenant_id"),
                                res.status_code,
                                res.text,
                            )
                    elif action == "delete_tenant":
                        tid = payload.get("tenant_id")
                        headers = {
                            "apikey": self.supabase_secret_key,
                            "Authorization": f"Bearer {self.supabase_secret_key}",
                        }
                        url = f"{self.supabase_url.rstrip('/')}/rest/v1/tenants?tenant_id=eq.{tid}"
                        with httpx.Client(timeout=10.0) as client:
                            res = client.delete(url, headers=headers)
                        if res.status_code in (200, 204, 404):
                            success = True
                        else:
                            logger.warning(
                                "[WAL REPLAY FAILED] Delete tenant %s failed with status %d",
                                tid,
                                res.status_code,
                            )
                except Exception as err:
                    err_str = str(err)
                    if "23505" in err_str or "duplicate key" in err_str.lower():
                        logger.info(
                            "[WAL DEDUPLICATION] Tenant %s already present in cloud ledger.",
                            payload.get("tenant_id"),
                        )
                        success = True
                    else:
                        logger.warning("[WAL REPLAY FAILED] Re-buffering item due to: %s", err)
                        success = False

                if success:
                    replayed_count += 1
                else:
                    remaining.append(item)

            if remaining:
                with open(wal_file, "w", encoding="utf-8") as f:
                    json.dump(remaining, f, indent=2)
            else:
                try:
                    wal_file.unlink()
                except Exception:
                    with open(wal_file, "w", encoding="utf-8") as f:
                        json.dump([], f)
                logger.info("[WAL RECOVERY] All offline transactions successfully synced to cloud ledger!")

            return replayed_count
        except Exception as e:
            logger.error("[WAL FLUSH ERROR] %s", e)
            return 0

    def sync_tenant_to_supabase(self, instance: TenantInstance) -> bool:
        """Synchronizes tenant state and encrypted configuration to Supabase PostgreSQL."""
        if not self.supabase_url or not self.supabase_secret_key:
            return False

        try:
            import httpx
            headers = {
                "apikey": self.supabase_secret_key,
                "Authorization": f"Bearer {self.supabase_secret_key}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates",
            }
            url = f"{self.supabase_url.rstrip('/')}/rest/v1/tenants"

            if instance.mode == ExecutionMode.PAPER and instance.paper_simulator:
                bal = instance.paper_simulator.wallet.balance
                eq = instance.paper_simulator.wallet.equity
                upnl = instance.paper_simulator.wallet.unrealized_pnl
                rpnl = instance.paper_simulator.wallet.realized_pnl
            else:
                bal = instance.live_balance if instance.last_sync_time > 0 else instance.config.initial_balance
                eq = instance.live_equity if instance.last_sync_time > 0 else bal
                upnl = instance.live_unrealized_pnl
                rpnl = instance.live_realized_pnl

            enc_key = encrypt_secret(instance.config.delta_api_key) if instance.config.delta_api_key else None
            enc_secret = encrypt_secret(instance.config.delta_api_secret) if instance.config.delta_api_secret else None

            payload = {
                "tenant_id": instance.config.tenant_id,
                "user_id": instance.config.user_id,
                "name": instance.config.name,
                "mode": instance.config.mode,
                "initial_balance": float(instance.config.initial_balance),
                "balance": round(float(bal), 4),
                "equity": round(float(eq), 4),
                "realized_pnl": round(float(rpnl), 4),
                "unrealized_pnl": round(float(upnl), 4),
                "leverage": int(instance.config.leverage),
                "risk_pct": float(instance.config.risk_pct),
                "timeframe": instance.config.timeframe,
                "max_drawdown_pct": float(instance.config.max_drawdown_pct),
                "delta_api_key": enc_key,
                "delta_api_secret": enc_secret,
                "symbols": instance.config.symbols,
                "is_running": instance.is_running,
                "is_active": instance.config.is_active,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            with httpx.Client(timeout=10.0) as client:
                res = client.post(url, headers=headers, json=payload)
            if res.status_code in (200, 201):
                self.flush_wal_buffer()
                return True
            else:
                logger.warning(
                    "Supabase returned status %d when syncing tenant %s. Queuing to WAL.",
                    res.status_code,
                    instance.config.tenant_id,
                )
                self._write_to_wal("upsert_tenant", payload)
                return False
        except Exception as e:
            logger.debug("Failed to sync tenant %s to Supabase: %s. Queuing to WAL.", instance.config.tenant_id, e)
            try:
                self._write_to_wal("upsert_tenant", payload)
            except Exception:
                pass
            return False

    def delete_tenant_from_supabase(self, tenant_id: str) -> bool:
        """Removes tenant record from Supabase PostgreSQL."""
        if not self.supabase_url or not self.supabase_secret_key:
            return False
        try:
            import httpx
            headers = {
                "apikey": self.supabase_secret_key,
                "Authorization": f"Bearer {self.supabase_secret_key}",
            }
            url = f"{self.supabase_url.rstrip('/')}/rest/v1/tenants?tenant_id=eq.{tenant_id}"
            with httpx.Client(timeout=10.0) as client:
                res = client.delete(url, headers=headers)
            if res.status_code in (200, 204):
                self.flush_wal_buffer()
                return True
            else:
                self._write_to_wal("delete_tenant", {"tenant_id": tenant_id})
                return False
        except Exception as e:
            logger.debug("Failed to delete tenant %s from Supabase: %s. Queuing to WAL.", tenant_id, e)
            self._write_to_wal("delete_tenant", {"tenant_id": tenant_id})
            return False

    # -------------------------------------------------------------------------
    # Shared Market Data Event Fan-Out (Multi-Tenant Dispatcher)
    # -------------------------------------------------------------------------

    async def handle_ticker(self, symbol: str, ticker: Ticker, orderbook: OrderBook | None) -> None:
        """Dispatches real-time ticker to each tenant's paper matching engine or state context."""
        for tenant in self.tenants.values():
            clean = symbol.strip().upper()
            if tenant.mode == ExecutionMode.PAPER and tenant.paper_simulator is not None:
                await tenant.paper_simulator.on_tick(
                    symbol=clean,
                    mark_price=ticker.mark_price,
                    orderbook=orderbook,
                )
            elif tenant.mode == ExecutionMode.LIVE:
                await tenant.state_manager.update_market_context(
                    symbol=clean,
                    mark_price=ticker.mark_price,
                )

    async def handle_candle_update(
        self, symbol: str, candle: Candle, orderbook: OrderBook | None
    ) -> None:
        """Dispatches intra-bar candle update to all tenants."""
        for tenant in self.tenants.values():
            clean = symbol.strip().upper()
            if tenant.mode == ExecutionMode.PAPER and tenant.paper_simulator is not None:
                await tenant.paper_simulator.on_tick(
                    symbol=clean,
                    mark_price=candle.close,
                    high=candle.high,
                    low=candle.low,
                    orderbook=orderbook,
                )
            elif tenant.mode == ExecutionMode.LIVE:
                await tenant.state_manager.update_market_context(
                    symbol=clean,
                    mark_price=candle.close,
                    high=candle.high,
                    low=candle.low,
                )

    async def handle_candle_close(
        self, symbol: str, candle: Candle, contract_specs: dict[str, ContractSpec]
    ) -> None:
        """
        Dispatches confirmed closed bar to each active tenant's BreakoutStrategyEngine.
        Evaluates signals, applies isolated tenant margin sizing, and routes orders.
        """
        clean_symbol = symbol.strip().upper()
        spec = contract_specs.get(clean_symbol)
        if not spec:
            return

        base_signal_id = f"SIG_{clean_symbol}_{int(candle.timestamp)}"

        for tenant in list(self.tenants.values()):
            if tenant.config.tenant_id == "default":
                continue  # Default tenant order is executed directly by the master orchestrator
            if not tenant.is_running or clean_symbol not in tenant.enabled_symbols:
                continue

            strategy = tenant.strategies.get(clean_symbol)
            if not strategy:
                continue

            tenant.register_contract_spec(spec)

            # Strategy Breakout Evaluation for this tenant
            order_intent = strategy.on_candle_close(candle)
            if order_intent is None:
                continue

            # Deterministic multi-tenant order ID mapping (UUIDv5)
            order_intent.base_signal_id = base_signal_id
            det_client_id = generate_deterministic_client_order_id(base_signal_id, tenant.config.tenant_id)
            order_intent.client_order_id = det_client_id
            tenant.order_to_base_id[det_client_id] = base_signal_id

            account_equity = await tenant.get_account_equity()

            # Pre-trade Risk Limit Verification for this tenant
            active_pos_count = tenant.get_active_positions_count()
            can_open, reject_reason = tenant.risk_manager.can_open_new_position(active_pos_count, symbol=clean_symbol)
            if not can_open:
                logger.warning(
                    "[TENANT %s:%s] Order intent rejected by RiskManager: %s",
                    tenant.config.tenant_id,
                    clean_symbol,
                    reject_reason,
                )
                order_intent.cancel(reject_reason or "RiskManager rejection", OrderStatus.CANCELLED_MANUAL)
                continue

            # Route to tenant simulator or live manager
            if tenant.mode == ExecutionMode.PAPER and tenant.paper_simulator is not None:
                try:
                    avail_margin = (
                        tenant.paper_simulator.wallet.available_balance
                        if tenant.paper_simulator.wallet.available_balance > 0
                        else account_equity
                    )
                    margin_res = tenant.risk_manager.size_position(
                        spec=spec,
                        trigger_price=order_intent.trigger_price,
                        stop_loss_price=order_intent.initial_stop_loss,
                        account_equity=account_equity,
                        available_margin=avail_margin,
                        risk_pct_override=tenant.config.risk_pct,
                    )
                    if not margin_res.is_margin_sufficient or margin_res.final_lots < 1:
                        logger.warning(
                            "[TENANT %s:%s] Paper sizing rejected by margin check: %s",
                            tenant.config.tenant_id,
                            clean_symbol,
                            margin_res.rejection_reason,
                        )
                        continue

                    order_intent.contract_lots = margin_res.final_lots
                    order_intent.product_id = spec.product_id
                    order_intent.leverage = tenant.config.leverage

                    side = "buy" if order_intent.signal_type == SignalType.BULLISH_BREAKOUT else "sell"
                    v_order = tenant.paper_simulator.submit_stop_order(
                        product_id=spec.product_id,
                        symbol=spec.symbol,
                        side=side,
                        size=order_intent.contract_lots,
                        stop_price=order_intent.trigger_price,
                        bracket_stop_loss_price=order_intent.initial_stop_loss,
                        bracket_take_profit_price=order_intent.take_profit,
                        leverage=tenant.config.leverage,
                        intent_id=order_intent.intent_id,
                        client_order_id=det_client_id,
                    )
                    order_intent.exchange_order_id = v_order.id
                    tenant.state_manager.register_intent(order_intent)
                    tenant.state_manager.bind_exchange_order_id(order_intent.intent_id, v_order.id)
                    tenant.risk_manager.record_trade_entry()
                    logger.info(
                        "[TENANT %s:%s] Placed paper breakout stop order #%d (%s, Lots=%d, Stop=%.4f)",
                        tenant.config.tenant_id,
                        clean_symbol,
                        v_order.id,
                        side.upper(),
                        v_order.size,
                        v_order.stop_price,
                    )
                except Exception as e:
                    logger.error(
                        "[TENANT %s:%s] Error submitting paper order: %s",
                        tenant.config.tenant_id,
                        clean_symbol,
                        e,
                    )
            elif tenant.mode == ExecutionMode.LIVE and tenant.execution_manager is not None:
                try:
                    await tenant.execution_manager.execute_signal(
                        order_intent=order_intent,
                        account_equity=account_equity,
                    )
                except Exception as e:
                    logger.error(
                        "[TENANT %s:%s] Error executing live signal: %s",
                        tenant.config.tenant_id,
                        clean_symbol,
                        e,
                    )

    async def fanout_order_intent(
        self,
        base_signal_id: str,
        master_intent: OrderIntent,
        spec: ContractSpec,
    ) -> list[str]:
        """
        Fans out a master breakout order intent across all active tenants using deterministic UUIDv5.
        Applies tenant-specific risk limits, leverage, and available margin constraints.
        Returns list of tenant IDs that executed the order.
        """
        executed_tenants: list[str] = []
        clean_symbol = spec.symbol.strip().upper()

        now_ts = time.time()
        self._inflight_signals = {
            k: v for k, v in self._inflight_signals.items() if now_ts - v < 5.0
        }
        if base_signal_id in self._inflight_signals:
            logger.warning(
                "[FANOUT ANTI-FLOOD] Signal %s already dispatched within 5s window. Dropping duplicate fan-out.",
                base_signal_id,
            )
            return []
        self._inflight_signals[base_signal_id] = now_ts

        for tenant in list(self.tenants.values()):
            if tenant.config.tenant_id == "default":
                continue  # Default tenant order is executed directly by the master orchestrator
            if not tenant.is_running or clean_symbol not in tenant.enabled_symbols:
                continue

            tenant.register_contract_spec(spec)
            account_equity = await tenant.get_account_equity()

            # Pre-trade Risk Limit Verification
            active_pos_count = tenant.get_active_positions_count()
            can_open, reject_reason = tenant.risk_manager.can_open_new_position(active_pos_count, symbol=clean_symbol)
            if not can_open:
                logger.warning(
                    "[FANOUT %s:%s] Order rejected by RiskManager: %s",
                    tenant.config.tenant_id,
                    clean_symbol,
                    reject_reason,
                )
                continue

            # Deterministic multi-tenant order ID mapping (UUIDv5)
            det_client_id = generate_deterministic_client_order_id(base_signal_id, tenant.config.tenant_id)
            tenant_intent = master_intent.model_copy()
            tenant_intent.intent_id = uuid.uuid4().hex[:12]
            tenant_intent.base_signal_id = base_signal_id
            tenant_intent.client_order_id = det_client_id
            tenant.order_to_base_id[det_client_id] = base_signal_id

            if tenant.mode == ExecutionMode.PAPER and tenant.paper_simulator is not None:
                try:
                    avail_margin = (
                        tenant.paper_simulator.wallet.available_balance
                        if tenant.paper_simulator.wallet.available_balance > 0
                        else account_equity
                    )
                    margin_res = tenant.risk_manager.size_position(
                        spec=spec,
                        trigger_price=tenant_intent.trigger_price,
                        stop_loss_price=tenant_intent.initial_stop_loss,
                        account_equity=account_equity,
                        available_margin=avail_margin,
                        risk_pct_override=tenant.config.risk_pct,
                    )
                    if not margin_res.is_margin_sufficient or margin_res.final_lots < 1:
                        logger.warning(
                            "[FANOUT %s:%s] Paper sizing rejected by margin check: %s",
                            tenant.config.tenant_id,
                            clean_symbol,
                            margin_res.rejection_reason,
                        )
                        continue

                    tenant_intent.contract_lots = margin_res.final_lots
                    tenant_intent.product_id = spec.product_id
                    tenant_intent.leverage = tenant.config.leverage

                    side = "buy" if tenant_intent.signal_type == SignalType.BULLISH_BREAKOUT else "sell"
                    v_order = tenant.paper_simulator.submit_stop_order(
                        product_id=spec.product_id,
                        symbol=spec.symbol,
                        side=side,
                        size=tenant_intent.contract_lots,
                        stop_price=tenant_intent.trigger_price,
                        bracket_stop_loss_price=tenant_intent.initial_stop_loss,
                        bracket_take_profit_price=tenant_intent.take_profit,
                        leverage=tenant.config.leverage,
                        intent_id=tenant_intent.intent_id,
                        client_order_id=det_client_id,
                    )
                    tenant_intent.exchange_order_id = v_order.id
                    tenant.state_manager.register_intent(tenant_intent)
                    tenant.state_manager.bind_exchange_order_id(tenant_intent.intent_id, v_order.id)
                    tenant.risk_manager.record_trade_entry()
                    executed_tenants.append(tenant.config.tenant_id)
                    logger.info(
                        "[FANOUT %s:%s] Placed paper breakout order #%d | ClientID=%s (Lots=%d, Stop=%.4f)",
                        tenant.config.tenant_id,
                        clean_symbol,
                        v_order.id,
                        det_client_id,
                        v_order.size,
                        v_order.stop_price,
                    )
                except Exception as e:
                    logger.error("[FANOUT %s:%s] Paper order failed: %s", tenant.config.tenant_id, clean_symbol, e)
            elif tenant.mode == ExecutionMode.LIVE and tenant.execution_manager is not None:
                try:
                    await tenant.execution_manager.execute_signal(
                        order_intent=tenant_intent,
                        account_equity=account_equity,
                    )
                    executed_tenants.append(tenant.config.tenant_id)
                except Exception as e:
                    logger.error("[FANOUT %s:%s] Live order failed: %s", tenant.config.tenant_id, clean_symbol, e)
            elif tenant.paper_simulator is not None:
                # Fallback to paper simulator if live credentials are not yet configured
                try:
                    avail_margin = (
                        tenant.paper_simulator.wallet.available_balance
                        if tenant.paper_simulator.wallet.available_balance > 0
                        else account_equity
                    )
                    margin_res = tenant.risk_manager.size_position(
                        spec=spec,
                        trigger_price=tenant_intent.trigger_price,
                        stop_loss_price=tenant_intent.initial_stop_loss,
                        account_equity=account_equity,
                        available_margin=avail_margin,
                        risk_pct_override=tenant.config.risk_pct,
                    )
                    if margin_res.is_margin_sufficient and margin_res.final_lots >= 1:
                        tenant_intent.contract_lots = margin_res.final_lots
                        tenant_intent.product_id = spec.product_id
                        tenant_intent.leverage = tenant.config.leverage
                        side = "buy" if tenant_intent.signal_type == SignalType.BULLISH_BREAKOUT else "sell"
                        v_order = tenant.paper_simulator.submit_stop_order(
                            product_id=spec.product_id,
                            symbol=spec.symbol,
                            side=side,
                            size=tenant_intent.contract_lots,
                            stop_price=tenant_intent.trigger_price,
                            bracket_stop_loss_price=tenant_intent.initial_stop_loss,
                            bracket_take_profit_price=tenant_intent.take_profit,
                            leverage=tenant.config.leverage,
                            intent_id=tenant_intent.intent_id,
                            client_order_id=det_client_id,
                        )
                        tenant_intent.exchange_order_id = v_order.id
                        tenant.state_manager.register_intent(tenant_intent)
                        tenant.state_manager.bind_exchange_order_id(tenant_intent.intent_id, v_order.id)
                        tenant.risk_manager.record_trade_entry()
                        executed_tenants.append(tenant.config.tenant_id)
                except Exception as e:
                    logger.error("[FANOUT %s:%s] Fallback paper order failed: %s", tenant.config.tenant_id, clean_symbol, e)

        return executed_tenants

    async def handle_session_rollover(self, symbol: str, session: SessionAnchorData) -> None:
        """Resets daily circuit breaker for all tenants at 00:00 UTC."""
        clean_symbol = symbol.strip().upper()
        for tenant in self.tenants.values():
            eq = await tenant.get_account_equity()
            tenant.risk_manager.reset_daily_circuit_breaker(current_equity=eq)
            logger.info(
                "[TENANT %s:%s] Reset daily circuit breaker at Equity: $%.2f",
                tenant.config.tenant_id,
                clean_symbol,
                eq,
            )
