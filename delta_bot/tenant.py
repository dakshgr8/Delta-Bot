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
from datetime import datetime, timezone
import json
import logging
import os
import re
import time
from typing import Any

from pydantic import BaseModel, Field

from delta_bot.client import DeltaRestClient
from delta_bot.config import BotConfig
from delta_bot.exceptions import ConfigurationError
from delta_bot.execution import ExecutionConfig, ExecutionManager
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


class TenantConfig(BaseModel):
    """Data specification and serialized configuration for an isolated tenant."""

    tenant_id: str
    user_id: str | None = None
    name: str
    mode: str = "live"  # Exclusively "live" for real trading
    initial_balance: float = 10000.0
    delta_api_key: str = ""
    delta_api_secret: str = ""
    delta_base_url: str = "https://api.delta.exchange"
    delta_ws_url: str = "wss://socket.delta.exchange"
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
        if d.get("delta_api_key"):
            k = d["delta_api_key"]
            d["delta_api_key_masked"] = f"{k[:4]}...{k[-4:]}" if len(k) >= 8 else "********"
            d["has_api_key"] = True
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
        return self.config.initial_balance

    def get_active_positions_count(self) -> int:
        if self.mode == ExecutionMode.PAPER and self.paper_simulator is not None:
            return len(self.paper_simulator._open_positions)
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
        else:
            self.execution_manager = None

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
            bal = self.config.initial_balance
            eq = bal
            upnl = 0.0
            rpnl = 0.0
            win_rate = 0.0
            drawdown = 0.0
            open_pos = len(self.state_manager.get_all_positions())
            resting_orders = (
                len(self.execution_manager.registry.active_orders()) if self.execution_manager else 0
            )
            avail_margin = bal

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
        }


class TenantManager:
    """Registry and lifecycle coordinator managing all isolated tenant instances."""

    def __init__(
        self,
        storage_path: str = "data/tenants.json",
        default_instance: TenantInstance | None = None,
        supabase_url: str | None = None,
        supabase_secret_key: str | None = None,
    ) -> None:
        self.storage_path = storage_path
        self.supabase_url = supabase_url
        self.supabase_secret_key = supabase_secret_key
        self.tenants: dict[str, TenantInstance] = {}
        self.active_tenant_id: str = "default"
        if default_instance:
            self.tenants[default_instance.config.tenant_id] = default_instance
            self.active_tenant_id = default_instance.config.tenant_id

    def get_tenant(self, tenant_id: str | None = None) -> TenantInstance | None:
        """Retrieves tenant instance by ID, falling back to active or default tenant."""
        if not tenant_id:
            tenant_id = self.active_tenant_id

        clean_id = tenant_id.strip()
        if clean_id in self.tenants:
            return self.tenants[clean_id]
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
        logger.info("[TENANT UPDATED] Updated settings for tenant ID '%s'", tenant_id)
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
        """Persists custom tenant configurations to JSON storage file."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
            serializable = {}
            for t_id, t in self.tenants.items():
                if t_id == "default":
                    continue  # Default tenant is synthesized from environment
                serializable[t_id] = t.config.model_dump()

            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
            logger.debug("Successfully saved %d tenants to %s", len(serializable), self.storage_path)
        except Exception as e:
            logger.error("Failed to persist tenants to %s: %s", self.storage_path, e)

    def load_from_disk(self, contract_specs: dict[str, ContractSpec]) -> None:
        """Loads and bootstraps saved tenant instances from JSON storage."""
        if not os.path.isfile(self.storage_path):
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            loaded = 0
            for t_id, c_dict in data.items():
                if t_id in self.tenants:
                    continue
                cfg = TenantConfig(**c_dict)
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
                cfg = TenantConfig(
                    tenant_id=t_id,
                    user_id=str(row["user_id"]) if row.get("user_id") else None,
                    name=row.get("name", t_id),
                    mode=row.get("mode", "paper"),
                    initial_balance=float(row.get("initial_balance", 10000.0)),
                    delta_api_key=row.get("delta_api_key") or "",
                    delta_api_secret=row.get("delta_api_secret") or "",
                    delta_base_url=row.get("delta_base_url") or "https://api.delta.exchange",
                    delta_ws_url=row.get("delta_ws_url") or "wss://socket.delta.exchange",
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

    def sync_tenant_to_supabase(self, instance: TenantInstance) -> bool:
        """Synchronizes tenant state and configuration to Supabase PostgreSQL."""
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

            bal = (
                instance.paper_simulator.wallet.balance
                if instance.paper_simulator
                else instance.config.initial_balance
            )
            eq = (
                instance.paper_simulator.wallet.equity
                if instance.paper_simulator
                else instance.config.initial_balance
            )
            upnl = (
                instance.paper_simulator.wallet.unrealized_pnl
                if instance.paper_simulator
                else 0.0
            )
            rpnl = (
                instance.paper_simulator.wallet.realized_pnl
                if instance.paper_simulator
                else 0.0
            )

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
                "delta_api_key": instance.config.delta_api_key or None,
                "delta_api_secret": instance.config.delta_api_secret or None,
                "symbols": instance.config.symbols,
                "is_running": instance.is_running,
                "is_active": instance.config.is_active,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            with httpx.Client(timeout=10.0) as client:
                res = client.post(url, headers=headers, json=payload)
            return res.status_code in (200, 201)
        except Exception as e:
            logger.debug("Failed to sync tenant %s to Supabase: %s", instance.config.tenant_id, e)
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
            return res.status_code in (200, 204)
        except Exception as e:
            logger.debug("Failed to delete tenant %s from Supabase: %s", tenant_id, e)
            return False

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            loaded = 0
            for t_id, c_dict in data.items():
                if t_id in self.tenants:
                    continue
                cfg = TenantConfig(**c_dict)
                instance = TenantInstance(config=cfg, contract_specs=contract_specs)
                self.tenants[t_id] = instance
                loaded += 1
            logger.info("Loaded %d custom tenants from %s", loaded, self.storage_path)
        except Exception as e:
            logger.error("Error reading tenants from %s: %s", self.storage_path, e)

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

        for tenant in list(self.tenants.values()):
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

            account_equity = await tenant.get_account_equity()

            # Pre-trade Risk Limit Verification for this tenant
            active_pos_count = tenant.get_active_positions_count()
            can_open, reject_reason = tenant.risk_manager.can_open_new_position(active_pos_count)
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
