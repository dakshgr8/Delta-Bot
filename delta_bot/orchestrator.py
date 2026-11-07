"""
delta_bot.orchestrator
~~~~~~~~~~~~~~~~~~~~~~

Final Integration, Orchestrator Daemon, and Paper/Live Deployment Engine (Phase 6).
Serves as the operational core binding all Delta-Bot modules:
- Task 1: Public Market Data Engine (DeltaWebSocketClient multi-symbol streaming & REST warm-up)
- Task 2: Private WebSocket Engine (PrivateWebSocketClient authenticated real-time updates)
- Task 3: Strategy & Signal Dispatcher (BreakoutStrategyEngine on confirmed candle closes)
- Task 4: State Reconciliation Worker (TradingStateManager periodic REST fallback sync)
- Task 5: Daily Session Rollover Engine (00:00 UTC PDH/PDL update & circuit breaker reset)
- Task 6: Watchdog & Health Telemetry Heartbeat (Terminal dashboard & system health monitoring)
- Dual Execution Modes: Paper Trading (in-memory simulator) vs Live Exchange Execution
- Graceful POSIX Signal Trapping (SIGINT / SIGTERM) & Emergency Flatten Controls
"""

import asyncio
from datetime import datetime, timezone
import logging
import signal
import sys
import time
from typing import Any

from delta_bot.client import DeltaRestClient
from delta_bot.config import BotConfig, get_config
from delta_bot.exceptions import ConfigurationError
from delta_bot.execution import ExecutionConfig, ExecutionManager
from delta_bot.models import (
    Candle,
    ContractSpec,
    ExecutionMode,
    OrderBook,
    OrderUpdate,
    PositionInfo,
    SessionAnchorData,
    Ticker,
    UserTrade,
)
from delta_bot.paper import PaperExchangeSimulator
from delta_bot.private_ws import PrivateWebSocketClient
from delta_bot.risk import RiskConfig, RiskManager
from delta_bot.state import TradingStateManager
from delta_bot.strategy import (
    BreakoutStrategyEngine,
    OrderIntent,
    OrderStatus,
    SignalType,
    StrategyConfig as BreakoutStrategyConfig,
)
from delta_bot.websocket_client import DeltaWebSocketClient

logger = logging.getLogger("delta_bot.orchestrator")


class TradingBotOrchestrator:
    """
    Unified Orchestrator Daemon executing the Delta-Bot algorithmic trading system.
    Coordinates networking, public/private market data, multi-symbol breakout strategies,
    real-time state reconciliation, and risk-managed execution across Paper and Live environments.
    """

    def __init__(
        self,
        config: BotConfig | None = None,
        mode: ExecutionMode | str | None = None,
        rest_client: DeltaRestClient | None = None,
        public_ws: DeltaWebSocketClient | None = None,
        private_ws: PrivateWebSocketClient | None = None,
        state_manager: TradingStateManager | None = None,
        risk_manager: RiskManager | None = None,
        paper_simulator: PaperExchangeSimulator | None = None,
    ) -> None:
        self.config: BotConfig = config or get_config()

        # Resolve Execution Mode
        if mode is not None:
            self.mode: ExecutionMode = (
                ExecutionMode.from_str(mode) if isinstance(mode, str) else mode
            )
        else:
            self.mode = ExecutionMode.from_str(self.config.execution_mode)

        # Enforce API Credential Safety in Live Mode
        if self.mode == ExecutionMode.LIVE and not self.config.has_valid_credentials():
            raise ConfigurationError(
                "ExecutionMode.LIVE requires valid DELTA_API_KEY and DELTA_API_SECRET. "
                "Configure valid credentials in your environment or use ExecutionMode.PAPER."
            )

        # Core Clients & State Subsystems
        self.rest_client: DeltaRestClient = rest_client or DeltaRestClient(config=self.config)
        self.public_ws: DeltaWebSocketClient = public_ws or DeltaWebSocketClient(config=self.config)
        self.state_manager: TradingStateManager = (
            state_manager or TradingStateManager(client=self.rest_client)
        )

        risk_cfg = RiskConfig(
            max_account_risk_pct=self.config.default_risk_pct,
            default_leverage=self.config.default_leverage,
        )
        self.risk_manager: RiskManager = risk_manager or RiskManager(
            config=risk_cfg,
            initial_equity=10000.0,
        )

        # Execution Engine based on Mode
        if self.mode == ExecutionMode.LIVE:
            self.private_ws: PrivateWebSocketClient | None = (
                private_ws or PrivateWebSocketClient(config=self.config)
            )
            exec_cfg = ExecutionConfig(
                leverage=self.config.default_leverage,
            )
            self.execution_manager: ExecutionManager | None = ExecutionManager(
                client=self.rest_client,
                config=exec_cfg,
                state_manager=self.state_manager,
            )
            self.paper_simulator: PaperExchangeSimulator | None = None
        else:
            self.private_ws = private_ws  # Typically None in pure paper mode
            self.execution_manager = None
            self.paper_simulator = paper_simulator or PaperExchangeSimulator(
                initial_balance=10000.0,
                state_manager=self.state_manager,
            )
            self.paper_simulator.set_state_manager(self.state_manager)

        # Multi-symbol Data Structures & Strategy Containers
        self.target_symbols: list[str] = [s.strip().upper() for s in self.config.target_symbols]
        self.timeframe: str = self.config.timeframe
        self.contract_specs: dict[str, ContractSpec] = {}
        self.strategies: dict[str, BreakoutStrategyEngine] = {}
        self.latest_tickers: dict[str, Ticker] = {}
        self.latest_orderbooks: dict[str, OrderBook] = {}

        # Daemon Lifecycle State
        self.is_running: bool = False
        self.start_time: float = 0.0
        self.enabled_symbols: set[str] = set(self.target_symbols)
        self._tasks: list[asyncio.Task[Any]] = []
        self._stop_event: asyncio.Event = asyncio.Event()
        self._shutdown_completed: bool = False

        # Initialize Strategy Engines per symbol
        for sym in self.target_symbols:
            strat_cfg = BreakoutStrategyConfig(
                symbol=sym,
                resolution=self.timeframe,
                risk_per_trade_pct=self.config.default_risk_pct,
            )
            strat = BreakoutStrategyEngine(config=strat_cfg)
            self._register_strategy_callbacks(strat)
            self.strategies[sym] = strat

        # Multi-Tenant Subsystem Initialization
        from delta_bot.tenant import TenantConfig, TenantInstance, TenantManager
        init_bal = (
            self.paper_simulator.wallet.initial_balance
            if self.paper_simulator and hasattr(self.paper_simulator, "wallet")
            else 10000.0
        )
        default_cfg = TenantConfig(
            tenant_id="default",
            name="Default Account",
            mode=self.mode.value,
            initial_balance=init_bal,
            delta_api_key=self.config.api_key_str,
            delta_api_secret=self.config.api_secret_str,
            delta_base_url=self.config.delta_base_url,
            delta_ws_url=self.config.delta_ws_url,
            risk_pct=self.config.default_risk_pct,
            leverage=self.config.default_leverage,
            timeframe=self.config.timeframe,
            symbols=self.target_symbols,
            is_active=True,
        )
        default_instance = TenantInstance(
            config=default_cfg,
            contract_specs=self.contract_specs,
            state_manager=self.state_manager,
            risk_manager=self.risk_manager,
            paper_simulator=self.paper_simulator,
            strategies=self.strategies,
        )
        self.tenant_manager = TenantManager(
            default_instance=default_instance,
            supabase_url=self.config.supabase_url,
            supabase_secret_key=self.config.supabase_secret_key_str,
        )
        self.tenant_manager.load_from_disk(contract_specs=self.contract_specs)
        if self.config.supabase_url and self.config.supabase_secret_key_str:
            self.tenant_manager.load_from_supabase(contract_specs=self.contract_specs)

    def set_symbol_trading_enabled(self, symbol: str, enabled: bool) -> bool:
        """Enables or disables signal evaluation and order placement for a symbol."""
        clean = symbol.strip().upper()
        if enabled:
            self.enabled_symbols.add(clean)
        else:
            self.enabled_symbols.discard(clean)
        logger.info("[%s] Trading signal evaluation %s", clean, "ENABLED" if enabled else "DISABLED")
        return clean in self.enabled_symbols

    def set_execution_mode(self, mode: ExecutionMode | str) -> ExecutionMode:
        """Dynamically switches between PAPER and LIVE operational execution modes."""
        new_mode = ExecutionMode.from_str(mode) if isinstance(mode, str) else mode
        if new_mode == ExecutionMode.LIVE:
            if not self.config.has_valid_credentials():
                raise ConfigurationError(
                    "Cannot switch to LIVE mode: DELTA_API_KEY and DELTA_API_SECRET are not configured or invalid in .env."
                )
            if self.execution_manager is None:
                exec_cfg = ExecutionConfig(leverage=self.config.default_leverage)
                self.execution_manager = ExecutionManager(
                    client=self.rest_client,
                    config=exec_cfg,
                    state_manager=self.state_manager,
                )
            self.mode = ExecutionMode.LIVE
            logger.warning("[OPERATIONAL MODE] Switched to LIVE trading on Delta Exchange")
        else:
            if self.paper_simulator is None:
                self.paper_simulator = PaperExchangeSimulator(
                    initial_balance=10000.0,
                    state_manager=self.state_manager,
                )
                self.paper_simulator.set_state_manager(self.state_manager)
                for spec in self.contract_specs.values():
                    self.paper_simulator.register_contract_spec(spec)
            self.mode = ExecutionMode.PAPER
            logger.info("[OPERATIONAL MODE] Switched to PAPER simulation mode")
        return self.mode

    # -------------------------------------------------------------------------
    # Strategy Callbacks & Inter-Process Wiring
    # -------------------------------------------------------------------------

    def _get_symbol_alias(self, symbol: str) -> str:
        """Resolves cross USD/USDT alias or canonical symbol."""
        sym = symbol.strip().upper()
        # 1. Check known contract specs
        spec = self.contract_specs.get(sym)
        if spec and spec.symbol.upper() != sym:
            return spec.symbol.upper()
        for k, v in self.contract_specs.items():
            if v.symbol.upper() == sym and k != sym:
                return k
        # 2. Check suffix conversion
        if sym.endswith("USD") and not sym.endswith("USDT"):
            return sym[:-3] + "USDT"
        if sym.endswith("USDT"):
            return sym[:-4] + "USD"
        return sym

    def _register_strategy_callbacks(self, strategy: BreakoutStrategyEngine) -> None:
        """Hooks asynchronous event listeners on individual strategy engines."""
        strategy.on_sl_ratchet(self._handle_sl_ratchet)
        strategy.on_order_cancel(self._handle_order_cancel)
        strategy.on_position_close(self._handle_position_close)

    async def _handle_sl_ratchet(
        self, intent: OrderIntent, old_sl: float, new_sl: float
    ) -> None:
        """Dispatches dynamic trailing stop ratchet to active position/order."""
        logger.info(
            "[%s] Trailing Stop Ratchet Triggered: %.4f -> %.4f",
            intent.symbol,
            old_sl,
            new_sl,
        )
        if self.mode == ExecutionMode.LIVE and self.execution_manager is not None:
            try:
                await self.execution_manager.handle_trailing_stop_ratchet(
                    order_intent=intent,
                    old_sl=old_sl,
                    new_sl=new_sl,
                )
            except Exception as e:
                logger.error(
                    "[%s] Failed to ratchet SL via ExecutionManager: %s",
                    intent.symbol,
                    e,
                    exc_info=True,
                )
        elif self.mode == ExecutionMode.PAPER and self.paper_simulator is not None:
            spec = self.contract_specs.get(intent.symbol)
            if spec:
                self.paper_simulator.edit_order_stop_loss(
                    product_id=spec.product_id,
                    order_id=intent.exchange_order_id or 0,
                    new_sl=new_sl,
                )

    async def _handle_order_cancel(self, intent: OrderIntent, reason: str) -> None:
        """Handles intermediate EMA invalidation or TTL expiration cancellations."""
        logger.warning(
            "[%s] Pending Order Cancelled: %s (Reason: %s)",
            intent.symbol,
            intent.intent_id,
            reason,
        )
        if self.mode == ExecutionMode.LIVE and self.execution_manager is not None:
            try:
                await self.execution_manager.handle_invalidation(intent)
            except Exception as e:
                logger.error(
                    "[%s] Error cancelling live order on invalidation: %s",
                    intent.symbol,
                    e,
                )
        elif self.mode == ExecutionMode.PAPER and self.paper_simulator is not None:
            spec = self.contract_specs.get(intent.symbol)
            if spec and intent.exchange_order_id:
                self.paper_simulator.cancel_order(
                    product_id=spec.product_id,
                    order_id=intent.exchange_order_id,
                )

    async def _handle_position_close(
        self, intent: OrderIntent, status: OrderStatus, exit_price: float
    ) -> None:
        """Logs strategy position close event and updates local accounting."""
        logger.info(
            "[%s] Position Closed: %s | Status=%s | ExitPrice=%.4f",
            intent.symbol,
            intent.intent_id,
            status.value,
            exit_price,
        )
        equity = await self.get_account_equity()
        self.risk_manager.record_trade_exit(
            pnl=intent.realized_pnl or 0.0,
            current_equity=equity,
            symbol=intent.symbol,
        )

    # -------------------------------------------------------------------------
    # Public & Private WebSocket Event Handlers
    # -------------------------------------------------------------------------

    async def _handle_candle_close(self, symbol: str, candle: Candle) -> None:
        """
        Task 3 Core: Evaluates BreakoutStrategyEngine upon confirmed candle closes.
        Orchestrates sizing, risk limits, and stop order dispatch.
        """
        clean_symbol = symbol.strip().upper()
        target_sym = clean_symbol
        if target_sym not in self.enabled_symbols:
            alt = self._get_symbol_alias(clean_symbol)
            if alt in self.enabled_symbols:
                target_sym = alt
            else:
                logger.debug("Trading disabled for symbol: %s, skipping breakout evaluation", clean_symbol)
                return

        strategy = self.strategies.get(target_sym) or self.strategies.get(clean_symbol)
        spec = self.contract_specs.get(target_sym) or self.contract_specs.get(clean_symbol)

        if not strategy or not spec:
            if not spec and self.rest_client:
                try:
                    spec = await self.rest_client.get_contract_spec(target_sym)
                    if spec:
                        self.contract_specs[target_sym] = spec
                        self.contract_specs[spec.symbol.upper()] = spec
                except Exception as e:
                    logger.debug("Lazy load ContractSpec for %s failed: %s", target_sym, e)

            if not strategy:
                strat_cfg = BreakoutStrategyConfig(
                    symbol=target_sym,
                    resolution=self.timeframe,
                    risk_per_trade_pct=self.config.default_risk_pct,
                )
                strategy = BreakoutStrategyEngine(config=strat_cfg)
                self._register_strategy_callbacks(strategy)
                self.strategies[target_sym] = strategy
                if target_sym != clean_symbol:
                    self.strategies[clean_symbol] = strategy

        if not strategy or not spec:
            logger.debug("Candle close received for untracked symbol: %s", symbol)
            return

        bar_cache = self.public_ws.get_bar_cache(clean_symbol, self.timeframe)
        session_anchor = self.public_ws.get_session_anchor(clean_symbol)
        account_equity = await self.get_account_equity()

        # Update circuit breaker high-watermark
        self.risk_manager.update_daily_equity(account_equity)

        # Strategy evaluation
        try:
            order_intent = await strategy.on_candle_close(
                candle=candle,
                session_anchor=session_anchor,
                contract_spec=spec,
                bar_cache=bar_cache,
                account_equity=account_equity,
            )
        except Exception as e:
            logger.error(
                "[%s] Error during strategy on_candle_close evaluation: %s",
                clean_symbol,
                e,
                exc_info=True,
            )
            order_intent = None

        # Dispatch confirmed candle close to all registered custom tenants for lifecycle management
        if hasattr(self, "tenant_manager"):
            await self._dispatch_candle_to_custom_tenants(clean_symbol, candle, spec)

        if order_intent is None:
            return

        logger.info(
            "[%s] BREAKOUT SIGNAL DETECTED: %s at Trigger=%.4f (Lots=%d, Risk=$%.2f)",
            clean_symbol,
            order_intent.signal_type.value,
            order_intent.trigger_price,
            order_intent.contract_lots,
            order_intent.capital_at_risk_usd,
        )

        # Dispatch breakout signal fan-out to all active tenants
        if hasattr(self, "tenant_manager"):
            base_sig_id = f"SIG_{clean_symbol}_{int(candle.timestamp)}"
            try:
                await self.tenant_manager.fanout_order_intent(base_sig_id, order_intent, spec)
            except Exception as fan_err:
                logger.error("[FANOUT ERROR] Failed to fan out signal %s: %s", base_sig_id, fan_err)

        # Check if default tenant execution is paused
        if hasattr(self, "tenant_manager") and "default" in self.tenant_manager.tenants:
            if not self.tenant_manager.tenants["default"].is_running:
                logger.debug("[%s] Default tenant is paused. Skipping default tenant order execution.", clean_symbol)
                return

        # Pre-trade Risk Limit Verification
        active_pos_count = self.get_active_positions_count()
        can_open, reject_reason = self.risk_manager.can_open_new_position(active_pos_count, symbol=clean_symbol)
        if not can_open:
            logger.warning(
                "[%s] Order Intent %s rejected by RiskManager: %s",
                clean_symbol,
                order_intent.intent_id,
                reject_reason,
            )
            order_intent.cancel(
                reject_reason or "RiskManager position rejection",
                OrderStatus.CANCELLED_MANUAL,
            )
            return

        # Execution Dispatch (Live vs Paper)
        if self.mode == ExecutionMode.LIVE and self.execution_manager is not None:
            try:
                await self.execution_manager.execute_signal(
                    order_intent=order_intent,
                    account_equity=account_equity,
                )
            except Exception as e:
                logger.error(
                    "[%s] Failed to execute live order intent %s: %s",
                    clean_symbol,
                    order_intent.intent_id,
                    e,
                    exc_info=True,
                )
        elif self.mode == ExecutionMode.PAPER and self.paper_simulator is not None:
            try:
                # Size lots with RiskManager dual constraints
                avail_margin = (
                    self.paper_simulator.wallet.available_balance
                    if self.paper_simulator.wallet.available_balance > 0
                    else account_equity
                )
                margin_res = self.risk_manager.size_position(
                    spec=spec,
                    trigger_price=order_intent.trigger_price,
                    stop_loss_price=order_intent.initial_stop_loss,
                    account_equity=account_equity,
                    available_margin=avail_margin,
                    risk_pct_override=self.config.default_risk_pct,
                )
                if not margin_res.is_margin_sufficient or margin_res.final_lots < 1:
                    logger.warning(
                        "[%s] Paper order sizing rejected by margin check: %s",
                        clean_symbol,
                        margin_res.rejection_reason,
                    )
                    return

                order_intent.contract_lots = margin_res.final_lots
                order_intent.product_id = spec.product_id
                order_intent.leverage = self.config.default_leverage

                side = "buy" if order_intent.signal_type == SignalType.BULLISH_BREAKOUT else "sell"
                v_order = self.paper_simulator.submit_stop_order(
                    product_id=spec.product_id,
                    symbol=spec.symbol,
                    side=side,
                    size=order_intent.contract_lots,
                    stop_price=order_intent.trigger_price,
                    bracket_stop_loss_price=order_intent.initial_stop_loss,
                    bracket_take_profit_price=order_intent.take_profit,
                    leverage=self.config.default_leverage,
                    intent_id=order_intent.intent_id,
                )
                order_intent.exchange_order_id = v_order.id
                self.state_manager.register_intent(order_intent)
                self.state_manager.bind_exchange_order_id(order_intent.intent_id, v_order.id)
                self.risk_manager.record_trade_entry()
            except Exception as e:
                logger.error(
                    "[%s] Failed to submit paper stop order: %s",
                    clean_symbol,
                    e,
                    exc_info=True,
                )

    async def _dispatch_candle_to_custom_tenants(
        self, clean_symbol: str, candle: Candle, spec: ContractSpec
    ) -> None:
        """Evaluates breakout strategy for all active custom tenants on confirmed bar close."""
        bar_cache = self.public_ws.get_bar_cache(clean_symbol, self.timeframe)
        session_anchor = self.public_ws.get_session_anchor(clean_symbol)
        latest_ticker = self.latest_tickers.get(clean_symbol)

        for t_id, tenant in self.tenant_manager.tenants.items():
            if t_id == "default":
                continue  # Default tenant is evaluated in main orchestrator loop
            if not tenant.is_running or not tenant.is_symbol_enabled(clean_symbol):
                continue

            t_strat = tenant.get_strategy(clean_symbol)
            if not t_strat:
                continue

            tenant.register_contract_spec(spec)
            try:
                t_equity = await tenant.get_account_equity()
                if tenant.mode == ExecutionMode.LIVE and t_equity <= 0:
                    continue  # Unfunded live account ($0.00), skip signal evaluation

                t_intent = await t_strat.on_candle_close(
                    candle=candle,
                    session_anchor=session_anchor,
                    contract_spec=spec,
                    bar_cache=bar_cache,
                    account_equity=t_equity,
                )
                if t_intent is not None:
                    await self._execute_tenant_intent(tenant, clean_symbol, spec, t_intent)
            except Exception as err:
                logger.error("[TENANT %s:%s] Error during breakout signal evaluation: %s", t_id, clean_symbol, err)

    async def _execute_tenant_intent(
        self, tenant: Any, clean_symbol: str, spec: ContractSpec, order_intent: OrderIntent
    ) -> None:
        """Executes a validated breakout order intent for an isolated tenant."""
        t_id = tenant.config.tenant_id
        t_equity = await tenant.get_account_equity()
        if tenant.mode == ExecutionMode.LIVE and t_equity <= 0:
            logger.warning(
                "[TENANT %s:%s] Order skipped: Live account balance is $0.00 on Delta Exchange India. Please deposit funds to trade.",
                t_id,
                clean_symbol,
            )
            order_intent.cancel("Zero account balance on exchange", OrderStatus.CANCELLED_MANUAL)
            return

        tenant.risk_manager.update_daily_equity(t_equity)

        active_count = tenant.get_active_positions_count()
        can_open, reject_reason = tenant.risk_manager.can_open_new_position(active_count, symbol=clean_symbol)
        if not can_open:
            logger.warning("[TENANT %s:%s] Order rejected by RiskManager: %s", t_id, clean_symbol, reject_reason)
            order_intent.cancel(reject_reason or "RiskManager rejection", OrderStatus.CANCELLED_MANUAL)
            return

        if tenant.mode == ExecutionMode.PAPER and tenant.paper_simulator is not None:
            avail_margin = (
                tenant.paper_simulator.wallet.available_balance
                if tenant.paper_simulator.wallet.available_balance > 0
                else t_equity
            )
            margin_res = tenant.risk_manager.size_position(
                spec=spec,
                trigger_price=order_intent.trigger_price,
                stop_loss_price=order_intent.initial_stop_loss,
                account_equity=t_equity,
                available_margin=avail_margin,
                risk_pct_override=tenant.config.risk_pct,
            )
            if not margin_res.is_margin_sufficient or margin_res.final_lots < 1:
                logger.warning("[TENANT %s:%s] Margin check failed: %s", t_id, clean_symbol, margin_res.rejection_reason)
                return

            order_intent.contract_lots = margin_res.final_lots
            order_intent.product_id = spec.product_id
            order_intent.leverage = tenant.config.leverage

            from delta_bot.tenant import generate_deterministic_client_order_id
            base_signal_id = order_intent.base_signal_id or f"SIG_{clean_symbol}_{int(time.time())}"
            det_client_id = generate_deterministic_client_order_id(base_signal_id, t_id)
            order_intent.base_signal_id = base_signal_id
            order_intent.client_order_id = det_client_id
            tenant.order_to_base_id[det_client_id] = base_signal_id

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
                "[TENANT %s:%s] Dispatched paper breakout order #%d (%s, Lots=%d, Stop=%.4f)",
                t_id,
                clean_symbol,
                v_order.id,
                side.upper(),
                v_order.size,
                v_order.stop_price,
            )
        elif tenant.mode == ExecutionMode.LIVE and tenant.execution_manager is not None:
            await tenant.execution_manager.execute_signal(order_intent=order_intent, account_equity=t_equity)

    async def _handle_candle_update(self, symbol: str, candle: Candle) -> None:
        """Receives forming candle updates for intra-bar price tracking."""
        clean_symbol = symbol.strip().upper()
        ob = self.latest_orderbooks.get(clean_symbol)

        if self.mode == ExecutionMode.PAPER and self.paper_simulator is not None:
            await self.paper_simulator.on_tick(
                symbol=clean_symbol,
                mark_price=candle.close,
                high=candle.high,
                low=candle.low,
                orderbook=ob,
            )
        elif self.mode == ExecutionMode.LIVE:
            await self.state_manager.update_market_context(
                symbol=clean_symbol,
                mark_price=candle.close,
                high=candle.high,
                low=candle.low,
            )

        if hasattr(self, "tenant_manager"):
            for t_id, tenant in self.tenant_manager.tenants.items():
                if t_id == "default":
                    continue
                if tenant.mode == ExecutionMode.PAPER and tenant.paper_simulator is not None:
                    await tenant.paper_simulator.on_tick(
                        symbol=clean_symbol,
                        mark_price=candle.close,
                        high=candle.high,
                        low=candle.low,
                        orderbook=ob,
                    )
                elif tenant.mode == ExecutionMode.LIVE:
                    await tenant.state_manager.update_market_context(
                        symbol=clean_symbol,
                        mark_price=candle.close,
                        high=candle.high,
                        low=candle.low,
                    )

    async def _handle_ticker(self, symbol: str, ticker: Ticker) -> None:
        """Processes real-time market ticks, mark price updates, and paper matching."""
        clean_symbol = symbol.strip().upper()
        self.latest_tickers[clean_symbol] = ticker
        alt = self._get_symbol_alias(clean_symbol)
        if alt != clean_symbol:
            self.latest_tickers[alt] = ticker
        ob = self.latest_orderbooks.get(clean_symbol) or self.latest_orderbooks.get(alt)

        if self.mode == ExecutionMode.PAPER and self.paper_simulator is not None:
            await self.paper_simulator.on_tick(
                symbol=clean_symbol,
                mark_price=ticker.mark_price,
                orderbook=ob,
            )
        elif self.mode == ExecutionMode.LIVE:
            await self.state_manager.update_market_context(
                symbol=clean_symbol,
                mark_price=ticker.mark_price,
            )

        if hasattr(self, "tenant_manager"):
            for t_id, tenant in self.tenant_manager.tenants.items():
                if t_id == "default":
                    continue
                if tenant.mode == ExecutionMode.PAPER and tenant.paper_simulator is not None:
                    await tenant.paper_simulator.on_tick(
                        symbol=clean_symbol,
                        mark_price=ticker.mark_price,
                        orderbook=ob,
                    )
                elif tenant.mode == ExecutionMode.LIVE:
                    await tenant.state_manager.update_market_context(
                        symbol=clean_symbol,
                        mark_price=ticker.mark_price,
                    )

    async def _handle_orderbook(self, symbol: str, orderbook: OrderBook) -> None:
        """Caches depth snapshots for slippage estimation and liquidity checks."""
        clean_symbol = symbol.strip().upper()
        self.latest_orderbooks[clean_symbol] = orderbook
        alt = self._get_symbol_alias(clean_symbol)
        if alt != clean_symbol:
            self.latest_orderbooks[alt] = orderbook

    async def _handle_session_rollover(
        self, symbol: str, session: SessionAnchorData
    ) -> None:
        """
        Task 5: Daily Session Rollover Engine (00:00 UTC).
        Locks previous day high/low (PDH/PDL) and resets circuit breakers.
        """
        clean_symbol = symbol.strip().upper()
        equity = await self.get_account_equity()

        logger.info(
            "====================================================================\n"
            "[SESSION ROLLOVER 00:00 UTC] Symbol: %s | Date: %s\n"
            "PDH: %s | PDL: %s | PDC: %s | Session VWAP Reset\n"
            "Resetting Daily Circuit Breaker at Equity: $%.2f\n"
            "====================================================================",
            clean_symbol,
            session.session_date,
            session.prior_day_high,
            session.prior_day_low,
            session.prior_day_close,
            equity,
        )

        # Reset daily circuit breaker baseline equity
        self.risk_manager.reset_daily_session(current_equity=equity)
        if hasattr(self, "tenant_manager"):
            for t_id, tenant in self.tenant_manager.tenants.items():
                if t_id == "default":
                    continue
                t_eq = await tenant.get_account_equity()
                tenant.risk_manager.reset_daily_session(current_equity=t_eq)

    # -------------------------------------------------------------------------
    # Private WebSocket Event Handlers (Live Mode)
    # -------------------------------------------------------------------------

    async def _handle_private_order_update(self, update: OrderUpdate) -> None:
        """Routes private user order updates to State Manager."""
        await self.state_manager.handle_order_update(update)

    async def _handle_private_position_update(self, pos: PositionInfo) -> None:
        """Routes private user position updates to State Manager."""
        await self.state_manager.handle_position_update(pos)

    async def _handle_private_user_trade(self, trade: UserTrade) -> None:
        """Routes fill execution trades to State Manager."""
        await self.state_manager.handle_user_trade(trade)

    # -------------------------------------------------------------------------
    # REST Historical Warm-Up Routine
    # -------------------------------------------------------------------------

    async def warmup_historical_candles(self, limit: int = 200) -> None:
        """
        Eliminates cold-start delay by priming CircularBarCache, SessionAnchorData,
        and indicators with historical candles via REST API.
        """
        logger.info(
            "Priming historical bar caches across %d symbols (Limit=%d, Timeframe=%s)...",
            len(self.target_symbols),
            limit,
            self.timeframe,
        )

        for sym in self.target_symbols:
            strategy = self.strategies.get(sym)
            if not strategy:
                continue

            bar_cache = self.public_ws.get_bar_cache(sym, self.timeframe)
            session_anchor = self.public_ws.get_session_anchor(sym)

            try:
                # 1. Fetch daily candles to prime PDH/PDL if available
                daily_candles = await self.rest_client.get_historical_candles(
                    symbol=sym,
                    resolution="1d",
                    limit=5,
                )
                if len(daily_candles) >= 2:
                    yesterday = daily_candles[-2]
                    session_anchor.prior_day_high = yesterday.high
                    session_anchor.prior_day_low = yesterday.low
                    session_anchor.prior_day_close = yesterday.close
                    logger.info(
                        "[%s] Seeded PDH=%.4f / PDL=%.4f from daily candles",
                        sym,
                        yesterday.high,
                        yesterday.low,
                    )
            except Exception as e:
                logger.debug("[%s] Optional daily candle warm-up failed: %s", sym, e)

            try:
                # 2. Warm up intraday resolution bars
                seeded = await strategy.warm_up_from_rest(
                    client=self.rest_client,
                    bar_cache=bar_cache,
                    session_anchor=session_anchor,
                    limit=limit,
                )
                logger.info("[%s] Warm-up complete: %d candles loaded into cache", sym, seeded)
            except Exception as e:
                logger.warning("[%s] REST warm-up failed: %s", sym, e)

    # -------------------------------------------------------------------------
    # Orchestrator Lifecycle: Start, Run, and Shutdown
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """
        Initializes connections, caches specifications, primes indicators,
        and launches all background daemon worker tasks.
        """
        logger.info(
            "Starting TradingBotOrchestrator in [%s] Mode | Target Symbols: %s | Timeframe: %s",
            self.mode.value.upper(),
            ", ".join(self.target_symbols),
            self.timeframe,
        )
        self.is_running = True
        self.start_time = time.time()

        # 1. Resolve and cache ContractSpec for all target symbols
        for sym in self.target_symbols:
            try:
                spec = await self.rest_client.get_contract_spec(sym)
                if spec.contract_type != "perpetual_futures":
                    logger.warning("Symbol %s is not a perpetual futures contract (got '%s'). Skipping.", sym, spec.contract_type)
                    continue
                self.contract_specs[sym] = spec
                self.contract_specs[spec.symbol.upper()] = spec
                # Register strategy alias mapping
                if spec.symbol.upper() != sym:
                    if sym in self.strategies and spec.symbol.upper() not in self.strategies:
                        self.strategies[spec.symbol.upper()] = self.strategies[sym]
                if self.mode == ExecutionMode.PAPER and self.paper_simulator is not None:
                    self.paper_simulator.register_contract_spec(spec)
                if hasattr(self, "tenant_manager"):
                    for t in self.tenant_manager.tenants.values():
                        t.register_contract_spec(spec)
                logger.info(
                    "Cached ContractSpec for %s (Exchange Symbol: %s): ProductID=%d, LotSize=%.4f, Tick=%.4f",
                    sym,
                    spec.symbol,
                    spec.product_id,
                    spec.contract_value,
                    spec.tick_size,
                )
            except Exception as e:
                logger.error("Failed to load ContractSpec for %s: %s", sym, e)

        # 2. Initialize RiskManager baseline capital
        initial_equity = await self.get_account_equity()
        self.risk_manager.reset_daily_session(current_equity=initial_equity)

        # 3. Perform Historical REST Warm-up
        await self.warmup_historical_candles(limit=200)

        # 4. Wire Public WebSocket Callbacks
        self.public_ws.on_candle_close(self._handle_candle_close)
        self.public_ws.on_candle_update(self._handle_candle_update)
        self.public_ws.on_ticker(self._handle_ticker)
        self.public_ws.on_orderbook(self._handle_orderbook)
        self.public_ws.on_session_rollover(self._handle_session_rollover)

        # 5. Connect Public Market Data WebSocket (Task 1)
        await self.public_ws.connect()
        sub_symbols: set[str] = set()
        for sym in self.target_symbols:
            sub_symbols.add(sym)
            spec = self.contract_specs.get(sym)
            if spec:
                sub_symbols.add(spec.symbol.upper())
        sub_list = list(sub_symbols)
        await self.public_ws.subscribe_candles(
            symbols=sub_list,
            resolutions=[self.timeframe],
        )
        await self.public_ws.subscribe_tickers(symbols=sub_list)
        if getattr(self.config, "enable_orderbook_stream", True):
            await self.public_ws.subscribe_orderbook(symbols=sub_list)
        else:
            logger.info("Orderbook streaming disabled via ENABLE_ORDERBOOK_STREAM=false (conserving bandwidth)")

        # 6. Initialize Live Subsystems (Tasks 2 & 4)
        if self.mode == ExecutionMode.LIVE and self.private_ws is not None:
            self.private_ws.on_order(self._handle_private_order_update)
            self.private_ws.on_position(self._handle_private_position_update)
            self.private_ws.on_trade(self._handle_private_user_trade)
            await self.private_ws.start()
            await self.state_manager.start_periodic_reconciliation()
            logger.info("Private WebSocket & Periodic REST State Reconciliation started")

        # 7. Launch Telemetry Heartbeat (Task 6)
        telemetry_task = asyncio.create_task(
            self._telemetry_heartbeat_loop(),
            name="Orchestrator-Watchdog-Telemetry",
        )
        self._tasks.append(telemetry_task)

        logger.info("TradingBotOrchestrator successfully initialized and listening for market events")

    async def run(self) -> None:
        """Runs the orchestrator daemon until stop signal is trapped."""
        await self.start()
        try:
            await self._stop_event.wait()
        except asyncio.CancelledError:
            logger.debug("Orchestrator run coroutine cancelled")
        finally:
            if not self._shutdown_completed:
                await self.shutdown(reason="Event loop exit")

    async def sync_exchange_leverage(self, leverage: int) -> dict[str, Any]:
        """
        Synchronizes order leverage across all registered perpetual futures contracts on Delta Exchange.
        Guarantees that new stop and limit orders adhere to the desired leverage tier (1x - 100x).
        """
        results: dict[str, Any] = {}
        target_lev = max(1, min(int(leverage), 100))
        for sym, spec in list(self.contract_specs.items()):
            if spec.contract_type != "perpetual_futures":
                continue
            try:
                await self.rest_client.set_order_leverage(spec.product_id, target_lev)
                results[sym] = {"status": "success", "product_id": spec.product_id, "leverage": target_lev}
                if self.execution_manager is not None:
                    self.execution_manager._configured_leverages[spec.product_id] = target_lev
                logger.info("Synchronized exchange order leverage for %s (id=%d) to %dx", sym, spec.product_id, target_lev)
            except Exception as exc:
                logger.warning("Could not set exchange order leverage for %s (id=%d): %s", sym, spec.product_id, exc)
                results[sym] = {"status": "error", "error": str(exc)}
        return results

    async def shutdown(self, reason: str = "SIGINT") -> None:
        """
        Gracefully halts the orchestrator daemon:
        - Rejects new signals
        - Cancels resting pending stop orders
        - Flattens open positions if FLATTEN_ON_SHUTDOWN=True
        - Stops reconciliation workers and disconnects WebSockets/REST pools
        """
        if self._shutdown_completed:
            return

        logger.warning(
            "====================================================================\n"
            "[SHUTDOWN] Initiating Graceful Daemon Teardown (Reason: %s)...\n"
            "====================================================================",
            reason,
        )
        self.is_running = False
        self._stop_event.set()

        # 1. Cancel resting stop orders across all target symbols
        logger.info("[SHUTDOWN] Cancelling resting pending breakout orders...")
        if self.mode == ExecutionMode.LIVE and self.execution_manager is not None:
            try:
                cancelled = await self.execution_manager.order_router.cancel_all_orders()
                logger.info("[SHUTDOWN] Cancelled %d resting live orders", len(cancelled))
            except Exception as e:
                logger.error("[SHUTDOWN] Error cancelling live orders: %s", e)
        elif self.mode == ExecutionMode.PAPER and self.paper_simulator is not None:
            try:
                cancelled = self.paper_simulator.cancel_all_orders()
                logger.info("[SHUTDOWN] Cancelled %d resting paper orders", len(cancelled))
            except Exception as e:
                logger.error("[SHUTDOWN] Error cancelling paper orders: %s", e)

        # 2. Evaluate FLATTEN_ON_SHUTDOWN policy
        if self.config.flatten_on_shutdown:
            logger.warning("[SHUTDOWN] FLATTEN_ON_SHUTDOWN is ENABLED: Closing active positions at market...")
            if self.mode == ExecutionMode.LIVE and self.execution_manager is not None:
                try:
                    res = await self.execution_manager.emergency_flatten()
                    logger.info("[SHUTDOWN] Emergency live flatten result: %s", res)
                except Exception as e:
                    logger.error("[SHUTDOWN] Live emergency flatten failed: %s", e)
            elif self.mode == ExecutionMode.PAPER and self.paper_simulator is not None:
                try:
                    flattened = self.paper_simulator.flatten_all()
                    logger.info("[SHUTDOWN] Paper flattened %d active positions", len(flattened))
                except Exception as e:
                    logger.error("[SHUTDOWN] Paper flatten failed: %s", e)
        else:
            logger.info(
                "[SHUTDOWN] FLATTEN_ON_SHUTDOWN is DISABLED. Leaving active positions intact "
                "with exchange-native stop loss and take profit bracket orders."
            )

        # 3. Stop Periodic Reconciliation
        try:
            await self.state_manager.stop_periodic_reconciliation()
        except Exception as e:
            logger.debug("[SHUTDOWN] Error stopping state reconciliation: %s", e)

        # 4. Stop Private WebSocket
        if self.private_ws is not None:
            try:
                await self.private_ws.stop()
            except Exception as e:
                logger.debug("[SHUTDOWN] Error disconnecting private WS: %s", e)

        # 5. Disconnect Public WebSocket
        try:
            await self.public_ws.disconnect()
        except Exception as e:
            logger.debug("[SHUTDOWN] Error disconnecting public WS: %s", e)

        # 6. Close REST client HTTP session pool
        try:
            await self.rest_client.close()
        except Exception as e:
            logger.debug("[SHUTDOWN] Error closing REST client: %s", e)

        # 7. Cancel daemon tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        self._shutdown_completed = True
        logger.info(
            "====================================================================\n"
            "[SHUTDOWN COMPLETE] Final Performance Snapshot:\n%s\n"
            "====================================================================",
            self.format_telemetry_dashboard(),
        )

    # -------------------------------------------------------------------------
    # Telemetry, Dashboard, & Account Metrics
    # -------------------------------------------------------------------------

    async def get_account_equity(self) -> float:
        """Returns real-time account equity (collateral + unrealized PnL)."""
        if self.mode == ExecutionMode.PAPER and self.paper_simulator is not None:
            return self.paper_simulator.wallet.equity

        try:
            balances = await self.rest_client.get_wallet_balances()
            # Default to USDT or USD collateral
            for b in balances:
                if b.asset_symbol in ("USDT", "USD"):
                    return b.balance
            if balances:
                return balances[0].balance
        except Exception as e:
            logger.debug("Failed to retrieve live wallet balance for equity: %s", e)

        return 10000.0

    def get_active_positions_count(self) -> int:
        """Counts current active positions across target products."""
        if self.mode == ExecutionMode.PAPER and self.paper_simulator is not None:
            return len(self.paper_simulator._open_positions)
        return len(self.state_manager.get_all_positions())

    def get_pending_orders_count(self) -> int:
        """Counts resting stop orders waiting for price trigger."""
        if self.mode == ExecutionMode.PAPER and self.paper_simulator is not None:
            return len(self.paper_simulator._pending_orders)
        if self.execution_manager is not None:
            return len(self.execution_manager.registry.active_orders())
        return 0

    def get_telemetry_snapshot(self) -> dict[str, Any]:
        """Provides an atomic structured snapshot of system telemetry."""
        uptime_sec = max(0.0, time.time() - self.start_time) if self.start_time > 0 else 0.0

        if self.mode == ExecutionMode.PAPER and self.paper_simulator is not None:
            paper_stats = self.paper_simulator.get_summary()
            balance = paper_stats["balance"]
            equity = paper_stats["equity"]
            realized_pnl = paper_stats["realized_pnl"]
            unrealized_pnl = paper_stats["unrealized_pnl"]
            win_rate = paper_stats["win_rate_pct"]
            drawdown = paper_stats["max_drawdown_pct"]
            total_trades = paper_stats["total_trades"]
            winning_trades = paper_stats["winning_trades"]
        else:
            balance = 10000.0
            equity = 10000.0
            realized_pnl = 0.0
            unrealized_pnl = 0.0
            win_rate = 0.0
            drawdown = 0.0
            total_trades = 0
            winning_trades = 0

        positions: list[str] = []
        if self.mode == ExecutionMode.PAPER and self.paper_simulator is not None:
            for pos in self.paper_simulator._open_positions.values():
                positions.append(
                    f"{pos.symbol}: {pos.side.upper()} {pos.size} lots @ {pos.entry_price:.4f} (UPnL: ${pos.unrealized_pnl:+.2f})"
                )
        else:
            for p in self.state_manager.get_all_positions():
                positions.append(
                    f"{p.symbol}: {p.size} lots @ {p.entry_price:.4f} (Mark: {p.mark_price:.4f})"
                )

        pending_orders: list[str] = []
        if self.mode == ExecutionMode.PAPER and self.paper_simulator is not None:
            for o in self.paper_simulator._pending_orders.values():
                pending_orders.append(
                    f"{o.symbol}: {o.side.upper()} STOP @ {o.stop_price:.4f} (Lots: {o.size})"
                )
        elif self.execution_manager is not None:
            for ao in self.execution_manager.registry.active_orders():
                pending_orders.append(
                    f"{ao.symbol}: {ao.side.upper()} STOP @ {ao.stop_price} (Lots: {ao.size})"
                )

        return {
            "mode": self.mode.value.upper(),
            "uptime_seconds": round(uptime_sec, 1),
            "is_running": self.is_running,
            "balance": balance,
            "equity": equity,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "win_rate_pct": win_rate,
            "max_drawdown_pct": drawdown,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "positions_count": len(positions),
            "positions": positions,
            "pending_orders_count": len(pending_orders),
            "pending_orders": pending_orders,
            "public_ws_connected": bool(self.public_ws.is_connected),
            "private_ws_connected": bool(self.private_ws.is_connected) if self.private_ws else False,
            "circuit_breaker_tripped": bool(self.risk_manager.is_circuit_breaker_tripped),
        }

    def format_telemetry_dashboard(self) -> str:
        """Formats an ASCII terminal telemetry dashboard for monitoring."""
        snap = self.get_telemetry_snapshot()
        uptime_h = int(snap["uptime_seconds"] // 3600)
        uptime_m = int((snap["uptime_seconds"] % 3600) // 60)
        uptime_s = int(snap["uptime_seconds"] % 60)
        uptime_str = f"{uptime_h:02d}h {uptime_m:02d}m {uptime_s:02d}s"

        cb_status = "TRIPPED" if snap["circuit_breaker_tripped"] else "NORMAL"
        pub_status = "CONNECTED" if snap["public_ws_connected"] else "DISCONNECTED"
        priv_status = (
            "CONNECTED"
            if snap["private_ws_connected"]
            else ("N/A (Paper)" if self.mode == ExecutionMode.PAPER else "DISCONNECTED")
        )

        lines = [
            "====================== DELTA-BOT HEALTH TELEMETRY ======================",
            f"Mode: {snap['mode']:<7} | Uptime: {uptime_str:<12} | Status: {'ACTIVE' if snap['is_running'] else 'STOPPED'}",
            f"Collateral: ${snap['equity']:,.2f} Equity | ${snap['balance']:,.2f} Balance | UPnL: ${snap['unrealized_pnl']:+,.2f}",
            f"Performance: Realized PnL: ${snap['realized_pnl']:+,.2f} | Win Rate: {snap['win_rate_pct']:.1f}% ({snap['winning_trades']}/{snap['total_trades']}) | Max DD: {snap['max_drawdown_pct']:.2f}%",
            f"Positions ({snap['positions_count']}): {', '.join(snap['positions']) if snap['positions'] else 'None'}",
            f"Pending Orders ({snap['pending_orders_count']}): {', '.join(snap['pending_orders']) if snap['pending_orders'] else 'None'}",
            f"Circuit Breaker: {cb_status:<8} | Public WS: {pub_status:<10} | Private WS: {priv_status}",
            "========================================================================",
        ]
        return "\n".join(lines)

    async def _telemetry_heartbeat_loop(self) -> None:
        """Task 6: Emits periodic terminal telemetry heartbeat every status_interval_sec."""
        interval = max(1.0, self.config.status_interval_sec)
        logger.debug("Telemetry heartbeat worker initiated (Interval=%.1fs)", interval)

        while self.is_running and not self._stop_event.is_set():
            try:
                await asyncio.sleep(interval)
                if self.is_running:
                    print(f"\n{self.format_telemetry_dashboard()}\n", flush=True)
            except asyncio.CancelledError:
                logger.debug("Telemetry heartbeat task cancelled")
                break
            except Exception as e:
                logger.error("Error in telemetry heartbeat loop: %s", e)

    # -------------------------------------------------------------------------
    # POSIX Signal Registration
    # -------------------------------------------------------------------------

    def setup_signal_handlers(self) -> None:
        """Traps POSIX SIGINT and SIGTERM to trigger graceful teardown."""
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(
                        self.shutdown(reason=f"POSIX signal {s.name}")
                    ),
                )
            logger.info("POSIX SIGINT and SIGTERM signal handlers installed")
        except (NotImplementedError, RuntimeError):
            # Windows or non-main event loops do not support add_signal_handler
            logger.debug("Signal handler registration not supported in current environment")
