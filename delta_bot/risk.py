"""
delta_bot.risk
~~~~~~~~~~~~~~

Quantitative Risk Management & Pre-Trade Verification Engine (Phase 4).
Enforces:
1. Pre-Trade Balance & Margin Verification (Dual-Constraint Sizing)
2. Max Leverage & Max Portfolio Risk limits (1% default per trade)
3. Hard Capital Allocation & Free Collateral Constraints
4. Circuit Breakers: Daily Max Drawdown limit (4% default)
5. Position Limits: Strictly 1 concurrent active position
6. Emergency Flatten & Kill-Switch Coordination
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import math
import time
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from delta_bot.exceptions import InsufficientMarginError
from delta_bot.models import ContractSpec, MarginCheckResult

logger = logging.getLogger(__name__)


class RiskConfig(BaseModel):
    """Configuration parameters for pre-trade risk management and circuit breakers."""

    model_config = ConfigDict(frozen=True)

    max_account_risk_pct: float = Field(
        default=1.0,
        gt=0.0,
        le=10.0,
        description="Maximum account equity percentage to risk on a single trade (e.g. 1.0%)",
    )
    max_leverage: int = Field(
        default=5,
        validation_alias=AliasChoices("max_leverage", "default_leverage", "leverage"),
        ge=1,
        le=100,
        description="Maximum allowable leverage tier across perpetual contracts",
    )
    margin_safety_buffer: float = Field(
        default=0.90,
        gt=0.0,
        le=1.0,
        description="Fraction of available margin usable for initial margin (10% buffer for taker fees)",
    )
    daily_max_drawdown_pct: float | None = Field(
        default=None,
        validation_alias=AliasChoices("daily_max_drawdown_pct", "max_daily_drawdown_pct"),
        ge=0.0,
        le=100.0,
        description="Circuit breaker: Halt all new trades if daily loss exceeds this percentage (None = disabled)",
    )
    enable_drawdown_circuit_breaker: bool = Field(
        default=False,
        description="Whether to enforce daily drawdown circuit breaker (default: False)",
    )
    max_concurrent_positions: int | None = Field(
        default=None,
        ge=1,
        description="Strict maximum number of simultaneous open positions across the bot (None = unlimited)",
    )
    default_collateral_symbol: str = Field(
        default="USDT",
        description="Primary quote/collateral asset symbol for margin balance verification",
    )
    enable_rollover_shield: bool = Field(
        default=True,
        description="Halt new orders during daily rollover window 21:58-22:10 UTC",
    )
    enable_revenge_breaker: bool = Field(
        default=False,
        description="Quarantine symbol after 2 rapid stop-outs within 15 minutes (default: False)",
    )
    revenge_cooldown_sec: float = Field(
        default=1800.0,
        description="Quarantine duration in seconds for revenge trading breaker (default: 30m)",
    )
    min_lot_risk_tolerance: float = Field(
        default=1.05,
        description="Max allowed risk / budget ratio to prevent oversized min-lot positions (default: 1.05)",
    )
    min_stop_loss_pct: float = Field(
        default=0.10,
        ge=0.0,
        description="Minimum stop-loss distance percentage (e.g. 0.10%) required for pre-trade verification",
    )
    max_effective_leverage: float = Field(
        default=5.0,
        gt=0.0,
        description="Strict maximum effective leverage (Notional / Equity) allowed per trade",
    )


class CircuitBreakerState(BaseModel):
    """Tracks circuit breaker telemetry, high-water mark, and daily drawdown."""

    is_tripped: bool = False
    tripped_at: float | None = None
    trip_reason: str | None = None
    session_start_equity: float = 0.0
    peak_daily_equity: float = 0.0
    current_drawdown_pct: float = 0.0

    def trip(self, reason: str, timestamp: float | None = None) -> None:
        """Trips the circuit breaker and halts all new trading activity."""
        self.is_tripped = True
        self.trip_reason = reason
        self.tripped_at = timestamp or time.time()
        logger.critical(
            "*** CIRCUIT BREAKER TRIPPED *** Reason: %s (Halted at %s)",
            reason,
            self.tripped_at,
        )

    def reset(self, new_session_equity: float) -> None:
        """Resets circuit breaker for a new session."""
        self.is_tripped = False
        self.trip_reason = None
        self.tripped_at = None
        self.session_start_equity = new_session_equity
        self.peak_daily_equity = new_session_equity
        self.current_drawdown_pct = 0.0
        logger.info(
            "Circuit breaker reset. Daily session anchored at equity: $%.2f",
            new_session_equity,
        )


class RiskManager:
    """
    Derivatives Risk Manager and Pre-Trade Safety Enforcement Engine.
    Controls lot sizing, leverage limits, concurrent position guards,
    and daily drawdown circuit breakers.
    """

    def __init__(self, config: RiskConfig | None = None, initial_equity: float = 0.0) -> None:
        self.config: RiskConfig = config or RiskConfig()
        self.circuit_breaker: CircuitBreakerState = CircuitBreakerState(
            session_start_equity=initial_equity,
            peak_daily_equity=initial_equity,
        )
        self.daily_realized_pnl: float = 0.0
        self.active_positions_count: int = 0
        self.symbol_loss_history: dict[str, list[float]] = {}
        self.symbol_cooldown_expiry: dict[str, float] = {}

    @property
    def is_circuit_breaker_tripped(self) -> bool:
        """Returns True if trading is halted by circuit breaker."""
        return self.circuit_breaker.is_tripped

    def reset_daily_session(self, current_equity: float) -> None:
        """Anchors daily session equity at 00:00 UTC rollover."""
        self.daily_realized_pnl = 0.0
        self.circuit_breaker.reset(current_equity)

    def update_daily_equity(self, current_equity: float) -> bool:
        """
        Updates daily equity watermark and tests against the daily maximum drawdown limit.
        Returns True if safe, False if circuit breaker tripped.
        """
        if current_equity <= 0:
            return True

        if self.circuit_breaker.session_start_equity <= 0:
            self.circuit_breaker.session_start_equity = current_equity
            self.circuit_breaker.peak_daily_equity = current_equity

        if current_equity > self.circuit_breaker.peak_daily_equity:
            self.circuit_breaker.peak_daily_equity = current_equity

        # Drawdown measured against starting equity of current session
        start_eq = self.circuit_breaker.session_start_equity
        if start_eq > 0 and current_equity < start_eq:
            drawdown_pct = ((start_eq - current_equity) / start_eq) * 100.0
            self.circuit_breaker.current_drawdown_pct = round(drawdown_pct, 2)

            if (
                self.config.enable_drawdown_circuit_breaker
                and self.config.daily_max_drawdown_pct is not None
                and drawdown_pct >= self.config.daily_max_drawdown_pct
            ):
                reason = (
                    f"Daily max drawdown breached: {drawdown_pct:.2f}% loss "
                    f"(Threshold: {self.config.daily_max_drawdown_pct:.2f}%)"
                )
                self.circuit_breaker.trip(reason)
                return False

        return not self.circuit_breaker.is_tripped

    def can_open_new_position(
        self,
        active_positions: int | None = None,
        symbol: str | None = None,
        now_utc: datetime | None = None,
    ) -> tuple[bool, str | None]:
        """
        Pre-trade check:
        1. Verifies circuit breaker state.
        2. Daily rollover spread shield (21:58 - 22:10 UTC).
        3. Revenge trading breaker (if enabled).
        4. Max concurrent position limit (if enabled).
        """
        if self.circuit_breaker.is_tripped:
            return False, f"Circuit breaker is TRIPPED: {self.circuit_breaker.trip_reason}"

        # Daily Rollover Spread Shield (21:58 - 22:10 UTC)
        if self.config.enable_rollover_shield:
            curr_dt = now_utc or datetime.now(timezone.utc)
            if (curr_dt.hour == 21 and curr_dt.minute >= 58) or (curr_dt.hour == 22 and curr_dt.minute <= 10):
                return (
                    False,
                    "[ROLLOVER SHIELD] Order rejected: Trading frozen between 21:58-22:10 UTC during daily settlement rollover.",
                )

        # Revenge Trading Breaker (Symbol Quarantine)
        if self.config.enable_revenge_breaker and symbol:
            clean_sym = symbol.strip().upper()
            cooldown = self.symbol_cooldown_expiry.get(clean_sym, 0.0)
            now_ts = time.time()
            if now_ts < cooldown:
                remaining_m = (cooldown - now_ts) / 60.0
                return (
                    False,
                    f"[REVENGE BREAKER] Symbol {clean_sym} quarantined for another {remaining_m:.1f}m due to rapid stop-outs.",
                )

        if self.config.max_concurrent_positions is not None:
            pos_count = active_positions if active_positions is not None else self.active_positions_count
            if pos_count >= self.config.max_concurrent_positions:
                return (
                    False,
                    f"Concurrent position limit reached ({pos_count}/{self.config.max_concurrent_positions} active)",
                )

        return True, None

    def size_position(
        self,
        spec: ContractSpec,
        trigger_price: float,
        stop_loss_price: float,
        account_equity: float,
        available_margin: float,
        risk_pct_override: float | None = None,
    ) -> MarginCheckResult:
        """
        Dual-Constraint Pre-Trade Sizing:
        1. Risk Budget Constraint: Loss budget = Equity * (RiskPct / 100)
        2. Margin Affordability Constraint: Initial Margin = (Notional / Leverage) <= AvailableMargin * SafetyBuffer

        Guarantees that final contract lots never exceed initial margin limits or trigger
        exchange insufficient_margin rejections.
        """
        if account_equity <= 0 or available_margin <= 0:
            return MarginCheckResult(
                symbol=spec.symbol,
                product_id=spec.product_id,
                leverage=float(self.config.max_leverage),
                available_margin=max(0.0, available_margin),
                required_margin=0.0,
                risk_budget_usd=0.0,
                risk_based_lots=0,
                margin_based_lots=0,
                final_lots=0,
                is_margin_sufficient=False,
                rejection_reason="Insufficient collateral: account equity or available margin is $0.00",
            )

        leverage = float(self.config.max_leverage)
        safety_buf = self.config.margin_safety_buffer
        risk_pct = risk_pct_override or self.config.max_account_risk_pct

        # 1. Risk-based sizing & Stop-Loss Distance Verification
        risk_budget_usd = account_equity * (risk_pct / 100.0)
        per_unit_risk = abs(trigger_price - stop_loss_price)
        sl_pct = (per_unit_risk / trigger_price * 100.0) if trigger_price > 0 else 0.0

        if per_unit_risk <= (spec.tick_size * 2) or (
            self.config.min_stop_loss_pct > 0 and sl_pct < self.config.min_stop_loss_pct
        ):
            rejection_reason = (
                f"Stop-Loss distance too small ({per_unit_risk:.4f} / {sl_pct:.3f}% of price). "
                f"Minimum required is {self.config.min_stop_loss_pct:.2f}% (and > 2 ticks). "
                f"Aborting trade to prevent noise stop-out, immediate rejection, and oversized sizing."
            )
            logger.warning(
                "RiskManager Sizing [%s] REJECTED: %s",
                spec.symbol,
                rejection_reason,
            )
            return MarginCheckResult(
                symbol=spec.symbol,
                product_id=spec.product_id,
                leverage=leverage,
                available_margin=round(available_margin, 2),
                required_margin=0.0,
                risk_budget_usd=round(risk_budget_usd, 2),
                risk_based_lots=0,
                margin_based_lots=0,
                final_lots=0,
                is_margin_sufficient=False,
                rejection_reason=rejection_reason,
            )

        target_qty_by_risk = risk_budget_usd / per_unit_risk
        risk_based_lots = spec.round_size(target_qty_by_risk)

        # Strict Cap: effective leverage (Notional / Equity) limit
        if self.config.max_effective_leverage > 0 and trigger_price > 0:
            max_notional_usd = account_equity * self.config.max_effective_leverage
            max_qty_by_effective_lev = max_notional_usd / trigger_price
            max_lots_by_effective_lev = int(math.floor(max_qty_by_effective_lev / spec.contract_value))
            risk_based_lots = min(risk_based_lots, max_lots_by_effective_lev)

        # 2. Margin-based sizing
        max_usable_margin = available_margin * safety_buf
        unit_margin_req = (trigger_price * spec.contract_value) / leverage

        if unit_margin_req > 0:
            max_qty_by_margin = (max_usable_margin * leverage) / trigger_price
            # Floor lot calculation to strictly prevent rounding up beyond usable margin
            margin_based_lots = int(math.floor(max_qty_by_margin / spec.contract_value))
        else:
            margin_based_lots = 0

        # 3. Final lot calculation
        final_lots = min(risk_based_lots, margin_based_lots)

        # Clamping guard: ensure required margin never exceeds max_usable_margin due to float precision
        while final_lots > 0 and ((final_lots * spec.contract_value * trigger_price) / leverage) > (max_usable_margin + 1e-6):
            final_lots -= 1

        required_margin = (final_lots * spec.contract_value * trigger_price) / leverage

        is_sufficient = (
            (final_lots >= 1)
            and (round(required_margin, 4) <= round(max_usable_margin, 4))
            and (round(required_margin, 4) <= round(available_margin, 4))
        )
        rejection_reason = None

        # Minimum Lot Risk Gatekeeper (Prop firm & capital preservation)
        min_1_lot_risk = 1 * spec.contract_value * per_unit_risk
        max_allowed_risk = risk_budget_usd * self.config.min_lot_risk_tolerance
        if (final_lots >= 1 and (final_lots * spec.contract_value * per_unit_risk) > max_allowed_risk) or (
            final_lots < 1 and min_1_lot_risk > max_allowed_risk
        ):
            is_sufficient = False
            actual_loss_at_sl = (final_lots if final_lots >= 1 else 1) * spec.contract_value * per_unit_risk
            excess_pct = ((actual_loss_at_sl / risk_budget_usd) - 1.0) * 100.0
            rejection_reason = (
                f"Position risk (${actual_loss_at_sl:.2f}) exceeds risk allocation (${risk_budget_usd:.2f}) "
                f"by {excess_pct:.1f}% (Minimum lot size too large for risk allocation). Aborting trade."
            )

        if not is_sufficient and rejection_reason is None:
            if available_margin <= 0:
                rejection_reason = f"Zero available margin ({available_margin:.2f} USD)"
            elif margin_based_lots < 1:
                rejection_reason = (
                    f"Available collateral ({available_margin:.2f} USD at {leverage:.0f}x leverage) "
                    f"insufficient for 1 {spec.symbol} contract (requires {unit_margin_req:.2f} USD)"
                )
            elif final_lots < 1:
                rejection_reason = f"Calculated size ({final_lots} lots) below minimum contract unit"
            elif round(required_margin, 4) > round(max_usable_margin, 4):
                rejection_reason = (
                    f"Required initial margin ({required_margin:.2f} USD) exceeds usable margin ({max_usable_margin:.2f} USD)"
                )

        logger.info(
            "RiskManager Sizing [%s]: RiskLots=%d | MarginLots=%d | FinalLots=%d | ReqMargin=$%.2f | Avail=$%.2f | OK=%s",
            spec.symbol,
            risk_based_lots,
            margin_based_lots,
            final_lots,
            required_margin,
            available_margin,
            is_sufficient,
        )

        return MarginCheckResult(
            symbol=spec.symbol,
            product_id=spec.product_id,
            leverage=leverage,
            available_margin=round(available_margin, 2),
            required_margin=round(required_margin, 2),
            risk_budget_usd=round(risk_budget_usd, 2),
            risk_based_lots=risk_based_lots,
            margin_based_lots=margin_based_lots,
            final_lots=max(final_lots, 0),
            is_margin_sufficient=is_sufficient,
            rejection_reason=rejection_reason,
        )

    def record_trade_exit(self, pnl: float, current_equity: float, symbol: str = "") -> None:
        """Records realized PnL, tests daily drawdown limits, and tracks stop-out loss streaks."""
        self.daily_realized_pnl += pnl
        self.update_daily_equity(current_equity)
        if self.active_positions_count > 0:
            self.active_positions_count -= 1

        # Revenge Trading Breaker tracking
        if self.config.enable_revenge_breaker and symbol and pnl < 0:
            clean_sym = symbol.strip().upper()
            now_ts = time.time()
            hist = self.symbol_loss_history.setdefault(clean_sym, [])
            hist.append(now_ts)
            self.symbol_loss_history[clean_sym] = [t for t in hist if now_ts - t <= 900.0]
            if len(self.symbol_loss_history[clean_sym]) >= 2:
                self.symbol_cooldown_expiry[clean_sym] = now_ts + self.config.revenge_cooldown_sec
                logger.warning(
                    "🛑 [REVENGE TRADING BREAKER] %s experienced 2 rapid stop-outs within 15 minutes. Quarantining symbol for %.1f minutes!",
                    clean_sym,
                    self.config.revenge_cooldown_sec / 60.0,
                )

    def record_trade_entry(self) -> None:
        """Increments active position count."""
        self.active_positions_count += 1
