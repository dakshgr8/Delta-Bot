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

import logging
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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
    daily_max_drawdown_pct: float = Field(
        default=4.0,
        gt=0.0,
        le=50.0,
        description="Circuit breaker: Halt all new trades if daily loss exceeds this percentage (e.g. 4.0%)",
    )
    max_concurrent_positions: int = Field(
        default=1,
        ge=1,
        description="Strict maximum number of simultaneous open positions across the bot (default: 1)",
    )
    default_collateral_symbol: str = Field(
        default="USDT",
        description="Primary quote/collateral asset symbol for margin balance verification",
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

    def __init__(self, config: RiskConfig | None = None, initial_equity: float = 10000.0) -> None:
        self.config: RiskConfig = config or RiskConfig()
        self.circuit_breaker: CircuitBreakerState = CircuitBreakerState(
            session_start_equity=initial_equity,
            peak_daily_equity=initial_equity,
        )
        self.daily_realized_pnl: float = 0.0
        self.active_positions_count: int = 0

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

            if drawdown_pct >= self.config.daily_max_drawdown_pct:
                reason = (
                    f"Daily max drawdown breached: {drawdown_pct:.2f}% loss "
                    f"(Threshold: {self.config.daily_max_drawdown_pct:.2f}%)"
                )
                self.circuit_breaker.trip(reason)
                return False

        return not self.circuit_breaker.is_tripped

    def can_open_new_position(self, active_positions: int | None = None) -> tuple[bool, str | None]:
        """
        Pre-trade check: Verifies circuit breaker and max concurrent position limit (strictly 1).
        Returns (True, None) if permitted, or (False, reason) if rejected.
        """
        if self.circuit_breaker.is_tripped:
            return False, f"Circuit breaker is TRIPPED: {self.circuit_breaker.trip_reason}"

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
        leverage = float(self.config.max_leverage)
        safety_buf = self.config.margin_safety_buffer
        risk_pct = risk_pct_override or self.config.max_account_risk_pct

        # 1. Risk-based sizing
        risk_budget_usd = account_equity * (risk_pct / 100.0)
        per_unit_risk = abs(trigger_price - stop_loss_price)
        if per_unit_risk <= (spec.tick_size * 2):
            per_unit_risk = trigger_price * 0.01

        target_qty_by_risk = risk_budget_usd / per_unit_risk
        risk_based_lots = spec.round_size(target_qty_by_risk)

        # 2. Margin-based sizing
        max_usable_margin = available_margin * safety_buf
        unit_margin_req = (trigger_price * spec.contract_value) / leverage

        if unit_margin_req > 0:
            max_qty_by_margin = (max_usable_margin * leverage) / trigger_price
            margin_based_lots = spec.round_size(max_qty_by_margin)
        else:
            margin_based_lots = 0

        # 3. Final lot calculation
        final_lots = min(risk_based_lots, margin_based_lots)
        required_margin = (final_lots * spec.contract_value * trigger_price) / leverage

        is_sufficient = (
            (final_lots >= 1)
            and (round(required_margin, 4) <= round(max_usable_margin, 4))
            and (round(required_margin, 4) <= round(available_margin, 4))
        )
        rejection_reason = None

        if not is_sufficient:
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

    def record_trade_exit(self, pnl: float, current_equity: float) -> None:
        """Records realized PnL and tests daily drawdown limits."""
        self.daily_realized_pnl += pnl
        self.update_daily_equity(current_equity)
        if self.active_positions_count > 0:
            self.active_positions_count -= 1

    def record_trade_entry(self) -> None:
        """Increments active position count."""
        self.active_positions_count += 1
