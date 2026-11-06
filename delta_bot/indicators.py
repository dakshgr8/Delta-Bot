"""
delta_bot.indicators
~~~~~~~~~~~~~~~~~~~~

Pure NumPy and mathematical indicator calculation library for quantitative trading.
No heavy external TA dependencies.
Implements:
- Exponential Moving Average (EMA)
- Simple Moving Average (SMA)
- Wilder's Average True Range (ATR) & ATR Moving Average
- Average Directional Index (ADX) with +DI / -DI
- Consolidated IndicatorSnapshot and streaming IndicatorPipeline
Guarantees zero lookahead bias and strict evaluation over confirmed closed bars.
"""

from typing import Any
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from delta_bot.models import Candle, CircularBarCache, SessionAnchorData


class IndicatorSnapshot(BaseModel):
    """
    Consolidated indicator readings computed strictly over confirmed closed bars.
    Provides complete state snapshot for market regime detection and signal generation.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(description="Trading symbol")
    timestamp_epoch_s: float = Field(description="Bar close timestamp in epoch seconds")
    close: float = Field(description="Bar closing price")
    high: float = Field(description="Bar high price")
    low: float = Field(description="Bar low price")

    # Trend & Moving Averages
    ema_20: float = Field(description="20-period Exponential Moving Average")
    session_vwap: float = Field(description="00:00 UTC Intraday Session VWAP")

    # Volatility (ATR)
    atr_14: float = Field(description="14-period Wilder's Average True Range")
    atr_sma_20: float = Field(description="20-period Simple Moving Average of ATR")
    is_volatility_expanded: bool = Field(
        description="True if ATR_14 >= SMA(ATR, 20) confirming momentum expansion"
    )

    # Regime (ADX)
    adx_14: float = Field(description="14-period Average Directional Index")
    plus_di_14: float = Field(description="14-period Positive Directional Indicator (+DI)")
    minus_di_14: float = Field(description="14-period Negative Directional Indicator (-DI)")
    is_trending: bool = Field(description="True if ADX >= 20.0 (non-consolidation regime)")

    # Daily Session Anchor Levels
    pdh: float | None = Field(default=None, description="Prior Day High")
    pdl: float | None = Field(default=None, description="Prior Day Low")
    pdc: float | None = Field(default=None, description="Prior Day Close")

    @property
    def plus_di(self) -> float:
        """Alias for plus_di_14."""
        return self.plus_di_14

    @property
    def minus_di(self) -> float:
        """Alias for minus_di_14."""
        return self.minus_di_14


# -----------------------------------------------------------------------------
# Vectorized Math Functions (Pure NumPy)
# -----------------------------------------------------------------------------

def calculate_sma(values: np.ndarray | list[float], period: int) -> np.ndarray:
    """
    Computes Simple Moving Average (SMA) over 1D array.
    Pads leading (period - 1) indices with NaN.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n < period or period <= 0:
        return np.full(n, np.nan, dtype=np.float64)

    ret = np.full(n, np.nan, dtype=np.float64)
    cumsum = np.cumsum(arr)
    ret[period - 1] = cumsum[period - 1] / period
    ret[period:] = (cumsum[period:] - cumsum[:-period]) / period
    return ret


def calculate_ema(values: np.ndarray | list[float], period: int) -> np.ndarray:
    """
    Computes Exponential Moving Average (EMA) using smoothing factor:
        alpha = 2 / (period + 1)
    Initial seed is the SMA of the first 'period' elements.
    Pads leading (period - 1) indices with NaN.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n < period or period <= 0:
        return np.full(n, np.nan, dtype=np.float64)

    ema = np.full(n, np.nan, dtype=np.float64)
    # Seed with initial SMA
    initial_sma = np.mean(arr[:period])
    ema[period - 1] = initial_sma

    alpha = 2.0 / (period + 1.0)
    for i in range(period, n):
        ema[i] = alpha * arr[i] + (1.0 - alpha) * ema[i - 1]

    return ema


def calculate_true_range(
    highs: np.ndarray | list[float],
    lows: np.ndarray | list[float],
    closes: np.ndarray | list[float],
) -> np.ndarray:
    """
    Computes True Range (TR) series:
        TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
    TR_0 = H_0 - L_0.
    """
    h = np.asarray(highs, dtype=np.float64)
    l = np.asarray(lows, dtype=np.float64)
    c = np.asarray(closes, dtype=np.float64)
    n = len(h)
    if n == 0:
        return np.array([], dtype=np.float64)

    tr = np.zeros(n, dtype=np.float64)
    tr[0] = h[0] - l[0]

    if n > 1:
        hl = h[1:] - l[1:]
        hc = np.abs(h[1:] - c[:-1])
        lc = np.abs(l[1:] - c[:-1])
        tr[1:] = np.maximum(hl, np.maximum(hc, lc))

    return tr


def calculate_atr(
    highs: np.ndarray | list[float],
    lows: np.ndarray | list[float],
    closes: np.ndarray | list[float],
    period: int = 14,
) -> np.ndarray:
    """
    Computes Wilder's Average True Range (ATR):
        ATR_0 = SMA(TR, period)
        ATR_t = (ATR_{t-1} * (period - 1) + TR_t) / period
    """
    tr = calculate_true_range(highs, lows, closes)
    n = len(tr)
    if n < period or period <= 0:
        return np.full(n, np.nan, dtype=np.float64)

    atr = np.full(n, np.nan, dtype=np.float64)
    # Initial SMA
    atr[period - 1] = np.mean(tr[:period])

    # Wilder's smoothing
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1.0) + tr[i]) / float(period)

    return atr


def calculate_adx(
    highs: np.ndarray | list[float],
    lows: np.ndarray | list[float],
    closes: np.ndarray | list[float],
    period: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Average Directional Index (ADX) alongside +DI and -DI using Wilder's smoothing:
    Returns tuple of (adx_series, plus_di_series, minus_di_series).
    Pads warmup periods with NaN.
    """
    h = np.asarray(highs, dtype=np.float64)
    l = np.asarray(lows, dtype=np.float64)
    c = np.asarray(closes, dtype=np.float64)
    n = len(h)

    nan_series = np.full(n, np.nan, dtype=np.float64)
    if n < (period * 2):
        return nan_series, nan_series, nan_series

    # 1. Directional Movements (+DM, -DM) and True Range (TR)
    up_move = h[1:] - h[:-1]
    down_move = l[:-1] - l[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = calculate_true_range(h, l, c)[1:]

    # 2. Wilder's Smoothing of TR, +DM, -DM
    m = len(tr)
    smooth_tr = np.full(m, np.nan, dtype=np.float64)
    smooth_plus_dm = np.full(m, np.nan, dtype=np.float64)
    smooth_minus_dm = np.full(m, np.nan, dtype=np.float64)

    smooth_tr[period - 1] = np.sum(tr[:period])
    smooth_plus_dm[period - 1] = np.sum(plus_dm[:period])
    smooth_minus_dm[period - 1] = np.sum(minus_dm[:period])

    for i in range(period, m):
        smooth_tr[i] = smooth_tr[i - 1] - (smooth_tr[i - 1] / period) + tr[i]
        smooth_plus_dm[i] = smooth_plus_dm[i - 1] - (smooth_plus_dm[i - 1] / period) + plus_dm[i]
        smooth_minus_dm[i] = smooth_minus_dm[i - 1] - (smooth_minus_dm[i - 1] / period) + minus_dm[i]

    # 3. +DI and -DI
    plus_di_m = np.divide(
        100.0 * smooth_plus_dm,
        smooth_tr,
        out=np.zeros_like(smooth_plus_dm),
        where=smooth_tr > 0,
    )
    minus_di_m = np.divide(
        100.0 * smooth_minus_dm,
        smooth_tr,
        out=np.zeros_like(smooth_minus_dm),
        where=smooth_tr > 0,
    )

    # 4. DX = 100 * |+DI - -DI| / (+DI + -DI)
    di_sum = plus_di_m + minus_di_m
    di_diff = np.abs(plus_di_m - minus_di_m)
    dx_m = np.divide(
        100.0 * di_diff,
        di_sum,
        out=np.zeros_like(di_diff),
        where=di_sum > 0,
    )

    # 5. ADX = Wilder's smoothing of DX over 'period'
    adx_m = np.full(m, np.nan, dtype=np.float64)
    dx_start = (period - 1) + period
    if m >= dx_start:
        adx_m[dx_start - 1] = np.mean(dx_m[period - 1 : dx_start])
        for i in range(dx_start, m):
            adx_m[i] = ((adx_m[i - 1] * (period - 1.0)) + dx_m[i]) / float(period)

    # Align back to length n by prepending NaN for index 0
    adx_out = np.concatenate(([np.nan], adx_m))
    plus_di_out = np.concatenate(([np.nan], plus_di_m))
    minus_di_out = np.concatenate(([np.nan], minus_di_m))

    return adx_out, plus_di_out, minus_di_out


# -----------------------------------------------------------------------------
# Streaming Indicator Pipeline
# -----------------------------------------------------------------------------

class IndicatorPipeline:
    """
    Maintains and updates technical indicators strictly over confirmed closed bars.
    Guarantees:
    - Zero lookahead bias
    - Fast calculation (NumPy vectorized slices)
    - Full parameterization for V2 Upgraded Breakout Strategy
    """

    def __init__(
        self,
        symbol: str,
        ema_period: int = 20,
        atr_period: int = 14,
        atr_sma_period: int = 20,
        adx_period: int = 14,
        adx_threshold: float = 20.0,
    ) -> None:
        self.symbol: str = symbol.upper()
        self.ema_period: int = ema_period
        self.atr_period: int = atr_period
        self.atr_sma_period: int = atr_sma_period
        self.adx_period: int = adx_period
        self.adx_threshold: float = adx_threshold

        # Historical snapshots history (latest is at index -1)
        self._snapshots: list[IndicatorSnapshot] = []

    @property
    def latest_snapshot(self) -> IndicatorSnapshot | None:
        """Returns the most recent confirmed indicator snapshot."""
        return self._snapshots[-1] if self._snapshots else None

    @property
    def previous_snapshot(self) -> IndicatorSnapshot | None:
        """Returns the indicator snapshot from bar t-1."""
        return self._snapshots[-2] if len(self._snapshots) >= 2 else None

    @property
    def has_sufficient_data(self) -> bool:
        """
        True if enough closed bars have been evaluated to produce valid (non-NaN)
        readings for EMA, ATR, ATR_SMA, and ADX.
        Required warmup: max(ema_period, adx_period * 2, atr_period + atr_sma_period).
        """
        req = max(self.ema_period, self.adx_period * 2, self.atr_period + self.atr_sma_period)
        return len(self._snapshots) >= req and self.latest_snapshot is not None

    def compute_from_cache(
        self,
        bar_cache: CircularBarCache,
        session_anchor: SessionAnchorData,
    ) -> IndicatorSnapshot | None:
        """
        Computes the complete IndicatorSnapshot from the 500-bar RingBuffer and SessionAnchor.
        Appends to internal snapshot history and returns the latest snapshot.
        """
        closed_bars = bar_cache.get_closed_bars()
        n = len(closed_bars)
        if n == 0:
            return None

        # Extract numpy arrays
        highs = np.array([b.high for b in closed_bars], dtype=np.float64)
        lows = np.array([b.low for b in closed_bars], dtype=np.float64)
        closes = np.array([b.close for b in closed_bars], dtype=np.float64)
        last_bar = closed_bars[-1]

        # 1. EMA 20
        ema_series = calculate_ema(closes, period=self.ema_period)
        ema_val = float(ema_series[-1]) if not np.isnan(ema_series[-1]) else last_bar.close

        # 2. ATR 14 and SMA(ATR, 20)
        atr_series = calculate_atr(highs, lows, closes, period=self.atr_period)
        atr_sma_series = calculate_sma(atr_series, period=self.atr_sma_period)

        atr_val = float(atr_series[-1]) if not np.isnan(atr_series[-1]) else (last_bar.high - last_bar.low)
        atr_sma_val = (
            float(atr_sma_series[-1])
            if not np.isnan(atr_sma_series[-1])
            else atr_val
        )
        is_vol_expanded = bool(atr_val >= atr_sma_val)

        # 3. ADX 14 with +DI / -DI
        adx_series, plus_di_series, minus_di_series = calculate_adx(
            highs, lows, closes, period=self.adx_period
        )
        adx_val = float(adx_series[-1]) if not np.isnan(adx_series[-1]) else 0.0
        plus_di_val = float(plus_di_series[-1]) if not np.isnan(plus_di_series[-1]) else 0.0
        minus_di_val = float(minus_di_series[-1]) if not np.isnan(minus_di_series[-1]) else 0.0
        is_trending = bool(adx_val >= self.adx_threshold)

        # 4. Session VWAP & Daily Anchor levels (PDH, PDL, PDC)
        vwap_val = session_anchor.session_vwap if session_anchor.session_vwap > 0 else last_bar.close
        pdh_val = session_anchor.prior_day_high
        pdl_val = session_anchor.prior_day_low
        pdc_val = session_anchor.prior_day_close

        snapshot = IndicatorSnapshot(
            symbol=self.symbol,
            timestamp_epoch_s=last_bar.candle_start_epoch_s,
            close=last_bar.close,
            high=last_bar.high,
            low=last_bar.low,
            ema_20=round(ema_val, 6),
            session_vwap=round(vwap_val, 6),
            atr_14=round(atr_val, 6),
            atr_sma_20=round(atr_sma_val, 6),
            is_volatility_expanded=is_vol_expanded,
            adx_14=round(adx_val, 2),
            plus_di_14=round(plus_di_val, 2),
            minus_di_14=round(minus_di_val, 2),
            is_trending=is_trending,
            pdh=round(pdh_val, 6) if pdh_val is not None else None,
            pdl=round(pdl_val, 6) if pdl_val is not None else None,
            pdc=round(pdc_val, 6) if pdc_val is not None else None,
        )

        self._snapshots.append(snapshot)
        # Retain last 500 snapshots in sync with bar cache
        if len(self._snapshots) > bar_cache.capacity:
            self._snapshots = self._snapshots[-bar_cache.capacity :]

        return snapshot
