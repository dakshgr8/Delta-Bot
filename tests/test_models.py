"""Tests for domain models (ContractSpec, Ticker, WalletBalance, ServerTimeInfo)."""

import pytest
from delta_bot.models import (
    Candle,
    ContractSpec,
    RateLimitState,
    ServerTimeInfo,
    Ticker,
    WalletBalance,
)


def test_contract_spec_quantization() -> None:
    """Verifies round_price and round_size quantization rules."""
    spec = ContractSpec(
        product_id=139,
        symbol="BTCUSDT",
        contract_type="perpetual_futures",
        contract_value=0.001,
        tick_size=0.5,
        maker_commission_rate=0.0001,
        taker_commission_rate=0.0001,
    )

    # Price rounding to 0.5 increments
    assert spec.round_price(75500.2) == 75500.0
    assert spec.round_price(75500.3) == 75500.5
    assert spec.round_price(75500.75) == 75501.0

    # Size rounding: native asset units to integer contracts (lots)
    assert spec.round_size(0.0054) == 5
    assert isinstance(spec.round_size(0.0054), int)
    assert spec.round_size(0.0056) == 6
    assert spec.asset_qty_to_contracts(0.005) == 5
    assert spec.contracts_to_asset_qty(5) == 0.005


def test_ticker_spread_and_bps() -> None:
    """Verifies calculated spread and basis points for top of book."""
    ticker = Ticker(
        product_id=139,
        symbol="BTCUSDT",
        mark_price=75000.0,
        best_bid=74990.0,
        best_ask=75010.0,
        bid_size=10.0,
        ask_size=12.0,
        open_interest=500.0,
    )

    assert ticker.spread == 20.0
    # mid = 75000.0, spread = 20.0 -> bps = (20 / 75000) * 10000 = 2.67 bps
    assert ticker.spread_bps == 2.67


def test_wallet_balance_utilization() -> None:
    """Verifies margin utilization percentage calculation."""
    wb = WalletBalance(
        asset_id=5,
        asset_symbol="USDT",
        balance=10000.0,
        available_balance=7500.0,
        order_margin=500.0,
        position_margin=2000.0,
        blocked_margin=2500.0,
    )

    assert wb.margin_utilization_pct == 25.0


def test_server_time_drift_threshold() -> None:
    """Verifies drift acceptability check (within 3.5 seconds)."""
    acceptable = ServerTimeInfo(
        server_epoch_s=100.0,
        local_epoch_s=98.0,
        clock_offset_s=2.0,
        round_trip_latency_ms=50.0,
    )
    assert acceptable.is_drift_acceptable is True

    unacceptable = ServerTimeInfo(
        server_epoch_s=100.0,
        local_epoch_s=95.0,
        clock_offset_s=5.0,
        round_trip_latency_ms=50.0,
    )
    assert unacceptable.is_drift_acceptable is False


def test_rate_limit_state() -> None:
    """Verifies rate limit exhaustion and utilization percentage."""
    state = RateLimitState(limit=1000, remaining=200, reset_ms=500.0)
    assert state.is_exhausted is False
    assert state.utilization_pct == 80.0

    exhausted = RateLimitState(limit=1000, remaining=0, reset_ms=1000.0)
    assert exhausted.is_exhausted is True


def test_candle_computed_properties() -> None:
    """Verifies Candle price math: hl2, hlc3, ohlc4, range, body, direction."""
    c = Candle(
        symbol="BTCUSDT",
        resolution="1m",
        open=75000.0,
        high=75200.0,
        low=74900.0,
        close=75150.0,
        volume=12.5,
        candle_start_time_usec=1700000000000000,
        is_closed=True,
    )

    assert c.candle_start_epoch_s == 1700000000.0
    assert c.hl2 == (75200.0 + 74900.0) / 2.0
    assert c.hlc3 == (75200.0 + 74900.0 + 75150.0) / 3.0
    assert c.ohlc4 == (75000.0 + 75200.0 + 74900.0 + 75150.0) / 4.0
    assert c.is_bullish is True
    assert c.is_bearish is False
    assert c.range == 300.0
    assert c.body == 150.0
