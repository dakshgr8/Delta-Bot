"""
tests/test_websocket.py
~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive unit test suite for the Market Data and WebSocket Engine (Phase 2).
Validates:
- SubscriptionDispatcher canonical payload construction and re-subscription mapping
- SequenceTracker dropped packet detection and sequence gap tracking
- CandleAggregator bar building: forming ticks vs confirmed closed bar guarantees
- CircularBarCache 500-bar RingBuffer retention and array export
- Daily Session Anchor 00:00 UTC rollover, PDH/PDL/PDC, and Intraday Session VWAP
- OrderBook level sorting and execution slippage estimation
- ChannelMultiplexer message dispatching
"""

import asyncio
from datetime import datetime, timezone
import json
import pytest

from delta_bot.models import (
    Candle,
    CircularBarCache,
    OrderBook,
    OrderBookLevel,
    SessionAnchorData,
    Ticker,
)
from delta_bot.websocket_client import (
    CandleAggregator,
    ChannelMultiplexer,
    SequenceTracker,
    SubscriptionDispatcher,
)


# -----------------------------------------------------------------------------
# 1. SubscriptionDispatcher Tests
# -----------------------------------------------------------------------------

def test_subscription_dispatcher_payload() -> None:
    """Verifies that SubscriptionDispatcher constructs canonical Delta JSON frames."""
    dispatcher = SubscriptionDispatcher()
    assert dispatcher.has_subscriptions is False

    # Add subscriptions
    dispatcher.add_subscription("v2/ticker", ["BTCUSDT", "ETHUSDT"])
    dispatcher.add_subscription("candlestick_1m", ["BTCUSDT"])
    assert dispatcher.has_subscriptions is True

    payload_str = dispatcher.build_subscription_payload()
    payload = json.loads(payload_str)

    assert payload["type"] == "subscribe"
    channels = {c["name"]: c["symbols"] for c in payload["payload"]["channels"]}

    assert "v2/ticker" in channels
    assert sorted(channels["v2/ticker"]) == ["BTCUSDT", "ETHUSDT"]
    assert "candlestick_1m" in channels
    assert channels["candlestick_1m"] == ["BTCUSDT"]

    # Test removing subscription
    dispatcher.remove_subscription("v2/ticker", ["ETHUSDT"])
    active = dispatcher.get_all_subscriptions()
    assert active["v2/ticker"] == {"BTCUSDT"}


# -----------------------------------------------------------------------------
# 2. SequenceTracker Tests
# -----------------------------------------------------------------------------

def test_sequence_tracker_gaps() -> None:
    """Verifies sequence gap identification and dropped packet counting."""
    tracker = SequenceTracker()
    stream = "ob:BTCUSDT"

    # First packet: baseline established, no gap
    has_gap, dropped = tracker.track_sequence(stream, 100)
    assert has_gap is False
    assert dropped == 0

    # Sequential packet: 101
    has_gap, dropped = tracker.track_sequence(stream, 101)
    assert has_gap is False
    assert dropped == 0

    # Gap packet: jump to 105 (missed 102, 103, 104 -> 3 dropped)
    has_gap, dropped = tracker.track_sequence(stream, 105)
    assert has_gap is True
    assert dropped == 3
    assert tracker.total_gaps_detected == 1
    assert tracker.total_packets_dropped == 3


# -----------------------------------------------------------------------------
# 3. CandleAggregator & RingBuffer Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_candle_aggregator_forming_vs_closed() -> None:
    """
    Guarantees strict separation between forming open candles and confirmed closed candles:
    Strategy signals MUST only fire on confirmed closed bars!
    """
    closed_candles: list[Candle] = []
    updated_candles: list[Candle] = []

    async def on_close(symbol: str, candle: Candle) -> None:
        closed_candles.append(candle)

    async def on_update(symbol: str, candle: Candle) -> None:
        updated_candles.append(candle)

    aggregator = CandleAggregator(
        capacity=500,
        on_candle_close=on_close,
        on_candle_update=on_update,
    )

    symbol = "BTCUSDT"
    res = "1m"
    t0_usec = 1712750400000000  # Minute 0: 00:00:00

    # Tick 1: Minute 0 forming candle starts
    msg1 = {
        "symbol": symbol,
        "resolution": res,
        "candle_start_time": t0_usec,
        "open": 75000.0,
        "high": 75050.0,
        "low": 74980.0,
        "close": 75020.0,
        "volume": 1.5,
    }
    await aggregator.process_candle_update(msg1)

    cache = aggregator.get_bar_cache(symbol, res)
    assert len(closed_candles) == 0, "Closed candle fired prematurely on forming bar!"
    assert cache.count == 0, "Forming bar prematurely placed into closed cache!"
    assert cache.open_candle is not None
    assert cache.open_candle.close == 75020.0
    assert cache.open_candle.is_closed is False

    # Tick 2: Minute 0 forming candle updates price
    msg2 = {
        "symbol": symbol,
        "resolution": res,
        "candle_start_time": t0_usec,
        "open": 75000.0,
        "high": 75100.0,
        "low": 74980.0,
        "close": 75080.0,
        "volume": 3.0,
    }
    await aggregator.process_candle_update(msg2)
    assert len(closed_candles) == 0
    assert cache.open_candle.high == 75100.0

    # Tick 3: Minute 1 arrives -> Minute 0 is now CONFIRMED CLOSED!
    t1_usec = t0_usec + 60_000_000  # Minute 1: 00:01:00
    msg3 = {
        "symbol": symbol,
        "resolution": res,
        "candle_start_time": t1_usec,
        "open": 75080.0,
        "high": 75090.0,
        "low": 75070.0,
        "close": 75085.0,
        "volume": 0.5,
    }
    await aggregator.process_candle_update(msg3)

    # Verification: Exactly 1 closed candle fired
    assert len(closed_candles) == 1, "Closed candle event was not fired upon transition!"
    closed_bar = closed_candles[0]
    assert closed_bar.is_closed is True
    assert closed_bar.open == 75000.0
    assert closed_bar.high == 75100.0
    assert closed_bar.low == 74980.0
    assert closed_bar.close == 75080.0
    assert closed_bar.volume == 3.0
    assert closed_bar.candle_start_time_usec == t0_usec

    # Cache state verification
    assert cache.count == 1
    assert cache.last_closed() == closed_bar
    assert cache.open_candle.candle_start_time_usec == t1_usec
    assert cache.open_candle.is_closed is False


def test_circular_bar_cache_retention() -> None:
    """Verifies that CircularBarCache retains up to capacity (e.g. 500) and drops oldest FIFO."""
    capacity = 5
    cache = CircularBarCache(symbol="BTCUSDT", resolution="1m", capacity=capacity)

    for i in range(10):
        c = Candle(
            symbol="BTCUSDT",
            resolution="1m",
            open=100.0 + i,
            high=105.0 + i,
            low=95.0 + i,
            close=102.0 + i,
            volume=1.0,
            candle_start_time_usec=1700000000000000 + (i * 60_000_000),
            is_closed=True,
        )
        cache.append(c)

    assert cache.count == capacity
    assert cache.is_full is True
    closed_bars = cache.get_closed_bars()
    assert len(closed_bars) == 5
    # Oldest retained should be i=5 (open=105.0), newest i=9 (open=109.0)
    assert closed_bars[0].open == 105.0
    assert closed_bars[-1].open == 109.0

    # Verify to_dict output
    data_dict = cache.to_dict()
    assert len(data_dict["close"]) == 5
    assert data_dict["close"][-1] == 111.0


# -----------------------------------------------------------------------------
# 4. Session Anchor & 00:00 UTC Rollover Tests
# -----------------------------------------------------------------------------

def test_session_anchor_vwap_and_rollover() -> None:
    """Verifies Intraday Session VWAP calculation and 00:00 UTC daily session rollover."""
    anchor = SessionAnchorData(
        symbol="BTCUSDT",
        session_date="2026-11-05",
        session_start_epoch_s=1762300800.0,  # 2026-11-05 00:00:00 UTC
    )

    # Bar 1: Price 100, Vol 10 -> Typical = 100
    anchor.update(price_or_candle=100.0, volume=10.0, high=100.0, low=100.0)
    assert anchor.session_vwap == 100.0
    assert anchor.session_high == 100.0
    assert anchor.session_low == 100.0

    # Bar 2: Price 110, Vol 10 -> Typical = 110. VWAP = (100*10 + 110*10) / 20 = 105.0
    anchor.update(price_or_candle=110.0, volume=10.0, high=115.0, low=105.0)
    assert anchor.session_vwap == 105.0
    assert anchor.session_high == 115.0
    assert anchor.session_low == 100.0

    # Trigger rollover to 2026-11-06
    new_anchor = anchor.rollover(
        next_session_date="2026-11-06",
        next_session_start_s=1762387200.0,
        next_open=112.0,
    )

    # Prior Day High, Low, Close captured
    assert new_anchor.session_date == "2026-11-06"
    assert new_anchor.prior_day_high == 115.0  # PDH
    assert new_anchor.prior_day_low == 100.0   # PDL
    assert new_anchor.prior_day_close == 110.0 # PDC
    # New session accumulators reset
    assert new_anchor.session_volume == 0.0
    assert new_anchor.session_vwap == 112.0
    assert new_anchor.session_open == 112.0


# -----------------------------------------------------------------------------
# 5. OrderBook & Slippage Estimation Tests
# -----------------------------------------------------------------------------

def test_orderbook_slippage_calculation() -> None:
    """Verifies OrderBook parsing, level sorting, and execution slippage simulation."""
    ob_data = {
        "symbol": "BTCUSDT",
        "product_id": 139,
        "last_sequence_no": 5001,
        "buy": [
            {"limit_price": "75000.0", "size": "10.0"},
            {"limit_price": "74990.0", "size": "20.0"},
        ],
        "sell": [
            {"limit_price": "75010.0", "size": "5.0"},
            {"limit_price": "75020.0", "size": "15.0"},
            {"limit_price": "75050.0", "size": "30.0"},
        ],
    }

    ob = OrderBook.from_api_dict(ob_data)

    assert ob.best_bid == 75000.0
    assert ob.best_ask == 75010.0
    assert ob.spread == 10.0
    assert ob.mid_price == 75005.0

    # Estimate buy slippage for size = 5 (all filled at top level 75010.0 -> 0 slippage)
    vwap, slippage_bps = ob.estimate_slippage(side="buy", size=5.0)
    assert vwap == 75010.0
    assert slippage_bps == 0.0

    # Estimate buy slippage for size = 15:
    # 5 filled at 75010.0 = 375,050
    # 10 filled at 75020.0 = 750,200
    # Total cost = 1,125,250 / 15 = 75016.666667
    # Slippage = (75016.666667 - 75010) / 75010 * 10000 = 0.89 bps
    vwap, slippage_bps = ob.estimate_slippage(side="buy", size=15.0)
    assert pytest.approx(vwap, 0.001) == 75016.667
    assert pytest.approx(slippage_bps, 0.01) == 0.89

    # Test liquidity exhaustion
    with pytest.raises(ValueError, match="Insufficient liquidity"):
        ob.estimate_slippage(side="buy", size=100.0)


# -----------------------------------------------------------------------------
# 6. ChannelMultiplexer Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_channel_multiplexer_dispatching() -> None:
    """Verifies that incoming JSON messages are accurately routed to respective callbacks."""
    received_tickers: list[Ticker] = []
    received_books: list[OrderBook] = []

    async def on_ticker(sym: str, t: Ticker) -> None:
        received_tickers.append(t)

    async def on_book(sym: str, b: OrderBook) -> None:
        received_books.append(b)

    aggregator = CandleAggregator()
    seq_tracker = SequenceTracker()
    multiplexer = ChannelMultiplexer(
        aggregator=aggregator,
        sequence_tracker=seq_tracker,
        on_ticker=on_ticker,
        on_orderbook=on_book,
    )

    # 1. Ticker message
    ticker_payload = json.dumps({
        "type": "v2/ticker",
        "symbol": "ETHUSDT",
        "product_id": 176,
        "mark_price": "2250.50",
        "quotes": {"best_bid": "2250.0", "best_ask": "2250.5"},
    })
    await multiplexer.handle_message(ticker_payload)
    assert len(received_tickers) == 1
    assert received_tickers[0].symbol == "ETHUSDT"
    assert received_tickers[0].mark_price == 2250.50

    # 2. Orderbook message
    ob_payload = json.dumps({
        "type": "l2_orderbook",
        "symbol": "ETHUSDT",
        "product_id": 176,
        "last_sequence_no": 1,
        "buy": [{"limit_price": "2250.0", "size": "10"}],
        "sell": [{"limit_price": "2250.5", "size": "10"}],
    })
    await multiplexer.handle_message(ob_payload)
    assert len(received_books) == 1
    assert received_books[0].best_bid == 2250.0

    # 3. Heartbeat frame (should be handled without error)
    await multiplexer.handle_message(json.dumps({"type": "heartbeat"}))
    await multiplexer.handle_message(json.dumps({"type": "pong"}))

    # 4. Malformed JSON should not crash
    await multiplexer.handle_message("NOT_JSON")


def test_circular_bar_cache_maxlen_initialization() -> None:
    """Verifies CircularBarCache(maxlen=50) initialization and seeding."""
    c = CircularBarCache(maxlen=50)
    assert hasattr(c, "is_full")
    assert c.is_full is False
    assert len(c) == 0

    candles = [
        Candle(
            symbol="BTCUSDT",
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1.0,
            candle_start_time_usec=1700000000000000 + (i * 60_000_000),
            is_closed=True,
        )
        for i in range(50)
    ]
    seeded = c.seed_historical_bars(candles)
    assert seeded == 50
    assert c.is_full is True
    assert len(c) == 50


@pytest.mark.asyncio
async def test_orderbook_gap_invalidation_and_resync() -> None:
    """
    Verifies that when an L2 sequence gap occurs:
    1. OrderBook is invalidated (is_valid=False, needs_resync=True)
    2. estimate_slippage() rejects calculations on corrupt book
    3. on_orderbook_resync callback is triggered to fetch fresh snapshot
    """
    resync_symbols: list[str] = []
    received_books: list[OrderBook] = []

    async def on_book(sym: str, b: OrderBook) -> None:
        received_books.append(b)

    async def on_resync(sym: str) -> None:
        resync_symbols.append(sym)

    aggregator = CandleAggregator()
    seq_tracker = SequenceTracker()
    multiplexer = ChannelMultiplexer(
        aggregator=aggregator,
        sequence_tracker=seq_tracker,
        on_orderbook=on_book,
        on_orderbook_resync=on_resync,
    )

    # Frame 1: Sequence 100
    f1 = json.dumps({
        "type": "l2_orderbook",
        "symbol": "BTCUSDT",
        "last_sequence_no": 100,
        "buy": [{"limit_price": "75000", "size": "10"}],
        "sell": [{"limit_price": "75010", "size": "10"}],
    })
    await multiplexer.handle_message(f1)
    assert received_books[0].is_valid is True
    assert len(resync_symbols) == 0

    # Frame 2: Gap jump to Sequence 105 (Dropped 4 packets)
    f2 = json.dumps({
        "type": "l2_orderbook",
        "symbol": "BTCUSDT",
        "last_sequence_no": 105,
        "buy": [{"limit_price": "75000", "size": "10"}],
        "sell": [{"limit_price": "75010", "size": "10"}],
    })
    await multiplexer.handle_message(f2)

    # Verification: Book is invalidated and resync triggered
    corrupted_book = received_books[1]
    assert corrupted_book.is_valid is False
    assert corrupted_book.needs_resync is True
    assert "BTCUSDT" in resync_symbols

    # Slippage estimation MUST raise error on corrupted book
    with pytest.raises(ValueError, match="invalid due to a detected sequence gap"):
        corrupted_book.estimate_slippage(side="buy", size=1.0)
