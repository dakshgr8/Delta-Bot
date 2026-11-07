"""Integration and unit tests for DeltaRestClient and error handling."""

import pytest
from pydantic import SecretStr

from delta_bot.client import DeltaRestClient
from delta_bot.config import BotConfig
from delta_bot.exceptions import (
    AuthenticationError,
    DeltaAPIError,
    InsufficientMarginError,
    RateLimitError,
)


@pytest.mark.asyncio
async def test_client_public_endpoints() -> None:
    """Verifies live exchange endpoints: get_server_time, get_products, get_ticker."""
    config = BotConfig()
    async with DeltaRestClient(config) as client:
        # Server time
        time_info = await client.get_server_time()
        assert time_info.server_epoch_s > 0
        assert time_info.round_trip_latency_ms >= 0

        # Products & Caching
        specs = await client.get_products(["perpetual_futures"])
        assert len(specs) > 0

        # Verify target perpetuals
        target_map = await client.get_target_perpetuals()
        assert "BTC" in target_map
        assert "ETH" in target_map
        assert "SOL" in target_map
        assert "XRP" in target_map

        # Verify caching without network call
        cached_btc = client.get_cached_contract_spec("BTCUSDT")
        assert cached_btc is not None
        assert cached_btc.symbol == "BTCUSDT"

        # Real-time ticker
        ticker = await client.get_ticker("BTCUSDT")
        assert ticker.symbol == "BTCUSDT"
        assert ticker.mark_price > 0

        # Historical closed candles for REST warm-up
        hist_candles = await client.get_historical_candles("BTCUSDT", resolution="1m", limit=10)
        assert len(hist_candles) > 0
        assert hist_candles[0].is_closed is True
        assert hist_candles[0].open > 0


@pytest.mark.asyncio
async def test_authenticated_endpoint_error_mapping() -> None:
    """Verifies that an unauthenticated / invalid credentials request raises AuthenticationError."""
    config = BotConfig(
        delta_api_key=SecretStr("INVALID_KEY"),
        delta_api_secret=SecretStr("INVALID_SECRET"),
    )
    async with DeltaRestClient(config) as client:
        with pytest.raises(AuthenticationError):
            await client.get_wallet_balances()


def test_exception_properties() -> None:
    """Verifies that custom exceptions hold proper diagnostic properties."""
    api_err = DeltaAPIError("General fail", status_code=400, error_code="bad_request")
    assert api_err.status_code == 400
    assert api_err.error_code == "bad_request"

    auth_err = AuthenticationError("Expired", status_code=401, error_code="expired_signature")
    assert auth_err.status_code == 401

    margin_err = InsufficientMarginError("No funds", status_code=400)
    assert margin_err.error_code == "insufficient_margin"

    rl_err = RateLimitError("Too Many Requests", status_code=429, retry_after_ms=1250.0)
    assert rl_err.retry_after_ms == 1250.0


@pytest.mark.asyncio
async def test_cross_region_fallback_xautusd() -> None:
    """Verifies that XAUTUSD (exclusive to Delta India) resolves via fallback when client is on Delta Global."""
    config = BotConfig(delta_base_url="https://api.delta.exchange")
    async with DeltaRestClient(config) as client:
        spec = await client.get_contract_spec("XAUTUSD")
        assert spec is not None
        assert spec.symbol == "XAUTUSD"
        assert spec.product_id == 131253
        assert spec.contract_type == "perpetual_futures"

        # Check alias lookup in cache
        alias_spec = client.get_cached_contract_spec("XAUTUSDT")
        assert alias_spec is not None
        assert alias_spec.product_id == 131253


@pytest.mark.asyncio
async def test_symbol_aliasing_and_historical_candles() -> None:
    """Verifies that requesting BTCUSDT against Delta India successfully resolves BTCUSD and fetches candles."""
    config = BotConfig(delta_base_url="https://api.india.delta.exchange")
    async with DeltaRestClient(config) as client:
        # Preload India perpetuals
        await client.get_products()
        spec = client.get_cached_contract_spec("BTCUSDT")
        assert spec is not None
        assert spec.symbol == "BTCUSD"

        # Candles query using BTCUSDT should resolve to BTCUSD on exchange
        candles = await client.get_historical_candles("BTCUSDT", resolution="1m", limit=5)
        assert len(candles) > 0
        assert candles[0].symbol == "BTCUSDT"
        assert candles[0].close > 0
