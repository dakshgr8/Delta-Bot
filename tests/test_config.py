"""Tests for BotConfig and DeltaEnvironment."""

import pytest
from pydantic import SecretStr

from delta_bot.config import BotConfig, DeltaEnvironment, DeltaWebSocketEnvironment


def test_config_defaults() -> None:
    """Verifies default configuration values."""
    config = BotConfig(
        delta_api_key=SecretStr("dummy_key"),
        delta_api_secret=SecretStr("dummy_secret"),
    )
    assert config.delta_base_url == "https://api.india.delta.exchange"
    assert config.delta_ws_url == "wss://socket.india.delta.exchange"
    assert config.target_symbols == ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "XAUTUSD"]
    assert config.request_timeout_sec == 10.0
    assert config.max_retries == 3
    assert config.backoff_base_sec == 0.5
    assert config.backoff_max_sec == 8.0
    assert "DeltaBot" in config.user_agent


def test_base_url_normalization() -> None:
    """Verifies trailing slashes and spaces are normalized in base URLs."""
    config = BotConfig(
        delta_base_url="https://api.india.delta.exchange/   ",
        delta_api_key=SecretStr("k"),
        delta_api_secret=SecretStr("s"),
    )
    assert config.delta_base_url == "https://api.india.delta.exchange"


def test_base_url_invalid_scheme() -> None:
    """Verifies ValueError is raised for invalid schemes."""
    with pytest.raises(ValueError, match="Invalid URL scheme"):
        BotConfig(
            delta_base_url="ftp://api.delta.exchange",
            delta_api_key=SecretStr("k"),
            delta_api_secret=SecretStr("s"),
        )


def test_credential_redaction() -> None:
    """Verifies private credentials do not leak in str() or repr()."""
    raw_secret = "SUPER_SECRET_KEY_9999"
    raw_key = "MY_PUBLIC_API_KEY_1234"
    config = BotConfig(
        delta_api_key=SecretStr(raw_key),
        delta_api_secret=SecretStr(raw_secret),
    )
    repr_output = repr(config)
    str_output = str(config)
    safe_dict = config.safe_summary()

    assert raw_secret not in repr_output
    assert raw_secret not in str_output
    assert safe_dict["delta_api_secret"] == "********"
    assert safe_dict["delta_api_key"] == "MY_P...1234"


def test_supported_environments() -> None:
    """Verifies DeltaEnvironment enum contains India, Global, and Testnet."""
    assert DeltaEnvironment.INDIA.value == "https://api.india.delta.exchange"
    assert DeltaEnvironment.GLOBAL.value == "https://api.delta.exchange"
    assert DeltaEnvironment.TESTNET.value == "https://testnet-api.delta.exchange"


def test_supported_ws_environments() -> None:
    """Verifies DeltaWebSocketEnvironment enum contains India and Global."""
    assert DeltaWebSocketEnvironment.INDIA.value == "wss://socket.india.delta.exchange"
    assert DeltaWebSocketEnvironment.GLOBAL.value == "wss://socket.delta.exchange"


def test_ws_url_normalization() -> None:
    """Verifies WebSocket URL normalization and validation."""
    config = BotConfig(delta_ws_url="wss://socket.india.delta.exchange/   ")
    assert config.delta_ws_url == "wss://socket.india.delta.exchange"
    assert config.ws_heartbeat_interval_sec == 20.0
    assert config.bar_cache_capacity == 500

    with pytest.raises(ValueError, match="Invalid WebSocket scheme"):
        BotConfig(delta_ws_url="http://socket.delta.exchange")
