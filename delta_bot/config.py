"""
delta_bot.config
~~~~~~~~~~~~~~~~

Production-grade typed configuration using Pydantic Settings.
Enforces credential masking, environment validation, and base URL normalization.
"""

from enum import Enum
from typing import Any
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeltaEnvironment(str, Enum):
    """Officially supported Delta Exchange environment endpoints."""

    INDIA = "https://api.india.delta.exchange"
    TESTNET_INDIA = "https://cdn-ind.testnet.deltaex.org"
    GLOBAL = "https://api.delta.exchange"
    TESTNET = "https://testnet-api.delta.exchange"
    TESTNET_GLOBAL = "https://testnet-api.delta.exchange"


class DeltaWebSocketEnvironment(str, Enum):
    """Officially supported Delta Exchange WebSocket endpoints."""

    INDIA = "wss://socket.india.delta.exchange"
    TESTNET_INDIA = "wss://socket-ind.testnet.deltaex.org"
    GLOBAL = "wss://socket.delta.exchange"
    TESTNET = "wss://socket-ind.testnet.deltaex.org"


class BotConfig(BaseSettings):
    """
    Centralized configuration engine reading from environment variables or .env file.
    Protects private keys via Pydantic SecretStr and explicit redaction routines.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # API Credentials (Masked automatically by SecretStr)
    delta_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("DELTA_API_KEY", "delta_api_key"),
        description="Delta Exchange API Key",
    )
    delta_api_secret: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("DELTA_API_SECRET", "delta_api_secret"),
        description="Delta Exchange API Secret",
    )

    # Base URL configuration (Defaults to Delta Exchange India)
    delta_base_url: str = Field(
        default=DeltaEnvironment.INDIA.value,
        validation_alias=AliasChoices("DELTA_BASE_URL", "delta_base_url"),
        description="Delta Exchange India base API URL",
    )

    # WebSocket URL configuration (Defaults to Delta Exchange India)
    delta_ws_url: str = Field(
        default=DeltaWebSocketEnvironment.INDIA.value,
        validation_alias=AliasChoices("DELTA_WS_URL", "delta_ws_url"),
        description="Delta Exchange India WebSocket API URL",
    )

    # Demo / Testnet Credentials & URLs
    delta_demo_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("DELTA_DEMO_API_KEY", "delta_demo_api_key"),
        description="Delta Exchange Demo / Testnet API Key",
    )
    delta_demo_api_secret: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("DELTA_DEMO_API_SECRET", "delta_demo_api_secret"),
        description="Delta Exchange Demo / Testnet API Secret",
    )
    delta_demo_base_url: str = Field(
        default=DeltaEnvironment.TESTNET_INDIA.value,
        validation_alias=AliasChoices("DELTA_DEMO_BASE_URL", "delta_demo_base_url"),
        description="Delta Exchange Demo / Testnet REST API URL",
    )
    delta_demo_ws_url: str = Field(
        default=DeltaWebSocketEnvironment.TESTNET_INDIA.value,
        validation_alias=AliasChoices("DELTA_DEMO_WS_URL", "delta_demo_ws_url"),
        description="Delta Exchange Demo / Testnet WebSocket API URL",
    )
    ws_heartbeat_interval_sec: float = Field(
        default=20.0,
        validation_alias=AliasChoices("WS_HEARTBEAT_INTERVAL_SEC", "ws_heartbeat_interval_sec"),
        ge=5.0,
        le=60.0,
        description="WebSocket ping heartbeat frequency in seconds",
    )
    ws_reconnect_max_attempts: int = Field(
        default=0,
        validation_alias=AliasChoices("WS_RECONNECT_MAX_ATTEMPTS", "ws_reconnect_max_attempts"),
        ge=0,
        description="Max reconnection attempts (0 for infinite/persistent)",
    )
    bar_cache_capacity: int = Field(
        default=500,
        validation_alias=AliasChoices("BAR_CACHE_CAPACITY", "bar_cache_capacity"),
        ge=50,
        le=5000,
        description="RingBuffer capacity for closed candlestick bars",
    )
    strict_orderbook_sequence_check: bool = Field(
        default=False,
        validation_alias=AliasChoices("STRICT_ORDERBOOK_SEQUENCE_CHECK", "strict_orderbook_sequence_check"),
        description="Whether to enforce strict consecutive sequence numbers on throttled L2 orderbook feeds",
    )
    enable_orderbook_stream: bool = Field(
        default=True,
        validation_alias=AliasChoices("ENABLE_ORDERBOOK_STREAM", "enable_orderbook_stream"),
        description="Whether to subscribe to continuous L2 orderbook feeds (consumes ~15GB/month)",
    )
    ws_broadcast_interval_sec: float = Field(
        default=2.0,
        validation_alias=AliasChoices("WS_BROADCAST_INTERVAL_SEC", "ws_broadcast_interval_sec"),
        ge=0.5,
        le=10.0,
        description="WebSocket live dashboard state broadcast frequency in seconds",
    )

    # HTTP Client timeouts & reliability
    request_timeout_sec: float = Field(
        default=10.0,
        validation_alias=AliasChoices("REQUEST_TIMEOUT_SEC", "request_timeout_sec"),
        ge=0.5,
        le=60.0,
        description="Client request timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        validation_alias=AliasChoices("MAX_RETRIES", "max_retries"),
        ge=0,
        le=10,
        description="Maximum retry attempts on transient network or server errors",
    )
    backoff_base_sec: float = Field(
        default=0.5,
        validation_alias=AliasChoices("BACKOFF_BASE_SEC", "backoff_base_sec"),
        ge=0.05,
        le=10.0,
        description="Base multiplier for exponential backoff",
    )
    backoff_max_sec: float = Field(
        default=8.0,
        validation_alias=AliasChoices("BACKOFF_MAX_SEC", "backoff_max_sec"),
        ge=0.5,
        le=60.0,
        description="Ceiling for exponential backoff delay",
    )

    # User Agent
    user_agent: str = Field(
        default="DeltaBot-QuantEngine/1.0.0 (Python 3.11+; AsyncIO; HighPerformance)",
        validation_alias=AliasChoices("USER_AGENT", "user_agent"),
        description="Custom User-Agent header sent to the exchange",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("LOG_LEVEL", "log_level"),
        description="Standard library log level",
    )

    # Deployment & Orchestrator Configuration (Phase 6)
    execution_mode: str = Field(
        default="live",
        validation_alias=AliasChoices("EXECUTION_MODE", "execution_mode"),
        description="Execution mode: 'live' (real exchange)",
    )
    flatten_on_shutdown: bool = Field(
        default=False,
        validation_alias=AliasChoices("FLATTEN_ON_SHUTDOWN", "flatten_on_shutdown"),
        description="Whether to market-close active positions when SIGINT/SIGTERM is received",
    )
    target_symbols_str: str = Field(
        default="BTCUSD,ETHUSD,SOLUSD,XRPUSD,XAUTUSD",
        validation_alias=AliasChoices("TARGET_SYMBOLS", "target_symbols"),
        description="Comma-separated target assets on Delta Exchange India (e.g. BTCUSD,ETHUSD,SOLUSD,XRPUSD,XAUTUSD)",
    )
    timeframe: str = Field(
        default="1m",
        validation_alias=AliasChoices("TIMEFRAME", "timeframe"),
        description="Candle resolution timeframe (e.g. 1m, 5m, 15m)",
    )
    default_leverage: int = Field(
        default=5,
        validation_alias=AliasChoices("DEFAULT_LEVERAGE", "default_leverage"),
        ge=1,
        le=100,
        description="Default leverage tier",
    )
    default_risk_pct: float = Field(
        default=1.0,
        validation_alias=AliasChoices("DEFAULT_RISK_PCT", "default_risk_pct"),
        ge=0.1,
        le=10.0,
        description="Account equity percentage at risk per breakout trade",
    )
    status_interval_sec: float = Field(
        default=60.0,
        validation_alias=AliasChoices("STATUS_INTERVAL_SEC", "status_interval_sec"),
        ge=5.0,
        le=600.0,
        description="Terminal heartbeat telemetry interval in seconds",
    )
    breakout_entry_buffer: float = Field(
        default=1.0,
        validation_alias=AliasChoices("BREAKOUT_ENTRY_BUFFER", "breakout_entry_buffer"),
        ge=0.0,
        le=50.0,
        description="Entry price buffer in tick size increments",
    )
    min_adx_threshold: float = Field(
        default=20.0,
        validation_alias=AliasChoices("MIN_ADX_THRESHOLD", "min_adx_threshold"),
        ge=5.0,
        le=60.0,
        description="Minimum ADX threshold for trend strength filter",
    )
    max_drawdown_pct: float = Field(
        default=5.0,
        validation_alias=AliasChoices("MAX_DRAWDOWN_PCT", "max_drawdown_pct"),
        ge=0.5,
        le=50.0,
        description="Circuit breaker daily maximum account drawdown percentage",
    )
    trailing_atr_multiplier: float = Field(
        default=2.0,
        validation_alias=AliasChoices("TRAILING_ATR_MULTIPLIER", "trailing_atr_multiplier"),
        ge=0.5,
        le=10.0,
        description="ATR multiplier for dynamic trailing stop ratchet",
    )
    order_ttl_bars: int = Field(
        default=3,
        validation_alias=AliasChoices("ORDER_TTL_BARS", "order_ttl_bars"),
        ge=1,
        le=20,
        description="Candle bar TTL before un-triggered breakout order invalidation",
    )

    # Dashboard & Web Control API Authentication (Phase 8)
    dashboard_auth_token: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("DASHBOARD_AUTH_TOKEN", "dashboard_auth_token"),
        description="Optional shared secret token protecting Dashboard control endpoints and WebSocket",
    )

    # Supabase Multi-Tenant Auth & Storage Settings
    supabase_url: str = Field(
        default="https://clkkzwgpumbnbtvsxkmw.supabase.co",
        validation_alias=AliasChoices("SUPABASE_URL", "supabase_url"),
        description="Supabase Project API URL",
    )
    supabase_key: str = Field(
        default="sb_publishable_rvcmboPnre20bT51oz82qQ_q3SvnIZY",
        validation_alias=AliasChoices("SUPABASE_KEY", "supabase_key", "SUPABASE_ANON_KEY"),
        description="Supabase Publishable / Anon Client Key",
    )
    supabase_secret_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("SUPABASE_SECRET_KEY", "supabase_secret_key", "SUPABASE_SERVICE_ROLE_KEY"),
        description="Supabase Secret / Admin Key",
    )
    supabase_db_url: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("SUPABASE_DB_URL", "supabase_db_url"),
        description="Supabase PostgreSQL Direct / Pooler Database URL",
    )

    @property
    def target_symbols(self) -> list[str]:
        """Parsed list of target contract symbols."""
        return [s.strip().upper() for s in self.target_symbols_str.split(",") if s.strip()]

    @field_validator("delta_base_url", mode="before")
    @classmethod
    def normalize_base_url(cls, v: Any) -> str:
        """Strips whitespace, removes trailing slashes, and verifies scheme."""
        if not isinstance(v, str):
            raise ValueError(f"Invalid base URL type: {type(v).__name__}")
        cleaned = v.strip().rstrip("/")
        parsed = urlparse(cleaned)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid URL scheme '{parsed.scheme}' in base URL: {cleaned}")
        if not parsed.netloc:
            raise ValueError(f"Missing host netloc in base URL: {cleaned}")
        return cleaned

    @field_validator("delta_ws_url", mode="before")
    @classmethod
    def normalize_ws_url(cls, v: Any) -> str:
        """Strips whitespace, removes trailing slashes, and verifies WebSocket scheme."""
        if not isinstance(v, str):
            raise ValueError(f"Invalid WebSocket URL type: {type(v).__name__}")
        cleaned = v.strip().rstrip("/")
        parsed = urlparse(cleaned)
        if parsed.scheme not in ("ws", "wss"):
            raise ValueError(f"Invalid WebSocket scheme '{parsed.scheme}' in URL: {cleaned}")
        if not parsed.netloc:
            raise ValueError(f"Missing host netloc in WebSocket URL: {cleaned}")
        return cleaned

    @property
    def is_demo_mode(self) -> bool:
        """Determines if the bot is currently in DEMO mode."""
        return self.execution_mode.strip().lower() in ("demo", "testnet", "paper")

    @property
    def active_base_url(self) -> str:
        """Returns the active Delta Exchange REST base URL (Demo Testnet or Real India)."""
        return self.delta_demo_base_url if self.is_demo_mode else self.delta_base_url

    @property
    def active_ws_url(self) -> str:
        """Returns the active Delta Exchange WebSocket URL (Demo Testnet or Real India)."""
        return self.delta_demo_ws_url if self.is_demo_mode else self.delta_ws_url

    @property
    def api_key_str(self) -> str:
        """Retrieves active raw unmasked API key for HMAC signing."""
        if self.is_demo_mode:
            k = self.delta_demo_api_key.get_secret_value().strip()
            return k if k else self.delta_api_key.get_secret_value()
        return self.delta_api_key.get_secret_value()

    @property
    def api_secret_str(self) -> str:
        """Retrieves active raw unmasked API secret for HMAC signing."""
        if self.is_demo_mode:
            s = self.delta_demo_api_secret.get_secret_value().strip()
            return s if s else self.delta_api_secret.get_secret_value()
        return self.delta_api_secret.get_secret_value()

    @property
    def dashboard_auth_token_str(self) -> str | None:
        """Retrieves raw unmasked dashboard token if configured and non-empty."""
        if self.dashboard_auth_token is None:
            return None
        val = self.dashboard_auth_token.get_secret_value().strip()
        return val if val else None

    @property
    def supabase_secret_key_str(self) -> str:
        """Retrieves raw unmasked Supabase Secret Key."""
        return self.supabase_secret_key.get_secret_value()

    @property
    def supabase_db_url_str(self) -> str:
        """Retrieves raw unmasked Supabase Database connection string."""
        return self.supabase_db_url.get_secret_value()

    def has_valid_credentials(self) -> bool:
        """Determines if credentials have been populated with non-dummy values."""
        key = self.api_key_str.strip()
        secret = self.api_secret_str.strip()
        dummy_markers = ("your_api_key", "dummy", "placeholder")
        if not key or not secret:
            return False
        for marker in dummy_markers:
            if marker in key.lower() or marker in secret.lower():
                return False
        return True

    def safe_summary(self) -> dict[str, Any]:
        """Provides a sanitized dictionary view suitable for structured logging."""
        raw_key = self.api_key_str
        masked_key = (
            f"{raw_key[:4]}...{raw_key[-4:]}" if len(raw_key) >= 10 else ("********" if raw_key else "<not set>")
        )
        return {
            "delta_base_url": self.delta_base_url,
            "delta_ws_url": self.delta_ws_url,
            "delta_api_key": masked_key,
            "delta_api_secret": "********" if self.api_secret_str else "<not set>",
            "request_timeout_sec": self.request_timeout_sec,
            "ws_heartbeat_interval_sec": self.ws_heartbeat_interval_sec,
            "bar_cache_capacity": self.bar_cache_capacity,
            "max_retries": self.max_retries,
            "backoff_base_sec": self.backoff_base_sec,
            "backoff_max_sec": self.backoff_max_sec,
            "user_agent": self.user_agent,
            "log_level": self.log_level,
            "dashboard_auth_token": "********" if self.dashboard_auth_token_str else "<not set>",
            "supabase_url": self.supabase_url,
            "supabase_key": f"{self.supabase_key[:8]}..." if self.supabase_key else "<not set>",
        }

    def __repr__(self) -> str:
        summary = self.safe_summary()
        items = [f"{k}={v!r}" for k, v in summary.items()]
        return f"BotConfig({', '.join(items)})"

    def __str__(self) -> str:
        return self.__repr__()


def get_config(**overrides: Any) -> BotConfig:
    """Helper function to obtain a configured BotConfig instance with optional overrides."""
    return BotConfig(**overrides)

