"""
Delta-Bot: High-Performance Crypto Trading Bot Foundation & REST API Client.
Quantitative infrastructure for Delta Exchange (India, Global, and Testnet).
"""

from delta_bot.auth import DeltaAuth, DeltaAuthEngine
from delta_bot.client import DeltaRestClient
from delta_bot.config import BotConfig, DeltaEnvironment, DeltaWebSocketEnvironment, get_config
from delta_bot.exceptions import (
    AuthenticationError,
    ConfigurationError,
    DeltaAPIError,
    DeltaClientError,
    DeltaConnectionError,
    DeltaServerError,
    InsufficientMarginError,
    NetworkTimeoutError,
    RateLimitError,
)
from delta_bot.execution import (
    ExecutionConfig,
    ExecutionManager,
    LocalRegistry,
    OrderExecutionManager,
    OrderRouter,
)
from delta_bot.http_client import AsyncHttpEngine
from delta_bot.indicators import (
    IndicatorPipeline,
    IndicatorSnapshot,
    calculate_adx,
    calculate_atr,
    calculate_ema,
    calculate_sma,
    calculate_true_range,
)
from delta_bot.models import (
    Candle,
    CircularBarCache,
    ContractSpec,
    ExchangeOrder,
    ExecutionMode,
    GhostPosition,
    LiquidationAlert,
    MarginCheckResult,
    OrderBook,
    OrderBookLevel,
    OrderUpdate,
    PositionInfo,
    RateLimitState,
    ServerTimeInfo,
    SessionAnchorData,
    Ticker,
    UserTrade,
    WalletBalance,
)
from delta_bot.orchestrator import (
    TradingBotOrchestrator,
)
from delta_bot.paper import (
    PaperExchangeSimulator,
    VirtualOrder,
    VirtualPosition,
    VirtualWallet,
)
from delta_bot.private_ws import (
    PrivateWebSocketClient,
)
from delta_bot.risk import (
    CircuitBreakerState,
    RiskConfig,
    RiskManager,
)
from delta_bot.server import (
    RingBufferLogHandler,
    WebSocketManager,
    create_app,
)
from delta_bot.state import (
    TradingStateManager,
)
from delta_bot.tenant import (
    TenantConfig,
    TenantInstance,
    TenantManager,
)
from delta_bot.strategy import (
    BreakoutStrategyEngine,
    OrderIntent,
    OrderStatus,
    SignalType,
    StrategyConfig,
)
from delta_bot.websocket_client import (
    CandleAggregator,
    ChannelMultiplexer,
    ConnectionLifecycleManager,
    DeltaWebSocketClient,
    SequenceTracker,
    SubscriptionDispatcher,
)

__version__ = "8.0.0"
__all__ = [
    # Web Dashboard & Control API (Phase 7)
    "create_app",
    "WebSocketManager",
    "RingBufferLogHandler",
    # Orchestrator & Deployment Daemon (Phase 6)
    "TradingBotOrchestrator",
    "PaperExchangeSimulator",
    "VirtualWallet",
    "VirtualOrder",
    "VirtualPosition",
    # REST Client & HTTP Engine
    "DeltaRestClient",
    "DeltaAuth",
    "DeltaAuthEngine",
    "AsyncHttpEngine",
    # WebSocket Engine (Phase 2 & Phase 5)
    "DeltaWebSocketClient",
    "PrivateWebSocketClient",
    "CandleAggregator",
    "ChannelMultiplexer",
    "ConnectionLifecycleManager",
    "SequenceTracker",
    "SubscriptionDispatcher",
    # Strategy & Signal Engine (Phase 3)
    "BreakoutStrategyEngine",
    "StrategyConfig",
    "OrderIntent",
    "SignalType",
    "OrderStatus",
    "IndicatorPipeline",
    "IndicatorSnapshot",
    "calculate_ema",
    "calculate_sma",
    "calculate_atr",
    "calculate_adx",
    "calculate_true_range",
    # Execution & Risk Management (Phase 4)
    "ExecutionManager",
    "OrderExecutionManager",
    "OrderRouter",
    "LocalRegistry",
    "ExecutionConfig",
    "ExecutionMode",
    "ExchangeOrder",
    "PositionInfo",
    "MarginCheckResult",
    "RiskManager",
    "RiskConfig",
    "CircuitBreakerState",
    # State Reconciliation & Phase 5
    "TradingStateManager",
    "OrderUpdate",
    "UserTrade",
    "GhostPosition",
    "LiquidationAlert",
    # Configuration
    "BotConfig",
    "get_config",
    "DeltaEnvironment",
    "DeltaWebSocketEnvironment",
    # Domain Models
    "Candle",
    "CircularBarCache",
    "ContractSpec",
    "OrderBook",
    "OrderBookLevel",
    "RateLimitState",
    "ServerTimeInfo",
    "SessionAnchorData",
    "Ticker",
    "WalletBalance",
    # Exceptions
    "DeltaClientError",
    "DeltaAPIError",
    "AuthenticationError",
    "ConfigurationError",
    "RateLimitError",
    "InsufficientMarginError",
    "DeltaServerError",
    "NetworkTimeoutError",
    "DeltaConnectionError",
]
