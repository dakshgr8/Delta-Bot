"""
delta_bot.exceptions
~~~~~~~~~~~~~~~~~~~

Comprehensive typed exception hierarchy for Delta Exchange algorithmic trading engine.
Designed for quantitative reliability, clear diagnostics, and deterministic error routing.
"""

from typing import Any


class DeltaClientError(Exception):
    """Base exception for all Delta Exchange bot and client errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message: str = message

    def __str__(self) -> str:
        return f"[{self.__class__.__name__}] {self.message}"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} message={self.message!r}>"


class ConfigurationError(DeltaClientError):
    """Raised when environment settings, execution mode, or credentials are misconfigured."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class NetworkTimeoutError(DeltaClientError):
    """Raised when an HTTP or network operation exceeds the configured timeout."""

    def __init__(
        self,
        message: str,
        timeout_sec: float | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.timeout_sec: float | None = timeout_sec
        self.endpoint: str | None = endpoint
        detail = f"Endpoint: {endpoint or 'unknown'} (timeout: {timeout_sec}s) - {message}"
        super().__init__(detail)


class DeltaConnectionError(DeltaClientError):
    """Raised when low-level network connectivity fails (DNS failure, connection dropped)."""

    def __init__(self, message: str, endpoint: str | None = None) -> None:
        self.endpoint: str | None = endpoint
        detail = f"Connection failure on {endpoint or 'unknown'}: {message}"
        super().__init__(detail)


class DeltaAPIError(DeltaClientError):
    """
    Base exception for exchange-returned API error responses.
    Captures status code, exchange error code, and context payload.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_code: str | None = None,
        response_body: dict[str, Any] | str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code: int | None = status_code
        self.error_code: str | None = error_code
        self.response_body: dict[str, Any] | str | None = response_body
        self.headers: dict[str, str] = headers or {}
        parts = [f"Status: {status_code}"]
        if error_code:
            parts.append(f"Code: {error_code}")
        parts.append(message)
        formatted_message = " | ".join(parts)
        super().__init__(formatted_message)


class AuthenticationError(DeltaAPIError):
    """
    Raised when API key authentication, timestamp synchronization, or HMAC signature fails.
    Common causes: invalid key/secret, signature expiration, or clock drift > 5 seconds.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 401,
        error_code: str | None = "authentication_failed",
        response_body: dict[str, Any] | str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            error_code=error_code,
            response_body=response_body,
            headers=headers,
        )


class RateLimitError(DeltaAPIError):
    """
    Raised when exchange rate limit quotas are exceeded (HTTP 429).
    Includes reset timing information from response headers.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 429,
        error_code: str | None = "rate_limit_exceeded",
        retry_after_ms: float | None = None,
        response_body: dict[str, Any] | str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.retry_after_ms: float | None = retry_after_ms
        suffix = f" (Retry-After: {retry_after_ms:.1f}ms)" if retry_after_ms else ""
        super().__init__(
            message=f"{message}{suffix}",
            status_code=status_code,
            error_code=error_code,
            response_body=response_body,
            headers=headers,
        )


class InsufficientMarginError(DeltaAPIError):
    """
    Raised when order placement or position modification fails due to inadequate wallet margin.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 400,
        error_code: str | None = "insufficient_margin",
        response_body: dict[str, Any] | str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            error_code=error_code,
            response_body=response_body,
            headers=headers,
        )


class DeltaServerError(DeltaAPIError):
    """
    Raised when Delta Exchange responds with 5xx server-side transient errors
    (e.g., 500 Internal Error, 502 Bad Gateway, 503 Service Unavailable, 504 Gateway Timeout).
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str | None = "server_error",
        response_body: dict[str, Any] | str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            error_code=error_code,
            response_body=response_body,
            headers=headers,
        )
