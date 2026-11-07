"""
delta_bot.http_client
~~~~~~~~~~~~~~~~~~~~~

Resilient asynchronous HTTP client engine built on httpx.
Features:
- Persistent TCP connection pooling & HTTP/2 keep-alive
- Automatic clock drift tracking via Delta's response headers
- Rate limit budget monitoring (X-RATE-LIMIT headers)
- Exponential backoff with full jitter on transient 429/5xx errors
- Strict exception mapping (DeltaAPIError, AuthenticationError, RateLimitError, InsufficientMarginError)
"""

import asyncio
from email.utils import parsedate_to_datetime
import json
import logging
import random
import time
from typing import Any

import httpx

from delta_bot.auth import DeltaAuthEngine
from delta_bot.config import BotConfig
from delta_bot.exceptions import (
    AuthenticationError,
    DeltaAPIError,
    DeltaConnectionError,
    DeltaServerError,
    InsufficientMarginError,
    NetworkTimeoutError,
    RateLimitError,
)
from delta_bot.models import RateLimitState

logger = logging.getLogger("delta_bot.http_client")


class AsyncHttpEngine:
    """
    Low-level async transport layer managing connections, retries, authentication,
    and exchange telemetry.
    """

    def __init__(self, config: BotConfig, auth_engine: DeltaAuthEngine) -> None:
        self._config: BotConfig = config
        self._auth_engine: DeltaAuthEngine = auth_engine
        self._base_url: str = config.delta_base_url
        self._rate_limit_state: RateLimitState = RateLimitState()
        self._client: httpx.AsyncClient | None = None
        self._closed: bool = False

    async def _get_client(self) -> httpx.AsyncClient:
        """Initializes or returns singleton httpx.AsyncClient with persistent pooling."""
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(
                max_keepalive_connections=50,
                max_connections=100,
                keepalive_expiry=30.0,
            )
            timeout = httpx.Timeout(
                timeout=self._config.request_timeout_sec,
                connect=5.0,
            )
            try:
                self._client = httpx.AsyncClient(
                    base_url=self._base_url,
                    limits=limits,
                    timeout=timeout,
                    http2=True,
                    follow_redirects=True,
                )
                logger.debug(
                    "Initialized persistent httpx.AsyncClient (HTTP/2 enabled, pool max=100, base_url=%s)",
                    self._base_url,
                )
            except Exception as ex:
                logger.warning(
                    "HTTP/2 initialization failed (%s); falling back to HTTP/1.1", ex
                )
                self._client = httpx.AsyncClient(
                    base_url=self._base_url,
                    limits=limits,
                    timeout=timeout,
                    http2=False,
                    follow_redirects=True,
                )
        return self._client

    @property
    def rate_limit_state(self) -> RateLimitState:
        """Returns the most recent exchange rate limit status."""
        return self._rate_limit_state

    def _extract_header_ci(self, headers: httpx.Headers, target_key: str) -> str | None:
        """Case-insensitive header retrieval helper."""
        target_lower = target_key.lower()
        for k, v in headers.items():
            if k.lower() == target_lower:
                return v
        return None

    def _update_telemetry(self, response: httpx.Response, reception_epoch_s: float) -> None:
        """
        Parses rate limit headers and server timestamp headers from every response.
        Dynamically corrects clock drift and tracks request budget.
        """
        headers = response.headers

        # 1. Server Time Telemetry (Request-In-Time is epoch microseconds)
        # Guard against CDN-cached responses (e.g. CloudFront age header) where timestamps are stale
        age_str = self._extract_header_ci(headers, "age")
        try:
            age_s = float(age_str) if age_str else 0.0
        except (ValueError, TypeError):
            age_s = 0.0

        if age_s < 1.0:
            req_in_time_str = self._extract_header_ci(headers, "request-in-time")
            if req_in_time_str:
                try:
                    server_epoch_s = float(req_in_time_str) / 1_000_000.0
                    self._auth_engine.update_clock_offset(server_epoch_s, reception_epoch_s)
                except (ValueError, TypeError) as ex:
                    logger.debug("Failed to parse Request-In-Time header %r: %s", req_in_time_str, ex)
            else:
                # Fallback to HTTP Date header
                date_str = self._extract_header_ci(headers, "date")
                if date_str:
                    try:
                        dt = parsedate_to_datetime(date_str)
                        server_epoch_s = dt.timestamp()
                        self._auth_engine.update_clock_offset(server_epoch_s, reception_epoch_s)
                    except Exception as ex:
                        logger.debug("Failed to parse Date header %r: %s", date_str, ex)

        # 2. Rate Limit Telemetry
        limit_val: int | None = None
        rem_val: int | None = None
        reset_ms_val: float | None = None

        limit_str = self._extract_header_ci(headers, "x-rate-limit-limit")
        rem_str = self._extract_header_ci(headers, "x-rate-limit-remaining")
        reset_str = self._extract_header_ci(headers, "x-rate-limit-reset")

        if limit_str and limit_str.isdigit():
            limit_val = int(limit_str)
        if rem_str and rem_str.isdigit():
            rem_val = int(rem_str)
        if reset_str:
            try:
                reset_ms_val = float(reset_str)
            except ValueError:
                reset_ms_val = None

        if limit_val is not None or rem_val is not None or reset_ms_val is not None:
            self._rate_limit_state = RateLimitState(
                limit=limit_val,
                remaining=rem_val,
                reset_ms=reset_ms_val,
                last_updated_epoch=reception_epoch_s,
            )
            if rem_val is not None and rem_val < 50:
                logger.warning(
                    "Exchange rate limit running low: %d remaining (Reset in %.1f ms)",
                    rem_val,
                    reset_ms_val or 0.0,
                )

    def _compute_backoff(self, attempt: int, custom_delay_s: float | None = None) -> float:
        """
        Calculates exponential backoff with full jitter to avoid thundering herd problem.
        """
        if custom_delay_s is not None and custom_delay_s > 0:
            return min(self._config.backoff_max_sec, custom_delay_s)

        base = self._config.backoff_base_sec * (2**attempt)
        jitter = random.uniform(0.5, 1.0)
        delay = min(self._config.backoff_max_sec, base * jitter)
        return max(0.05, delay)

    def _parse_error_payload(self, body_data: Any) -> tuple[str, str | None, dict[str, Any] | None]:
        """Extracts readable error message, error code, and context payload."""
        if not isinstance(body_data, dict):
            return str(body_data), None, None

        error_field = body_data.get("error")
        error_code = body_data.get("code")
        context = body_data.get("context")

        if isinstance(error_field, dict):
            code = error_field.get("code", error_code)
            msg = error_field.get("message") or error_field.get("description") or str(error_field)
            ctx = error_field.get("context", context)
            return msg, code, ctx

        if isinstance(error_field, str):
            return error_field, error_code, context

        if "message" in body_data:
            return str(body_data["message"]), error_code, context

        return "Unknown exchange error", error_code, context

    def _classify_and_raise(self, response: httpx.Response, body_data: Any) -> None:
        """Maps HTTP status codes and error bodies to specific typed exceptions."""
        headers_dict = dict(response.headers)
        status = response.status_code
        msg, code, _ = self._parse_error_payload(body_data)

        # 1. Rate Limit Exceeded (429)
        if status == 429 or code == "rate_limit_exceeded":
            reset_ms = self._rate_limit_state.reset_ms
            raise RateLimitError(
                message=msg or "Rate limit quota exceeded",
                status_code=status,
                error_code=code or "rate_limit_exceeded",
                retry_after_ms=reset_ms,
                response_body=body_data,
                headers=headers_dict,
            )

        # 2. Authentication Failures (401 or auth error codes)
        auth_error_codes = {
            "unauthorized",
            "invalid_api_key",
            "expired_signature",
            "signature_mismatch",
            "SignatureExpired",
            "InvalidApiKey",
            "ip_not_whitelisted_for_api_key",
            "ip_blocked_for_api_key",
        }
        if status == 401 or (code and code in auth_error_codes):
            raise AuthenticationError(
                message=msg or "Authentication failed or signature expired",
                status_code=status,
                error_code=code or "unauthorized",
                response_body=body_data,
                headers=headers_dict,
            )

        # 3. Insufficient Margin
        if code == "insufficient_margin" or "insufficient margin" in msg.lower():
            raise InsufficientMarginError(
                message=msg or "Insufficient available margin for requested operation",
                status_code=status,
                error_code=code or "insufficient_margin",
                response_body=body_data,
                headers=headers_dict,
            )

        # 4. Server Errors (5xx)
        if status >= 500:
            raise DeltaServerError(
                message=msg or f"Exchange internal error (HTTP {status})",
                status_code=status,
                error_code=code or "server_error",
                response_body=body_data,
                headers=headers_dict,
            )

        # 5. Generic API Error
        raise DeltaAPIError(
            message=msg or f"Exchange returned failure (HTTP {status})",
            status_code=status,
            error_code=code,
            response_body=body_data,
            headers=headers_dict,
        )

    async def request(
        self,
        method: str,
        path: str,
        query: str | dict[str, Any] | None = None,
        payload: str | dict[str, Any] | list[Any] | None = None,
        is_authenticated: bool = True,
    ) -> Any:
        """
        Executes an asynchronous HTTP request with automatic signing, clock sync,
        rate limit telemetry, and resilient exponential backoff retry logic.
        """
        client = await self._get_client()
        attempt = 0
        max_retries = self._config.max_retries

        while True:
            attempt += 1
            headers = self._auth_engine.build_headers(
                method=method,
                path=path,
                query=query,
                payload=payload,
                is_authenticated=is_authenticated,
            )

            # Query params formatting
            target_path = "/" + path.strip().lstrip("/")
            request_params: dict[str, Any] | None = None
            if isinstance(query, dict):
                request_params = query
            elif isinstance(query, str) and query:
                # Append query string directly to target path if not empty
                clean_q = query if query.startswith("?") else f"?{query}"
                target_path = f"{target_path}{clean_q}"

            # Payload formatting
            req_content: str | None = None
            if payload is not None and payload != "":
                if isinstance(payload, (dict, list)):
                    req_content = json.dumps(payload, separators=(",", ":"))
                else:
                    req_content = str(payload)

            send_time = time.time()
            try:
                response = await client.request(
                    method=method.upper(),
                    url=target_path,
                    params=request_params,
                    content=req_content,
                    headers=headers,
                )
                receive_time = time.time()
                self._update_telemetry(response, receive_time)

                # Parse JSON body
                try:
                    body_data = response.json()
                except json.JSONDecodeError:
                    body_data = response.text

                # Check if exchange returned an HTTP error or business-logic failure
                is_http_ok = 200 <= response.status_code < 300
                is_exchange_success = (
                    isinstance(body_data, dict) and body_data.get("success", True)
                )

                if is_http_ok and is_exchange_success:
                    # Successful request
                    return body_data.get("result", body_data) if isinstance(body_data, dict) else body_data

                # Classify transient vs permanent error
                status = response.status_code
                is_transient = status in (429, 500, 502, 503, 504)

                if is_transient and attempt <= max_retries:
                    retry_delay: float | None = None
                    if status == 429:
                        # Extract reset delay if available
                        reset_ms = self._rate_limit_state.reset_ms
                        if reset_ms:
                            retry_delay = (reset_ms / 1000.0) + 0.1

                    backoff_delay = self._compute_backoff(attempt, retry_delay)
                    logger.warning(
                        "Transient HTTP %d on [%s %s] (Attempt %d/%d). Retrying in %.2fs...",
                        status,
                        method,
                        path,
                        attempt,
                        max_retries,
                        backoff_delay,
                    )
                    await asyncio.sleep(backoff_delay)
                    continue

                # Permanent failure or retry quota exhausted
                self._classify_and_raise(response, body_data)

            except (httpx.ConnectError, httpx.RemoteProtocolError) as conn_ex:
                if attempt <= max_retries:
                    backoff_delay = self._compute_backoff(attempt)
                    logger.warning(
                        "Network connection error on [%s %s]: %s. Retrying (%d/%d) in %.2fs...",
                        method,
                        path,
                        conn_ex,
                        attempt,
                        max_retries,
                        backoff_delay,
                    )
                    await asyncio.sleep(backoff_delay)
                    continue
                raise DeltaConnectionError(
                    message=f"Persistent connection drop after {attempt} attempts: {conn_ex}",
                    endpoint=path,
                ) from conn_ex

            except (httpx.TimeoutException, TimeoutError) as timeout_ex:
                if attempt <= max_retries:
                    backoff_delay = self._compute_backoff(attempt)
                    logger.warning(
                        "Network timeout on [%s %s]: %s. Retrying (%d/%d) in %.2fs...",
                        method,
                        path,
                        timeout_ex,
                        attempt,
                        max_retries,
                        backoff_delay,
                    )
                    await asyncio.sleep(backoff_delay)
                    continue
                raise NetworkTimeoutError(
                    message=f"Request timed out after {attempt} attempts: {timeout_ex}",
                    timeout_sec=self._config.request_timeout_sec,
                    endpoint=path,
                ) from timeout_ex

    async def close(self) -> None:
        """Closes the underlying httpx connection pool gracefully."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            logger.debug("Closed AsyncHttpEngine connection pool.")
        self._closed = True

    async def __aenter__(self) -> "AsyncHttpEngine":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
