"""
delta_bot.auth
~~~~~~~~~~~~~~

Delta Exchange HMAC-SHA256 request authentication and clock synchronization engine.
Strictly generates canonical signature strings compliant with Delta's 5-second replay window:
    METHOD + timestamp + path + query_string + payload
"""

import hashlib
import hmac
import json
import logging
import time
from typing import Any
from urllib.parse import urlencode

from pydantic import SecretStr

from delta_bot.exceptions import AuthenticationError

logger = logging.getLogger("delta_bot.auth")


class DeltaAuthEngine:
    """
    Cryptographic authentication engine for Delta Exchange.
    Calculates HMAC-SHA256 digests and maintains real-time clock offset synchronization.
    """

    def __init__(
        self,
        api_key: SecretStr | str,
        api_secret: SecretStr | str,
        user_agent: str = "DeltaBot-QuantEngine/1.0.0",
        initial_clock_offset_s: float = 0.0,
    ) -> None:
        self._api_key: SecretStr = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
        self._api_secret: SecretStr = api_secret if isinstance(api_secret, SecretStr) else SecretStr(api_secret)
        self._user_agent: str = user_agent
        self._clock_offset_s: float = float(initial_clock_offset_s)

    @property
    def clock_offset_s(self) -> float:
        """Current estimated delta between exchange server time and host local time in seconds."""
        return self._clock_offset_s

    def update_credentials(self, api_key: SecretStr | str, api_secret: SecretStr | str) -> None:
        """Updates API key and secret for subsequent signature generation."""
        self._api_key = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
        self._api_secret = api_secret if isinstance(api_secret, SecretStr) else SecretStr(api_secret)

    def update_clock_offset(self, server_epoch_s: float, local_epoch_s: float) -> float:
        """
        Updates clock offset estimation using round-trip synchronized server telemetry.
        Positive offset means host clock is behind exchange server.
        """
        offset = server_epoch_s - local_epoch_s
        # Apply smoothing / update
        old_offset = self._clock_offset_s
        self._clock_offset_s = offset
        logger.debug(
            "Clock offset updated: %.4fs (previous: %.4fs, shift: %.4fs)",
            offset,
            old_offset,
            offset - old_offset,
        )
        return self._clock_offset_s

    def get_synced_timestamp(self) -> str:
        """
        Generates current UTC epoch time in seconds as a string, compensated by clock offset.
        Guarantees request falls within Delta's strict 5-second acceptance window.
        """
        synced_epoch = time.time() + self._clock_offset_s
        return str(int(synced_epoch))

    def _normalize_query_string(self, query: str | dict[str, Any] | None) -> str:
        """
        Formats query string strictly for Delta signature requirements.
        If parameters exist, must start with '?' (e.g. '?contract_types=perpetual_futures').
        If empty, returns empty string ''.
        """
        if not query:
            return ""

        if isinstance(query, dict):
            # Sort keys for deterministic hashing
            encoded = urlencode(sorted([(k, v) for k, v in query.items() if v is not None]))
            return f"?{encoded}" if encoded else ""

        query_str = str(query).strip()
        if not query_str:
            return ""
        if not query_str.startswith("?"):
            return f"?{query_str}"
        return query_str

    def _normalize_payload(self, payload: str | dict[str, Any] | list[Any] | None) -> str:
        """
        Normalizes body payload to compact JSON or empty string.
        Delta requires no spaces after separators: (',', ':').
        """
        if payload is None or payload == "":
            return ""

        if isinstance(payload, (dict, list)):
            return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

        return str(payload)

    def create_signature(
        self,
        method: str,
        path: str,
        query: str | dict[str, Any] | None = None,
        payload: str | dict[str, Any] | list[Any] | None = None,
        timestamp: str | None = None,
    ) -> tuple[str, str]:
        """
        Computes HMAC-SHA256 hex digest according to Delta specification:
            signature_string = METHOD + timestamp + path + query_string + payload
        Returns tuple of (signature_hex, timestamp_used).
        """
        api_secret_raw = self._api_secret.get_secret_value()
        if not api_secret_raw:
            raise AuthenticationError(
                message="Cannot sign request: DELTA_API_SECRET is empty or unconfigured",
                status_code=401,
                error_code="missing_credentials",
            )

        ts = timestamp if timestamp is not None else self.get_synced_timestamp()
        normalized_method = method.strip().upper()
        normalized_path = "/" + path.strip().lstrip("/")
        normalized_query = self._normalize_query_string(query)
        normalized_payload = self._normalize_payload(payload)

        signature_data = f"{normalized_method}{ts}{normalized_path}{normalized_query}{normalized_payload}"

        digest = hmac.new(
            key=api_secret_raw.encode("utf-8"),
            msg=signature_data.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()

        logger.debug(
            "Constructed HMAC signature for [%s %s%s] with timestamp %s",
            normalized_method,
            normalized_path,
            normalized_query,
            ts,
        )
        return digest, ts

    def build_headers(
        self,
        method: str,
        path: str,
        query: str | dict[str, Any] | None = None,
        payload: str | dict[str, Any] | list[Any] | None = None,
        is_authenticated: bool = True,
    ) -> dict[str, str]:
        """
        Assembles all mandatory HTTP request headers for Delta Exchange.
        If is_authenticated is True:
            - 'api-key': DELTA_API_KEY
            - 'timestamp': str(timestamp)
            - 'signature': HMAC-SHA256 hex digest
            - 'Content-Type': 'application/json'
            - 'User-Agent': custom user-agent string
        """
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self._user_agent,
        }

        if not is_authenticated:
            return headers

        api_key_raw = self._api_key.get_secret_value()
        if not api_key_raw:
            raise AuthenticationError(
                message="Authenticated endpoint requires DELTA_API_KEY",
                status_code=401,
                error_code="missing_api_key",
            )

        signature, ts = self.create_signature(
            method=method,
            path=path,
            query=query,
            payload=payload,
        )

        headers["api-key"] = api_key_raw
        headers["timestamp"] = ts
        headers["signature"] = signature
        return headers

    def create_websocket_auth_payload(self, timestamp: str | None = None) -> dict[str, Any]:
        """
        Builds Delta Exchange private WebSocket authentication frame.
        Specification:
            Message = "GET" + timestamp + "/live"
            HMAC-SHA256 signature generated with API secret
        Returns:
            {
                "type": "auth",
                "payload": {
                    "api-key": API_KEY,
                    "signature": HEX_DIGEST,
                    "timestamp": str(TS)
                }
            }
        """
        api_key_raw = self._api_key.get_secret_value()
        if not api_key_raw:
            raise AuthenticationError(
                message="WebSocket authentication requires DELTA_API_KEY",
                status_code=401,
                error_code="missing_credentials",
            )
        signature, ts = self.create_signature(
            method="GET",
            path="/live",
            query="",
            payload="",
            timestamp=timestamp,
        )
        return {
            "type": "auth",
            "payload": {
                "api-key": api_key_raw,
                "signature": signature,
                "timestamp": str(ts),
            },
        }

    def __repr__(self) -> str:
        key_raw = self._api_key.get_secret_value()
        masked_key = f"{key_raw[:4]}...{key_raw[-4:]}" if len(key_raw) >= 10 else ("********" if key_raw else "<unset>")
        return (
            f"DeltaAuthEngine(api_key={masked_key!r}, "
            f"clock_offset_s={self._clock_offset_s:.4f}, user_agent={self._user_agent!r})"
        )


# Canonical alias
DeltaAuth = DeltaAuthEngine
