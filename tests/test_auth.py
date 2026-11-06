"""Tests for DeltaAuthEngine HMAC signing and clock synchronization."""

import hashlib
import hmac
import time
import pytest
from pydantic import SecretStr

from delta_bot.auth import DeltaAuth, DeltaAuthEngine
from delta_bot.exceptions import AuthenticationError


def test_delta_auth_alias_and_query_formatting() -> None:
    """
    Verifies DeltaAuth alias and canonical query string formatting:
    - Empty query/dict/None must evaluate to '' (not '?' or None)
    - Dict query parameters must be sorted deterministically
    """
    assert DeltaAuth is DeltaAuthEngine
    assert "DeltaAuthEngine" in repr(DeltaAuth)

    auth = DeltaAuth(api_key="k", api_secret="s")
    assert auth._normalize_query_string("") == ""
    assert auth._normalize_query_string(None) == ""
    assert auth._normalize_query_string({}) == ""
    # Deterministic alphabetical ordering
    assert (
        auth._normalize_query_string({"page_size": 30, "contract_types": "perpetual_futures"})
        == "?contract_types=perpetual_futures&page_size=30"
    )


def test_hmac_canonical_format() -> None:
    """
    Verifies that the signature string strictly follows:
        METHOD + timestamp + path + query_string + payload
    and HMAC-SHA256 matches manual digest.
    """
    api_key = "test_key_abc"
    api_secret = "test_secret_xyz"
    auth = DeltaAuthEngine(api_key=api_key, api_secret=api_secret)

    method = "GET"
    timestamp = "1712750400"
    path = "/v2/wallet/balances"
    query = ""
    payload = ""

    sig, ts = auth.create_signature(
        method=method,
        path=path,
        query=query,
        payload=payload,
        timestamp=timestamp,
    )

    expected_canonical = f"{method}{timestamp}{path}{query}{payload}"
    expected_digest = hmac.new(
        api_secret.encode("utf-8"),
        expected_canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    assert sig == expected_digest
    assert ts == timestamp


def test_hmac_with_query_and_payload() -> None:
    """Verifies HMAC generation when query string and POST JSON payload are provided."""
    api_key = "key1"
    api_secret = "secret1"
    auth = DeltaAuthEngine(api_key=api_key, api_secret=api_secret)

    method = "POST"
    timestamp = "1712750500"
    path = "/v2/orders"
    payload = {"product_id": 139, "size": 10, "order_type": "limit_order"}
    query = {"source": "bot"}

    sig, ts = auth.create_signature(
        method=method,
        path=path,
        query=query,
        payload=payload,
        timestamp=timestamp,
    )

    expected_canonical = 'POST1712750500/v2/orders?source=bot{"product_id":139,"size":10,"order_type":"limit_order"}'
    expected_digest = hmac.new(
        api_secret.encode("utf-8"),
        expected_canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    assert sig == expected_digest


def test_clock_offset_sync() -> None:
    """Verifies clock offset tracking and timestamp adjustment."""
    auth = DeltaAuthEngine(api_key="k", api_secret="s")
    assert auth.clock_offset_s == 0.0

    # Simulate server being 3.2 seconds ahead of host
    server_epoch = 1712750403.2
    local_epoch = 1712750400.0
    auth.update_clock_offset(server_epoch, local_epoch)

    assert pytest.approx(auth.clock_offset_s, 0.001) == 3.2

    # Verify get_synced_timestamp() uses offset
    host_time = time.time()
    synced_ts = int(auth.get_synced_timestamp())
    expected_ts = int(host_time + 3.2)
    assert abs(synced_ts - expected_ts) <= 1


def test_missing_credentials_raises_auth_error() -> None:
    """Verifies AuthenticationError is raised when trying to sign with empty secret."""
    auth = DeltaAuthEngine(api_key="", api_secret="")
    with pytest.raises(AuthenticationError, match="DELTA_API_SECRET is empty"):
        auth.create_signature(method="GET", path="/v2/wallet/balances")
