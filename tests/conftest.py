"""
tests/conftest.py
~~~~~~~~~~~~~~~~~
Global pytest configuration and environment isolation.
Ensures test fixtures execute in an isolated environment without
picking up credentials or tokens from local developer .env files.
"""

import os
import pytest


@pytest.fixture(autouse=True)
def isolate_test_environment(monkeypatch):
    """Isolates test environment from local .env developer tokens."""
    monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "")
    monkeypatch.setenv("SUPABASE_URL", "")
    monkeypatch.setenv("SUPABASE_KEY", "")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "")
    monkeypatch.setenv("SUPABASE_DB_URL", "")
