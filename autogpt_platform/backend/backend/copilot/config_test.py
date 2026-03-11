"""Unit tests for ChatConfig."""

import pytest

from .config import ChatConfig

# Env vars that the ChatConfig validators read — must be cleared so they don't
# override the explicit constructor values we pass in each test.
_E2B_ENV_VARS = (
    "CHAT_USE_E2B_SANDBOX",
    "CHAT_E2B_API_KEY",
    "E2B_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_e2b_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _E2B_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestE2BActive:
    """Tests for the e2b_active property — single source of truth for E2B usage."""

    def test_both_enabled_and_key_present_returns_true(self):
        """e2b_active is True when use_e2b_sandbox=True and e2b_api_key is set."""
        cfg = ChatConfig(use_e2b_sandbox=True, e2b_api_key="test-key")
        assert cfg.e2b_active is True

    def test_enabled_but_missing_key_returns_false(self):
        """e2b_active is False when use_e2b_sandbox=True but e2b_api_key is absent."""
        cfg = ChatConfig(use_e2b_sandbox=True, e2b_api_key=None)
        assert cfg.e2b_active is False

    def test_disabled_returns_false(self):
        """e2b_active is False when use_e2b_sandbox=False regardless of key."""
        cfg = ChatConfig(use_e2b_sandbox=False, e2b_api_key="test-key")
        assert cfg.e2b_active is False
