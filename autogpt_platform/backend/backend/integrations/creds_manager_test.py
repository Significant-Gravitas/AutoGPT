"""Tests for creds_manager hook system: register, invoke, and CRUD integration."""

import pytest

from backend.integrations.creds_manager import (
    _invoke_creds_changed_hook,
    register_creds_changed_hook,
    unregister_creds_changed_hook,
)


@pytest.fixture(autouse=True)
def _reset_hook():
    """Ensure global hook state is clean before and after every test."""
    unregister_creds_changed_hook()
    yield
    unregister_creds_changed_hook()


class TestRegisterCredsChangedHook:
    def test_register_and_invoke(self):
        calls: list[tuple[str, str]] = []
        register_creds_changed_hook(lambda u, p: calls.append((u, p)))

        _invoke_creds_changed_hook("user-1", "github")
        assert calls == [("user-1", "github")]

    def test_double_register_raises(self):
        register_creds_changed_hook(lambda u, p: None)
        with pytest.raises(RuntimeError, match="already registered"):
            register_creds_changed_hook(lambda u, p: None)

    def test_unregister_then_reregister(self):
        register_creds_changed_hook(lambda u, p: None)
        unregister_creds_changed_hook()
        # Should not raise after unregister.
        register_creds_changed_hook(lambda u, p: None)


class TestInvokeCredsChangedHook:
    def test_noop_when_no_hook_registered(self):
        # Must not raise even when no hook is registered.
        _invoke_creds_changed_hook("user-1", "github")

    def test_hook_exception_is_swallowed(self):
        def bad_hook(user_id: str, provider: str) -> None:
            raise ValueError("boom")

        register_creds_changed_hook(bad_hook)
        # Must not propagate the exception.
        _invoke_creds_changed_hook("user-1", "github")

    def test_hook_receives_correct_args(self):
        calls: list[tuple[str, str]] = []
        register_creds_changed_hook(lambda u, p: calls.append((u, p)))

        _invoke_creds_changed_hook("user-a", "github")
        _invoke_creds_changed_hook("user-b", "slack")

        assert calls == [("user-a", "github"), ("user-b", "slack")]
