"""Tests for the bounded alert rate limiter in retry.py."""

import threading

from backend.util.retry import (
    _ALERT_RATE_LIMITER_MAX_ENTRIES,
    _alert_rate_limiter,
    _rate_limiter_lock,
    should_send_alert,
)


class TestAlertRateLimiterBounded:
    """Verify the rate limiter dict does not grow without bound."""

    def setup_method(self):
        """Clear the global rate limiter before each test."""
        with _rate_limiter_lock:
            _alert_rate_limiter.clear()

    def test_evicts_oldest_when_over_capacity(self):
        # Fill the rate limiter to capacity + extra
        for i in range(_ALERT_RATE_LIMITER_MAX_ENTRIES + 100):
            exc = RuntimeError(f"unique_error_{i}")
            should_send_alert(f"func_{i}", exc, f"ctx_{i}")

        with _rate_limiter_lock:
            assert len(_alert_rate_limiter) <= _ALERT_RATE_LIMITER_MAX_ENTRIES

    def test_recent_entries_survive_eviction(self):
        # Fill past capacity
        for i in range(_ALERT_RATE_LIMITER_MAX_ENTRIES + 50):
            exc = RuntimeError(f"unique_error_{i}")
            should_send_alert(f"func_{i}", exc, f"ctx_{i}")

        # The most recent entry should still be present
        last_key = f"ctx_{_ALERT_RATE_LIMITER_MAX_ENTRIES + 49}:func_{_ALERT_RATE_LIMITER_MAX_ENTRIES + 49}:RuntimeError:unique_error_{_ALERT_RATE_LIMITER_MAX_ENTRIES + 49}"
        with _rate_limiter_lock:
            assert last_key in _alert_rate_limiter

    def test_rate_limits_duplicate_alerts(self):
        exc = RuntimeError("same error")
        # First call should return True (alert sent)
        assert should_send_alert("func", exc, "ctx") is True
        # Immediate second call should return False (rate limited)
        assert should_send_alert("func", exc, "ctx") is False

    def teardown_method(self):
        with _rate_limiter_lock:
            _alert_rate_limiter.clear()
