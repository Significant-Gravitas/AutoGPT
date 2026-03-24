"""Tests for the circuit breaker module."""

import time

import pytest

from backend.util.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
)


class TestCircuitBreaker:
    """Core circuit breaker state-machine tests."""

    def test_starts_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=10)
        assert cb.state == CircuitState.CLOSED

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=10)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_opens_at_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=10)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_pre_call_raises_when_open(self):
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=60)
        cb.record_failure()
        cb.record_failure()
        with pytest.raises(CircuitOpenError) as exc_info:
            cb.pre_call()
        assert exc_info.value.key == "test"
        assert exc_info.value.retry_after > 0

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=10)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        # Should be able to tolerate more failures now
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_transitions_to_half_open_after_recovery(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_allows_probe_call(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        # Should not raise
        cb.pre_call()

    def test_half_open_success_closes(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.pre_call()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.pre_call()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerRegistry:
    """Registry returns the same breaker for the same key."""

    def test_returns_same_instance(self):
        reg = CircuitBreakerRegistry()
        b1 = reg.get("openai")
        b2 = reg.get("openai")
        assert b1 is b2

    def test_different_keys_different_instances(self):
        reg = CircuitBreakerRegistry()
        b1 = reg.get("openai")
        b2 = reg.get("anthropic")
        assert b1 is not b2
