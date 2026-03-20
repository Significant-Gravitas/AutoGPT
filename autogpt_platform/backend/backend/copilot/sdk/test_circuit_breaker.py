"""Tests for the tool call circuit breaker in tool_adapter.py."""

import pytest

from backend.copilot.sdk.tool_adapter import (
    _MAX_CONSECUTIVE_TOOL_FAILURES,
    _check_circuit_breaker,
    _clear_tool_failures,
    _consecutive_tool_failures,
    _record_tool_failure,
)


@pytest.fixture(autouse=True)
def _reset_tracker():
    """Reset the circuit breaker tracker for each test."""
    token = _consecutive_tool_failures.set({})
    yield
    _consecutive_tool_failures.reset(token)


class TestCircuitBreaker:
    def test_no_trip_below_threshold(self):
        """Circuit breaker should not trip before reaching the limit."""
        args = {"file_path": "/tmp/test.txt"}
        for _ in range(_MAX_CONSECUTIVE_TOOL_FAILURES - 1):
            assert _check_circuit_breaker("write_file", args) is None
            _record_tool_failure("write_file", args)
        # Still under the limit
        assert _check_circuit_breaker("write_file", args) is None

    def test_trips_at_threshold(self):
        """Circuit breaker should trip after reaching the failure limit."""
        args = {"file_path": "/tmp/test.txt"}
        for _ in range(_MAX_CONSECUTIVE_TOOL_FAILURES):
            assert _check_circuit_breaker("write_file", args) is None
            _record_tool_failure("write_file", args)
        # Now it should trip
        result = _check_circuit_breaker("write_file", args)
        assert result is not None
        assert "STOP" in result
        assert "write_file" in result

    def test_different_args_tracked_separately(self):
        """Different args should have separate failure counters."""
        args_a = {"file_path": "/tmp/a.txt"}
        args_b = {"file_path": "/tmp/b.txt"}
        for _ in range(_MAX_CONSECUTIVE_TOOL_FAILURES):
            _record_tool_failure("write_file", args_a)
        # args_a should trip
        assert _check_circuit_breaker("write_file", args_a) is not None
        # args_b should NOT trip
        assert _check_circuit_breaker("write_file", args_b) is None

    def test_empty_args_tracked(self):
        """Empty args ({}) — the exact failure pattern from the bug — should be tracked."""
        args = {}
        for _ in range(_MAX_CONSECUTIVE_TOOL_FAILURES):
            _record_tool_failure("write_file", args)
        assert _check_circuit_breaker("write_file", args) is not None

    def test_clear_resets_counter(self):
        """Clearing failures should reset the counter."""
        args = {}
        for _ in range(_MAX_CONSECUTIVE_TOOL_FAILURES):
            _record_tool_failure("write_file", args)
        _clear_tool_failures("write_file")
        assert _check_circuit_breaker("write_file", args) is None

    def test_success_clears_failures(self):
        """A successful call should reset the failure counter."""
        args = {}
        for _ in range(_MAX_CONSECUTIVE_TOOL_FAILURES - 1):
            _record_tool_failure("write_file", args)
        # Success clears failures
        _clear_tool_failures("write_file")
        # Should be able to fail again without tripping
        for _ in range(_MAX_CONSECUTIVE_TOOL_FAILURES - 1):
            _record_tool_failure("write_file", args)
        assert _check_circuit_breaker("write_file", args) is None

    def test_no_tracker_returns_none(self):
        """If tracker is not initialized, circuit breaker should not trip."""
        _consecutive_tool_failures.set(None)  # type: ignore[arg-type]
        _record_tool_failure("write_file", {})  # should not raise
        assert _check_circuit_breaker("write_file", {}) is None
