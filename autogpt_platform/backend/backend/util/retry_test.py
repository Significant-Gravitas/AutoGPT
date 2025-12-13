import asyncio
import threading
import time
from unittest.mock import Mock, patch

import pytest

from backend.util.retry import (
    ALERT_RATE_LIMIT_SECONDS,
    _alert_rate_limiter,
    _rate_limiter_lock,
    _send_critical_retry_alert,
    conn_retry,
    create_retry_decorator,
    should_send_alert,
)


def test_conn_retry_sync_function():
    retry_count = 0

    @conn_retry("Test", "Test function", max_retry=2, max_wait=0.1)
    def test_function():
        nonlocal retry_count
        retry_count -= 1
        if retry_count > 0:
            raise ValueError("Test error")
        return "Success"

    retry_count = 2
    res = test_function()
    assert res == "Success"

    retry_count = 100
    with pytest.raises(ValueError) as e:
        test_function()
        assert str(e.value) == "Test error"


@pytest.mark.asyncio
async def test_conn_retry_async_function():
    retry_count = 0

    @conn_retry("Test", "Test function", max_retry=2, max_wait=0.1)
    async def test_function():
        nonlocal retry_count
        await asyncio.sleep(1)
        retry_count -= 1
        if retry_count > 0:
            raise ValueError("Test error")
        return "Success"

    retry_count = 2
    res = await test_function()
    assert res == "Success"

    retry_count = 100
    with pytest.raises(ValueError) as e:
        await test_function()
        assert str(e.value) == "Test error"


class TestRetryRateLimiting:
    """Test the rate limiting functionality for critical retry alerts."""

    def setup_method(self):
        """Reset rate limiter state before each test."""
        with _rate_limiter_lock:
            _alert_rate_limiter.clear()

    def test_should_send_alert_allows_first_occurrence(self):
        """Test that the first occurrence of an error allows alert."""
        exc = ValueError("test error")
        assert should_send_alert("test_func", exc, "test_context") is True

    def test_should_send_alert_rate_limits_duplicate(self):
        """Test that duplicate errors are rate limited."""
        exc = ValueError("test error")

        # First call should be allowed
        assert should_send_alert("test_func", exc, "test_context") is True

        # Second call should be rate limited
        assert should_send_alert("test_func", exc, "test_context") is False

    def test_should_send_alert_allows_different_errors(self):
        """Test that different errors are allowed even if same function."""
        exc1 = ValueError("error 1")
        exc2 = ValueError("error 2")

        # First error should be allowed
        assert should_send_alert("test_func", exc1, "test_context") is True

        # Different error should also be allowed
        assert should_send_alert("test_func", exc2, "test_context") is True

    def test_should_send_alert_allows_different_contexts(self):
        """Test that same error in different contexts is allowed."""
        exc = ValueError("test error")

        # First context should be allowed
        assert should_send_alert("test_func", exc, "context1") is True

        # Different context should also be allowed
        assert should_send_alert("test_func", exc, "context2") is True

    def test_should_send_alert_allows_different_functions(self):
        """Test that same error in different functions is allowed."""
        exc = ValueError("test error")

        # First function should be allowed
        assert should_send_alert("func1", exc, "test_context") is True

        # Different function should also be allowed
        assert should_send_alert("func2", exc, "test_context") is True

    def test_should_send_alert_respects_time_window(self):
        """Test that alerts are allowed again after the rate limit window."""
        exc = ValueError("test error")

        # First call should be allowed
        assert should_send_alert("test_func", exc, "test_context") is True

        # Immediately after should be rate limited
        assert should_send_alert("test_func", exc, "test_context") is False

        # Mock time to simulate passage of rate limit window
        current_time = time.time()
        with patch("backend.util.retry.time.time") as mock_time:
            # Simulate time passing beyond rate limit window
            mock_time.return_value = current_time + ALERT_RATE_LIMIT_SECONDS + 1
            assert should_send_alert("test_func", exc, "test_context") is True

    def test_should_send_alert_thread_safety(self):
        """Test that rate limiting is thread-safe."""
        exc = ValueError("test error")
        results = []

        def check_alert():
            result = should_send_alert("test_func", exc, "test_context")
            results.append(result)

        # Create multiple threads trying to send the same alert
        threads = [threading.Thread(target=check_alert) for _ in range(10)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Only one thread should have been allowed to send the alert
        assert sum(results) == 1
        assert len([r for r in results if r is True]) == 1
        assert len([r for r in results if r is False]) == 9

    @patch("backend.util.clients.get_notification_manager_client")
    def test_send_critical_retry_alert_rate_limiting(self, mock_get_client):
        """Test that _send_critical_retry_alert respects rate limiting."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        exc = ValueError("spend_credits API error")

        # First alert should be sent
        _send_critical_retry_alert("spend_credits", 50, exc, "Service communication")
        assert mock_client.discord_system_alert.call_count == 1

        # Second identical alert should be rate limited (not sent)
        _send_critical_retry_alert("spend_credits", 50, exc, "Service communication")
        assert mock_client.discord_system_alert.call_count == 1  # Still 1, not 2

        # Different error should be allowed
        exc2 = ValueError("different API error")
        _send_critical_retry_alert("spend_credits", 50, exc2, "Service communication")
        assert mock_client.discord_system_alert.call_count == 2

    @patch("backend.util.clients.get_notification_manager_client")
    def test_send_critical_retry_alert_handles_notification_failure(
        self, mock_get_client
    ):
        """Test that notification failures don't break the rate limiter."""
        mock_client = Mock()
        mock_client.discord_system_alert.side_effect = Exception("Notification failed")
        mock_get_client.return_value = mock_client

        exc = ValueError("test error")

        # Should not raise exception even if notification fails
        _send_critical_retry_alert("test_func", 50, exc, "test_context")

        # Rate limiter should still work for subsequent calls
        assert should_send_alert("test_func", exc, "test_context") is False

    def test_error_signature_generation(self):
        """Test that error signatures are generated correctly for rate limiting."""
        # Test with long exception message (should be truncated to 100 chars)
        long_message = "x" * 200
        exc = ValueError(long_message)

        # Should not raise exception and should work normally
        assert should_send_alert("test_func", exc, "test_context") is True
        assert should_send_alert("test_func", exc, "test_context") is False

    def test_real_world_scenario_spend_credits_spam(self):
        """Test the real-world scenario that was causing spam."""
        # Simulate the exact error that was causing issues
        exc = Exception(
            "HTTP 500: Server error '500 Internal Server Error' for url 'http://autogpt-database-manager.prod-agpt.svc.cluster.local:8005/spend_credits'"
        )

        # First 50 attempts reach threshold - should send alert
        with patch(
            "backend.util.clients.get_notification_manager_client"
        ) as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            _send_critical_retry_alert(
                "_call_method_sync", 50, exc, "Service communication"
            )
            assert mock_client.discord_system_alert.call_count == 1

            # Next 950 failures should not send alerts (rate limited)
            for _ in range(950):
                _send_critical_retry_alert(
                    "_call_method_sync", 50, exc, "Service communication"
                )

            # Still only 1 alert sent total
            assert mock_client.discord_system_alert.call_count == 1

    @patch("backend.util.clients.get_notification_manager_client")
    def test_retry_decorator_with_excessive_failures(self, mock_get_client):
        """Test retry decorator behavior when it hits the alert threshold."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        @create_retry_decorator(
            max_attempts=60, max_wait=0.1
        )  # More than EXCESSIVE_RETRY_THRESHOLD, but fast
        def always_failing_function():
            raise ValueError("persistent failure")

        with pytest.raises(ValueError):
            always_failing_function()

        # Should have sent exactly one alert at the threshold
        assert mock_client.discord_system_alert.call_count == 1
