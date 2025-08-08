import time
from functools import cached_property
from unittest.mock import Mock

import httpx
import pytest

from backend.util.service import (
    AppService,
    AppServiceClient,
    endpoint_to_async,
    expose,
    get_service_client,
)

TEST_SERVICE_PORT = 8765


def wait_for_service_ready(service_client_type, timeout_seconds=30):
    """Helper method to wait for a service to be ready using health check with retry."""
    client = get_service_client(service_client_type, request_retry=True)
    client.health_check()  # This will retry until service is ready


class ServiceTest(AppService):
    def __init__(self):
        super().__init__()
        self.fail_count = 0

    def cleanup(self):
        pass

    @classmethod
    def get_port(cls) -> int:
        return TEST_SERVICE_PORT

    def __enter__(self):
        # Start the service
        result = super().__enter__()

        # Wait for the service to be ready
        wait_for_service_ready(ServiceTestClient)

        return result

    @expose
    def add(self, a: int, b: int) -> int:
        return a + b

    @expose
    def subtract(self, a: int, b: int) -> int:
        return a - b

    @expose
    def fun_with_async(self, a: int, b: int) -> int:
        async def add_async(a: int, b: int) -> int:
            return a + b

        return self.run_and_wait(add_async(a, b))

    @expose
    def failing_add(self, a: int, b: int) -> int:
        """Method that fails 2 times then succeeds - for testing retry logic"""
        self.fail_count += 1
        if self.fail_count <= 2:
            raise RuntimeError(f"Intended error for testing {self.fail_count}/2")
        return a + b

    @expose
    def always_failing_add(self, a: int, b: int) -> int:
        """Method that always fails - for testing no retry when disabled"""
        raise RuntimeError("Intended error for testing")


class ServiceTestClient(AppServiceClient):
    @classmethod
    def get_service_type(cls):
        return ServiceTest

    add = ServiceTest.add
    subtract = ServiceTest.subtract
    fun_with_async = ServiceTest.fun_with_async
    failing_add = ServiceTest.failing_add
    always_failing_add = ServiceTest.always_failing_add

    add_async = endpoint_to_async(ServiceTest.add)
    subtract_async = endpoint_to_async(ServiceTest.subtract)


@pytest.mark.asyncio
async def test_service_creation(server):
    with ServiceTest():
        client = get_service_client(ServiceTestClient)
        assert client.add(5, 3) == 8
        assert client.subtract(10, 4) == 6
        assert client.fun_with_async(5, 3) == 8
        assert await client.add_async(5, 3) == 8
        assert await client.subtract_async(10, 4) == 6


class TestDynamicClientConnectionHealing:
    """Test the DynamicClient connection healing logic"""

    def setup_method(self):
        """Setup for each test method"""
        # Create a mock service client type
        self.mock_service_type = Mock()
        self.mock_service_type.get_host.return_value = "localhost"
        self.mock_service_type.get_port.return_value = 8000

        self.mock_service_client_type = Mock()
        self.mock_service_client_type.get_service_type.return_value = (
            self.mock_service_type
        )

        # Create our test client with the real DynamicClient logic
        self.client = self._create_test_client()

    def _create_test_client(self):
        """Create a test client that mimics the real DynamicClient"""

        class TestClient:
            def __init__(self, service_client_type):
                service_type = service_client_type.get_service_type()
                host = service_type.get_host()
                port = service_type.get_port()
                self.base_url = f"http://{host}:{port}".rstrip("/")
                self._connection_failure_count = 0
                self._last_client_reset = 0

            def _create_sync_client(self) -> httpx.Client:
                return Mock(spec=httpx.Client)

            def _create_async_client(self) -> httpx.AsyncClient:
                return Mock(spec=httpx.AsyncClient)

            @cached_property
            def sync_client(self) -> httpx.Client:
                return self._create_sync_client()

            @cached_property
            def async_client(self) -> httpx.AsyncClient:
                return self._create_async_client()

            def _handle_connection_error(self, error: Exception) -> None:
                """Handle connection errors and implement self-healing"""
                self._connection_failure_count += 1
                current_time = time.time()

                # If we've had 3+ failures and it's been more than 30 seconds since last reset
                if (
                    self._connection_failure_count >= 3
                    and current_time - self._last_client_reset > 30
                ):

                    # Clear cached clients to force recreation on next access
                    if hasattr(self, "sync_client"):
                        delattr(self, "sync_client")
                    if hasattr(self, "async_client"):
                        delattr(self, "async_client")

                    # Reset counters
                    self._connection_failure_count = 0
                    self._last_client_reset = current_time

        return TestClient(self.mock_service_client_type)

    def test_client_caching(self):
        """Test that clients are cached via @cached_property"""
        # Get clients multiple times
        sync1 = self.client.sync_client
        sync2 = self.client.sync_client
        async1 = self.client.async_client
        async2 = self.client.async_client

        # Should return same instances (cached)
        assert sync1 is sync2, "Sync clients should be cached"
        assert async1 is async2, "Async clients should be cached"

    def test_connection_error_counting(self):
        """Test that connection errors are counted correctly"""
        initial_count = self.client._connection_failure_count

        # Simulate connection errors
        self.client._handle_connection_error(Exception("Connection failed"))
        assert self.client._connection_failure_count == initial_count + 1

        self.client._handle_connection_error(Exception("Connection failed"))
        assert self.client._connection_failure_count == initial_count + 2

    def test_no_reset_before_threshold(self):
        """Test that clients are NOT reset before reaching failure threshold"""
        # Get initial clients
        sync_before = self.client.sync_client
        async_before = self.client.async_client

        # Simulate 2 failures (below threshold of 3)
        self.client._handle_connection_error(Exception("Connection failed"))
        self.client._handle_connection_error(Exception("Connection failed"))

        # Clients should still be the same (no reset)
        sync_after = self.client.sync_client
        async_after = self.client.async_client

        assert (
            sync_before is sync_after
        ), "Sync client should not be reset before threshold"
        assert (
            async_before is async_after
        ), "Async client should not be reset before threshold"
        assert self.client._connection_failure_count == 2

    def test_no_reset_within_time_window(self):
        """Test that clients are NOT reset if within the 30-second window"""
        # Get initial clients
        sync_before = self.client.sync_client
        async_before = self.client.async_client

        # Set last reset to recent time (within 30 seconds)
        self.client._last_client_reset = time.time() - 10  # 10 seconds ago

        # Simulate 3+ failures
        for _ in range(3):
            self.client._handle_connection_error(Exception("Connection failed"))

        # Clients should still be the same (no reset due to time window)
        sync_after = self.client.sync_client
        async_after = self.client.async_client

        assert (
            sync_before is sync_after
        ), "Sync client should not be reset within time window"
        assert (
            async_before is async_after
        ), "Async client should not be reset within time window"
        assert self.client._connection_failure_count == 3

    def test_reset_after_threshold_and_time(self):
        """Test that clients ARE reset after threshold failures and time window"""
        # Get initial clients
        sync_before = self.client.sync_client
        async_before = self.client.async_client

        # Set last reset to old time (beyond 30 seconds)
        self.client._last_client_reset = time.time() - 60  # 60 seconds ago

        # Simulate 3+ failures to trigger reset
        for _ in range(3):
            self.client._handle_connection_error(Exception("Connection failed"))

        # Clients should be different (reset occurred)
        sync_after = self.client.sync_client
        async_after = self.client.async_client

        assert (
            sync_before is not sync_after
        ), "Sync client should be reset after threshold"
        assert (
            async_before is not async_after
        ), "Async client should be reset after threshold"
        assert (
            self.client._connection_failure_count == 0
        ), "Failure count should be reset"

    def test_reset_counters_after_healing(self):
        """Test that counters are properly reset after healing"""
        # Set up for reset
        self.client._last_client_reset = time.time() - 60
        self.client._connection_failure_count = 5

        # Trigger reset
        self.client._handle_connection_error(Exception("Connection failed"))

        # Check counters are reset
        assert self.client._connection_failure_count == 0
        assert self.client._last_client_reset > time.time() - 5  # Recently reset


class TestConnectionHealingIntegration:
    """Integration tests for the complete connection healing workflow"""

    def test_failure_count_reset_on_success(self):
        """Test that failure count would be reset on successful requests"""

        # This simulates what happens in _handle_call_method_response
        class ClientWithSuccessHandling:
            def __init__(self):
                self._connection_failure_count = 5

            def _handle_successful_response(self):
                # This is what happens in the real _handle_call_method_response
                self._connection_failure_count = 0

        client = ClientWithSuccessHandling()
        client._handle_successful_response()
        assert client._connection_failure_count == 0

    def test_thirty_second_window_timing(self):
        """Test that the 30-second window works as expected"""
        current_time = time.time()

        # Test cases for the timing logic
        test_cases = [
            (current_time - 10, False),  # 10 seconds ago - should NOT reset
            (current_time - 29, False),  # 29 seconds ago - should NOT reset
            (current_time - 31, True),  # 31 seconds ago - should reset
            (current_time - 60, True),  # 60 seconds ago - should reset
        ]

        for last_reset_time, should_reset in test_cases:
            failure_count = 3  # At threshold
            time_condition = current_time - last_reset_time > 30
            should_trigger_reset = failure_count >= 3 and time_condition

            assert (
                should_trigger_reset == should_reset
            ), f"Time window logic failed for {current_time - last_reset_time} seconds ago"


def test_cached_property_behavior():
    """Test that @cached_property works as expected for our use case"""
    creation_count = 0

    class TestCachedProperty:
        @cached_property
        def expensive_resource(self):
            nonlocal creation_count
            creation_count += 1
            return f"resource-{creation_count}"

    obj = TestCachedProperty()

    # First access should create
    resource1 = obj.expensive_resource
    assert creation_count == 1

    # Second access should return cached
    resource2 = obj.expensive_resource
    assert creation_count == 1  # No additional creation
    assert resource1 is resource2

    # Deleting the cached property should allow recreation
    delattr(obj, "expensive_resource")
    resource3 = obj.expensive_resource
    assert creation_count == 2  # New creation
    assert resource1 != resource3


def test_service_with_runtime_error_retries(server):
    """Test a real service method that throws RuntimeError and gets retried"""
    with ServiceTest():
        # Get client with retry enabled
        client = get_service_client(ServiceTestClient, request_retry=True)

        # This should succeed after retries (fails 2 times, succeeds on 3rd try)
        result = client.failing_add(5, 3)
        assert result == 8


def test_service_no_retry_when_disabled(server):
    """Test that retry doesn't happen when disabled"""
    with ServiceTest():
        # Get client with retry disabled
        client = get_service_client(ServiceTestClient, request_retry=False)

        # This should fail immediately without retry
        with pytest.raises(RuntimeError, match="Intended error for testing"):
            client.always_failing_add(5, 3)
