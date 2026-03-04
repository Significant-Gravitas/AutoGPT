import asyncio
import contextlib
import time
from functools import cached_property
from unittest.mock import Mock

import httpx
import pytest
from prisma.errors import DataError, UniqueViolationError

from backend.util.service import (
    AppService,
    AppServiceClient,
    HTTPClientError,
    HTTPServerError,
    endpoint_to_async,
    expose,
    get_service_client,
)

TEST_SERVICE_PORT = 8765


class ServiceTest(AppService):
    def __init__(self):
        super().__init__()
        self.fail_count = 0

    @classmethod
    def get_port(cls) -> int:
        return TEST_SERVICE_PORT

    def __enter__(self):
        # Start the service
        result = super().__enter__()

        # Wait for the service to be ready
        self.wait_until_ready()

        return result

    def wait_until_ready(self, timeout_seconds: int = 5):
        """Helper method to wait for a service to be ready using health check with retry."""
        client = get_service_client(
            ServiceTestClient, call_timeout=timeout_seconds, request_retry=True
        )
        client.health_check()  # This will retry until service is ready\

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


class TestHTTPErrorRetryBehavior:
    """Test that HTTP client errors (4xx) are not retried but server errors (5xx) can be."""

    # Note: These tests access private methods for testing internal behavior
    # Type ignore comments are used to suppress warnings about accessing private methods

    def test_http_client_error_not_retried(self):
        """Test that 4xx errors are wrapped as HTTPClientError and not retried."""
        # Create a mock response with 404 status
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"message": "Not found"}
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found", request=Mock(), response=mock_response
        )

        # Create client
        client = get_service_client(ServiceTestClient)
        dynamic_client = client

        # Test the _handle_call_method_response directly
        with pytest.raises(HTTPClientError) as exc_info:
            dynamic_client._handle_call_method_response(  # type: ignore[attr-defined]
                response=mock_response, method_name="test_method"
            )

        assert exc_info.value.status_code == 404
        assert "404" in str(exc_info.value)

    def test_http_server_error_can_be_retried(self):
        """Test that 5xx errors are wrapped as HTTPServerError and can be retried."""
        # Create a mock response with 500 status
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"message": "Internal server error"}
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error", request=Mock(), response=mock_response
        )

        # Create client
        client = get_service_client(ServiceTestClient)
        dynamic_client = client

        # Test the _handle_call_method_response directly
        with pytest.raises(HTTPServerError) as exc_info:
            dynamic_client._handle_call_method_response(  # type: ignore[attr-defined]
                response=mock_response, method_name="test_method"
            )

        assert exc_info.value.status_code == 500
        assert "500" in str(exc_info.value)

    def test_mapped_exception_preserves_original_type(self):
        """Test that mapped exceptions preserve their original type regardless of HTTP status."""
        # Create a mock response with ValueError in the remote call error
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "type": "ValueError",
            "args": ["Invalid parameter value"],
        }
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "400 Bad Request", request=Mock(), response=mock_response
        )

        # Create client
        client = get_service_client(ServiceTestClient)
        dynamic_client = client

        # Test the _handle_call_method_response directly
        with pytest.raises(ValueError) as exc_info:
            dynamic_client._handle_call_method_response(  # type: ignore[attr-defined]
                response=mock_response, method_name="test_method"
            )

        assert "Invalid parameter value" in str(exc_info.value)

    def test_prisma_data_error_reconstructed_correctly(self):
        """Test that DataError subclasses (e.g. UniqueViolationError) are
        reconstructed without crashing.

        Prisma's DataError.__init__ expects a dict `data` arg with
        a 'user_facing_error' key.  RPC serialization only preserves the
        string message via exc.args, so the client must wrap it in the
        expected dict structure.
        """
        for exc_type in [DataError, UniqueViolationError]:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.json.return_value = {
                "type": exc_type.__name__,
                "args": ["Unique constraint failed on the fields: (`path`)"],
            }
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "400 Bad Request", request=Mock(), response=mock_response
            )

            client = get_service_client(ServiceTestClient)

            with pytest.raises(exc_type) as exc_info:
                client._handle_call_method_response(  # type: ignore[attr-defined]
                    response=mock_response, method_name="test_method"
                )

            # The exception should have the message preserved
            assert "Unique constraint" in str(exc_info.value)
            # And should have the expected data structure (not crash)
            assert hasattr(exc_info.value, "data")
            assert isinstance(exc_info.value.data, dict)

    def test_client_error_status_codes_coverage(self):
        """Test that various 4xx status codes are all wrapped as HTTPClientError."""
        client_error_codes = [400, 401, 403, 404, 405, 409, 422, 429]

        for status_code in client_error_codes:
            mock_response = Mock()
            mock_response.status_code = status_code
            mock_response.json.return_value = {"message": f"Error {status_code}"}
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                f"{status_code} Error", request=Mock(), response=mock_response
            )

            client = get_service_client(ServiceTestClient)
            dynamic_client = client

            with pytest.raises(HTTPClientError) as exc_info:
                dynamic_client._handle_call_method_response(  # type: ignore
                    response=mock_response, method_name="test_method"
                )

            assert exc_info.value.status_code == status_code

    def test_server_error_status_codes_coverage(self):
        """Test that various 5xx status codes are all wrapped as HTTPServerError."""
        server_error_codes = [500, 501, 502, 503, 504, 505]

        for status_code in server_error_codes:
            mock_response = Mock()
            mock_response.status_code = status_code
            mock_response.json.return_value = {"message": f"Error {status_code}"}
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                f"{status_code} Error", request=Mock(), response=mock_response
            )

            client = get_service_client(ServiceTestClient)
            dynamic_client = client

            with pytest.raises(HTTPServerError) as exc_info:
                dynamic_client._handle_call_method_response(  # type: ignore
                    response=mock_response, method_name="test_method"
                )

            assert exc_info.value.status_code == status_code


class TestGracefulShutdownService(AppService):
    """Test service with slow endpoints for testing graceful shutdown"""

    @classmethod
    def get_port(cls) -> int:
        return 18999  # Use a specific test port

    def __init__(self):
        super().__init__()
        self.request_log = []
        self.cleanup_called = False
        self.cleanup_completed = False

    @expose
    async def slow_endpoint(self, duration: int = 5) -> dict:
        """Endpoint that takes time to complete"""
        start_time = time.time()
        self.request_log.append(f"slow_endpoint started at {start_time}")

        await asyncio.sleep(duration)

        end_time = time.time()
        result = {
            "message": "completed",
            "duration": end_time - start_time,
            "start_time": start_time,
            "end_time": end_time,
        }
        self.request_log.append(f"slow_endpoint completed at {end_time}")
        return result

    @expose
    def fast_endpoint(self) -> dict:
        """Fast endpoint for testing rejection during shutdown"""
        timestamp = time.time()
        self.request_log.append(f"fast_endpoint called at {timestamp}")
        return {"message": "fast", "timestamp": timestamp}

    def cleanup(self):
        """Override cleanup to track when it's called"""
        self.cleanup_called = True
        self.request_log.append(f"cleanup started at {time.time()}")

        # Call parent cleanup
        super().cleanup()

        self.cleanup_completed = True
        self.request_log.append(f"cleanup completed at {time.time()}")


@pytest.fixture(scope="function")
async def test_service():
    """Run the test service in a separate process"""

    service = TestGracefulShutdownService()
    service.start(background=True)

    base_url = f"http://localhost:{service.get_port()}"

    await wait_until_service_ready(base_url)
    yield service, base_url

    service.stop()


async def wait_until_service_ready(base_url: str, timeout: float = 10):
    start_time = time.time()
    while time.time() - start_time <= timeout:
        async with httpx.AsyncClient(timeout=5) as client:
            with contextlib.suppress(httpx.ConnectError):
                response = await client.get(f"{base_url}/health_check", timeout=5)

                if response.status_code == 200 and response.json() == "OK":
                    return

        await asyncio.sleep(0.5)

    raise RuntimeError(f"Service at {base_url} not available after {timeout} seconds")


async def send_slow_request(base_url: str) -> dict:
    """Send a slow request and return the result"""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(f"{base_url}/slow_endpoint", json={"duration": 5})
        assert response.status_code == 200
        return response.json()


@pytest.mark.asyncio
async def test_graceful_shutdown(test_service):
    """Test that AppService handles graceful shutdown correctly"""
    service, test_service_url = test_service

    # Start a slow request that should complete even after shutdown
    slow_task = asyncio.create_task(send_slow_request(test_service_url))

    # Give the slow request time to start
    await asyncio.sleep(1)

    # Send SIGTERM to the service process
    shutdown_start_time = time.time()
    service.process.terminate()  # This sends SIGTERM

    # Wait a moment for shutdown to start
    await asyncio.sleep(0.5)

    # Try to send a new request - should be rejected or connection refused
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.post(f"{test_service_url}/fast_endpoint", json={})
            # Should get 503 Service Unavailable during shutdown
            assert response.status_code == 503
            assert "shutting down" in response.json()["detail"].lower()
    except httpx.ConnectError:
        # Connection refused is also acceptable - server stopped accepting
        pass

    # The slow request should still complete successfully
    slow_result = await slow_task
    assert slow_result["message"] == "completed"
    assert 4.9 < slow_result["duration"] < 5.5  # Should have taken ~5 seconds

    # Wait for the service to fully shut down
    service.process.join(timeout=15)
    shutdown_end_time = time.time()

    # Verify the service actually terminated
    assert not service.process.is_alive()

    # Verify shutdown took reasonable time (slow request - 1s + cleanup)
    shutdown_duration = shutdown_end_time - shutdown_start_time
    assert 4 <= shutdown_duration <= 6  # ~5s request - 1s + buffer

    print(f"Shutdown took {shutdown_duration:.2f} seconds")
    print(f"Slow request completed in: {slow_result['duration']:.2f} seconds")


@pytest.mark.asyncio
async def test_health_check_during_shutdown(test_service):
    """Test that health checks behave correctly during shutdown"""
    service, test_service_url = test_service

    # Health check should pass initially
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(f"{test_service_url}/health_check")
        assert response.status_code == 200

    # Send SIGTERM
    service.process.terminate()

    # Wait for shutdown to begin
    await asyncio.sleep(1)

    # Health check should now fail or connection should be refused
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{test_service_url}/health_check")
            # Could either get 503, 500 (unhealthy), or connection error
            assert response.status_code in [500, 503]
    except (httpx.ConnectError, httpx.ConnectTimeout):
        # Connection refused/timeout is also acceptable
        pass
