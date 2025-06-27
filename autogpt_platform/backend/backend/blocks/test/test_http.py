"""Comprehensive tests for HTTP block with HostScopedCredentials functionality."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.http import (
    HttpCredentials,
    HttpMethod,
    SendAuthenticatedWebRequestBlock,
)
from backend.data.model import HostScopedCredentials
from backend.util.request import Response


class TestHttpBlockWithHostScopedCredentials:
    """Test suite for HTTP block integration with HostScopedCredentials."""

    @pytest.fixture
    def http_block(self):
        """Create an HTTP block instance."""
        return SendAuthenticatedWebRequestBlock()

    @pytest.fixture
    def mock_response(self):
        """Mock a successful HTTP response."""
        response = MagicMock(spec=Response)
        response.status = 200
        response.headers = {"content-type": "application/json"}
        response.json.return_value = {"success": True, "data": "test"}
        return response

    @pytest.fixture
    def exact_match_credentials(self):
        """Create host-scoped credentials for exact domain matching."""
        return HostScopedCredentials(
            provider="http",
            host="api.example.com",
            headers={
                "Authorization": SecretStr("Bearer exact-match-token"),
                "X-API-Key": SecretStr("api-key-123"),
            },
            title="Exact Match API Credentials",
        )

    @pytest.fixture
    def wildcard_credentials(self):
        """Create host-scoped credentials with wildcard pattern."""
        return HostScopedCredentials(
            provider="http",
            host="*.github.com",
            headers={
                "Authorization": SecretStr("token ghp_wildcard123"),
            },
            title="GitHub Wildcard Credentials",
        )

    @pytest.fixture
    def non_matching_credentials(self):
        """Create credentials that don't match test URLs."""
        return HostScopedCredentials(
            provider="http",
            host="different.api.com",
            headers={
                "Authorization": SecretStr("Bearer non-matching-token"),
            },
            title="Non-matching Credentials",
        )

    @pytest.mark.asyncio
    @patch("backend.blocks.http.Requests")
    async def test_http_block_with_exact_host_match(
        self,
        mock_requests_class,
        http_block,
        exact_match_credentials,
        mock_response,
    ):
        """Test HTTP block with exact host matching credentials."""
        # Setup mocks
        mock_requests = AsyncMock()
        mock_requests.request.return_value = mock_response
        mock_requests_class.return_value = mock_requests

        # Prepare input data
        input_data = SendAuthenticatedWebRequestBlock.Input(
            url="https://api.example.com/data",
            method=HttpMethod.GET,
            headers={"User-Agent": "test-agent"},
            credentials=cast(
                HttpCredentials,
                {
                    "id": exact_match_credentials.id,
                    "provider": "http",
                    "type": "host_scoped",
                    "title": exact_match_credentials.title,
                },
            ),
        )

        # Execute with credentials provided by execution manager
        result = []
        async for output_name, output_data in http_block.run(
            input_data,
            credentials=exact_match_credentials,
            graph_exec_id="test-exec-id",
        ):
            result.append((output_name, output_data))

        # Verify request headers include both credential and user headers
        mock_requests.request.assert_called_once()
        call_args = mock_requests.request.call_args
        expected_headers = {
            "Authorization": "Bearer exact-match-token",
            "X-API-Key": "api-key-123",
            "User-Agent": "test-agent",
        }
        assert call_args.kwargs["headers"] == expected_headers

        # Verify response handling
        assert len(result) == 1
        assert result[0][0] == "response"
        assert result[0][1] == {"success": True, "data": "test"}

    @pytest.mark.asyncio
    @patch("backend.blocks.http.Requests")
    async def test_http_block_with_wildcard_host_match(
        self,
        mock_requests_class,
        http_block,
        wildcard_credentials,
        mock_response,
    ):
        """Test HTTP block with wildcard host pattern matching."""
        # Setup mocks
        mock_requests = AsyncMock()
        mock_requests.request.return_value = mock_response
        mock_requests_class.return_value = mock_requests

        # Test with subdomain that should match *.github.com
        input_data = SendAuthenticatedWebRequestBlock.Input(
            url="https://api.github.com/user",
            method=HttpMethod.GET,
            headers={},
            credentials=cast(
                HttpCredentials,
                {
                    "id": wildcard_credentials.id,
                    "provider": "http",
                    "type": "host_scoped",
                    "title": wildcard_credentials.title,
                },
            ),
        )

        # Execute with wildcard credentials
        result = []
        async for output_name, output_data in http_block.run(
            input_data,
            credentials=wildcard_credentials,
            graph_exec_id="test-exec-id",
        ):
            result.append((output_name, output_data))

        # Verify wildcard matching works
        mock_requests.request.assert_called_once()
        call_args = mock_requests.request.call_args
        expected_headers = {"Authorization": "token ghp_wildcard123"}
        assert call_args.kwargs["headers"] == expected_headers

    @pytest.mark.asyncio
    @patch("backend.blocks.http.Requests")
    async def test_http_block_with_non_matching_credentials(
        self,
        mock_requests_class,
        http_block,
        non_matching_credentials,
        mock_response,
    ):
        """Test HTTP block when credentials don't match the target URL."""
        # Setup mocks
        mock_requests = AsyncMock()
        mock_requests.request.return_value = mock_response
        mock_requests_class.return_value = mock_requests

        # Test with URL that doesn't match the credentials
        input_data = SendAuthenticatedWebRequestBlock.Input(
            url="https://api.example.com/data",
            method=HttpMethod.GET,
            headers={"User-Agent": "test-agent"},
            credentials=cast(
                HttpCredentials,
                {
                    "id": non_matching_credentials.id,
                    "provider": "http",
                    "type": "host_scoped",
                    "title": non_matching_credentials.title,
                },
            ),
        )

        # Execute with non-matching credentials
        result = []
        async for output_name, output_data in http_block.run(
            input_data,
            credentials=non_matching_credentials,
            graph_exec_id="test-exec-id",
        ):
            result.append((output_name, output_data))

        # Verify only user headers are included (no credential headers)
        mock_requests.request.assert_called_once()
        call_args = mock_requests.request.call_args
        expected_headers = {"User-Agent": "test-agent"}
        assert call_args.kwargs["headers"] == expected_headers

    @pytest.mark.asyncio
    @patch("backend.blocks.http.Requests")
    async def test_user_headers_override_credential_headers(
        self,
        mock_requests_class,
        http_block,
        exact_match_credentials,
        mock_response,
    ):
        """Test that user-provided headers take precedence over credential headers."""
        # Setup mocks
        mock_requests = AsyncMock()
        mock_requests.request.return_value = mock_response
        mock_requests_class.return_value = mock_requests

        # Test with user header that conflicts with credential header
        input_data = SendAuthenticatedWebRequestBlock.Input(
            url="https://api.example.com/data",
            method=HttpMethod.POST,
            headers={
                "Authorization": "Bearer user-override-token",  # Should override
                "Content-Type": "application/json",  # Additional user header
            },
            credentials=cast(
                HttpCredentials,
                {
                    "id": exact_match_credentials.id,
                    "provider": "http",
                    "type": "host_scoped",
                    "title": exact_match_credentials.title,
                },
            ),
        )

        # Execute with conflicting headers
        result = []
        async for output_name, output_data in http_block.run(
            input_data,
            credentials=exact_match_credentials,
            graph_exec_id="test-exec-id",
        ):
            result.append((output_name, output_data))

        # Verify user headers take precedence
        mock_requests.request.assert_called_once()
        call_args = mock_requests.request.call_args
        expected_headers = {
            "X-API-Key": "api-key-123",  # From credentials
            "Authorization": "Bearer user-override-token",  # User override
            "Content-Type": "application/json",  # User header
        }
        assert call_args.kwargs["headers"] == expected_headers

    @pytest.mark.asyncio
    @patch("backend.blocks.http.Requests")
    async def test_auto_discovered_credentials_flow(
        self,
        mock_requests_class,
        http_block,
        mock_response,
    ):
        """Test the auto-discovery flow where execution manager provides matching credentials."""
        # Create auto-discovered credentials
        auto_discovered_creds = HostScopedCredentials(
            provider="http",
            host="*.example.com",
            headers={
                "Authorization": SecretStr("Bearer auto-discovered-token"),
            },
            title="Auto-discovered Credentials",
        )

        # Setup mocks
        mock_requests = AsyncMock()
        mock_requests.request.return_value = mock_response
        mock_requests_class.return_value = mock_requests

        # Test with empty credentials field (triggers auto-discovery)
        input_data = SendAuthenticatedWebRequestBlock.Input(
            url="https://api.example.com/data",
            method=HttpMethod.GET,
            headers={},
            credentials=cast(
                HttpCredentials,
                {
                    "id": "",  # Empty ID triggers auto-discovery in execution manager
                    "provider": "http",
                    "type": "host_scoped",
                    "title": "",
                },
            ),
        )

        # Execute with auto-discovered credentials provided by execution manager
        result = []
        async for output_name, output_data in http_block.run(
            input_data,
            credentials=auto_discovered_creds,  # Execution manager found these
            graph_exec_id="test-exec-id",
        ):
            result.append((output_name, output_data))

        # Verify auto-discovered credentials were applied
        mock_requests.request.assert_called_once()
        call_args = mock_requests.request.call_args
        expected_headers = {"Authorization": "Bearer auto-discovered-token"}
        assert call_args.kwargs["headers"] == expected_headers

        # Verify response handling
        assert len(result) == 1
        assert result[0][0] == "response"
        assert result[0][1] == {"success": True, "data": "test"}

    @pytest.mark.asyncio
    @patch("backend.blocks.http.Requests")
    async def test_multiple_header_credentials(
        self,
        mock_requests_class,
        http_block,
        mock_response,
    ):
        """Test credentials with multiple headers are all applied."""
        # Create credentials with multiple headers
        multi_header_creds = HostScopedCredentials(
            provider="http",
            host="api.example.com",
            headers={
                "Authorization": SecretStr("Bearer multi-token"),
                "X-API-Key": SecretStr("api-key-456"),
                "X-Client-ID": SecretStr("client-789"),
                "X-Custom-Header": SecretStr("custom-value"),
            },
            title="Multi-Header Credentials",
        )

        # Setup mocks
        mock_requests = AsyncMock()
        mock_requests.request.return_value = mock_response
        mock_requests_class.return_value = mock_requests

        # Test with credentials containing multiple headers
        input_data = SendAuthenticatedWebRequestBlock.Input(
            url="https://api.example.com/data",
            method=HttpMethod.GET,
            headers={"User-Agent": "test-agent"},
            credentials=cast(
                HttpCredentials,
                {
                    "id": multi_header_creds.id,
                    "provider": "http",
                    "type": "host_scoped",
                    "title": multi_header_creds.title,
                },
            ),
        )

        # Execute with multi-header credentials
        result = []
        async for output_name, output_data in http_block.run(
            input_data,
            credentials=multi_header_creds,
            graph_exec_id="test-exec-id",
        ):
            result.append((output_name, output_data))

        # Verify all headers are included
        mock_requests.request.assert_called_once()
        call_args = mock_requests.request.call_args
        expected_headers = {
            "Authorization": "Bearer multi-token",
            "X-API-Key": "api-key-456",
            "X-Client-ID": "client-789",
            "X-Custom-Header": "custom-value",
            "User-Agent": "test-agent",
        }
        assert call_args.kwargs["headers"] == expected_headers

    @pytest.mark.asyncio
    @patch("backend.blocks.http.Requests")
    async def test_credentials_with_complex_url_patterns(
        self,
        mock_requests_class,
        http_block,
        mock_response,
    ):
        """Test credentials matching various URL patterns."""
        # Test cases for different URL patterns
        test_cases = [
            {
                "host_pattern": "api.example.com",
                "test_url": "https://api.example.com/v1/users",
                "should_match": True,
            },
            {
                "host_pattern": "*.example.com",
                "test_url": "https://api.example.com/v1/users",
                "should_match": True,
            },
            {
                "host_pattern": "*.example.com",
                "test_url": "https://subdomain.example.com/data",
                "should_match": True,
            },
            {
                "host_pattern": "api.example.com",
                "test_url": "https://api.different.com/data",
                "should_match": False,
            },
        ]

        # Setup mocks
        mock_requests = AsyncMock()
        mock_requests.request.return_value = mock_response
        mock_requests_class.return_value = mock_requests

        for case in test_cases:
            # Reset mock for each test case
            mock_requests.reset_mock()

            # Create credentials for this test case
            test_creds = HostScopedCredentials(
                provider="http",
                host=case["host_pattern"],
                headers={
                    "Authorization": SecretStr(f"Bearer {case['host_pattern']}-token"),
                },
                title=f"Credentials for {case['host_pattern']}",
            )

            input_data = SendAuthenticatedWebRequestBlock.Input(
                url=case["test_url"],
                method=HttpMethod.GET,
                headers={"User-Agent": "test-agent"},
                credentials=cast(
                    HttpCredentials,
                    {
                        "id": test_creds.id,
                        "provider": "http",
                        "type": "host_scoped",
                        "title": test_creds.title,
                    },
                ),
            )

            # Execute with test credentials
            result = []
            async for output_name, output_data in http_block.run(
                input_data,
                credentials=test_creds,
                graph_exec_id="test-exec-id",
            ):
                result.append((output_name, output_data))

            # Verify headers based on whether pattern should match
            mock_requests.request.assert_called_once()
            call_args = mock_requests.request.call_args
            headers = call_args.kwargs["headers"]

            if case["should_match"]:
                # Should include both user and credential headers
                expected_auth = f"Bearer {case['host_pattern']}-token"
                assert headers["Authorization"] == expected_auth
                assert headers["User-Agent"] == "test-agent"
            else:
                # Should only include user headers
                assert "Authorization" not in headers
                assert headers["User-Agent"] == "test-agent"
