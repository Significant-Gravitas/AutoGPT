"""Tests for validate_callback_url function in external integrations API."""

from unittest.mock import patch

import pytest

from backend.api.external.v1.integrations import validate_callback_url


class MockConfig:
    """Mock configuration for testing."""

    def __init__(self, allowed_origins: list[str]):
        self.external_oauth_callback_origins = allowed_origins


class MockSettings:
    """Mock settings for testing."""

    def __init__(self, allowed_origins: list[str]):
        self.config = MockConfig(allowed_origins)


@pytest.fixture
def mock_settings_localhost_3000():
    """Mock settings with localhost:3000 allowed."""
    return MockSettings(["http://localhost:3000"])


@pytest.fixture
def mock_settings_multiple_origins():
    """Mock settings with multiple allowed origins."""
    return MockSettings(
        [
            "http://localhost:3000",
            "https://app.example.com",
            "https://staging.example.com:8443",
        ]
    )


class TestValidateCallbackUrl:
    """Test cases for validate_callback_url function."""

    def test_exact_match_allowed(self, mock_settings_localhost_3000):
        """Test that exact origin match is allowed."""
        with patch(
            "backend.api.external.v1.integrations.settings",
            mock_settings_localhost_3000,
        ):
            assert validate_callback_url("http://localhost:3000/callback") is True
            assert validate_callback_url("http://localhost:3000/oauth/complete") is True

    def test_different_port_rejected(self, mock_settings_localhost_3000):
        """Test that localhost with different port is rejected (security fix)."""
        with patch(
            "backend.api.external.v1.integrations.settings",
            mock_settings_localhost_3000,
        ):
            # These should be REJECTED - different ports on localhost
            assert validate_callback_url("http://localhost:8080/callback") is False
            assert validate_callback_url("http://localhost:9999/callback") is False
            assert validate_callback_url("http://localhost/callback") is False
            assert validate_callback_url("http://localhost:5000/callback") is False

    def test_different_scheme_rejected(self, mock_settings_localhost_3000):
        """Test that different scheme is rejected."""
        with patch(
            "backend.api.external.v1.integrations.settings",
            mock_settings_localhost_3000,
        ):
            # HTTPS vs HTTP should be rejected
            assert validate_callback_url("https://localhost:3000/callback") is False

    def test_different_host_rejected(self, mock_settings_localhost_3000):
        """Test that different host is rejected."""
        with patch(
            "backend.api.external.v1.integrations.settings",
            mock_settings_localhost_3000,
        ):
            assert validate_callback_url("http://127.0.0.1:3000/callback") is False
            assert validate_callback_url("http://example.com:3000/callback") is False

    def test_multiple_allowed_origins(self, mock_settings_multiple_origins):
        """Test validation with multiple allowed origins."""
        with patch(
            "backend.api.external.v1.integrations.settings",
            mock_settings_multiple_origins,
        ):
            # All configured origins should be allowed
            assert validate_callback_url("http://localhost:3000/callback") is True
            assert validate_callback_url("https://app.example.com/oauth") is True
            assert (
                validate_callback_url("https://staging.example.com:8443/callback")
                is True
            )

            # Non-configured origins should be rejected
            assert validate_callback_url("http://localhost:4000/callback") is False
            assert validate_callback_url("https://other.example.com/callback") is False
            assert (
                validate_callback_url("https://staging.example.com/callback") is False
            )

    def test_invalid_url_rejected(self, mock_settings_localhost_3000):
        """Test that invalid URLs are rejected."""
        with patch(
            "backend.api.external.v1.integrations.settings",
            mock_settings_localhost_3000,
        ):
            assert validate_callback_url("not-a-url") is False
            assert validate_callback_url("") is False

    def test_empty_allowed_origins(self):
        """Test that no origins allowed rejects everything."""
        mock_settings = MockSettings([])
        with patch(
            "backend.api.external.v1.integrations.settings",
            mock_settings,
        ):
            assert validate_callback_url("http://localhost:3000/callback") is False
            assert validate_callback_url("https://example.com/callback") is False

    def test_path_does_not_affect_validation(self, mock_settings_localhost_3000):
        """Test that different paths on same origin are all allowed."""
        with patch(
            "backend.api.external.v1.integrations.settings",
            mock_settings_localhost_3000,
        ):
            assert validate_callback_url("http://localhost:3000/") is True
            assert validate_callback_url("http://localhost:3000/callback") is True
            assert validate_callback_url("http://localhost:3000/oauth/complete") is True
            assert (
                validate_callback_url("http://localhost:3000/deep/nested/path") is True
            )

    def test_query_params_do_not_affect_validation(self, mock_settings_localhost_3000):
        """Test that query parameters don't affect origin validation."""
        with patch(
            "backend.api.external.v1.integrations.settings",
            mock_settings_localhost_3000,
        ):
            assert (
                validate_callback_url("http://localhost:3000/callback?code=abc") is True
            )
            assert (
                validate_callback_url(
                    "http://localhost:3000/callback?state=xyz&code=abc"
                )
                is True
            )
