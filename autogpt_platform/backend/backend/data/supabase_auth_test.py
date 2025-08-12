"""
Tests for Supabase auth integration.
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.data.supabase_auth import (
    clear_user_auth_cache,
    get_user_auth_data_from_supabase,
)


@pytest.fixture
def mock_supabase_user():
    """Create a mock Supabase user object."""
    user = MagicMock()
    user.id = "test-user-123"
    user.email = "test@example.com"
    user.role = "authenticated"
    user.app_metadata = {
        "role": "admin",
        "organization": "test-org",
    }
    return user


@pytest.fixture
def mock_supabase_response(mock_supabase_user):
    """Create a mock Supabase response."""
    response = MagicMock()
    response.user = mock_supabase_user
    return response


class TestGetUserAuthDataFromSupabase:
    """Test get_user_auth_data_from_supabase function."""

    @pytest.mark.asyncio
    async def test_successful_fetch_with_app_metadata_role(
        self, mock_supabase_response
    ):
        """Test successful fetch with role in app_metadata."""
        with patch("backend.data.supabase_auth.get_supabase") as mock_get_supabase:
            # Setup mock
            mock_client = MagicMock()
            mock_client.auth.admin.get_user_by_id.return_value = mock_supabase_response
            mock_get_supabase.return_value = mock_client

            # Clear cache first to ensure fresh fetch
            clear_user_auth_cache("test-user-123")

            # Call function
            result = await get_user_auth_data_from_supabase("test-user-123")

            # Assertions
            assert result is not None
            assert result["role"] == "admin"  # Should use app_metadata role
            assert result["email"] == "test@example.com"
            assert result["app_organization"] == "test-org"

            # Verify Supabase was called
            mock_client.auth.admin.get_user_by_id.assert_called_once_with(
                "test-user-123"
            )

    @pytest.mark.asyncio
    async def test_successful_fetch_without_app_metadata_role(self, mock_supabase_user):
        """Test successful fetch when role is not in app_metadata."""
        # Remove role from app_metadata
        mock_supabase_user.app_metadata = {"organization": "test-org"}
        mock_response = MagicMock()
        mock_response.user = mock_supabase_user

        with patch("backend.data.supabase_auth.get_supabase") as mock_get_supabase:
            # Setup mock
            mock_client = MagicMock()
            mock_client.auth.admin.get_user_by_id.return_value = mock_response
            mock_get_supabase.return_value = mock_client

            # Clear cache first
            clear_user_auth_cache("test-user-123")

            # Call function
            result = await get_user_auth_data_from_supabase("test-user-123")

            # Assertions
            assert result is not None
            assert result["role"] == "authenticated"  # Should use user.role
            assert result["email"] == "test@example.com"
            assert result["app_organization"] == "test-org"

    @pytest.mark.asyncio
    async def test_user_not_found(self):
        """Test when user is not found in Supabase."""
        with patch("backend.data.supabase_auth.get_supabase") as mock_get_supabase:
            # Setup mock
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.user = None
            mock_client.auth.admin.get_user_by_id.return_value = mock_response
            mock_get_supabase.return_value = mock_client

            # Clear cache first
            clear_user_auth_cache("nonexistent-user")

            # Call function
            result = await get_user_auth_data_from_supabase("nonexistent-user")

            # Assertions
            assert result is None

    @pytest.mark.asyncio
    async def test_supabase_error(self):
        """Test when Supabase raises an error."""
        with patch("backend.data.supabase_auth.get_supabase") as mock_get_supabase:
            # Setup mock to raise error
            mock_client = MagicMock()
            mock_client.auth.admin.get_user_by_id.side_effect = Exception(
                "Supabase error"
            )
            mock_get_supabase.return_value = mock_client

            # Clear cache first
            clear_user_auth_cache("test-user-123")

            # Call function
            result = await get_user_auth_data_from_supabase("test-user-123")

            # Assertions
            assert result is None

    @pytest.mark.asyncio
    async def test_import_error(self):
        """Test when Supabase integration cannot be imported."""
        with patch(
            "backend.data.supabase_auth.get_supabase",
            side_effect=ImportError("Module not found"),
        ):
            # Clear cache first
            clear_user_auth_cache("test-user-123")

            # Call function
            result = await get_user_auth_data_from_supabase("test-user-123")

            # Assertions
            assert result is None

    @pytest.mark.asyncio
    async def test_caching(self, mock_supabase_response):
        """Test that results are cached."""
        with patch("backend.data.supabase_auth.get_supabase") as mock_get_supabase:
            # Setup mock
            mock_client = MagicMock()
            mock_client.auth.admin.get_user_by_id.return_value = mock_supabase_response
            mock_get_supabase.return_value = mock_client

            # Clear cache first
            clear_user_auth_cache("test-user-123")

            # First call
            result1 = await get_user_auth_data_from_supabase("test-user-123")

            # Second call (should use cache)
            result2 = await get_user_auth_data_from_supabase("test-user-123")

            # Assertions
            assert result1 == result2
            # Supabase should only be called once due to caching
            mock_client.auth.admin.get_user_by_id.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_app_metadata(self):
        """Test when user has no app_metadata."""
        mock_user = MagicMock()
        mock_user.id = "test-user-123"
        mock_user.email = "test@example.com"
        mock_user.role = "authenticated"
        mock_user.app_metadata = None  # No app_metadata

        mock_response = MagicMock()
        mock_response.user = mock_user

        with patch("backend.data.supabase_auth.get_supabase") as mock_get_supabase:
            # Setup mock
            mock_client = MagicMock()
            mock_client.auth.admin.get_user_by_id.return_value = mock_response
            mock_get_supabase.return_value = mock_client

            # Clear cache first
            clear_user_auth_cache("test-user-123")

            # Call function
            result = await get_user_auth_data_from_supabase("test-user-123")

            # Assertions
            assert result is not None
            assert result["role"] == "authenticated"
            assert result["email"] == "test@example.com"
            # Should not have any app_ prefixed keys
            assert not any(key.startswith("app_") for key in result.keys())


class TestClearUserAuthCache:
    """Test clear_user_auth_cache function."""

    def test_clear_cache(self):
        """Test clearing cache for a user."""
        # This is a simple function that doesn't throw errors
        # Just ensure it runs without error
        clear_user_auth_cache("test-user-123")
        # Should not raise any exceptions
