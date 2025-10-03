"""
Tests for cache invalidation in V1 API routes.

This module tests that caches are properly invalidated when data is modified
through POST, PUT, PATCH, and DELETE operations.
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

import backend.server.routers.cache as cache
from backend.data import graph as graph_db


@pytest.fixture
def mock_user_id():
    """Generate a mock user ID for testing."""
    return str(uuid.uuid4())


@pytest.fixture
def mock_graph_id():
    """Generate a mock graph ID for testing."""
    return str(uuid.uuid4())


class TestGraphCacheInvalidation:
    """Test cache invalidation for graph operations."""

    @pytest.mark.asyncio
    async def test_create_graph_clears_list_cache(self, mock_user_id):
        """Test that creating a graph clears the graphs list cache."""
        # Setup
        cache.get_cached_graphs.cache_clear()

        # Pre-populate cache
        with patch.object(
            graph_db, "list_graphs_paginated", new_callable=AsyncMock
        ) as mock_list:
            # Use a simple dict instead of MagicMock to make it pickleable
            mock_list.return_value = {
                "graphs": [],
                "total_count": 0,
                "page": 1,
                "page_size": 250,
            }

            # First call should hit the database
            await cache.get_cached_graphs(mock_user_id, 1, 250)
            assert mock_list.call_count == 1

            # Second call should use cache
            await cache.get_cached_graphs(mock_user_id, 1, 250)
            assert mock_list.call_count == 1  # Still 1, used cache

            # Simulate cache invalidation (what happens in create_new_graph)
            cache.get_cached_graphs.cache_delete(mock_user_id, 1, 250)

            # Next call should hit database again
            await cache.get_cached_graphs(mock_user_id, 1, 250)
            assert mock_list.call_count == 2  # Incremented, cache was cleared

    @pytest.mark.asyncio
    async def test_delete_graph_clears_multiple_caches(
        self, mock_user_id, mock_graph_id
    ):
        """Test that deleting a graph clears all related caches."""
        # Clear all caches first
        cache.get_cached_graphs.cache_clear()
        cache.get_cached_graph.cache_clear()
        cache.get_cached_graph_all_versions.cache_clear()
        cache.get_cached_graph_executions.cache_clear()

        # Setup mocks
        with (
            patch.object(
                graph_db, "list_graphs_paginated", new_callable=AsyncMock
            ) as mock_list,
            patch.object(graph_db, "get_graph", new_callable=AsyncMock) as mock_get,
            patch.object(
                graph_db, "get_graph_all_versions", new_callable=AsyncMock
            ) as mock_versions,
        ):
            mock_list.return_value = {
                "graphs": [],
                "total_count": 0,
                "page": 1,
                "page_size": 250,
            }
            mock_get.return_value = {"id": mock_graph_id}
            mock_versions.return_value = []

            # Pre-populate all caches (use consistent argument style)
            await cache.get_cached_graphs(mock_user_id, 1, 250)
            await cache.get_cached_graph(mock_graph_id, None, mock_user_id)
            await cache.get_cached_graph_all_versions(mock_graph_id, mock_user_id)

            initial_calls = {
                "list": mock_list.call_count,
                "get": mock_get.call_count,
                "versions": mock_versions.call_count,
            }

            # Use cached values (no additional DB calls)
            await cache.get_cached_graphs(mock_user_id, 1, 250)
            await cache.get_cached_graph(mock_graph_id, None, mock_user_id)
            await cache.get_cached_graph_all_versions(mock_graph_id, mock_user_id)

            # Verify cache was used
            assert mock_list.call_count == initial_calls["list"]
            assert mock_get.call_count == initial_calls["get"]
            assert mock_versions.call_count == initial_calls["versions"]

            # Simulate delete_graph cache invalidation
            # Use positional arguments for cache_delete to match how we called the functions
            result1 = cache.get_cached_graphs.cache_delete(mock_user_id, 1, 250)
            result2 = cache.get_cached_graph.cache_delete(
                mock_graph_id, None, mock_user_id
            )
            result3 = cache.get_cached_graph_all_versions.cache_delete(
                mock_graph_id, mock_user_id
            )

            # Verify that the cache entries were actually deleted
            assert result1, "Failed to delete graphs cache entry"
            assert result2, "Failed to delete graph cache entry"
            assert result3, "Failed to delete graph versions cache entry"

            # Next calls should hit database
            await cache.get_cached_graphs(mock_user_id, 1, 250)
            await cache.get_cached_graph(mock_graph_id, None, mock_user_id)
            await cache.get_cached_graph_all_versions(mock_graph_id, mock_user_id)

            # Verify database was called again
            assert mock_list.call_count == initial_calls["list"] + 1
            assert mock_get.call_count == initial_calls["get"] + 1
            assert mock_versions.call_count == initial_calls["versions"] + 1

    @pytest.mark.asyncio
    async def test_update_graph_clears_caches(self, mock_user_id, mock_graph_id):
        """Test that updating a graph clears the appropriate caches."""
        # Clear caches
        cache.get_cached_graph.cache_clear()
        cache.get_cached_graph_all_versions.cache_clear()
        cache.get_cached_graphs.cache_clear()

        with (
            patch.object(graph_db, "get_graph", new_callable=AsyncMock) as mock_get,
            patch.object(
                graph_db, "get_graph_all_versions", new_callable=AsyncMock
            ) as mock_versions,
            patch.object(
                graph_db, "list_graphs_paginated", new_callable=AsyncMock
            ) as mock_list,
        ):
            mock_get.return_value = {"id": mock_graph_id, "version": 1}
            mock_versions.return_value = [{"version": 1}]
            mock_list.return_value = {
                "graphs": [],
                "total_count": 0,
                "page": 1,
                "page_size": 250,
            }

            # Populate caches
            await cache.get_cached_graph(mock_graph_id, None, mock_user_id)
            await cache.get_cached_graph_all_versions(mock_graph_id, mock_user_id)
            await cache.get_cached_graphs(mock_user_id, 1, 250)

            initial_calls = {
                "get": mock_get.call_count,
                "versions": mock_versions.call_count,
                "list": mock_list.call_count,
            }

            # Verify cache is being used
            await cache.get_cached_graph(mock_graph_id, None, mock_user_id)
            await cache.get_cached_graph_all_versions(mock_graph_id, mock_user_id)
            await cache.get_cached_graphs(mock_user_id, 1, 250)

            assert mock_get.call_count == initial_calls["get"]
            assert mock_versions.call_count == initial_calls["versions"]
            assert mock_list.call_count == initial_calls["list"]

            # Simulate update_graph cache invalidation
            cache.get_cached_graph.cache_delete(mock_graph_id, None, mock_user_id)
            cache.get_cached_graph_all_versions.cache_delete(
                mock_graph_id, mock_user_id
            )
            cache.get_cached_graphs.cache_delete(mock_user_id, 1, 250)

            # Next calls should hit database
            await cache.get_cached_graph(mock_graph_id, None, mock_user_id)
            await cache.get_cached_graph_all_versions(mock_graph_id, mock_user_id)
            await cache.get_cached_graphs(mock_user_id, 1, 250)

            assert mock_get.call_count == initial_calls["get"] + 1
            assert mock_versions.call_count == initial_calls["versions"] + 1
            assert mock_list.call_count == initial_calls["list"] + 1


class TestUserPreferencesCacheInvalidation:
    """Test cache invalidation for user preferences operations."""

    @pytest.mark.asyncio
    async def test_update_preferences_clears_cache(self, mock_user_id):
        """Test that updating preferences clears the preferences cache."""
        # Clear cache
        cache.get_cached_user_preferences.cache_clear()

        with patch.object(
            cache.user_db, "get_user_notification_preference", new_callable=AsyncMock
        ) as mock_get_prefs:
            mock_prefs = {"email_notifications": True, "push_notifications": False}
            mock_get_prefs.return_value = mock_prefs

            # First call hits database
            result1 = await cache.get_cached_user_preferences(mock_user_id)
            assert mock_get_prefs.call_count == 1
            assert result1 == mock_prefs

            # Second call uses cache
            result2 = await cache.get_cached_user_preferences(mock_user_id)
            assert mock_get_prefs.call_count == 1  # Still 1
            assert result2 == mock_prefs

            # Simulate update_preferences cache invalidation
            cache.get_cached_user_preferences.cache_delete(mock_user_id)

            # Change the mock return value to simulate updated preferences
            mock_prefs_updated = {
                "email_notifications": False,
                "push_notifications": True,
            }
            mock_get_prefs.return_value = mock_prefs_updated

            # Next call should hit database and get new value
            result3 = await cache.get_cached_user_preferences(mock_user_id)
            assert mock_get_prefs.call_count == 2
            assert result3 == mock_prefs_updated

    @pytest.mark.asyncio
    async def test_timezone_cache_operations(self, mock_user_id):
        """Test timezone cache and its operations."""
        # Clear cache
        cache.get_cached_user_timezone.cache_clear()

        with patch.object(
            cache.user_db, "get_user_by_id", new_callable=AsyncMock
        ) as mock_get_user:
            # Use a simple object that supports attribute access
            class MockUser:
                def __init__(self, timezone):
                    self.timezone = timezone

            mock_user = MockUser("America/New_York")
            mock_get_user.return_value = mock_user

            # First call hits database
            result1 = await cache.get_cached_user_timezone(mock_user_id)
            assert mock_get_user.call_count == 1
            assert result1["timezone"] == "America/New_York"

            # Second call uses cache
            result2 = await cache.get_cached_user_timezone(mock_user_id)
            assert mock_get_user.call_count == 1  # Still 1
            assert result2["timezone"] == "America/New_York"

            # Clear cache manually (simulating what would happen after update)
            cache.get_cached_user_timezone.cache_delete(mock_user_id)

            # Change timezone
            mock_user_updated = MockUser("Europe/London")
            mock_get_user.return_value = mock_user_updated

            # Next call should hit database
            result3 = await cache.get_cached_user_timezone(mock_user_id)
            assert mock_get_user.call_count == 2
            assert result3["timezone"] == "Europe/London"


class TestExecutionCacheInvalidation:
    """Test cache invalidation for execution operations."""

    @pytest.mark.asyncio
    async def test_execution_cache_cleared_on_graph_delete(
        self, mock_user_id, mock_graph_id
    ):
        """Test that execution caches are cleared when a graph is deleted."""
        # Clear cache
        cache.get_cached_graph_executions.cache_clear()

        with patch.object(
            cache.execution_db, "get_graph_executions_paginated", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = {
                "executions": [],
                "total_count": 0,
                "page": 1,
                "page_size": 25,
            }

            # Populate cache for multiple pages
            for page in range(1, 4):
                await cache.get_cached_graph_executions(
                    mock_graph_id, mock_user_id, page, 25
                )

            initial_calls = mock_exec.call_count

            # Verify cache is used
            for page in range(1, 4):
                await cache.get_cached_graph_executions(
                    mock_graph_id, mock_user_id, page, 25
                )

            assert mock_exec.call_count == initial_calls  # No new calls

            # Simulate graph deletion clearing execution caches
            for page in range(1, 10):  # Clear more pages as done in delete_graph
                cache.get_cached_graph_executions.cache_delete(
                    mock_graph_id, mock_user_id, page, 25
                )

            # Next calls should hit database
            for page in range(1, 4):
                await cache.get_cached_graph_executions(
                    mock_graph_id, mock_user_id, page, 25
                )

            assert mock_exec.call_count == initial_calls + 3  # 3 new calls


class TestCacheInfo:
    """Test cache information and metrics."""

    def test_cache_info_returns_correct_metrics(self):
        """Test that cache_info returns correct metrics."""
        # Clear all caches
        cache.get_cached_graphs.cache_clear()
        cache.get_cached_graph.cache_clear()

        # Get initial info
        info_graphs = cache.get_cached_graphs.cache_info()
        info_graph = cache.get_cached_graph.cache_info()

        assert info_graphs["size"] == 0
        assert info_graph["size"] == 0

        # Note: We can't directly test cache population without real async context,
        # but we can verify the cache_info structure
        assert "size" in info_graphs
        assert "maxsize" in info_graphs
        assert "ttl_seconds" in info_graphs

    def test_cache_clear_removes_all_entries(self):
        """Test that cache_clear removes all entries."""
        # This test verifies the cache_clear method exists and can be called
        cache.get_cached_graphs.cache_clear()
        cache.get_cached_graph.cache_clear()
        cache.get_cached_graph_all_versions.cache_clear()
        cache.get_cached_graph_executions.cache_clear()
        cache.get_cached_graphs_executions.cache_clear()
        cache.get_cached_user_preferences.cache_clear()
        cache.get_cached_user_timezone.cache_clear()

        # After clear, all caches should be empty
        assert cache.get_cached_graphs.cache_info()["size"] == 0
        assert cache.get_cached_graph.cache_info()["size"] == 0
        assert cache.get_cached_graph_all_versions.cache_info()["size"] == 0
        assert cache.get_cached_graph_executions.cache_info()["size"] == 0
        assert cache.get_cached_graphs_executions.cache_info()["size"] == 0
        assert cache.get_cached_user_preferences.cache_info()["size"] == 0
        assert cache.get_cached_user_timezone.cache_info()["size"] == 0
