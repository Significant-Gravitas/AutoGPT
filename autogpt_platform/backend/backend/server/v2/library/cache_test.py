"""
Tests for cache invalidation in Library API routes.

This module tests that library caches are properly invalidated when data is modified.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import backend.server.v2.library.cache as library_cache
import backend.server.v2.library.db as library_db


@pytest.fixture
def mock_user_id():
    """Generate a mock user ID for testing."""
    return str(uuid.uuid4())


@pytest.fixture
def mock_library_agent_id():
    """Generate a mock library agent ID for testing."""
    return str(uuid.uuid4())


class TestLibraryAgentCacheInvalidation:
    """Test cache invalidation for library agent operations."""

    @pytest.mark.asyncio
    async def test_add_agent_clears_list_cache(self, mock_user_id):
        """Test that adding an agent clears the library agents list cache."""
        # Clear cache
        library_cache.get_cached_library_agents.cache_clear()

        with patch.object(
            library_db, "list_library_agents", new_callable=AsyncMock
        ) as mock_list:
            mock_response = MagicMock(agents=[], total_count=0, page=1, page_size=20)
            mock_list.return_value = mock_response

            # First call hits database
            await library_cache.get_cached_library_agents(mock_user_id, 1, 20)
            assert mock_list.call_count == 1

            # Second call uses cache
            await library_cache.get_cached_library_agents(mock_user_id, 1, 20)
            assert mock_list.call_count == 1  # Still 1, cache used

            # Simulate adding an agent (cache invalidation)
            for page in range(1, 5):
                library_cache.get_cached_library_agents.cache_delete(
                    mock_user_id, page, 15
                )
                library_cache.get_cached_library_agents.cache_delete(
                    mock_user_id, page, 20
                )

            # Next call should hit database
            await library_cache.get_cached_library_agents(mock_user_id, 1, 20)
            assert mock_list.call_count == 2

    @pytest.mark.asyncio
    async def test_delete_agent_clears_multiple_caches(
        self, mock_user_id, mock_library_agent_id
    ):
        """Test that deleting an agent clears both specific and list caches."""
        # Clear caches
        library_cache.get_cached_library_agent.cache_clear()
        library_cache.get_cached_library_agents.cache_clear()

        with (
            patch.object(
                library_db, "get_library_agent", new_callable=AsyncMock
            ) as mock_get,
            patch.object(
                library_db, "list_library_agents", new_callable=AsyncMock
            ) as mock_list,
        ):
            mock_agent = MagicMock(id=mock_library_agent_id, name="Test Agent")
            mock_get.return_value = mock_agent
            mock_list.return_value = MagicMock(agents=[mock_agent])

            # Populate caches
            await library_cache.get_cached_library_agent(
                mock_library_agent_id, mock_user_id
            )
            await library_cache.get_cached_library_agents(mock_user_id, 1, 20)

            initial_calls = {
                "get": mock_get.call_count,
                "list": mock_list.call_count,
            }

            # Verify cache is used
            await library_cache.get_cached_library_agent(
                mock_library_agent_id, mock_user_id
            )
            await library_cache.get_cached_library_agents(mock_user_id, 1, 20)

            assert mock_get.call_count == initial_calls["get"]
            assert mock_list.call_count == initial_calls["list"]

            # Simulate delete_library_agent cache invalidation
            library_cache.get_cached_library_agent.cache_delete(
                mock_library_agent_id, mock_user_id
            )
            for page in range(1, 5):
                library_cache.get_cached_library_agents.cache_delete(
                    mock_user_id, page, 15
                )
                library_cache.get_cached_library_agents.cache_delete(
                    mock_user_id, page, 20
                )

            # Next calls should hit database
            await library_cache.get_cached_library_agent(
                mock_library_agent_id, mock_user_id
            )
            await library_cache.get_cached_library_agents(mock_user_id, 1, 20)

            assert mock_get.call_count == initial_calls["get"] + 1
            assert mock_list.call_count == initial_calls["list"] + 1

    @pytest.mark.asyncio
    async def test_favorites_cache_operations(self, mock_user_id):
        """Test that favorites cache works independently."""
        # Clear cache
        library_cache.get_cached_library_agent_favorites.cache_clear()

        with patch.object(
            library_db, "list_favorite_library_agents", new_callable=AsyncMock
        ) as mock_favs:
            mock_response = MagicMock(agents=[], total_count=0, page=1, page_size=20)
            mock_favs.return_value = mock_response

            # First call hits database
            await library_cache.get_cached_library_agent_favorites(mock_user_id, 1, 20)
            assert mock_favs.call_count == 1

            # Second call uses cache
            await library_cache.get_cached_library_agent_favorites(mock_user_id, 1, 20)
            assert mock_favs.call_count == 1  # Cache used

            # Clear cache
            library_cache.get_cached_library_agent_favorites.cache_delete(
                mock_user_id, 1, 20
            )

            # Next call hits database
            await library_cache.get_cached_library_agent_favorites(mock_user_id, 1, 20)
            assert mock_favs.call_count == 2


class TestLibraryPresetCacheInvalidation:
    """Test cache invalidation for library preset operations."""

    @pytest.mark.asyncio
    async def test_preset_cache_operations(self, mock_user_id):
        """Test preset cache and invalidation."""
        # Clear cache
        library_cache.get_cached_library_presets.cache_clear()
        library_cache.get_cached_library_preset.cache_clear()

        preset_id = str(uuid.uuid4())

        with (
            patch.object(
                library_db, "list_presets", new_callable=AsyncMock
            ) as mock_list,
            patch.object(library_db, "get_preset", new_callable=AsyncMock) as mock_get,
        ):
            mock_preset = MagicMock(id=preset_id, name="Test Preset")
            mock_list.return_value = MagicMock(presets=[mock_preset])
            mock_get.return_value = mock_preset

            # Populate caches
            await library_cache.get_cached_library_presets(mock_user_id, 1, 20)
            await library_cache.get_cached_library_preset(preset_id, mock_user_id)

            initial_calls = {
                "list": mock_list.call_count,
                "get": mock_get.call_count,
            }

            # Verify cache is used
            await library_cache.get_cached_library_presets(mock_user_id, 1, 20)
            await library_cache.get_cached_library_preset(preset_id, mock_user_id)

            assert mock_list.call_count == initial_calls["list"]
            assert mock_get.call_count == initial_calls["get"]

            # Clear specific preset cache
            library_cache.get_cached_library_preset.cache_delete(
                preset_id, mock_user_id
            )

            # Clear list cache
            library_cache.get_cached_library_presets.cache_delete(mock_user_id, 1, 20)

            # Next calls should hit database
            await library_cache.get_cached_library_presets(mock_user_id, 1, 20)
            await library_cache.get_cached_library_preset(preset_id, mock_user_id)

            assert mock_list.call_count == initial_calls["list"] + 1
            assert mock_get.call_count == initial_calls["get"] + 1


class TestLibraryCacheMetrics:
    """Test library cache metrics and management."""

    def test_cache_info_structure(self):
        """Test that cache_info returns expected structure."""
        info = library_cache.get_cached_library_agents.cache_info()

        assert "size" in info
        assert "maxsize" in info
        assert "ttl_seconds" in info
        assert info["maxsize"] == 1000  # As defined in cache.py
        assert info["ttl_seconds"] == 600  # 10 minutes

    def test_all_library_caches_can_be_cleared(self):
        """Test that all library caches can be cleared."""
        # Clear all library caches
        library_cache.get_cached_library_agents.cache_clear()
        library_cache.get_cached_library_agent_favorites.cache_clear()
        library_cache.get_cached_library_agent.cache_clear()
        library_cache.get_cached_library_agent_by_graph_id.cache_clear()
        library_cache.get_cached_library_agent_by_store_version.cache_clear()
        library_cache.get_cached_library_presets.cache_clear()
        library_cache.get_cached_library_preset.cache_clear()

        # Verify all are empty
        assert library_cache.get_cached_library_agents.cache_info()["size"] == 0
        assert (
            library_cache.get_cached_library_agent_favorites.cache_info()["size"] == 0
        )
        assert library_cache.get_cached_library_agent.cache_info()["size"] == 0
        assert (
            library_cache.get_cached_library_agent_by_graph_id.cache_info()["size"] == 0
        )
        assert (
            library_cache.get_cached_library_agent_by_store_version.cache_info()["size"]
            == 0
        )
        assert library_cache.get_cached_library_presets.cache_info()["size"] == 0
        assert library_cache.get_cached_library_preset.cache_info()["size"] == 0

    def test_cache_ttl_values(self):
        """Test that cache TTL values are set correctly."""
        # Library agents - 10 minutes
        assert (
            library_cache.get_cached_library_agents.cache_info()["ttl_seconds"] == 600
        )

        # Favorites - 5 minutes (more dynamic)
        assert (
            library_cache.get_cached_library_agent_favorites.cache_info()["ttl_seconds"]
            == 300
        )

        # Individual agent - 30 minutes
        assert (
            library_cache.get_cached_library_agent.cache_info()["ttl_seconds"] == 1800
        )

        # Presets - 30 minutes
        assert (
            library_cache.get_cached_library_presets.cache_info()["ttl_seconds"] == 1800
        )
        assert (
            library_cache.get_cached_library_preset.cache_info()["ttl_seconds"] == 1800
        )
