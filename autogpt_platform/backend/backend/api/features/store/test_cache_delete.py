#!/usr/bin/env python3
"""
Test suite for verifying cache_delete functionality in store routes.
Tests that specific cache entries can be deleted while preserving others.
"""

from unittest.mock import AsyncMock, patch

import pytest

from backend.util.models import Pagination

from . import cache as store_cache
from .model import StoreAgent, StoreAgentsResponse


class TestCacheDeletion:
    """Test cache deletion functionality for store routes."""

    @pytest.mark.asyncio
    async def test_store_agents_cache_delete(self):
        """Test that specific agent list cache entries can be deleted."""
        # Mock the database function
        mock_response = StoreAgentsResponse(
            agents=[
                StoreAgent(
                    slug="test-agent",
                    agent_name="Test Agent",
                    agent_image="https://example.com/image.jpg",
                    creator="testuser",
                    creator_avatar="https://example.com/avatar.jpg",
                    sub_heading="Test subheading",
                    description="Test description",
                    runs=100,
                    rating=4.5,
                )
            ],
            pagination=Pagination(
                total_items=1,
                total_pages=1,
                current_page=1,
                page_size=20,
            ),
        )

        with patch(
            "backend.api.features.store.db.get_store_agents",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_db:
            # Clear cache first
            store_cache._get_cached_store_agents.cache_clear()

            # First call - should hit database
            result1 = await store_cache._get_cached_store_agents(
                featured=False,
                creator=None,
                sorted_by=None,
                search_query="test",
                category=None,
                page=1,
                page_size=20,
            )
            assert mock_db.call_count == 1
            assert result1.agents[0].agent_name == "Test Agent"

            # Second call with same params - should use cache
            await store_cache._get_cached_store_agents(
                featured=False,
                creator=None,
                sorted_by=None,
                search_query="test",
                category=None,
                page=1,
                page_size=20,
            )
            assert mock_db.call_count == 1  # No additional DB call

            # Third call with different params - should hit database
            await store_cache._get_cached_store_agents(
                featured=True,  # Different param
                creator=None,
                sorted_by=None,
                search_query="test",
                category=None,
                page=1,
                page_size=20,
            )
            assert mock_db.call_count == 2  # New DB call

            # Delete specific cache entry
            deleted = store_cache._get_cached_store_agents.cache_delete(
                featured=False,
                creator=None,
                sorted_by=None,
                search_query="test",
                category=None,
                page=1,
                page_size=20,
            )
            assert deleted is True  # Entry was deleted

            # Try to delete non-existent entry
            deleted = store_cache._get_cached_store_agents.cache_delete(
                featured=False,
                creator="nonexistent",
                sorted_by=None,
                search_query="test",
                category=None,
                page=1,
                page_size=20,
            )
            assert deleted is False  # Entry didn't exist

            # Call with deleted params - should hit database again
            await store_cache._get_cached_store_agents(
                featured=False,
                creator=None,
                sorted_by=None,
                search_query="test",
                category=None,
                page=1,
                page_size=20,
            )
            assert mock_db.call_count == 3  # New DB call after deletion

            # Call with featured=True - should still be cached
            await store_cache._get_cached_store_agents(
                featured=True,
                creator=None,
                sorted_by=None,
                search_query="test",
                category=None,
                page=1,
                page_size=20,
            )
            assert mock_db.call_count == 3  # No additional DB call

    @pytest.mark.asyncio
    async def test_cache_info_after_deletions(self):
        """Test that cache_info correctly reflects deletions."""
        # Clear all caches first
        store_cache._get_cached_store_agents.cache_clear()

        mock_response = StoreAgentsResponse(
            agents=[],
            pagination=Pagination(
                total_items=0,
                total_pages=1,
                current_page=1,
                page_size=20,
            ),
        )

        with patch(
            "backend.api.features.store.db.get_store_agents",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            # Add multiple entries
            for i in range(5):
                await store_cache._get_cached_store_agents(
                    featured=False,
                    creator=f"creator{i}",
                    sorted_by=None,
                    search_query=None,
                    category=None,
                    page=1,
                    page_size=20,
                )

            # Check cache size
            info = store_cache._get_cached_store_agents.cache_info()
            assert info["size"] == 5

            # Delete some entries
            for i in range(2):
                deleted = store_cache._get_cached_store_agents.cache_delete(
                    featured=False,
                    creator=f"creator{i}",
                    sorted_by=None,
                    search_query=None,
                    category=None,
                    page=1,
                    page_size=20,
                )
                assert deleted is True

            # Check cache size after deletion
            info = store_cache._get_cached_store_agents.cache_info()
            assert info["size"] == 3

    @pytest.mark.asyncio
    async def test_cache_delete_with_complex_params(self):
        """Test cache deletion with various parameter combinations."""
        mock_response = StoreAgentsResponse(
            agents=[],
            pagination=Pagination(
                total_items=0,
                total_pages=1,
                current_page=1,
                page_size=20,
            ),
        )

        with patch(
            "backend.api.features.store.db.get_store_agents",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_db:
            store_cache._get_cached_store_agents.cache_clear()

            # Test with all parameters
            await store_cache._get_cached_store_agents(
                featured=True,
                creator="testuser",
                sorted_by="rating",
                search_query="AI assistant",
                category="productivity",
                page=2,
                page_size=50,
            )
            assert mock_db.call_count == 1

            # Delete with exact same parameters
            deleted = store_cache._get_cached_store_agents.cache_delete(
                featured=True,
                creator="testuser",
                sorted_by="rating",
                search_query="AI assistant",
                category="productivity",
                page=2,
                page_size=50,
            )
            assert deleted is True

            # Try to delete with slightly different parameters
            deleted = store_cache._get_cached_store_agents.cache_delete(
                featured=True,
                creator="testuser",
                sorted_by="rating",
                search_query="AI assistant",
                category="productivity",
                page=2,
                page_size=51,  # Different page_size
            )
            assert deleted is False  # Different parameters, not in cache


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
