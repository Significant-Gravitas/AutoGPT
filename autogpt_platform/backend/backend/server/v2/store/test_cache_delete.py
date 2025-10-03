#!/usr/bin/env python3
"""
Test suite for verifying cache_delete functionality in store routes.
Tests that specific cache entries can be deleted while preserving others.
"""

import datetime
from unittest.mock import AsyncMock, patch

import pytest

from backend.server.v2.store import routes
from backend.server.v2.store.model import (
    ProfileDetails,
    StoreAgent,
    StoreAgentDetails,
    StoreAgentsResponse,
)
from backend.util.models import Pagination


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
            "backend.server.v2.store.db.get_store_agents",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_db:
            # Clear cache first
            routes._get_cached_store_agents.cache_clear()

            # First call - should hit database
            result1 = await routes._get_cached_store_agents(
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
            await routes._get_cached_store_agents(
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
            await routes._get_cached_store_agents(
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
            deleted = routes._get_cached_store_agents.cache_delete(
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
            deleted = routes._get_cached_store_agents.cache_delete(
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
            await routes._get_cached_store_agents(
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
            await routes._get_cached_store_agents(
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
    async def test_agent_details_cache_delete(self):
        """Test that specific agent details cache entries can be deleted."""
        mock_response = StoreAgentDetails(
            store_listing_version_id="version1",
            slug="test-agent",
            agent_name="Test Agent",
            agent_video="https://example.com/video.mp4",
            agent_image=["https://example.com/image.jpg"],
            creator="testuser",
            creator_avatar="https://example.com/avatar.jpg",
            sub_heading="Test subheading",
            description="Test description",
            categories=["productivity"],
            runs=100,
            rating=4.5,
            versions=[],
            last_updated=datetime.datetime(2024, 1, 1),
        )

        with patch(
            "backend.server.v2.store.db.get_store_agent_details",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_db:
            # Clear cache first
            routes._get_cached_agent_details.cache_clear()

            # First call - should hit database
            await routes._get_cached_agent_details(
                username="testuser", agent_name="testagent"
            )
            assert mock_db.call_count == 1

            # Second call - should use cache
            await routes._get_cached_agent_details(
                username="testuser", agent_name="testagent"
            )
            assert mock_db.call_count == 1  # No additional DB call

            # Delete specific entry
            deleted = routes._get_cached_agent_details.cache_delete(
                username="testuser", agent_name="testagent"
            )
            assert deleted is True

            # Call again - should hit database
            await routes._get_cached_agent_details(
                username="testuser", agent_name="testagent"
            )
            assert mock_db.call_count == 2  # New DB call after deletion

    @pytest.mark.asyncio
    async def test_user_profile_cache_delete(self):
        """Test that user profile cache entries can be deleted."""
        mock_response = ProfileDetails(
            name="Test User",
            username="testuser",
            description="Test profile",
            links=["https://example.com"],
        )

        with patch(
            "backend.server.v2.store.db.get_user_profile",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_db:
            # Clear cache first
            routes._get_cached_user_profile.cache_clear()

            # First call - should hit database
            await routes._get_cached_user_profile("user123")
            assert mock_db.call_count == 1

            # Second call - should use cache
            await routes._get_cached_user_profile("user123")
            assert mock_db.call_count == 1

            # Different user - should hit database
            await routes._get_cached_user_profile("user456")
            assert mock_db.call_count == 2

            # Delete specific user's cache
            deleted = routes._get_cached_user_profile.cache_delete("user123")
            assert deleted is True

            # user123 should hit database again
            await routes._get_cached_user_profile("user123")
            assert mock_db.call_count == 3

            # user456 should still be cached
            await routes._get_cached_user_profile("user456")
            assert mock_db.call_count == 3  # No additional DB call

    @pytest.mark.asyncio
    async def test_cache_info_after_deletions(self):
        """Test that cache_info correctly reflects deletions."""
        # Clear all caches first
        routes._get_cached_store_agents.cache_clear()

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
            "backend.server.v2.store.db.get_store_agents",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            # Add multiple entries
            for i in range(5):
                await routes._get_cached_store_agents(
                    featured=False,
                    creator=f"creator{i}",
                    sorted_by=None,
                    search_query=None,
                    category=None,
                    page=1,
                    page_size=20,
                )

            # Check cache size
            info = routes._get_cached_store_agents.cache_info()
            assert info["size"] == 5

            # Delete some entries
            for i in range(2):
                deleted = routes._get_cached_store_agents.cache_delete(
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
            info = routes._get_cached_store_agents.cache_info()
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
            "backend.server.v2.store.db.get_store_agents",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_db:
            routes._get_cached_store_agents.cache_clear()

            # Test with all parameters
            await routes._get_cached_store_agents(
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
            deleted = routes._get_cached_store_agents.cache_delete(
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
            deleted = routes._get_cached_store_agents.cache_delete(
                featured=True,
                creator="testuser",
                sorted_by="rating",
                search_query="AI assistant",
                category="productivity",
                page=2,
                page_size=51,  # Different page_size
            )
            assert deleted is False  # Different parameters, not in cache

    @pytest.mark.asyncio
    async def test_clear_submissions_cache_page_size_consistency(self):
        """
        Test that _clear_submissions_cache uses the correct page_size.
        This test ensures that if the default page_size in routes changes,
        the hardcoded value in _clear_submissions_cache must also change.
        """
        from backend.server.v2.store.model import StoreSubmissionsResponse

        mock_response = StoreSubmissionsResponse(
            submissions=[],
            pagination=Pagination(
                total_items=0,
                total_pages=1,
                current_page=1,
                page_size=20,
            ),
        )

        with patch(
            "backend.server.v2.store.db.get_store_submissions",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            # Clear cache first
            routes._get_cached_submissions.cache_clear()

            # Populate cache with multiple pages using the default page_size
            DEFAULT_PAGE_SIZE = 20  # This should match the default in routes.py
            user_id = "test_user"

            # Add entries for pages 1-5
            for page in range(1, 6):
                await routes._get_cached_submissions(
                    user_id=user_id, page=page, page_size=DEFAULT_PAGE_SIZE
                )

            # Verify cache has entries
            cache_info_before = routes._get_cached_submissions.cache_info()
            assert cache_info_before["size"] == 5

            # Call _clear_submissions_cache
            routes._clear_submissions_cache(user_id, num_pages=20)

            # All entries should be cleared
            cache_info_after = routes._get_cached_submissions.cache_info()
            assert (
                cache_info_after["size"] == 0
            ), "Cache should be empty after _clear_submissions_cache"

    @pytest.mark.asyncio
    async def test_clear_submissions_cache_detects_page_size_mismatch(self):
        """
        Test that detects if _clear_submissions_cache is using wrong page_size.
        If this test fails, it means the hardcoded page_size in _clear_submissions_cache
        doesn't match the default page_size used in the routes.
        """
        from backend.server.v2.store.model import StoreSubmissionsResponse

        mock_response = StoreSubmissionsResponse(
            submissions=[],
            pagination=Pagination(
                total_items=0,
                total_pages=1,
                current_page=1,
                page_size=20,
            ),
        )

        with patch(
            "backend.server.v2.store.db.get_store_submissions",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            # Clear cache first
            routes._get_cached_submissions.cache_clear()

            # WRONG_PAGE_SIZE simulates what happens if someone changes
            # the default page_size in routes but forgets to update _clear_submissions_cache
            WRONG_PAGE_SIZE = 25  # Different from the hardcoded value in cache.py
            user_id = "test_user"

            # Populate cache with the "wrong" page_size
            for page in range(1, 6):
                await routes._get_cached_submissions(
                    user_id=user_id, page=page, page_size=WRONG_PAGE_SIZE
                )

            # Verify cache has entries
            cache_info_before = routes._get_cached_submissions.cache_info()
            assert cache_info_before["size"] == 5

            # Call _clear_submissions_cache (which uses page_size=20 hardcoded)
            routes._clear_submissions_cache(user_id, num_pages=20)

            # If page_size is mismatched, entries won't be cleared
            cache_info_after = routes._get_cached_submissions.cache_info()

            # This assertion will FAIL if _clear_submissions_cache uses wrong page_size
            assert (
                cache_info_after["size"] == 5
            ), "Cache entries with different page_size should NOT be cleared (this is expected)"

    @pytest.mark.asyncio
    async def test_my_agents_cache_needs_clearing_too(self):
        """
        Test that demonstrates _get_cached_my_agents also needs cache clearing.
        Currently there's no _clear_my_agents_cache function, but there should be.
        """
        from backend.server.v2.store.model import MyAgentsResponse

        mock_response = MyAgentsResponse(
            agents=[],
            pagination=Pagination(
                total_items=0,
                total_pages=1,
                current_page=1,
                page_size=20,
            ),
        )

        with patch(
            "backend.server.v2.store.db.get_my_agents",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            routes._get_cached_my_agents.cache_clear()

            DEFAULT_PAGE_SIZE = 20
            user_id = "test_user"

            # Populate cache
            for page in range(1, 6):
                await routes._get_cached_my_agents(
                    user_id=user_id, page=page, page_size=DEFAULT_PAGE_SIZE
                )

            cache_info = routes._get_cached_my_agents.cache_info()
            assert cache_info["size"] == 5

            # NOTE: Currently there's no _clear_my_agents_cache function
            # If we implement one, it should clear all pages consistently
            # For now we document this as a TODO


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
