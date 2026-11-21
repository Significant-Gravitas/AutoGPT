"""Deleting one store agent list cache entry leaves the others cached."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.util.models import Pagination

from . import cache as store_cache
from .db import StoreAgentsSortOptions
from .model import StoreAgent, StoreAgentsResponse


class TestCacheDeletion:
    async def test_store_agents_cache_delete(self, mock_db):
        mock_db.return_value = StoreAgentsResponse(
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
                    agent_graph_id="test-graph-id",
                )
            ],
            pagination=_pagination(total_items=1),
        )

        result1 = await store_cache.get_cached_store_agents(**_params())
        assert mock_db.call_count == 1
        assert result1.agents[0].agent_name == "Test Agent"

        await store_cache.get_cached_store_agents(**_params())
        assert mock_db.call_count == 1

        await store_cache.get_cached_store_agents(**_params(featured=True))
        assert mock_db.call_count == 2

        deleted = store_cache.get_cached_store_agents.cache_delete(**_params())
        assert deleted is True

        deleted = store_cache.get_cached_store_agents.cache_delete(
            **_params(creator="nonexistent")
        )
        assert deleted is False

        await store_cache.get_cached_store_agents(**_params())
        assert mock_db.call_count == 3

        # The featured=True entry survived the other entry's delete.
        await store_cache.get_cached_store_agents(**_params(featured=True))
        assert mock_db.call_count == 3

    async def test_cache_info_after_deletions(self, mock_db):
        for i in range(5):
            await store_cache.get_cached_store_agents(
                **_params(creator=f"creator{i}", search_query=None)
            )

        info = store_cache.get_cached_store_agents.cache_info()
        assert info["size"] == 5

        for i in range(2):
            deleted = store_cache.get_cached_store_agents.cache_delete(
                **_params(creator=f"creator{i}", search_query=None)
            )
            assert deleted is True

        info = store_cache.get_cached_store_agents.cache_info()
        assert info["size"] == 3

    async def test_cache_delete_with_complex_params(self, mock_db):
        params = _params(
            featured=True,
            creator="testuser",
            sorted_by=StoreAgentsSortOptions.RATING,
            search_query="AI assistant",
            category="productivity",
            page=2,
            page_size=50,
        )
        await store_cache.get_cached_store_agents(**params)
        assert mock_db.call_count == 1

        deleted = store_cache.get_cached_store_agents.cache_delete(**params)
        assert deleted is True

        deleted = store_cache.get_cached_store_agents.cache_delete(
            **{**params, "page_size": 51}
        )
        assert deleted is False


@pytest.fixture
def mock_db():
    with patch(
        "backend.api.features.store.db.get_store_agents",
        new_callable=AsyncMock,
        return_value=StoreAgentsResponse(
            agents=[], pagination=_pagination(total_items=0)
        ),
    ) as mock:
        store_cache.get_cached_store_agents.cache_clear()
        yield mock


def _params(**overrides) -> dict:
    return {
        "featured": False,
        "creator": None,
        "sorted_by": None,
        "search_query": "test",
        "category": None,
        "page": 1,
        "page_size": 20,
        **overrides,
    }


def _pagination(total_items: int) -> Pagination:
    return Pagination(
        total_items=total_items, total_pages=1, current_page=1, page_size=20
    )
