"""The public store search cache, against the real Redis."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.search import embeddings, rate_limit
from backend.util import cache as cache_util

from . import cache as store_cache
from . import hybrid_search as store_hybrid_search


@pytest.mark.asyncio(loop_scope="session")
async def test_a_search_ranked_by_text_alone_is_recomputed_and_a_full_one_cached():
    # Unique per run: the cache is the real Redis, shared with later runs.
    query = f"sheets {uuid.uuid4().hex}"
    rows = AsyncMock(return_value=[_agent_row()])
    slot = AsyncMock(return_value=False)
    vector = AsyncMock(return_value=[0.1] * embeddings.EMBEDDING_DIM)
    with (
        patch.object(store_hybrid_search, "query_raw_with_schema", rows),
        patch.object(embeddings, "take_public_search_embedding_slot", slot),
        patch.object(embeddings, "generate_embedding", vector),
    ):
        for _ in range(2):
            await _search(query)
        assert rows.await_count == 2

        slot.return_value = True
        for _ in range(2):
            await _search(query)
        assert rows.await_count == 3


@pytest.mark.asyncio(loop_scope="session")
async def test_with_the_embedding_budget_off_a_text_ranked_search_is_cached():
    rows = AsyncMock(return_value=[_agent_row()])
    with (
        patch.object(store_hybrid_search, "query_raw_with_schema", rows),
        patch.object(
            rate_limit.settings.config, "public_search_embeddings_per_minute", 0
        ),
    ):
        query = f"sheets {uuid.uuid4().hex}"
        for _ in range(2):
            await _search(query)

    assert rows.await_count == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_a_search_ranked_by_text_alone_answers_while_the_cache_is_unreachable():
    with (
        patch.object(
            store_hybrid_search,
            "query_raw_with_schema",
            AsyncMock(return_value=[_agent_row()]),
        ),
        patch.object(
            embeddings,
            "take_public_search_embedding_slot",
            AsyncMock(return_value=False),
        ),
        patch.object(
            cache_util, "_get_redis", MagicMock(side_effect=ConnectionError("down"))
        ),
    ):
        response = await _search(f"sheets {uuid.uuid4().hex}")

    assert [agent.slug for agent in response.agents] == ["sheets-agent"]


async def _search(query: str):
    return await store_cache.get_public_store_agents(
        featured=False,
        creator=None,
        sorted_by=None,
        search_query=query,
        category=None,
        page=1,
        page_size=20,
    )


def _agent_row() -> dict:
    return {
        "slug": "sheets-agent",
        "agent_name": "Sheets Agent",
        "agent_image": ["sheets.png"],
        "creator_username": "creator",
        "creator_avatar": "avatar.png",
        "sub_heading": "Works with sheets",
        "description": "Reads and writes sheets",
        "runs": 10,
        "rating": 4.5,
        "categories": ["data"],
        "featured": False,
        "is_available": True,
        "updated_at": "2025-01-01T00:00:00Z",
        "semantic_score": 0.0,
        "lexical_score": 0.5,
        "category_score": 0.0,
        "recency_score": 0.1,
        "popularity_score": 0.2,
        "combined_score": 0.3,
        "total_count": 1,
    }
