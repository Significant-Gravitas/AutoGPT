"""The public store search cache, against the real Redis."""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.search import embeddings, rate_limit
from backend.util import cache as cache_util
from backend.util.cache_test_helpers import callers_in_flight

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
        patch.object(
            cache_util,
            "_get_async_redis",
            AsyncMock(side_effect=ConnectionError("down")),
        ),
    ):
        response = await _search(f"sheets {uuid.uuid4().hex}")

    assert [agent.slug for agent in response.agents] == ["sheets-agent"]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_burst_of_distinct_searches_runs_a_capped_number_of_queries_at_once(
    monkeypatch: pytest.MonkeyPatch,
):
    """The public store search runs a database query for each search it
    hasn't cached, and anyone can send distinct ones. The cache computes
    misses of different searches side by side, but at most
    ``MAX_CONCURRENT_MISSES`` at once per process.

    Redis here answers at once and keeps nothing, so no read suspends: every
    search has reached its query or its turn before the test looks."""
    monkeypatch.setattr(cache_util, "_get_async_redis", _a_redis_that_keeps_nothing)
    cap = cache_util.MAX_CONCURRENT_MISSES
    searches = 3 * cap
    everyone_in = callers_in_flight(monkeypatch, searches)
    at_the_cap, finish = asyncio.Event(), asyncio.Event()
    running = peak = 0

    async def slow_query(**kwargs):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        if running == cap:
            at_the_cap.set()
        await finish.wait()
        running -= 1
        return {"agents": [], "search_query": kwargs["search_query"]}

    queries = [f"sheets {uuid.uuid4().hex}" for _ in range(searches)]
    with patch.object(store_cache.store_db, "get_store_agents", slow_query):
        calls = [
            asyncio.create_task(
                store_cache.get_cached_store_agents(
                    featured=False,
                    creator=None,
                    sorted_by=None,
                    search_query=query,
                    category=None,
                    page=1,
                    page_size=20,
                    public_search=True,
                )
            )
            for query in queries
        ]
        try:
            async with asyncio.timeout(10):
                await everyone_in.wait()
                await at_the_cap.wait()
                # Every search has missed: the cap's worth query, the rest wait.
                assert running == cap
                finish.set()
                results = await asyncio.gather(*calls)
        finally:
            finish.set()
            async with asyncio.timeout(10):
                await asyncio.gather(*calls, return_exceptions=True)

    assert [result["search_query"] for result in results] == queries
    assert peak == cap


class _RedisThatKeepsNothing:
    async def get(self, key: str) -> None:
        return None

    async def getex(self, key: str, ex: int | None = None) -> None:
        return None

    async def set(self, key: str, value: bytes, ex: int | None = None) -> bool:
        return True


async def _a_redis_that_keeps_nothing() -> _RedisThatKeepsNothing:
    return _RedisThatKeepsNothing()


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


def test_closing_or_keeping_an_account_clears_the_store_caches(mocker):
    clear = mocker.patch.object(store_cache, "clear_all_caches")

    store_cache.forget_store_listings()

    clear.assert_called_once_with()


def test_a_cache_that_cannot_be_cleared_is_only_logged(mocker):
    mocker.patch.object(
        store_cache, "clear_all_caches", side_effect=ConnectionError("redis down")
    )

    store_cache.forget_store_listings()
