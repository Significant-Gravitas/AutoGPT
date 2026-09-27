"""The ingestion worker hands graphiti's extraction only the earlier episodes
the recall policy allows, and writes each episode holding the graph's write
lock, mocked at the graphiti boundary and on the in-memory Redis.

The live runs are ``ingest_recall_integration_test.py`` and
``recall_inflight_integration_test.py``.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_utils import RELEVANT_SCHEMA_LIMIT

from . import ingest, scope_lock
from .recall import is_recallable_episode, recallable_episode_predicate
from .recall_fake_redis import FakeRedis
from .scope import write_lock_key

_NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
_LOCK = write_lock_key("user_test")
# What the episode store holds, newest first as the query orders them.
_STORED = [
    {"uuid": "ep-newest", "redacted": False, "entity_edges": ["kept"]},
    {"uuid": "ep-redacted", "redacted": True, "entity_edges": []},
    {"uuid": "ep-cites-a-forgotten-fact", "redacted": False, "entity_edges": ["gone"]},
    {"uuid": "ep-oldest", "redacted": False, "entity_edges": []},
]


async def _policy_filtered_store(query: str, **params: object):
    """The graph's answer to a recallable-episodes read, filtered with the
    predicate's Python twin; any other read would be unfiltered."""
    assert recallable_episode_predicate("e") in query, "an unfiltered episode read"
    rows = [
        {"uuid": episode["uuid"]}
        for episode in _STORED
        if is_recallable_episode(
            episode["entity_edges"], {"gone"}, redacted=episode["redacted"]
        )
    ]
    return rows, [], None


def _payload(**extra: object) -> dict:
    return {
        "name": "conversation_s1",
        "episode_body": "User: Bob leads Atlas",
        "source": EpisodeType.message,
        "source_description": "User message in session s1",
        "reference_time": _NOW,
        "group_id": "user_test",
        **extra,
    }


async def _run_worker(
    client: MagicMock, monkeypatch: pytest.MonkeyPatch, *payloads: dict
) -> None:
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    for payload in payloads or (_payload(),):
        queue.put_nowait(payload)
    monkeypatch.setattr(ingest, "_WORKER_IDLE_TIMEOUT", 0.05)
    with (
        patch.object(ingest, "get_graphiti_client", AsyncMock(return_value=client)),
        patch.object(ingest, "ensure_indices_once", AsyncMock()),
    ):
        await ingest._ingestion_worker("test-user", "user_test", queue)


def _graphiti_client() -> MagicMock:
    client = MagicMock()
    client.add_episode = AsyncMock()
    client.driver.execute_query = AsyncMock(side_effect=_policy_filtered_store)
    return client


class TestExtractionContext:
    @pytest.mark.asyncio
    async def test_add_episode_gets_only_recallable_earlier_episodes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _graphiti_client()

        await _run_worker(client, monkeypatch)

        client.add_episode.assert_awaited_once()
        assert client.add_episode.await_args is not None
        previous = client.add_episode.await_args.kwargs["previous_episode_uuids"]
        assert previous == ["ep-oldest", "ep-newest"], "oldest first, as graphiti"
        assert "ep-redacted" not in previous
        [read] = client.driver.execute_query.await_args_list
        assert read.kwargs == {
            "group_id": "user_test",
            "reference_time": _NOW,
            "source": "message",
            "limit": RELEVANT_SCHEMA_LIMIT,
        }

    @pytest.mark.asyncio
    async def test_a_failed_read_still_writes_without_earlier_episodes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``None`` would let graphiti pick for itself, forgotten episodes
        included; the episode is still written."""
        client = _graphiti_client()
        client.driver.execute_query = AsyncMock(side_effect=RuntimeError("down"))

        await _run_worker(client, monkeypatch)

        assert client.add_episode.await_args is not None
        assert client.add_episode.await_args.kwargs["previous_episode_uuids"] == []


class TestWriteLock:
    @pytest.mark.asyncio
    async def test_the_episode_is_written_holding_the_graphs_lock(
        self, lock_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _graphiti_client()
        held: list[bool] = []
        client.add_episode.side_effect = lambda **_: held.append(
            _LOCK in lock_redis.values
        )

        await _run_worker(client, monkeypatch)

        assert held == [True]
        assert _LOCK not in lock_redis.values, "released afterwards"

    @pytest.mark.asyncio
    async def test_a_graph_a_forget_keeps_locked_is_tried_again_once(
        self, lock_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The first attempt writes nothing; the forget finishes and the
        episode, back at the end of the queue, is written, and only then
        does the enqueuer's completion count it."""
        client = _graphiti_client()
        lock_redis.values[_LOCK] = "a forget's token"
        completion = ingest.IngestionCompletion()
        completion.register()
        requeue = ingest._requeue_once

        def forget_finishes(*args, **kwargs) -> bool:
            assert completion.registered == 1 and not client.add_episode.called
            del lock_redis.values[_LOCK]
            return requeue(*args, **kwargs)

        monkeypatch.setattr(ingest, "INGEST_LOCK_WAIT_SECONDS", 0)
        monkeypatch.setattr(ingest, "_requeue_once", forget_finishes)
        await _run_worker(client, monkeypatch, _payload(_completion=completion))

        client.add_episode.assert_awaited_once()
        assert "_lock_retried" not in client.add_episode.await_args.kwargs
        assert await completion.wait(0)

    @pytest.mark.asyncio
    async def test_a_graph_locked_for_both_waits_drops_the_episode(
        self,
        lock_redis: FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        client = _graphiti_client()
        lock_redis.values[_LOCK] = "a forget's token"
        completion = ingest.IngestionCompletion()
        completion.register()

        monkeypatch.setattr(ingest, "INGEST_LOCK_WAIT_SECONDS", 0)
        await _run_worker(client, monkeypatch, _payload(_completion=completion))

        client.add_episode.assert_not_called()
        assert "requeued once" in caplog.text and "dropping" in caplog.text
        assert await completion.wait(0), "a dropped episode is no longer pending"

    @pytest.mark.asyncio
    async def test_without_redis_the_episode_is_written_anyway(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _graphiti_client()
        down = AsyncMock(side_effect=ConnectionError("redis down"))

        with patch.object(scope_lock, "get_redis_async", down):
            await _run_worker(client, monkeypatch)

        client.add_episode.assert_awaited_once()
        assert "write lock unavailable" in caplog.text
