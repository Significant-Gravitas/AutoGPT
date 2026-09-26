"""The ingestion worker hands graphiti's extraction only the earlier episodes
the recall policy allows, and hands its result to ``recall_ingest`` to keep
every forget, mocked at the graphiti boundary.

The live runs are ``ingest_recall_integration_test.py`` and
``recall_ingest_integration_test.py``; ``recall_ingest_test.py`` pins the
repair itself.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_utils import RELEVANT_SCHEMA_LIMIT

from . import ingest, recall_ingest_plan
from .recall import is_recallable_episode, recallable_episode_predicate

_NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
# What the episode store holds, newest first as the query orders them.
_STORED = [
    {"uuid": "ep-newest", "redacted": False, "entity_edges": ["kept"]},
    {"uuid": "ep-redacted", "redacted": True, "entity_edges": []},
    {"uuid": "ep-cites-a-forgotten-fact", "redacted": False, "entity_edges": ["gone"]},
    {"uuid": "ep-oldest", "redacted": False, "entity_edges": []},
]


async def _policy_filtered_store(query: str, **params: object):
    """The graph's answer to a recallable-episodes read, filtered with the
    predicate's Python twin, or to the forgotten-fact snapshot (none here);
    any other read would be unfiltered."""
    if query == recall_ingest_plan._SNAPSHOT_QUERY:
        return [], [], None
    assert recallable_episode_predicate("e") in query, "an unfiltered episode read"
    rows = [
        {"uuid": episode["uuid"]}
        for episode in _STORED
        if is_recallable_episode(
            episode["entity_edges"], {"gone"}, redacted=episode["redacted"]
        )
    ]
    return rows, [], None


async def _run_worker_on_one_episode(
    client: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    queue.put_nowait(
        {
            "name": "conversation_s1",
            "episode_body": "User: Bob leads Atlas",
            "source": EpisodeType.message,
            "source_description": "User message in session s1",
            "reference_time": _NOW,
            "group_id": "user_test",
        }
    )
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

        await _run_worker_on_one_episode(client, monkeypatch)

        client.add_episode.assert_awaited_once()
        assert client.add_episode.await_args is not None
        previous = client.add_episode.await_args.kwargs["previous_episode_uuids"]
        assert previous == ["ep-oldest", "ep-newest"], "oldest first, as graphiti"
        assert "ep-redacted" not in previous
        read, snapshot = client.driver.execute_query.await_args_list
        assert snapshot.args == (recall_ingest_plan._SNAPSHOT_QUERY,)
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

        await _run_worker_on_one_episode(client, monkeypatch)

        assert client.add_episode.await_args is not None
        assert client.add_episode.await_args.kwargs["previous_episode_uuids"] == []

    @pytest.mark.asyncio
    async def test_the_result_goes_through_the_forget_repair(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What graphiti wrote is checked against the snapshot taken before
        it ran, with the same earlier episodes and extraction instructions,
        and with when the episode was said and when its ingestion began."""
        client = _graphiti_client()
        keep = AsyncMock()
        monkeypatch.setattr(ingest, "keep_forgotten", keep)
        before = datetime.now(timezone.utc)

        await _run_worker_on_one_episode(client, monkeypatch)

        keep.assert_awaited_once()
        assert keep.await_args is not None
        passed, run, snapshot, result = keep.await_args.args
        assert (passed, snapshot) == (client, {})
        assert result is client.add_episode.return_value
        assert (run.group_id, run.reference_time) == ("user_test", _NOW)
        assert (run.previous, run.instructions) == (["ep-oldest", "ep-newest"], None)
        assert run.started_at >= before, "stamped before add_episode ran"
