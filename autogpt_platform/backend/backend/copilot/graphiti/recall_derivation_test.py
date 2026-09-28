"""Unit tests for ``recall_derivation``: what a dream write rests on is
recorded on its episode, then on the facts only dream episodes state, right
after the ingestion worker writes it, under the graph's write lock.

On FalkorDB, through ``dream/apply.py`` and the production worker, the facts
a dream write produced or merged into are checked in
``recall_cascade_integration_test.py``.
"""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.nodes import EpisodeType

from . import ingest, recall_derivation
from .recall_citations import Citations
from .recall_fake_redis import FakeRedis
from .scope import write_lock_key

_CITED = Citations(fact_uuids=["f1", "f1", "f2"], episode_uuids=["ep1"])


def _driver() -> MagicMock:
    driver = MagicMock()
    driver.execute_query = AsyncMock(return_value=([], [], None))
    return driver


class TestRecord:
    @pytest.mark.asyncio
    async def test_records_the_episode_then_the_facts_it_touched(self) -> None:
        driver = _driver()

        await recall_derivation.record(driver, "user_a", "dream-ep", ["e1"], _CITED)

        episode, facts = driver.execute_query.await_args_list
        assert episode.args[0] == recall_derivation.RECORD_EPISODE_QUERY
        assert episode.kwargs == {
            "episode": "dream-ep",
            "facts": ["f1", "f2"],
            "episodes": ["ep1"],
        }
        assert facts.args[0] == recall_derivation.STAMP_FACTS_QUERY
        assert facts.kwargs == {"uuids": ["e1"], "group_id": "user_a"}

    @pytest.mark.asyncio
    async def test_a_write_that_touched_no_fact_records_its_episode_only(
        self,
    ) -> None:
        driver = _driver()

        await recall_derivation.record(driver, "user_a", "dream-ep", [], _CITED)

        [call] = driver.execute_query.await_args_list
        assert call.args[0] == recall_derivation.RECORD_EPISODE_QUERY

    @pytest.mark.asyncio
    async def test_a_failure_is_logged_and_the_write_stands(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        driver = _driver()
        driver.execute_query.side_effect = RuntimeError("down")

        await recall_derivation.record(driver, "user_a", "dream-ep", ["e1"], _CITED)

        assert "Failed to record what dream episode dream-ep" in caplog.text


class TestTheStamp:
    def test_a_fact_is_stamped_only_when_every_source_is_a_recorded_episode(
        self,
    ) -> None:
        query = recall_derivation.STAMP_FACTS_QUERY
        assert "all(x IN coalesce(e.episodes, []) WHERE x IN found)" in query
        assert "all(s IN sources WHERE s.derived_from_facts IS NOT NULL)" in query
        assert "e.forgotten_at IS NULL" in query

    def test_the_stamp_is_the_union_of_its_sources_records(self) -> None:
        query = recall_derivation.STAMP_FACTS_QUERY
        assert "acc + [x IN s.derived_from_facts WHERE NOT x IN acc]" in query
        assert "SET e.derived_from_facts = reduce(" in query


def _payload(citations: Citations | None) -> dict:
    return {
        "name": "dream_p1_consolidate_000",
        "episode_body": '{"content": "Alice works on Atlas"}',
        "source": EpisodeType.json,
        "source_description": "dream-pass consolidation",
        "reference_time": datetime(2026, 9, 28, tzinfo=timezone.utc),
        "group_id": "user_test",
        "_citations": citations,
        "_completion": None,
    }


def _written(episode: str, edges: dict[str, list[str]]) -> SimpleNamespace:
    """What ``add_episode`` returned: its episode and each edge's episodes."""
    return SimpleNamespace(
        episode=SimpleNamespace(uuid=episode),
        edges=[
            SimpleNamespace(uuid=uuid, episodes=episodes, expired_at=None)
            for uuid, episodes in edges.items()
        ],
    )


async def _work(
    payload: dict,
    written: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    record: AsyncMock,
) -> None:
    """One payload through the worker, recording through ``record``."""
    client = MagicMock()
    client.add_episode = AsyncMock(return_value=written)
    client.driver.execute_query = AsyncMock(
        return_value=([{"facts": ["f1"], "episodes": []}], [], None)
    )
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    queue.put_nowait(payload)
    monkeypatch.setattr(ingest, "_WORKER_IDLE_TIMEOUT", 0.05)
    with (
        patch.object(ingest, "get_graphiti_client", AsyncMock(return_value=client)),
        patch.object(ingest, "ensure_indices_once", AsyncMock()),
        patch.object(ingest, "previous_episode_uuids", AsyncMock(return_value=[])),
        patch.object(ingest, "record_derivation", record),
    ):
        await ingest._ingestion_worker("test-user", "user_test", queue)


class TestTheWorker:
    @pytest.mark.asyncio
    async def test_records_a_dream_write_on_the_facts_it_produced_or_merged_into(
        self, lock_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An edge the episode invalidated does not name it and is left out;
        the record is made while the write still holds the lock."""
        held: list[bool] = []
        cited = Citations(fact_uuids=["f1"])
        written = _written(
            "dream-ep",
            {"new": ["dream-ep"], "merged": ["user-ep", "dream-ep"], "old": ["x"]},
        )

        async def under_the_lock(*args, **kwargs) -> None:
            held.append(write_lock_key("user_test") in lock_redis.values)

        recorded = AsyncMock(side_effect=under_the_lock)
        await _work(_payload(cited), written, monkeypatch, recorded)

        recorded.assert_awaited_once()
        _, group, episode, touched, citations = recorded.await_args.args
        assert (group, episode, touched) == ("user_test", "dream-ep", ["new", "merged"])
        assert citations == cited
        assert held == [True]

    @pytest.mark.asyncio
    async def test_a_chat_write_records_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        written = _written("chat-ep", {"new": ["chat-ep"]})
        recorded = AsyncMock()

        await _work(_payload(None), written, monkeypatch, recorded)

        recorded.assert_not_awaited()
