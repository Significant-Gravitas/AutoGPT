"""Unit tests for the ingestion worker's side of ``recall_derivation``
(``marked_write.py``, called from ``ingest._write_locked``): a dream write's
citations are marked before the write, failing closed, and recorded after
it, both under the graph's write lock; a record that fails is noted for the
reaper and reported as ``provenance_pending``, and a marked write that
raised as failed. The record itself is pinned in
``recall_derivation_test.py``.
"""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.nodes import EpisodeType

from . import ingest, marked_write
from .recall_citations import Citations
from .recall_fake_redis import FakeRedis
from .scope import write_lock_key


def _payload(citations: Citations | None, completion=None) -> dict:
    return {
        "name": "dream_p1_consolidate_000",
        "episode_body": '{"content": "Alice works on Atlas"}',
        "source": EpisodeType.json,
        "source_description": "dream-pass consolidation",
        "reference_time": datetime(2026, 9, 28, tzinfo=timezone.utc),
        "group_id": "user_test",
        "_citations": citations,
        "_completion": completion,
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


class _Worker(SimpleNamespace):
    """What the worker was handed and what it called."""

    add_episode: AsyncMock
    mark: AsyncMock
    record: AsyncMock
    noted: AsyncMock


async def _work(
    payload: dict,
    written: SimpleNamespace | Exception,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mark: AsyncMock | None = None,
    record: AsyncMock | None = None,
) -> _Worker:
    """One payload through the worker, marking through ``mark`` and
    recording through ``record``."""
    client = MagicMock()
    client.add_episode = AsyncMock(side_effect=[written])
    client.driver.execute_query = AsyncMock(
        return_value=([{"facts": ["f1"], "episodes": []}], [], None)
    )
    worker = _Worker(
        add_episode=client.add_episode,
        mark=mark or AsyncMock(return_value="m1"),
        record=record or AsyncMock(return_value=True),
        noted=AsyncMock(),
    )
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    queue.put_nowait(payload)
    monkeypatch.setattr(ingest, "_WORKER_IDLE_TIMEOUT", 0.05)
    with (
        patch.object(ingest, "get_graphiti_client", AsyncMock(return_value=client)),
        patch.object(ingest, "ensure_indices_once", AsyncMock()),
        patch.object(ingest, "previous_episode_uuids", AsyncMock(return_value=[])),
        patch.object(marked_write, "mark", worker.mark),
        patch.object(marked_write, "record", worker.record),
        patch.object(marked_write, "note_pending", worker.noted),
    ):
        await ingest._ingestion_worker("test-user", "user_test", queue)
    return worker


class TestTheWorker:
    @pytest.mark.asyncio
    async def test_marks_before_the_write_and_records_after_it_under_the_lock(
        self, lock_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An edge the episode invalidated does not name it and is left out."""
        held: list[str] = []
        cited = Citations(fact_uuids=["f1"])
        written = _written(
            "dream-ep",
            {"new": ["dream-ep"], "merged": ["user-ep", "dream-ep"], "old": ["x"]},
        )

        def step(name: str, answer: object):
            async def run(*args, **kwargs) -> object:
                locked = write_lock_key("user_test") in lock_redis.values
                held.append(f"{name}:{'locked' if locked else 'unlocked'}")
                return answer

            return run

        worker = await _work(
            _payload(cited),
            written,
            monkeypatch,
            mark=AsyncMock(side_effect=step("mark", "m1")),
            record=AsyncMock(side_effect=step("record", True)),
        )

        assert held == ["mark:locked", "record:locked"]
        _, group, name, citations = worker.mark.await_args.args
        assert (group, name, citations) == (
            "user_test",
            "dream_p1_consolidate_000",
            cited,
        )
        _, group, marker, episode, touched, citations = worker.record.await_args.args
        assert (group, marker, episode) == ("user_test", "m1", "dream-ep")
        assert (touched, citations) == (["new", "merged"], cited)
        worker.noted.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_marker_that_cannot_be_written_drops_the_write(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        completion = ingest.IngestionCompletion()
        completion.register()

        worker = await _work(
            _payload(Citations(fact_uuids=["f1"]), completion),
            _written("dream-ep", {"new": ["dream-ep"]}),
            monkeypatch,
            mark=AsyncMock(side_effect=RuntimeError("down")),
        )

        worker.add_episode.assert_not_awaited()
        worker.record.assert_not_awaited()
        assert (completion.failed, completion.provenance_pending) == (1, 0)
        assert await completion.wait(0), "the drain is not held up"

    @pytest.mark.asyncio
    async def test_a_failed_record_is_noted_and_reported_pending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        completion = ingest.IngestionCompletion()
        completion.register()

        worker = await _work(
            _payload(Citations(fact_uuids=["f1"]), completion),
            _written("dream-ep", {"new": ["dream-ep"]}),
            monkeypatch,
            record=AsyncMock(return_value=False),
        )

        worker.noted.assert_awaited_once_with("user_test")
        assert (completion.failed, completion.provenance_pending) == (0, 1)

    @pytest.mark.asyncio
    async def test_a_marked_write_that_raised_is_noted_and_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It may have landed in part: its marker stays for the reaper."""
        completion = ingest.IngestionCompletion()
        completion.register()

        worker = await _work(
            _payload(Citations(fact_uuids=["f1"]), completion),
            RuntimeError("graphiti down"),
            monkeypatch,
        )

        worker.noted.assert_awaited_once_with("user_test")
        worker.record.assert_not_awaited()
        assert (completion.failed, completion.provenance_pending) == (1, 0)

    @pytest.mark.asyncio
    async def test_a_chat_write_marks_and_records_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker = await _work(
            _payload(None),
            _written("chat-ep", {"new": ["chat-ep"]}),
            monkeypatch,
        )

        worker.mark.assert_not_awaited()
        worker.record.assert_not_awaited()
        worker.add_episode.assert_awaited_once()
