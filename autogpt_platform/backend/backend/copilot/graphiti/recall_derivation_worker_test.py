"""Unit tests for the ingestion worker's side of ``recall_derivation``
(``marked_write.py``, called from ``ingest._write_locked``): a dream write's
citations are marked first, under the uuid drawn for its episode, failing
closed, and only then checked against forgets (one resting on a forget is
dropped and its marker withdrawn); the episode is placed under that uuid,
``write_pending``, and graphiti's ``add_episode`` handed it; the write is
recorded after it, all under the graph's write lock. A record that fails is
noted for the reaper and reported as ``provenance_pending``; a marked write
that raised marks its marker aborted, is noted and counts as failed. The
record itself is pinned in ``recall_derivation_test.py``.
"""

import asyncio
from collections.abc import Callable
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


def _landed(edges: Callable[[str], dict[str, list[str]]]) -> Callable:
    """graphiti's ``add_episode``: the episode it was handed, with ``edges``."""

    async def add_episode(**kwargs) -> SimpleNamespace:
        uuid = kwargs["uuid"] or "graphiti-drawn"
        return _written(uuid, edges(uuid))

    return add_episode


class _Worker(SimpleNamespace):
    """What the worker was handed and what it called, in ``order``."""

    add_episode: AsyncMock
    mark: AsyncMock
    record: AsyncMock
    abort: AsyncMock
    withdraw: AsyncMock
    noted: AsyncMock
    queries: AsyncMock
    order: list[str]


def _tracked(name: str, order: list[str], mock: AsyncMock) -> AsyncMock:
    """``mock``, noting ``name`` in ``order`` each time it is awaited."""

    async def run(*args, **kwargs) -> object:
        order.append(name)
        return await mock(*args, **kwargs)

    return AsyncMock(side_effect=run)


async def _work(
    payload: dict,
    written: Callable | Exception,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mark: AsyncMock | None = None,
    record: AsyncMock | None = None,
    forgotten: str | None = None,
) -> _Worker:
    """One payload through the worker, marking through ``mark``, recording
    through ``record``, and finding the write resting on a forget when
    ``forgotten`` says why."""
    order: list[str] = []
    client = MagicMock()

    async def check(driver, citations: Citations | None) -> str | None:
        if citations is None:
            return None
        order.append("check")
        return forgotten

    async def add_episode(**kwargs):
        order.append("add_episode")
        if isinstance(written, Exception):
            raise written
        return await written(**kwargs)

    async def query(cypher: str, **params):
        if cypher == marked_write.PLACE_EPISODE_QUERY:
            order.append("place")
        return [{"facts": ["f1"], "episodes": []}], [], None

    client.add_episode = AsyncMock(side_effect=add_episode)
    client.driver.execute_query = AsyncMock(side_effect=query)
    worker = _Worker(
        add_episode=client.add_episode,
        mark=_tracked("mark", order, mark or AsyncMock(return_value="m1")),
        record=_tracked("record", order, record or AsyncMock(return_value=True)),
        abort=_tracked("abort", order, AsyncMock()),
        withdraw=_tracked("withdraw", order, AsyncMock()),
        noted=AsyncMock(),
        queries=client.driver.execute_query,
        order=order,
    )
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    queue.put_nowait(payload)
    monkeypatch.setattr(ingest, "_WORKER_IDLE_TIMEOUT", 0.05)
    with (
        patch.object(ingest, "get_graphiti_client", AsyncMock(return_value=client)),
        patch.object(ingest, "ensure_indices_once", AsyncMock()),
        patch.object(ingest, "previous_episode_uuids", AsyncMock(return_value=[])),
        patch.object(ingest, "rests_on_a_forget", AsyncMock(side_effect=check)),
        patch.object(marked_write, "mark", worker.mark),
        patch.object(marked_write, "record", worker.record),
        patch.object(marked_write, "abort", worker.abort),
        patch.object(marked_write, "withdraw", worker.withdraw),
        patch.object(marked_write, "note_pending", worker.noted),
    ):
        await ingest._ingestion_worker("test-user", "user_test", queue)
    return worker


class TestTheWorker:
    @pytest.mark.asyncio
    async def test_marks_checks_places_writes_and_records_under_one_uuid(
        self, lock_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An edge the episode invalidated does not name it and is left out."""
        held: list[str] = []
        cited = Citations(fact_uuids=["f1"])

        def step(name: str, answer: object):
            async def run(*args, **kwargs) -> object:
                locked = write_lock_key("user_test") in lock_redis.values
                held.append(f"{name}:{'locked' if locked else 'unlocked'}")
                return answer

            return run

        worker = await _work(
            _payload(cited),
            _landed(
                lambda uuid: {"new": [uuid], "merged": ["user-ep", uuid], "old": ["x"]}
            ),
            monkeypatch,
            mark=AsyncMock(side_effect=step("mark", "m1")),
            record=AsyncMock(side_effect=step("record", True)),
        )

        assert held == ["mark:locked", "record:locked"]
        _, group, episode, name, citations = worker.mark.await_args.args
        assert (group, name, citations) == (
            "user_test",
            "dream_p1_consolidate_000",
            cited,
        )
        assert worker.order == ["mark", "check", "place", "add_episode", "record"]
        [place] = [
            call
            for call in worker.queries.await_args_list
            if call.args[0] == marked_write.PLACE_EPISODE_QUERY
        ]
        assert (place.kwargs["uuid"], place.kwargs["group_id"]) == (
            episode,
            "user_test",
        )
        assert place.kwargs["content"] == '{"content": "Alice works on Atlas"}'
        assert worker.add_episode.await_args.kwargs["uuid"] == episode
        _, group, marker, recorded, touched, citations = worker.record.await_args.args
        assert (group, marker, recorded) == ("user_test", "m1", episode)
        assert (touched, citations) == (["new", "merged"], cited)
        worker.noted.assert_not_awaited()
        worker.abort.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_marker_that_cannot_be_written_drops_the_write(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        completion = ingest.IngestionCompletion()
        completion.register()

        worker = await _work(
            _payload(Citations(fact_uuids=["f1"]), completion),
            _landed(lambda uuid: {"new": [uuid]}),
            monkeypatch,
            mark=AsyncMock(side_effect=RuntimeError("down")),
        )

        worker.add_episode.assert_not_awaited()
        worker.record.assert_not_awaited()
        assert worker.order == ["mark"], "nothing checked or placed"
        assert (completion.failed, completion.provenance_pending) == (1, 0)
        assert await completion.wait(0), "the drain is not held up"

    @pytest.mark.asyncio
    async def test_a_write_resting_on_a_forget_is_dropped_its_marker_withdrawn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Marked before the check, so a forget landing after the check
        (where the lock does not hold) finds the marker."""
        completion = ingest.IngestionCompletion()
        completion.register()

        worker = await _work(
            _payload(Citations(fact_uuids=["f1"]), completion),
            _landed(lambda uuid: {"new": [uuid]}),
            monkeypatch,
            forgotten="cites a fact that is forgotten or gone",
        )

        assert worker.order == ["mark", "check", "withdraw"]
        assert worker.withdraw.await_args.args[1] == "m1"
        worker.add_episode.assert_not_awaited()
        assert (completion.dropped_forgotten, completion.failed) == (1, 0)

    @pytest.mark.asyncio
    async def test_a_failed_record_is_noted_and_reported_pending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        completion = ingest.IngestionCompletion()
        completion.register()

        worker = await _work(
            _payload(Citations(fact_uuids=["f1"]), completion),
            _landed(lambda uuid: {"new": [uuid]}),
            monkeypatch,
            record=AsyncMock(return_value=False),
        )

        worker.noted.assert_awaited_once_with("user_test")
        assert (completion.failed, completion.provenance_pending) == (0, 1)

    @pytest.mark.asyncio
    async def test_a_marked_write_that_raised_is_aborted_noted_and_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It may have landed in part: reconcile resolves its marker once it
        has found no episode under its uuid."""
        completion = ingest.IngestionCompletion()
        completion.register()

        worker = await _work(
            _payload(Citations(fact_uuids=["f1"]), completion),
            RuntimeError("graphiti down"),
            monkeypatch,
        )

        assert worker.order == ["mark", "check", "place", "add_episode", "abort"]
        assert worker.abort.await_args.args[1] == "m1"
        worker.noted.assert_awaited_once_with("user_test")
        worker.record.assert_not_awaited()
        assert (completion.failed, completion.provenance_pending) == (1, 0)

    @pytest.mark.asyncio
    async def test_a_chat_write_marks_places_and_records_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker = await _work(
            _payload(None),
            _landed(lambda uuid: {"new": [uuid]}),
            monkeypatch,
        )

        worker.mark.assert_not_awaited()
        worker.record.assert_not_awaited()
        assert worker.order == ["add_episode"]
        assert worker.add_episode.await_args.kwargs["uuid"] is None
