"""Unit tests for ``recall_derivation``: a dream write's citations are marked
before the ingestion worker writes it, failing closed, then recorded on its
episode and on the facts only dream episodes state, and the marker dropped,
all under the graph's write lock; a record that fails leaves the marker and
is reported as ``provenance_pending``.

On FalkorDB, through ``dream/apply.py`` and the production worker:
``recall_cascade_integration_test.py`` (the facts a dream write produced or
merged into) and ``recall_provenance_integration_test.py`` (a record that
failed, reconciled before the next forget).
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


class TestMark:
    @pytest.mark.asyncio
    async def test_marks_the_complete_citations_under_the_episodes_name(
        self,
    ) -> None:
        driver = _driver()

        marker = await recall_derivation.mark(driver, "user_a", "dream_p_1", _CITED)

        [call] = driver.execute_query.await_args_list
        assert call.args[0] == recall_derivation.MARK_QUERY
        params = dict(call.kwargs)
        assert params.pop("now")
        assert params == {
            "uuid": marker,
            "group_id": "user_a",
            "name": "dream_p_1",
            "facts": ["f1", "f2"],
            "episodes": ["ep1"],
        }

    @pytest.mark.asyncio
    async def test_a_marker_that_cannot_be_written_raises(self) -> None:
        driver = _driver()
        driver.execute_query.side_effect = RuntimeError("down")

        with pytest.raises(RuntimeError):
            await recall_derivation.mark(driver, "user_a", "dream_p_1", _CITED)

    def test_a_marker_holds_uuids_and_names_only(self) -> None:
        query = recall_derivation.MARK_QUERY
        assert "CREATE (:DreamCitations {" in query
        for text in ("content", "fact:", "rationale", "source_description"):
            assert text not in query


class TestRecord:
    @pytest.mark.asyncio
    async def test_records_the_episode_then_the_facts_then_drops_the_marker(
        self,
    ) -> None:
        driver = _driver()

        recorded = await recall_derivation.record(
            driver, "user_a", "m1", "dream-ep", ["e1"], _CITED
        )

        episode, facts, drop = driver.execute_query.await_args_list
        assert episode.args[0] == recall_derivation.RECORD_EPISODE_QUERY
        assert episode.kwargs == {
            "episode": "dream-ep",
            "facts": ["f1", "f2"],
            "episodes": ["ep1"],
        }
        assert facts.args[0] == recall_derivation.STAMP_FACTS_QUERY
        assert facts.kwargs == {"uuids": ["e1"], "group_id": "user_a"}
        assert (drop.args[0], drop.kwargs) == (
            recall_derivation.DROP_MARKER_QUERY,
            {"uuid": "m1"},
        )
        assert recorded is True

    @pytest.mark.asyncio
    async def test_a_write_that_touched_no_fact_records_its_episode_only(
        self,
    ) -> None:
        driver = _driver()

        await recall_derivation.record(driver, "user_a", "m1", "dream-ep", [], _CITED)

        queries = [call.args[0] for call in driver.execute_query.await_args_list]
        assert queries == [
            recall_derivation.RECORD_EPISODE_QUERY,
            recall_derivation.DROP_MARKER_QUERY,
        ]

    @pytest.mark.asyncio
    async def test_a_failure_leaves_the_marker_and_says_so(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        driver = _driver()
        driver.execute_query.side_effect = [([], [], None), RuntimeError("down")]

        recorded = await recall_derivation.record(
            driver, "user_a", "m1", "dream-ep", ["e1"], _CITED
        )

        assert recorded is False
        assert driver.execute_query.await_count == 2, "the marker is not dropped"
        assert "its marker stays for reconcile" in caplog.text


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
        patch.object(ingest, "mark_derivation", worker.mark),
        patch.object(ingest, "record_derivation", worker.record),
        patch.object(ingest, "note_pending", worker.noted),
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
