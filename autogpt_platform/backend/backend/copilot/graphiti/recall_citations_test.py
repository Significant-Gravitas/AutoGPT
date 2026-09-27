"""Unit tests for ``recall_citations``: a dream write is checked against
forgets, by what it cites or, citing nothing, by its statement, and the
ingestion worker drops one that rests on a forget.

The live runs, through the real dream write helpers and the production
worker, are in ``recall_dream_citations_integration_test.py``.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.nodes import EpisodeType

from . import ingest, recall
from .recall_citations import Citations, rests_on_a_forget
from .recall_fake_redis import FakeRedis
from .scope import write_lock_key

_NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
_STATEMENT = "Alice works on Atlas"


def _driver(*answers: list[dict]) -> MagicMock:
    """A driver answering each query in turn with ``answers``' rows."""
    driver = MagicMock()
    driver.execute_query = AsyncMock(side_effect=[(rows, [], None) for rows in answers])
    return driver


def _still(facts: list[str], episodes: list[str]) -> list[dict]:
    return [{"facts": facts, "episodes": episodes}]


class TestRestsOnAForget:
    @pytest.mark.asyncio
    async def test_a_write_that_carries_no_citations_is_not_checked(self) -> None:
        driver = _driver()

        assert await rests_on_a_forget(driver, None) is None
        driver.execute_query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_everything_it_cites_still_there_lets_it_through(self) -> None:
        driver = _driver(_still(["f1", "f2"], ["ep1"]))
        cited = Citations(fact_uuids=["f1", "f2"], episode_uuids=["ep1"])

        assert await rests_on_a_forget(driver, cited) is None
        [call] = driver.execute_query.await_args_list
        query = call.args[0]
        assert query.startswith(recall.forgotten_facts_clause())
        assert "NOT (fact.uuid IN forgotten)" in query
        assert recall.recallable_episode_predicate("episode") in query
        assert call.kwargs == {"fact_uuids": ["f1", "f2"], "episode_uuids": ["ep1"]}

    @pytest.mark.parametrize(
        "still, reason",
        [
            (_still(["f1"], ["ep1"]), "cites a fact that is forgotten"),
            (_still(["f1", "f2"], []), "cites an episode that is hidden"),
            ([], "cites a fact that is forgotten"),
        ],
        ids=["a fact forgotten or deleted", "an episode hidden", "no answer"],
    )
    @pytest.mark.asyncio
    async def test_anything_it_cites_forgotten_or_gone_drops_it(
        self, still: list[dict], reason: str
    ) -> None:
        cited = Citations(fact_uuids=["f1", "f2"], episode_uuids=["ep1"])

        found = await rests_on_a_forget(_driver(still), cited)

        assert found is not None and found.startswith(reason)

    @pytest.mark.asyncio
    async def test_citing_nothing_its_statement_is_compared_with_forgotten_ones(
        self,
    ) -> None:
        driver = _driver([{"uuid": "forgotten-edge"}])

        found = await rests_on_a_forget(driver, Citations(statement=_STATEMENT))

        assert found == "restates a forgotten fact"
        [call] = driver.execute_query.await_args_list
        assert recall.forgotten_fact_predicate("e") in call.args[0]
        assert "coalesce(e.fact_redacted, e.fact)" in call.args[0]
        assert "LIMIT 1" in call.args[0]
        assert call.kwargs == {"statement": _STATEMENT, "space": r"\s+"}

    @pytest.mark.asyncio
    async def test_citing_nothing_it_also_rests_on_everything_the_pass_read(
        self,
    ) -> None:
        uncited = Citations(fact_uuids=["f1"], statement=_STATEMENT)

        forgotten_read = _driver(_still([], []))
        assert await rests_on_a_forget(forgotten_read, uncited) is not None
        assert forgotten_read.execute_query.await_count == 1

        all_live = _driver(_still(["f1"], []), [])
        assert await rests_on_a_forget(all_live, uncited) is None
        assert all_live.execute_query.await_count == 2


def _payload(citations: Citations, completion: ingest.IngestionCompletion) -> dict:
    return {
        "name": "dream_p1_consolidate_000",
        "episode_body": '{"content": "Alice works on Atlas"}',
        "source": EpisodeType.json,
        "source_description": "dream-pass consolidation",
        "reference_time": _NOW,
        "group_id": "user_test",
        "_citations": citations,
        "_completion": completion,
    }


def _client(still: list[dict]) -> MagicMock:
    client = MagicMock()
    client.add_episode = AsyncMock()
    client.driver.execute_query = AsyncMock(return_value=(still, [], None))
    return client


async def _work(client: MagicMock, monkeypatch: pytest.MonkeyPatch, payload: dict):
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    queue.put_nowait(payload)
    monkeypatch.setattr(ingest, "_WORKER_IDLE_TIMEOUT", 0.05)
    with (
        patch.object(ingest, "get_graphiti_client", AsyncMock(return_value=client)),
        patch.object(ingest, "ensure_indices_once", AsyncMock()),
        patch.object(ingest, "previous_episode_uuids", AsyncMock(return_value=[])),
    ):
        await ingest._ingestion_worker("test-user", "user_test", queue)


class TestTheWorker:
    @pytest.mark.asyncio
    async def test_drops_a_dream_write_resting_on_a_forget_and_counts_it(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _client(_still([], []))
        completion = ingest.IngestionCompletion()
        completion.register()

        with caplog.at_level("INFO"):
            payload = _payload(Citations(fact_uuids=["f1"]), completion)
            await _work(client, monkeypatch, payload)

        client.add_episode.assert_not_called()
        assert completion.dropped_forgotten == 1
        assert await completion.wait(0), "a dropped write is no longer pending"
        assert "Dropped dream write 'dream_p1_consolidate_000'" in caplog.text

    @pytest.mark.asyncio
    async def test_writes_one_whose_citations_are_all_live(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _client(_still(["f1"], []))
        completion = ingest.IngestionCompletion()
        completion.register()

        payload = _payload(Citations(fact_uuids=["f1"]), completion)
        await _work(client, monkeypatch, payload)

        client.add_episode.assert_awaited_once()
        assert "_citations" not in client.add_episode.await_args.kwargs
        assert completion.dropped_forgotten == 0

    @pytest.mark.asyncio
    async def test_a_write_that_found_the_graph_locked_is_checked_on_its_retry(
        self, lock_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Its citations go back on the queue with it: the forget that held
        the lock reached them, so the second attempt drops it."""
        lock = write_lock_key("user_test")
        lock_redis.values[lock] = "the forget's token"
        client = _client(_still([], []))
        completion = ingest.IngestionCompletion()
        completion.register()
        requeue = ingest._requeue_once

        def forget_finishes(*args, **kwargs) -> bool:
            client.driver.execute_query.assert_not_awaited()
            del lock_redis.values[lock]
            return requeue(*args, **kwargs)

        monkeypatch.setattr(ingest, "INGEST_LOCK_WAIT_SECONDS", 0)
        monkeypatch.setattr(ingest, "_requeue_once", forget_finishes)
        payload = _payload(Citations(fact_uuids=["f1"]), completion)
        await _work(client, monkeypatch, payload)

        client.add_episode.assert_not_called()
        assert completion.dropped_forgotten == 1
