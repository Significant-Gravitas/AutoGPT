"""A dream write's provenance is durable before its fact exists, on a live
FalkorDB whose own Redis holds the graph's write lock, through
``dream/apply.py`` and the production ingestion worker
(``recall_derivation.py``, ``recall_reconcile.py``,
``provenance_pending.py``).

Codex's probe: the record written after the write fails. Before the marker,
the pass reported success, the derived fact was live with no record, and
forgetting what it was derived from left it live. Now the marker written
before the write keeps the citations, the pass reports the write as
``provenance_pending``, and the next forget completes the record before it
cascades, even one that was waiting for the write's lock; the reaper's sweep
completes it without a forget. A marker that cannot be written drops the
write.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_provenance_integration_test.py
"""

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from graphiti_core.driver.falkordb_driver import FalkorDriverSession
from pytest_mock import MockerFixture

from backend.copilot.dream import apply
from backend.copilot.dream.schemas import ConsolidatedFact, DreamOperations
from backend.data import redis_client

from .falkordb_driver import AutoGPTFalkorDriver
from .memory_model import ForgetResult
from .provenance_pending import PENDING_KEY, Swept, sweep_pending
from .recall_cascade_fixtures import FLOUR, SUPPLIES, derivation, dream, gather
from .recall_derivation import MARK_QUERY, MARKER_LABEL, RECORD_EPISODE_QUERY
from .recall_forget import retract
from .recall_integration_fixtures import (
    count,
    ingest_facts,
    live_facts,
    patch_recall_boundaries,
    rows,
    stop_ingestion_workers,
)
from .scope import MemoryScope

_MARKERS = f"MATCH (m:{MARKER_LABEL}) RETURN count(m) AS c"


@pytest.fixture(autouse=True)
def boundaries(mocker, scope_graph, stub_graphiti_client):
    patch_recall_boundaries(mocker, scope_graph[0], stub_graphiti_client)


@pytest_asyncio.fixture(loop_scope="function")
async def dream_apply(mocker: MockerFixture) -> AsyncIterator[None]:
    """apply's Postgres side (the dream session and its summary) stubbed."""
    mocker.patch.object(apply, "_create_dream_session", AsyncMock(return_value="s"))
    mocker.patch.object(apply, "_write_dream_summary_message", AsyncMock())
    mocker.patch(
        "backend.copilot.dream.registry.ensure_dream_system_scheduled",
        AsyncMock(return_value=None),
    )
    yield
    await stop_ingestion_workers()


@contextmanager
def _failing(query: str, times: int | None = None) -> Iterator[None]:
    """Every run of ``query`` raises while the block is open, or only its
    first ``times`` runs."""
    original = AutoGPTFalkorDriver.execute_query
    runs = [0]

    async def failing(self, cypher_query_, **params):
        if cypher_query_ == query:
            runs[0] += 1
            if times is None or runs[0] <= times:
                raise RuntimeError("injected failure")
        return await original(self, cypher_query_, **params)

    with patch.object(AutoGPTFalkorDriver, "execute_query", failing):
        yield


async def _said(driver, scope: MemoryScope, build) -> tuple[str, str, DreamOperations]:
    """The user's flour fact and chat turn, and a consolidation citing both."""
    said, edges = await ingest_facts(driver, scope, build, [FLOUR], session_id="s-1")
    flour = edges[FLOUR[2]]
    write = ConsolidatedFact(
        content=SUPPLIES[2],
        confidence=0.9,
        source_fact_uuids=[flour],
        source_episode_uuids=[said],
    )
    return flour, said, DreamOperations(writes=[write])


async def _supplies(driver) -> str | None:
    """The consolidation's fact, forgotten or not; None if never written."""
    found = await rows(
        driver,
        "MATCH ()-[e:RELATES_TO]->() "
        "WHERE coalesce(e.fact_redacted, e.fact) = $sentence RETURN e.uuid AS uuid",
        sentence=SUPPLIES[2],
    )
    return found[0]["uuid"] if found else None


async def _pending() -> set[str]:
    redis = await redis_client.get_redis_async()
    return set(redis.sets.get(PENDING_KEY, set()))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_record_that_failed_is_reconciled_before_the_next_forget(
    scope_graph, stub_graphiti_client, dream_apply
) -> None:
    driver, scope = scope_graph
    flour, _, ops = await _said(driver, scope, stub_graphiti_client)
    read = await gather(scope)

    with _failing(RECORD_EPISODE_QUERY):
        stats = await dream(
            driver, scope, stub_graphiti_client, ops, read, SUPPLIES, "p-fail"
        )

    supplies = await _supplies(driver)
    assert supplies is not None and supplies in await live_facts(driver)
    assert (stats["consolidated_count"], stats["provenance_pending"]) == (1, 1)
    assert (await derivation(driver, supplies))["facts"] is None
    assert await count(driver, _MARKERS) == 1
    assert await _pending() == {scope.group_id}

    result = await retract(scope, [flour])

    assert (result.derived, result.failures) == ([supplies], [])
    assert await live_facts(driver) == {}
    assert (await derivation(driver, supplies))["facts"] == [flour]
    assert await count(driver, _MARKERS) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_marker_that_cannot_be_written_drops_the_write(
    scope_graph, stub_graphiti_client, dream_apply
) -> None:
    driver, scope = scope_graph
    _, _, ops = await _said(driver, scope, stub_graphiti_client)
    read = await gather(scope)

    with _failing(MARK_QUERY):
        stats = await dream(
            driver, scope, stub_graphiti_client, ops, read, SUPPLIES, "p-nomark"
        )

    assert (stats["failed_writes"], stats["provenance_pending"]) == (1, 0)
    assert await _supplies(driver) is None
    dreamed = (
        "MATCH (ep:Episodic) WHERE ep.name STARTS WITH 'dream_' RETURN count(ep) AS c"
    )
    assert await count(driver, dreamed) == 0
    assert await count(driver, _MARKERS) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_the_reaper_sweep_completes_a_record_without_a_forget(
    scope_graph, stub_graphiti_client, dream_apply
) -> None:
    driver, scope = scope_graph
    flour, said, ops = await _said(driver, scope, stub_graphiti_client)
    read = await gather(scope)
    with _failing(RECORD_EPISODE_QUERY):
        await dream(driver, scope, stub_graphiti_client, ops, read, SUPPLIES, "p-r")

    swept = await sweep_pending()

    assert swept == Swept(graphs=1, completed=1)
    supplies = await _supplies(driver)
    assert supplies is not None
    record = await derivation(driver, supplies)
    assert (record["facts"], record["episodes"]) == ([flour], [said])
    assert await count(driver, _MARKERS) == 0
    assert await _pending() == set()


@contextmanager
def _forget_while_saving(
    scope: MemoryScope, flour: str
) -> Iterator[tuple[list[asyncio.Future], list[bool]]]:
    """Start a forget of ``flour`` right before graphiti saves the dream
    write's edges, while the worker holds the graph's write lock, and note
    whether it is still waiting for the lock 0.3 s later."""
    original = FalkorDriverSession.run
    started: list[asyncio.Future] = []
    waited: list[bool] = []

    async def run(session: FalkorDriverSession, query: Any, **params: Any) -> Any:
        text = query if isinstance(query, str) else " ".join(q for q, _ in query)
        if not started and "SET r = edge" in text:
            started.append(asyncio.ensure_future(retract(scope, [flour])))
            await asyncio.sleep(0.3)
            waited.append(not started[0].done())
        return await original(session, query, **params)

    with patch.object(FalkorDriverSession, "run", run):
        yield started, waited


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_forget_waiting_for_a_write_whose_record_fails_retracts_it(
    scope_graph, stub_graphiti_client, dream_apply
) -> None:
    driver, scope = scope_graph
    flour, _, ops = await _said(driver, scope, stub_graphiti_client)
    read = await gather(scope)

    with (
        _failing(RECORD_EPISODE_QUERY, times=1),
        _forget_while_saving(scope, flour) as (started, waited),
    ):
        stats = await dream(
            driver, scope, stub_graphiti_client, ops, read, SUPPLIES, "p-race"
        )
        result: ForgetResult = await asyncio.wait_for(started[0], 30)

    assert waited == [True], "the forget waited for the write's lock"
    supplies = await _supplies(driver)
    assert stats["provenance_pending"] == 1
    assert (result.derived, result.failures) == ([supplies], [])
    assert await live_facts(driver) == {}
    assert await count(driver, _MARKERS) == 0
