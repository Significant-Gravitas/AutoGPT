"""A forget and a dream write at the same time, and a forget repeated after
its cascade failed, on a live FalkorDB whose own Redis holds the graph's
write lock (``scope_lock.py``), through ``dream/apply.py`` and the
production ingestion worker.

The worker records what a dream write was derived from before it lets the
lock go (``recall_derivation.py``), so a forget waiting for the write finds
the new fact and retracts it; a write waiting for a forget is checked after
it and dropped (``recall_citations.py``). A forget whose cascade stopped
partway goes on from the facts it already retracted when repeated.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_cascade_lock_integration_test.py
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from graphiti_core.driver.falkordb_driver import FalkorDriverSession
from pytest_mock import MockerFixture

from backend.copilot.dream import apply
from backend.copilot.dream.schemas import ConsolidatedFact, DreamOperations

from . import recall_forget
from .falkordb_driver import AutoGPTFalkorDriver
from .memory_model import ForgetResult, MemoryForgetFailureCode
from .recall import FORGOTTEN_FACT
from .recall_cascade_fixtures import (
    FLOUR,
    SUPPLIES,
    build_bakery,
    dream,
    fact_uuid,
    gather,
)
from .recall_forget import retract
from .recall_hide import SCRUB_FACTS_QUERY
from .recall_integration_fixtures import (
    count,
    edge_row,
    ingest_facts,
    live_facts,
    patch_recall_boundaries,
    stop_ingestion_workers,
)


@pytest.fixture(autouse=True)
def boundaries(mocker, scope_graph, stub_graphiti_client):
    patch_recall_boundaries(mocker, scope_graph[0], stub_graphiti_client)


@pytest_asyncio.fixture(loop_scope="function")
async def dream_apply(mocker: MockerFixture) -> AsyncIterator[None]:
    mocker.patch.object(apply, "_create_dream_session", AsyncMock(return_value="s"))
    mocker.patch.object(apply, "_write_dream_summary_message", AsyncMock())
    mocker.patch(
        "backend.copilot.dream.registry.ensure_dream_system_scheduled",
        AsyncMock(return_value=None),
    )
    yield
    await stop_ingestion_workers()


@contextmanager
def _forget_while_saving(
    forget: Callable[[], Awaitable[ForgetResult]],
) -> Iterator[tuple[list[asyncio.Future], list[bool]]]:
    """Start ``forget`` right before graphiti saves the dream write's edges,
    holding the graph's write lock, and note whether it is still waiting
    for the lock 0.6 s later."""
    original = FalkorDriverSession.run
    started: list[asyncio.Future] = []
    waited: list[bool] = []

    async def run(session: FalkorDriverSession, query: Any, **params: Any) -> Any:
        text = query if isinstance(query, str) else " ".join(q for q, _ in query)
        if not started and "SET r = edge" in text:
            started.append(asyncio.ensure_future(forget()))
            await asyncio.sleep(0.6)
            waited.append(not started[0].done())
        return await original(session, query, **params)

    with patch.object(FalkorDriverSession, "run", run):
        yield started, waited


def _consolidation(flour: str, said: str) -> DreamOperations:
    return DreamOperations(
        writes=[
            ConsolidatedFact(
                content=SUPPLIES[2],
                confidence=0.9,
                source_fact_uuids=[flour],
                source_episode_uuids=[said],
            )
        ]
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_forget_waiting_for_a_dream_write_retracts_what_it_wrote(
    scope_graph, stub_graphiti_client, dream_apply
) -> None:
    driver, scope = scope_graph
    build = stub_graphiti_client
    said, edges = await ingest_facts(driver, scope, build, [FLOUR], session_id="s-1")
    flour = edges[FLOUR[2]]
    read = await gather(scope)

    with _forget_while_saving(lambda: retract(scope, [flour])) as (started, waited):
        stats = await dream(
            driver, scope, build, _consolidation(flour, said), read, SUPPLIES, "p-1"
        )
    result = await asyncio.wait_for(started[0], 30)

    assert waited == [True], "the forget did not wait for the dream write"
    assert (stats["consolidated_count"], stats["dropped_forgotten"]) == (1, 0)
    supplies = await fact_uuid(driver, SUPPLIES[2])
    assert (result.derived, result.failures) == ([supplies], [])
    assert await live_facts(driver) == {}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_dream_write_waiting_for_a_forget_is_dropped(
    scope_graph, stub_graphiti_client, dream_apply
) -> None:
    driver, scope = scope_graph
    build = stub_graphiti_client
    said, edges = await ingest_facts(driver, scope, build, [FLOUR], session_id="s-1")
    flour = edges[FLOUR[2]]
    read = await gather(scope)
    entered, release = asyncio.Event(), asyncio.Event()
    query = AutoGPTFalkorDriver.execute_query

    async def hold_the_forget(self, cypher_query_, **params):
        if cypher_query_ == recall_forget._EXISTING_EDGES_QUERY:
            entered.set()
            await release.wait()
        return await query(self, cypher_query_, **params)

    with patch.object(AutoGPTFalkorDriver, "execute_query", hold_the_forget):
        forget = asyncio.ensure_future(retract(scope, [flour]))
        await asyncio.wait_for(entered.wait(), 10)
        write = asyncio.ensure_future(
            dream(
                driver, scope, build, _consolidation(flour, said), read, SUPPLIES, "p"
            )
        )
        await asyncio.sleep(0.6)
        assert not write.done(), "the dream write did not wait for the forget"
        release.set()
        result = await asyncio.wait_for(forget, 30)
        stats = await asyncio.wait_for(write, 30)

    assert (result.deleted, result.derived) == ([flour], [])
    assert stats["dropped_forgotten"] == 1
    dreamed = (
        "MATCH (ep:Episodic) WHERE ep.name STARTS WITH 'dream_' RETURN count(ep) AS c"
    )
    assert await count(driver, dreamed) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_forget_whose_cascade_failed_finishes_it_when_repeated(
    scope_graph, stub_graphiti_client, dream_apply
) -> None:
    """The consolidation is retracted, then scrubbing its sentence fails:
    the forget reports a clean-up error, and forgetting again finishes the
    scrub and retracts what rests on it."""
    driver, scope = scope_graph
    bakery = await build_bakery(driver, scope, stub_graphiti_client)
    query = AutoGPTFalkorDriver.execute_query

    async def fail_its_scrub(self, cypher_query_, **params):
        if cypher_query_ == SCRUB_FACTS_QUERY and bakery.supplies in params["uuids"]:
            raise RuntimeError("scrub lost")
        return await query(self, cypher_query_, **params)

    with patch.object(AutoGPTFalkorDriver, "execute_query", fail_its_scrub):
        first = await retract(scope, [bakery.flour])

    assert first.derived == [bakery.supplies]
    assert [(f.uuid, f.code) for f in first.failures] == [
        (bakery.flour, MemoryForgetFailureCode.CLEANUP_ERROR)
    ]
    assert {bakery.boule_flour, bakery.weekly} <= set(await live_facts(driver))

    second = await retract(scope, [bakery.flour])

    assert second.failures == []
    assert sorted(second.derived) == sorted([bakery.boule_flour, bakery.weekly])
    assert (await edge_row(driver, bakery.supplies))["fact"] == FORGOTTEN_FACT
    assert set(await live_facts(driver)) == {bakery.boule, bakery.cafe}
