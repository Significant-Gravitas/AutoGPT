"""A dream's writes after a forget that answered once the dream had read the
graph, through ``dream/apply.py`` and the production ingestion worker on a
live FalkorDB.

The dream reads Alice's fact and its episode, the user forgets the fact
(soft or hard), then the dream writes a consolidation or a proposal resting
on what it read: the worker drops it under the graph's write lock
(``recall_citations.py``), no edge is added, the fact stays out of recall
and the pass reports the drop. A write citing nothing the pass read never
gets that far: apply drops it first (``dream/citations.py``). A write resting
only on live memory still lands. Reproduced first by an independent
validation (``r5-quality-delayed-operations.py``, the four stale-dream
cases). A forget made after the dream's writes landed is
``recall_cascade_integration_test.py``.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_dream_citations_integration_test.py
"""

from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from backend.copilot.dream import apply, fetch
from backend.copilot.dream.fetch import DreamInput
from backend.copilot.dream.schemas import (
    ConsolidatedFact,
    DreamOperations,
    ProposedFinding,
)

from . import ingest, recall
from .falkordb_driver import AutoGPTFalkorDriver
from .recall_forget import retract
from .recall_integration_fixtures import (
    ALICE,
    BOB,
    BuildClient,
    Fact,
    ingest_facts,
    live_facts,
    patch_recall_boundaries,
    rows,
    scripted_responses,
    stop_ingestion_workers,
)
from .scope import MemoryScope

_BOREALIS: Fact = ("Bob", "Borealis", "Bob will lead Borealis")


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


async def _gather(scope: MemoryScope) -> DreamInput:
    sessions = SimpleNamespace(get_user_chat_sessions=AsyncMock(return_value=[]))
    with patch.object(fetch, "chat_db", return_value=sessions):
        return await fetch.gather_dream_input(scope)


async def _apply(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    build: BuildClient,
    ops: DreamOperations,
    read: DreamInput | None,
    *,
    extracted: Fact = ALICE,
    pass_id: str = "pass-1",
) -> dict:
    """``ops`` through ``apply_operations`` and the worker, graphiti's model
    extracting ``extracted`` from what it writes; the pass's stats."""
    client = build(driver, scripted_responses([extracted]))
    with patch.object(ingest, "get_graphiti_client", AsyncMock(return_value=client)):
        return await apply.apply_operations(
            scope,
            pass_id,
            ops,
            known_fact_uuids=read.known_fact_uuids if read else None,
            known_episode_uuids=read.known_episode_uuids if read else None,
            ingestion_drain_timeout=25,
        )


def _resting_on(operation: str, read: DreamInput, content: str) -> DreamOperations:
    """A write of ``content`` citing what the pass read, as its model would."""
    facts, episodes = sorted(read.known_fact_uuids), sorted(read.known_episode_uuids)
    if operation == "consolidate":
        fact = ConsolidatedFact(
            content=content,
            confidence=0.9,
            source_episode_uuids=episodes,
            source_fact_uuids=facts,
        )
        return DreamOperations(writes=[fact])
    finding = ProposedFinding(
        content=content,
        confidence=0.9,
        rationale="read before the forget",
        source_fact_uuids=facts,
        source_episode_uuids=episodes,
    )
    return DreamOperations(proposals=[finding])


def _only_bob(read: DreamInput) -> DreamInput:
    """What ``read`` holds of Bob's fact and his chat turn, and nothing else."""
    facts = {fact.uuid for fact in read.facts if fact.fact == BOB[2]}
    episodes = {ep.uuid for ep in read.episodes if BOB[2] in (ep.content or "")}
    return read.model_copy(
        update={"known_fact_uuids": facts, "known_episode_uuids": episodes}
    )


async def _edges(driver: AutoGPTFalkorDriver) -> set[str]:
    found = await rows(driver, "MATCH ()-[e:RELATES_TO]->() RETURN e.uuid AS uuid")
    return {row["uuid"] for row in found}


async def _recalled(scope: MemoryScope) -> set[str]:
    return {fact.fact for fact in await recall.search_facts(scope, "Atlas", limit=20)}


async def _dream_episodes(driver: AutoGPTFalkorDriver) -> int:
    found = await rows(
        driver,
        "MATCH (ep:Episodic) WHERE ep.name STARTS WITH 'dream_' "
        "RETURN count(ep) AS c",
    )
    return int(found[0]["c"])


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("hard", [False, True], ids=["soft", "hard"])
@pytest.mark.parametrize("operation", ["consolidate", "propose"])
async def test_a_dream_write_resting_on_a_fact_forgotten_since_it_read_is_dropped(
    scope_graph, stub_graphiti_client, dream_apply, operation: str, hard: bool
) -> None:
    driver, scope = scope_graph
    _, edges = await ingest_facts(
        driver, scope, stub_graphiti_client, [ALICE], session_id="s-1"
    )
    read = await _gather(scope)
    assert {fact.fact for fact in read.facts} == {ALICE[2]}, "read before"
    await retract(scope, list(edges.values()), hard=hard)
    before = await _edges(driver)

    stats = await _apply(
        driver,
        scope,
        stub_graphiti_client,
        _resting_on(operation, read, ALICE[2]),
        read,
    )

    assert stats["dropped_forgotten"] == 1
    assert await _edges(driver) == before, "the dream added an edge"
    assert await _dream_episodes(driver) == 0
    assert ALICE[2] not in await _recalled(scope)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("hard", [False, True], ids=["soft", "hard"])
async def test_an_uncited_dream_write_is_dropped_before_it_is_queued(
    scope_graph, stub_graphiti_client, dream_apply, hard: bool
) -> None:
    """A restatement of the forgotten fact citing nothing, and one citing
    only a uuid the pass never read, are dropped by apply and counted: the
    worker never sees them."""
    driver, scope = scope_graph
    _, edges = await ingest_facts(
        driver, scope, stub_graphiti_client, [ALICE], session_id="s-1"
    )
    read = await _gather(scope)
    await retract(scope, list(edges.values()), hard=hard)
    before = await _edges(driver)
    restated = ConsolidatedFact(content="  alice WORKS on\tAtlas ", confidence=0.9)
    made_up = restated.model_copy(update={"source_fact_uuids": ["made-up"]})

    stats = await _apply(
        driver,
        scope,
        stub_graphiti_client,
        DreamOperations(writes=[restated, made_up]),
        read,
    )

    assert (stats["uncited_writes_dropped"], stats["dropped_forgotten"]) == (2, 0)
    assert await _edges(driver) == before, "the dream added an edge"
    assert await _dream_episodes(driver) == 0
    assert ALICE[2] not in await _recalled(scope)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["consolidate", "propose"])
async def test_a_dream_write_resting_only_on_live_memory_still_lands(
    scope_graph, stub_graphiti_client, dream_apply, operation: str
) -> None:
    """Alice is forgotten after the pass read; a write citing only Bob's
    fact and episode is written, like one from a pass that read after the
    forget."""
    driver, scope = scope_graph
    _, alice = await ingest_facts(
        driver, scope, stub_graphiti_client, [ALICE], session_id="s-1"
    )
    await ingest_facts(driver, scope, stub_graphiti_client, [BOB], session_id="s-2")
    read = await _gather(scope)
    await retract(scope, list(alice.values()))

    ops = _resting_on(operation, _only_bob(read), _BOREALIS[2])
    cited = await _apply(
        driver, scope, stub_graphiti_client, ops, read, extracted=_BOREALIS
    )
    read = await _gather(scope)
    ops = _resting_on(operation, read, "Bob plans Borealis")
    fresh = await _apply(
        driver,
        scope,
        stub_graphiti_client,
        ops,
        read,
        extracted=_BOREALIS,
        pass_id="pass-2",
    )

    assert (cited["dropped_forgotten"], fresh["dropped_forgotten"]) == (0, 0)
    assert await _dream_episodes(driver) == 2, "both written"
    assert _BOREALIS[2] in (await live_facts(driver)).values()
    assert ALICE[2] not in await _recalled(scope)
