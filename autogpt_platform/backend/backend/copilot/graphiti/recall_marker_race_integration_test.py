"""A forget landing while a dream write is in flight, where the graph's write
lock does not hold, on a live FalkorDB through ``dream/apply.py`` and the
production ingestion worker (``recall_forget_writes.py``,
``recall_landing.py``).

The lock admits two writers at once when Redis cannot be reached (both go
ahead without it) or when a holder loses its lease (its key expires while
it writes, and the forget takes the lock). Codex's race: the forget
reported success while the write was still in flight, and the write then
landed a live derived fact resting on a purged root.

Here the forget runs to the end right before graphiti saves the write's
facts (its episode is saved, its facts are not), so it cannot reach them:
it reports ``cleanup_error``, since a write citing its root may still land.
The write lands, and its writer settles it before clearing its marker: its
fact is retracted (erased, for a hard forget) a few statements after
graphiti saved it, the dream episode hidden. A read in those few
statements can see the fact live; a writer that died in between leaves its
marker, which the next reconcile (the forget's retry, which the
``cleanup_error`` asks for) completes and settles. Forgetting again reports
the forget done.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_marker_race_integration_test.py
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture
from redis.asyncio import Redis

from backend.copilot.dream import apply
from backend.copilot.dream.schemas import ConsolidatedFact, DreamOperations

from . import marked_write, scope_lock
from .memory_model import MemoryForgetFailureCode
from .recall import FORGOTTEN_FACT
from .recall_cascade_fixtures import FLOUR, SUPPLIES, derivation, dream, gather
from .recall_cascade_walk import derived_reason
from .recall_derivation import MARKER_LABEL
from .recall_forget import retract
from .recall_integration_fixtures import (
    count,
    edge_row,
    episode_row,
    ingest_facts,
    live_facts,
    patch_recall_boundaries,
    rows,
    sentence_properties,
    stop_ingestion_workers,
)
from .recall_marker_fixtures import forget_mid_save
from .scope import MemoryScope, write_lock_key

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


def _unfencing(
    overlap: str, scope: MemoryScope, redis: Redis, mocker: MockerFixture
) -> Callable[[], Awaitable[None]]:
    """What takes the lock's protection away: Redis down for every writer
    from the start, or the writer's lease lost right before the forget."""
    if overlap == "redis_down":
        down = AsyncMock(side_effect=ConnectionError("redis down"))
        mocker.patch.object(scope_lock, "get_redis_async", down)

    async def unfence() -> None:
        if overlap == "lease_lost":
            key = write_lock_key(scope.group_id)
            assert await redis.get(key), "the writer held the lock"
            await redis.delete(key)

    return unfence


async def _dreamed(driver, kind: str) -> str:
    """The one fact or episode carrying a dream's record."""
    match = "()-[x:RELATES_TO]->()" if kind == "fact" else "(x:Episodic)"
    [row] = await rows(
        driver,
        f"MATCH {match} WHERE x.derived_from_facts IS NOT NULL RETURN x.uuid AS uuid",
    )
    return row["uuid"]


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overlap", "hard"), [("redis_down", True), ("lease_lost", False)]
)
async def test_a_forget_while_a_write_is_in_flight_and_the_write_landing(
    scope_graph,
    stub_graphiti_client,
    dream_apply,
    live_lock: Redis,
    mocker: MockerFixture,
    overlap: str,
    hard: bool,
) -> None:
    driver, scope = scope_graph
    said, edges = await ingest_facts(
        driver, scope, stub_graphiti_client, [FLOUR], session_id="s-1"
    )
    flour = edges[FLOUR[2]]
    write = ConsolidatedFact(
        content=SUPPLIES[2],
        confidence=0.9,
        source_fact_uuids=[flour],
        source_episode_uuids=[said],
    )
    read = await gather(scope)
    unfence = _unfencing(overlap, scope, live_lock, mocker)

    with forget_mid_save(scope, flour, hard, unfence) as forgotten:
        stats = await dream(
            driver,
            scope,
            stub_graphiti_client,
            DreamOperations(writes=[write]),
            read,
            SUPPLIES,
            f"p-{overlap}",
        )

    [first] = forgotten
    assert [(f.uuid, f.code) for f in first.failures] == [
        (flour, MemoryForgetFailureCode.CLEANUP_ERROR)
    ], "a write citing the root may still land"
    assert "may still land" in first.failures[0].reason
    assert (stats["provenance_pending"], stats["failed_writes"]) == (0, 0)
    supplies = await _dreamed(driver, "fact")
    assert await live_facts(driver) == {}, "its writer settled it on landing"
    record = await derivation(driver, supplies)
    assert (record["facts"], record["reason"]) == ([flour], derived_reason(flour))
    fact = await edge_row(driver, supplies)
    assert fact["fact"] == FORGOTTEN_FACT
    assert (fact["fact_redacted"] == "") is hard, "erased only for a hard forget"
    episode = await episode_row(driver, await _dreamed(driver, "episode"))
    assert episode["redacted_at"] is not None, "the dream episode is hidden"
    assert await count(driver, _MARKERS) == 0
    if hard:
        assert await sentence_properties(driver, SUPPLIES[2]) == set()
    retried = await retract(scope, [flour], hard=hard)
    assert retried.failures == [], "forgetting again reports it done"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_writer_dying_after_its_write_landed_is_settled_by_the_retry(
    scope_graph,
    stub_graphiti_client,
    dream_apply,
    live_lock: Redis,
    mocker: MockerFixture,
) -> None:
    """The residual window: the writer dies between graphiti's save and its
    record, so its fact is live until the next reconcile, which the
    forget's ``cleanup_error`` asks for: the retry completes the marker,
    settles the write and erases the fact."""
    driver, scope = scope_graph
    said, edges = await ingest_facts(
        driver, scope, stub_graphiti_client, [FLOUR], session_id="s-1"
    )
    flour = edges[FLOUR[2]]
    write = ConsolidatedFact(
        content=SUPPLIES[2],
        confidence=0.9,
        source_fact_uuids=[flour],
        source_episode_uuids=[said],
    )
    read = await gather(scope)
    unfence = _unfencing("redis_down", scope, live_lock, mocker)
    mocker.patch.object(marked_write, "recorded", AsyncMock())

    with forget_mid_save(scope, flour, True, unfence) as forgotten:
        await dream(
            driver,
            scope,
            stub_graphiti_client,
            DreamOperations(writes=[write]),
            read,
            SUPPLIES,
            "p-dies",
        )
    live = await live_facts(driver)
    retried = await retract(scope, [flour], hard=True)

    [first] = forgotten
    assert [f.code for f in first.failures] == [MemoryForgetFailureCode.CLEANUP_ERROR]
    assert list(live.values()) == [SUPPLIES[2]], "live until the next reconcile"
    assert (retried.failures, retried.resumed) == ([], [flour])
    assert await live_facts(driver) == {}
    fact = await edge_row(driver, await _dreamed(driver, "fact"))
    assert (fact["fact"], fact["fact_redacted"]) == (FORGOTTEN_FACT, "")
    assert await count(driver, _MARKERS) == 0
