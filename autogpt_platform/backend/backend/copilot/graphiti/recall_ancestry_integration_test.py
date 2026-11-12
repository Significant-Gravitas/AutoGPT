"""A dream write citing a derived fact that went out of use after the pass
read it, whose own source a forget then reached, on a live FalkorDB through
``dream/apply.py`` and the production ingestion worker
(``recall_sources.py``, ``recall_citations.py``, ``recall_landing.py``).

A pass reads the consolidation ``supplies``, derived from the user's flour
fact, while it is live, and proposes a fact citing it. The dream then
supersedes ``supplies``, and the user forgets the flour fact: the forget's
cascade walks through ``supplies``, no longer live and so left as it is.
Before, the proposal was then written live, resting on the forgotten fact
through ``supplies`` and carrying its content. Now:

- where the lock holds, the check before the write walks up from
  ``supplies`` to the forgotten flour fact and drops the write;
- where it does not (Redis down), with the forget landing while the write
  is in flight, the forget reports ``cleanup_error`` (the write cites a
  fact its cascade walked through), and the write's settle walks up the
  same way and retracts what the write made, erasing it for a hard forget,
  under the flour fact's name;
- a superseded fact whose own source is still there leaves the write alone.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_ancestry_integration_test.py
"""

from collections.abc import AsyncIterator
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from backend.copilot.dream import apply
from backend.copilot.dream.fetch import DreamInput
from backend.copilot.dream.schemas import (
    ConsolidatedFact,
    DreamOperations,
    ProposedFinding,
)

from . import scope_lock
from .falkordb_driver import AutoGPTFalkorDriver
from .memory_model import MemoryForgetFailureCode
from .recall import FORGOTTEN_FACT
from .recall_cascade_fixtures import (
    FLOUR,
    SUPPLIES,
    WEEKLY,
    dream,
    dream_once,
    fact_uuid,
    gather,
)
from .recall_cascade_walk import derived_reason
from .recall_derivation import MARKER_LABEL
from .recall_forget import retract
from .recall_integration_fixtures import (
    BuildClient,
    count,
    edge_row,
    ingest_facts,
    live_facts,
    patch_recall_boundaries,
    rows,
    stop_ingestion_workers,
)
from .recall_marker_fixtures import forget_mid_save
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


async def _read_then_supersede(
    driver: AutoGPTFalkorDriver, scope: MemoryScope, build: BuildClient
) -> tuple[str, str, DreamInput]:
    """The flour fact, the consolidation ``supplies`` derived from it, and
    what a pass read while ``supplies`` was live; then the dream supersedes
    ``supplies``, as a demotion leaves a fact."""
    said, edges = await ingest_facts(driver, scope, build, [FLOUR], session_id="s-1")
    flour = edges[FLOUR[2]]
    consolidation = ConsolidatedFact(
        content=SUPPLIES[2],
        confidence=0.9,
        source_fact_uuids=[flour],
        source_episode_uuids=[said],
    )
    await dream_once(driver, scope, build, consolidation, SUPPLIES, "pass-1")
    supplies = await fact_uuid(driver, SUPPLIES[2])
    read = await gather(scope)
    assert supplies in read.known_fact_uuids, "the pass read it live"
    await driver.execute_query(
        "MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() "
        "SET e.status = 'superseded', e.expiration_reason = 'stale_fact', "
        "e.expired_at = $now",
        uuid=supplies,
        now=datetime.now(timezone.utc).isoformat(),
    )
    return flour, supplies, read


def _citing(supplies: str) -> DreamOperations:
    return DreamOperations(
        proposals=[
            ProposedFinding(
                content=WEEKLY[2],
                confidence=0.6,
                rationale="the bakery's flour",
                source_fact_uuids=[supplies],
            )
        ]
    )


async def _weekly(driver: AutoGPTFalkorDriver, supplies: str) -> list[str]:
    """The proposal's fact, forgotten, erased or not; none if never
    written."""
    found = await rows(
        driver,
        "MATCH ()-[e:RELATES_TO]->() "
        "WHERE $supplies IN coalesce(e.derived_from_facts, []) "
        "OR coalesce(e.fact_redacted, e.fact) = $sentence "
        "RETURN e.uuid AS uuid",
        supplies=supplies,
        sentence=WEEKLY[2],
    )
    return [row["uuid"] for row in found]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_write_resting_on_a_forget_through_a_superseded_fact_is_dropped(
    scope_graph, stub_graphiti_client, dream_apply
) -> None:
    """Fenced: the forget lands before the write, under the lock."""
    driver, scope = scope_graph
    flour, supplies, read = await _read_then_supersede(
        driver, scope, stub_graphiti_client
    )

    forgotten = await retract(scope, [flour])
    stats = await dream(
        driver, scope, stub_graphiti_client, _citing(supplies), read, WEEKLY, "pass-2"
    )

    assert (forgotten.failures, forgotten.passed) == ([], [supplies])
    assert stats["dropped_forgotten"] == 1, "dropped by the check before the write"
    assert await _weekly(driver, supplies) == [], "never written"
    assert await live_facts(driver) == {}
    assert await count(driver, _MARKERS) == 0, "its marker withdrawn"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_superseded_fact_whose_source_stays_leaves_the_write_alone(
    scope_graph, stub_graphiti_client, dream_apply
) -> None:
    driver, scope = scope_graph
    _, supplies, read = await _read_then_supersede(driver, scope, stub_graphiti_client)

    stats = await dream(
        driver, scope, stub_graphiti_client, _citing(supplies), read, WEEKLY, "pass-2"
    )

    assert (stats["dropped_forgotten"], stats["proposal_count"]) == (0, 1)
    [weekly] = await _weekly(driver, supplies)
    assert weekly in await live_facts(driver)
    assert await count(driver, _MARKERS) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_write_in_flight_resting_through_a_superseded_fact_is_settled(
    scope_graph, stub_graphiti_client, dream_apply, mocker: MockerFixture
) -> None:
    """Unfenced (Redis down): a hard forget of the flour fact runs right
    before graphiti saves the proposal's fact."""
    driver, scope = scope_graph
    flour, supplies, read = await _read_then_supersede(
        driver, scope, stub_graphiti_client
    )
    down = AsyncMock(side_effect=ConnectionError("redis down"))
    mocker.patch.object(scope_lock, "get_redis_async", down)

    with forget_mid_save(scope, flour, True, AsyncMock()) as forgotten:
        await dream(
            driver,
            scope,
            stub_graphiti_client,
            _citing(supplies),
            read,
            WEEKLY,
            "pass-2",
        )

    [first] = forgotten
    assert [f.code for f in first.failures] == [MemoryForgetFailureCode.CLEANUP_ERROR]
    assert first.passed == [supplies], "the write cites what its cascade walked"
    [weekly] = await _weekly(driver, supplies)
    assert await live_facts(driver) == {}, "its writer settled it on landing"
    row = await edge_row(driver, weekly)
    assert (row["reason"], row["fact"], row["fact_redacted"]) == (
        derived_reason(flour),
        FORGOTTEN_FACT,
        "",
    ), "retracted under the flour fact's name, and erased: it was purged"
    assert await count(driver, _MARKERS) == 0
    retried = await retract(scope, [flour], hard=True)
    assert (retried.failures, retried.resumed) == ([], [flour])
