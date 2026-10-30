"""The derivation backfill on a live FalkorDB, on dream facts written the way
they were before derivation records: no record anywhere, and the citations
only in each dream episode's description, in its old form (a consolidation
listed its episodes, a proposal its facts).

It records them so a later forget reaches them, counts the one it cannot
attribute (a proposal that cited only an episode), and, asked to, cascades
from a forget made before it ran. The script's unit tests are
``migrations/backfill_derivations_test.py``.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/backfill_derivations_integration_test.py
"""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from backend.copilot.dream import apply

from .falkordb_driver import AutoGPTFalkorDriver
from .migrations.backfill_derivations import backfill_graph
from .recall_cascade_fixtures import Bakery, build_bakery, derivation
from .recall_forget import retract
from .recall_integration_fixtures import (
    edge_row,
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


async def _as_before_records(driver: AutoGPTFalkorDriver, bakery: Bakery) -> None:
    """Take every record away and give each dream episode the description
    the dream wrote before records existed."""
    await driver.execute_query(
        "MATCH (ep:Episodic) "
        "SET ep.derived_from_facts = NULL, ep.derived_from_episodes = NULL"
    )
    await driver.execute_query(
        "MATCH ()-[e:RELATES_TO]->() "
        "SET e.derived_from_facts = NULL, e.derived_from_episodes = NULL"
    )
    described = {
        bakery.supplies_episode: (
            f"dream-pass consolidation; src_episodes={bakery.said}"
        ),
        bakery.boule_flour_episode: (
            "dream-pass proposal; rationale=the bakery's flour; "
            f"src_facts={bakery.supplies},{bakery.boule}"
        ),
        bakery.weekly_episode: "dream-pass proposal; rationale=the bakery's flour",
    }
    for uuid, description in described.items():
        await driver.execute_query(
            "MATCH (ep:Episodic {uuid: $uuid}) SET ep.source_description = $d",
            uuid=uuid,
            d=description,
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_dry_run_counts_and_writes_nothing(
    scope_graph, stub_graphiti_client, dream_apply
) -> None:
    driver, scope = scope_graph
    bakery = await build_bakery(driver, scope, stub_graphiti_client)
    await _as_before_records(driver, bakery)

    found = await backfill_graph(driver, apply=False, cascade_forgets=True)

    assert (found.episodes, found.facts, found.unattributed) == (3, 3, 1)
    assert (found.roots, found.derived) == (0, 0)
    assert (await derivation(driver, bakery.supplies))["facts"] is None
    assert (await derivation(driver, bakery.supplies_episode))["facts"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_apply_records_them_so_a_later_forget_reaches_them(
    scope_graph, stub_graphiti_client, dream_apply
) -> None:
    driver, scope = scope_graph
    bakery = await build_bakery(driver, scope, stub_graphiti_client)
    await _as_before_records(driver, bakery)

    found = await backfill_graph(driver, apply=True, cascade_forgets=False)
    again = await backfill_graph(driver, apply=True, cascade_forgets=False)

    assert (found.episodes, found.facts, found.unattributed) == (3, 3, 1)
    assert (again.episodes, again.facts, again.busy) == (0, 0, 0), "idempotent"
    supplies = await derivation(driver, bakery.supplies)
    assert (supplies["facts"], supplies["episodes"]) == ([], [bakery.said])
    boule_flour = await derivation(driver, bakery.boule_flour)
    assert boule_flour["facts"] == [bakery.supplies, bakery.boule]
    result = await retract(scope, [bakery.flour])
    assert sorted(result.derived) == sorted([bakery.supplies, bakery.boule_flour])
    assert bakery.weekly in await live_facts(driver), "cited an episode only"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_it_cascades_from_a_forget_made_before_it_ran(
    scope_graph, stub_graphiti_client, dream_apply
) -> None:
    driver, scope = scope_graph
    bakery = await build_bakery(driver, scope, stub_graphiti_client)
    await _as_before_records(driver, bakery)
    earlier = await retract(scope, [bakery.flour])
    assert earlier.derived == [], "nothing recorded to follow yet"

    found = await backfill_graph(driver, apply=True, cascade_forgets=True)

    # The forgotten fact, and the chat turn it hid, which a record now names.
    assert (found.roots, found.derived, found.failed) == (2, 2, 0)
    live = set(await live_facts(driver))
    assert live == {bakery.boule, bakery.cafe, bakery.weekly}
    for uuid in (bakery.supplies, bakery.boule_flour):
        row = await edge_row(driver, uuid)
        assert row["reason"] == f"derived_from_forgotten:{bakery.flour}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_forged_or_ambiguous_description_attributes_nothing(
    scope_graph, stub_graphiti_client, dream_apply
) -> None:
    """Codex's probe on older dream text: a rationale that forges a citation
    of a uuid the graph lacks, one naming a key twice, and a project-scoped
    write citing the global cafe fact. The parser that took the last key
    linked each to what it named; now none is attributed, and forgetting the
    cafe retracts nothing."""
    driver, scope = scope_graph
    bakery = await build_bakery(driver, scope, stub_graphiti_client)
    await _as_before_records(driver, bakery)
    ghost = "11111111-2222-3333-4444-555555555555"
    forged = {
        bakery.supplies_episode: (
            f"dream-pass proposal; rationale=ordinary; src_facts={ghost}"
        ),
        bakery.boule_flour_episode: (
            "dream-pass proposal; rationale=ordinary; "
            f"src_facts={bakery.cafe}; src_facts={bakery.supplies}"
        ),
        bakery.weekly_episode: (
            f"dream-pass proposal; rationale=weekly; src_facts={bakery.cafe}"
        ),
    }
    for uuid, description in forged.items():
        await driver.execute_query(
            "MATCH (ep:Episodic {uuid: $uuid}) SET ep.source_description = $d",
            uuid=uuid,
            d=description,
        )
    await driver.execute_query(
        "MATCH (ep:Episodic {uuid: $uuid}) SET ep.content = $content",
        uuid=bakery.weekly_episode,
        content='{"content": "orders weekly", "scope": "project:ordering"}',
    )

    found = await backfill_graph(driver, apply=True, cascade_forgets=False)

    assert (found.ambiguous, found.rejected) == (1, 2)
    for uuid in (bakery.supplies, bakery.boule_flour, bakery.weekly):
        assert (await derivation(driver, uuid))["facts"] == [], uuid
    assert (await retract(scope, [bakery.cafe])).derived == []
