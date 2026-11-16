"""The scope rule on a live FalkorDB, through ``dream/apply.py`` and the
production ingestion worker (``dream/citations.py``).

Codex's probe: a scripted model writes a ``project:unrelated`` conclusion
about the bakery citing an unrelated ``real:global`` fact the pass read (the
cafe). Before the check it was written: forgetting the bakery's real source
missed it, and forgetting the cafe retracted it. Now it is dropped unwritten,
and so is a write citing a fact of its own scope beside one of another:
trimmed to its own scope it could restate the other fact and escape that
fact's forget. Episode citations are not scoped: a project write citing only
the user's chat turn is written, and forgetting the fact that turn stated
hides the turn and retracts the write through ``derived_from_episodes``.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_citation_scope_integration_test.py
"""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from backend.copilot.dream import apply
from backend.copilot.dream.schemas import ConsolidatedFact, DreamOperations

from .recall_cascade_fixtures import CAFE, FLOUR, derivation, dream, fact_uuid, gather
from .recall_forget import retract
from .recall_integration_fixtures import (
    Fact,
    ingest_facts,
    live_facts,
    patch_recall_boundaries,
    rows,
    stop_ingestion_workers,
)

_QUARTERLY: Fact = (
    "Sunrise Bakery",
    "Quarterly Plan",
    "Sunrise Bakery changes suppliers every quarter",
)
_BREAD: Fact = (
    "Bread Program",
    "Hill Country Mills",
    "The bread program runs on Hill Country Mills flour",
)
_COUNTS = (
    "consolidated_count",
    "uncited_writes_dropped",
    "cross_scope_citations_dropped",
)


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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_write_citing_only_another_scope_is_never_written(
    scope_graph, stub_graphiti_client, dream_apply
) -> None:
    driver, scope = scope_graph
    _, facts = await ingest_facts(
        driver, scope, stub_graphiti_client, [FLOUR, CAFE], session_id="s-scope"
    )
    cafe = facts[CAFE[2]]
    wrong = ConsolidatedFact(
        content=_QUARTERLY[2],
        scope="project:unrelated",
        confidence=0.9,
        source_fact_uuids=[cafe],
    )

    stats = await dream(
        driver,
        scope,
        stub_graphiti_client,
        DreamOperations(writes=[wrong]),
        await gather(scope),
        _QUARTERLY,
        "pass-wrong-scope",
    )

    assert [stats[key] for key in _COUNTS] == [0, 1, 1]
    assert await _written(driver, _QUARTERLY[2]) == []
    assert (await retract(scope, [cafe])).derived == [], "nothing rests on it"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_write_citing_a_fact_of_another_scope_beside_its_own_is_dropped(
    scope_graph, stub_graphiti_client, dream_apply
) -> None:
    """The flour fact is the bread project's; the cafe fact is global. A
    bread-project write citing both is dropped whole, and neither forget has
    anything to retract."""
    driver, scope = scope_graph
    _, facts = await ingest_facts(
        driver, scope, stub_graphiti_client, [FLOUR, CAFE], session_id="s-bread"
    )
    flour, cafe = facts[FLOUR[2]], facts[CAFE[2]]
    await driver.execute_query(
        "MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() SET e.scope = 'project:bread'",
        uuid=flour,
    )
    write = ConsolidatedFact(
        content=_BREAD[2],
        scope="project:bread",
        confidence=0.9,
        source_fact_uuids=[flour, cafe],
    )

    stats = await dream(
        driver,
        scope,
        stub_graphiti_client,
        DreamOperations(writes=[write]),
        await gather(scope),
        _BREAD,
        "pass-bread",
    )

    assert [stats[key] for key in _COUNTS] == [0, 1, 1]
    assert await _written(driver, _BREAD[2]) == []
    assert (await retract(scope, [cafe])).derived == []
    assert (await retract(scope, [flour])).derived == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_project_write_citing_only_a_chat_turn_is_written_and_reached(
    scope_graph, stub_graphiti_client, dream_apply
) -> None:
    """The user's chat turn named the flour supplier; a bread-project write
    cites only that turn. It is written with the turn as its source, and
    forgetting the flour fact hides the turn and retracts the write."""
    driver, scope = scope_graph
    said, facts = await ingest_facts(
        driver, scope, stub_graphiti_client, [FLOUR], session_id="s-turn"
    )
    write = ConsolidatedFact(
        content=_BREAD[2],
        scope="project:bread",
        confidence=0.9,
        source_episode_uuids=[said],
    )

    stats = await dream(
        driver,
        scope,
        stub_graphiti_client,
        DreamOperations(writes=[write]),
        await gather(scope),
        _BREAD,
        "pass-turn",
    )

    assert [stats[key] for key in _COUNTS] == [1, 0, 0]
    bread = await fact_uuid(driver, _BREAD[2])
    record = await derivation(driver, bread)
    assert (record["facts"], record["episodes"]) == ([], [said])
    assert (await retract(scope, [facts[FLOUR[2]]])).derived == [bread]
    assert bread not in await live_facts(driver)


async def _written(driver, sentence: str) -> list[str]:
    """The facts whose sentence is ``sentence``."""
    found = await rows(
        driver,
        "MATCH ()-[e:RELATES_TO {fact: $sentence}]->() RETURN e.uuid AS uuid",
        sentence=sentence,
    )
    return [row["uuid"] for row in found]
