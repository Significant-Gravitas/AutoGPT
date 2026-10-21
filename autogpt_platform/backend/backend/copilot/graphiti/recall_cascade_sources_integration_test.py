"""Which facts carry a dream's record, when graphiti merges a statement into
an existing fact, and so which a forget's cascade may retract, on a live
FalkorDB through ``dream/apply.py`` and the production ingestion worker
(``recall_derivation.py``, ``recall_cascade.py``).

A fact a dream write merges into that a user's own episode states is never
stamped, and a derived fact the user later states word for word keeps its
record but stays: both have a source the forget did not reach. A dream write
graphiti's model merges into a derived fact, which rewrites the fact's
attributes, leaves it the union of both writes' records.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_cascade_sources_integration_test.py
"""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from backend.copilot.dream import apply
from backend.copilot.dream.schemas import ConsolidatedFact, ProposedFinding

from .falkordb_driver import AutoGPTFalkorDriver
from .recall_cascade_fixtures import (
    BOULE,
    BOULE_FLOUR,
    FLOUR,
    SUPPLIES,
    derivation,
    dream_episode,
    dream_once,
    fact_uuid,
)
from .recall_forget import retract
from .recall_integration_fixtures import (
    BuildClient,
    Fact,
    episode_row,
    ingest_facts,
    live_facts,
    patch_recall_boundaries,
    stop_ingestion_workers,
)
from .scope import MemoryScope

_PARAPHRASE: Fact = (
    "Hill Country Mills",
    "Sunrise Bakery",
    "Sunrise Bakery's flour comes from Hill Country Mills",
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


async def _said(
    driver: AutoGPTFalkorDriver, scope: MemoryScope, build: BuildClient, fact: Fact
) -> tuple[str, str]:
    """``fact`` in a chat turn of its own: the turn's episode and the fact."""
    episode, edges = await ingest_facts(
        driver, scope, build, [fact], session_id=f"s-{fact[2][:12]}"
    )
    return episode, edges[fact[2]]


async def _supplies(
    driver: AutoGPTFalkorDriver, scope: MemoryScope, build: BuildClient
) -> tuple[str, str, str]:
    """The flour supplier said, then consolidated by a dream pass citing
    it: the fact, the consolidation and its dream episode."""
    said, flour = await _said(driver, scope, build, FLOUR)
    consolidation = ConsolidatedFact(
        content=SUPPLIES[2],
        confidence=0.9,
        source_fact_uuids=[flour],
        source_episode_uuids=[said],
    )
    await dream_once(driver, scope, build, consolidation, SUPPLIES, "pass-1")
    return (
        flour,
        await fact_uuid(driver, SUPPLIES[2]),
        await dream_episode(driver, "pass-1"),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_dream_write_merged_into_a_users_fact_leaves_it_alone(
    scope_graph, stub_graphiti_client, dream_apply
) -> None:
    """The dream restates the user's own fact word for word, citing the
    flour supplier: graphiti adds its episode to the user's fact, which is
    not stamped and outlives the forget; the dream's text is hidden."""
    driver, scope = scope_graph
    build = stub_graphiti_client
    _, flour = await _said(driver, scope, build, FLOUR)
    stated, boule = await _said(driver, scope, build, BOULE)
    restating = ConsolidatedFact(
        content=BOULE[2], confidence=0.9, source_fact_uuids=[flour, boule]
    )
    await dream_once(driver, scope, build, restating, BOULE, "pass-1")
    restated = await dream_episode(driver, "pass-1")
    before = await derivation(driver, boule)
    assert before["sources"] == [stated, restated], "merged into the user's fact"
    assert before["facts"] is None, "never stamped"

    result = await retract(scope, [flour])

    assert (result.derived, result.failures) == ([], [])
    assert boule in await live_facts(driver)
    assert (await derivation(driver, restated))["redacted_at"] is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_derived_fact_the_user_states_again_stays_and_so_does_its_kin(
    scope_graph, stub_graphiti_client, dream_apply
) -> None:
    """The user later says the consolidation word for word: graphiti adds
    their turn to it and keeps its record, but the forget passes it over,
    and the proposal resting on it, since the user stated it themselves."""
    driver, scope = scope_graph
    build = stub_graphiti_client
    flour, supplies, supplies_episode = await _supplies(driver, scope, build)
    _, boule = await _said(driver, scope, build, BOULE)
    proposal = ProposedFinding(
        content=BOULE_FLOUR[2],
        confidence=0.6,
        rationale="the bakery's flour",
        source_fact_uuids=[supplies, boule],
    )
    await dream_once(driver, scope, build, proposal, BOULE_FLOUR, "pass-2")
    again, _ = await _said(driver, scope, build, SUPPLIES)
    row = await derivation(driver, supplies)
    assert row["sources"] == [supplies_episode, again]
    assert row["facts"] == [flour], "graphiti's exact-text merge keeps the record"

    result = await retract(scope, [flour])

    assert (result.derived, result.failures) == ([], [])
    live = await live_facts(driver)
    assert {supplies, boule, await fact_uuid(driver, BOULE_FLOUR[2])} <= set(live)
    assert (await episode_row(driver, supplies_episode))["redacted_at"] is not None
    assert (await episode_row(driver, again))["redacted_at"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_dream_write_merged_into_a_derived_fact_keeps_both_records(
    scope_graph, stub_graphiti_client, dream_apply
) -> None:
    """A later pass paraphrases the consolidation citing only the user's
    bakery fact, and graphiti's model merges it in, rewriting the fact's
    attributes: the fact's record is rebuilt as the union of both dream
    episodes', so forgetting the flour supplier still retracts it."""
    driver, scope = scope_graph
    build = stub_graphiti_client
    flour, supplies, supplies_episode = await _supplies(driver, scope, build)
    _, boule = await _said(driver, scope, build, BOULE)
    said = (await derivation(driver, supplies_episode))["episodes"]
    paraphrase = ConsolidatedFact(
        content=_PARAPHRASE[2], confidence=0.9, source_fact_uuids=[boule]
    )
    await dream_once(
        driver,
        scope,
        build,
        paraphrase,
        _PARAPHRASE,
        "pass-2",
        merge_into=SUPPLIES[2],
    )
    merged = await derivation(driver, supplies)
    assert merged["sources"] == [
        supplies_episode,
        await dream_episode(driver, "pass-2"),
    ]
    assert (merged["facts"], merged["episodes"]) == ([flour, boule], said)

    result = await retract(scope, [flour])

    assert result.derived == [supplies]
    assert boule in await live_facts(driver)
