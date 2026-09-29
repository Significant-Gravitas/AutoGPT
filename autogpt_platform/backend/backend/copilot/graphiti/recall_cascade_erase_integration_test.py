"""A hard forget erases the sentences the dream derived from the fact it
forgot, on a live FalkorDB, the dream's facts written through
``dream/apply.py`` and the production ingestion worker (``recall_erase.py``).

The bakery is ``recall_cascade_fixtures.py``'s. A soft forget of its flour
supplier keeps each derived sentence as an audit copy
(``recall_cascade_integration_test.py``); a hard one leaves none anywhere in
the graph, while a user's own words stay.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_cascade_erase_integration_test.py
"""

from collections.abc import AsyncIterator
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from backend.copilot.dream import apply

from . import recall
from .falkordb_driver import AutoGPTFalkorDriver
from .recall import FORGOTTEN_FACT
from .recall_cascade_fixtures import (
    BOULE,
    BOULE_FLOUR,
    FLOUR,
    SUPPLIES,
    WEEKLY,
    build_bakery,
)
from .recall_forget import retract
from .recall_integration_fixtures import (
    Fact,
    edge_row,
    episode_row,
    ingest_facts,
    live_facts,
    patch_recall_boundaries,
    rows,
    sentence_properties,
    stop_ingestion_workers,
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


async def _user_turn(driver: AutoGPTFalkorDriver, fact: str) -> str:
    """The user's chat turn that stated ``fact``."""
    [row] = await rows(
        driver,
        "MATCH (ep:Episodic) "
        "WHERE $fact IN ep.entity_edges AND ep.derived_from_facts IS NULL "
        "RETURN ep.uuid AS uuid",
        fact=fact,
    )
    return row["uuid"]


# Said after the forget, between the entities the erased facts still join, so
# graphiti's dedup compares it with them, embeddings gone.
_PAYS: Fact = (
    "Sunrise Bakery",
    "Hill Country Mills",
    "Sunrise Bakery pays Hill Country Mills every month",
)


async def _keys(driver: AutoGPTFalkorDriver, uuids: list[str]) -> dict[str, set[str]]:
    """Every property each of the edges ``uuids`` holds."""
    found = await rows(
        driver,
        "MATCH ()-[e:RELATES_TO]->() WHERE e.uuid IN $uuids "
        "RETURN e.uuid AS uuid, keys(e) AS keys",
        uuids=uuids,
    )
    return {row["uuid"]: set(row["keys"]) for row in found}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_hard_forget_erases_the_sentences_the_dream_derived(
    scope_graph, stub_graphiti_client, dream_apply
) -> None:
    """The forgotten fact is deleted and its chat turn emptied. What the dream
    derived is retracted softly, its edges and reasons kept, and its text
    erased: each derived fact reads the placeholder with blank audit copies,
    the superseded proposal the walk only went through included, and each
    dream episode loses its body and rationale. Each derived fact's
    embedding goes too, and graphiti's searches still run past it. The
    user's turn about the boule, hidden too because graphiti listed the
    consolidation among its edges (as it lists an edge an episode
    invalidated), keeps its text."""
    driver, scope = scope_graph
    bakery = await build_bakery(driver, scope, stub_graphiti_client)
    await driver.execute_query(
        "MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() "
        "SET e.status = 'superseded', e.expiration_reason = 'stale_fact', "
        "e.expired_at = $now",
        uuid=bakery.boule_flour,
        now=datetime.now(timezone.utc).isoformat(),
    )
    boule_turn = await _user_turn(driver, bakery.boule)
    await driver.execute_query(
        "MATCH (ep:Episodic {uuid: $uuid}) "
        "SET ep.entity_edges = ep.entity_edges + [$edge]",
        uuid=boule_turn,
        edge=bakery.supplies,
    )

    before = await _keys(driver, bakery.derived())
    assert all("fact_embedding" in keys for keys in before.values())

    result = await retract(scope, [bakery.flour], hard=True)

    assert (result.deleted, result.failures) == ([bakery.flour], [])
    assert sorted(result.derived) == sorted([bakery.supplies, bakery.weekly])
    assert await edge_row(driver, bakery.flour) == {}
    assert (await episode_row(driver, bakery.said))["hard_deleted_at"] is not None
    retired = {uuid: await edge_row(driver, uuid) for uuid in bakery.derived()}
    assert {uuid: row["reason"] for uuid, row in retired.items()} == {
        bakery.supplies: f"derived_from_forgotten:{bakery.flour}",
        bakery.weekly: f"derived_from_forgotten:{bakery.flour}",
        bakery.boule_flour: "stale_fact",
    }
    for row in retired.values():
        assert (row["fact"], row["fact_redacted"]) == (FORGOTTEN_FACT, "")
        assert (row["name"], row["name_redacted"]) == (FORGOTTEN_FACT, "")
    after = await _keys(driver, bakery.derived())
    assert {uuid: before[uuid] - keys for uuid, keys in after.items()} == dict.fromkeys(
        bakery.derived(), {"fact_embedding"}
    ), "the embedding goes, and nothing else the edge had"
    searched = await recall.search_facts(scope, "Sunrise Bakery flour", limit=10)
    assert {fact.uuid for fact in searched} <= {bakery.boule, bakery.cafe}
    _, pays = await ingest_facts(driver, scope, stub_graphiti_client, [_PAYS])
    assert pays[_PAYS[2]] in await live_facts(driver), "dedup ran past them"
    for sentence in (FLOUR[2], SUPPLIES[2], BOULE_FLOUR[2], WEEKLY[2]):
        assert await sentence_properties(driver, sentence) == set(), sentence
    dream_texts = await rows(
        driver,
        "MATCH (ep:Episodic) WHERE ep.uuid IN $uuids "
        "RETURN ep.content AS content, ep.source_description AS description, "
        "ep.redacted_at IS NOT NULL AS hidden ORDER BY ep.name",
        uuids=[
            bakery.supplies_episode,
            bakery.boule_flour_episode,
            bakery.weekly_episode,
        ],
    )
    assert [(r["content"], r["description"], r["hidden"]) for r in dream_texts] == [
        ("", "dream-pass consolidation", True),
        ("", "dream-pass proposal", True),
        ("", "dream-pass proposal", True),
    ]
    turn = await episode_row(driver, boule_turn)
    assert turn["redacted_at"] is not None and BOULE[2] in turn["content"]
    assert set(await live_facts(driver)) == {bakery.boule, bakery.cafe, pays[_PAYS[2]]}
