"""The recall policy against a live FalkorDB: a forgotten fact and the
episode text it came from can no longer be recalled; a superseded fact is not
recalled either, while its episode, which no forget touched, still is.

Facts go in through graphiti's real ``add_episode`` (only the LLM boundary is
scripted, see ``recall_integration_fixtures.py``) and are read back through
the same ``recall`` functions warm context, ``memory_search``,
``memory_forget_search`` and the settings page use. The unit siblings
(``recall_test.py``, ``recall_forget_test.py``) pin the Cypher; this file
proves it does what it says on FalkorDB. The forget contract's hard cases are
in ``recall_forget_integration_test.py``, ``recall_hard_forget_integration_test.py``
and ``recall_dream_writers_integration_test.py``.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_integration_test.py
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from graphiti_core.edges import EntityEdge

from backend.api.features.memory import routes as memory_routes
from backend.copilot.dream.fetch import _fetch_active_facts, _fetch_recent_episodes
from backend.copilot.dream.hidden_sessions import hidden_session_ids
from backend.copilot.dream.ratification import try_ratify_on_hit
from backend.copilot.dream.ratification_hits import get_hit_count
from backend.copilot.model import ChatSession
from backend.copilot.tools import graphiti_search
from backend.copilot.tools.models import MemorySearchResponse

from . import context
from .falkordb_driver import open_driver
from .guarded_writes import WriteOutcome, supersede_unless_recalled
from .memory_model import MemoryForgetFailureCode
from .recall import FORGOTTEN_FACT, forgotten_fact_predicate, is_forgotten
from .recall_forget import retract
from .recall_integration_fixtures import (
    ALICE,
    BOB,
    CAROL,
    Fact,
    capture_spawned_tasks,
    edge_row,
    ingest_facts,
    patch_recall_boundaries,
    recalled_episodes,
    recalled_facts,
    rows,
)
from .recall_stamp import RecallProtection
from .scope import MemoryScope

_LONG_AGO = "2025-01-01T00:00:00+00:00"
# (status, expiration_reason, expired_at, invalid_at, forgotten_at) -> forgotten?
_EDGE_SHAPES = {
    ("active", None, None, None, None): False,
    ("retracted", "user_signal", _LONG_AGO, None, _LONG_AGO): True,
    ("retracted", "user_signal", _LONG_AGO, None, None): True,
    ("superseded", "user_signal", _LONG_AGO, None, None): True,
    ("superseded", "stale_fact", _LONG_AGO, None, None): False,
    ("superseded", "stale_fact", _LONG_AGO, _LONG_AGO, _LONG_AGO): True,
    ("active", None, _LONG_AGO, _LONG_AGO, None): False,
    ("active", None, _LONG_AGO, None, None): True,
}


@pytest.fixture(autouse=True)
def boundaries(mocker, scope_graph, stub_graphiti_client):
    patch_recall_boundaries(mocker, scope_graph[0], stub_graphiti_client)


@pytest.fixture
def hit_tasks(mocker) -> list[asyncio.Task]:
    return capture_spawned_tasks(mocker)


async def _settings_view(scope: MemoryScope) -> tuple[int, set[str]]:
    """The settings page's fact count and fact list for the scope."""
    overview = await memory_routes._get_overview_impl(scope.owner_user_id, None)
    listing = await memory_routes._list_facts_impl(scope.owner_user_id, None, 50)
    return overview.facts, {item.uuid for item in listing.items}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retracted_fact_leaves_recall_and_hides_its_episode(
    scope_graph, stub_graphiti_client
) -> None:
    driver, scope = scope_graph
    episode, edges = await ingest_facts(
        driver, scope, stub_graphiti_client, [ALICE, BOB]
    )
    alice, bob = edges[ALICE[2]], edges[BOB[2]]
    assert await recalled_facts(scope) == {alice, bob}
    assert await recalled_episodes(scope) == {episode}
    assert await _settings_view(scope) == (2, {alice, bob})

    first = await retract(scope, [alice])

    assert first.deleted == [alice] and first.failures == []
    row = await edge_row(driver, alice)
    assert row["status"] == "retracted"
    assert row["reason"] == "user_signal"
    assert row["expired_at"] is not None
    assert row["forgotten_at"] is not None
    assert row["invalid_at"] is None, "a forget is not a world change (Snodgrass)"
    assert (row["fact"], row["fact_redacted"]) == (FORGOTTEN_FACT, ALICE[2])
    assert await recalled_facts(scope) == {bob}
    assert await _settings_view(scope) == (1, {bob})
    # The episode holds Alice's sentence too, so it goes at once; Bob's fact
    # stays recallable as a fact.
    assert first.redacted_episodes == [episode]
    assert await recalled_episodes(scope) == set()

    second = await retract(scope, [bob])

    assert second.redacted_episodes == [episode]
    assert await recalled_facts(scope) == set()
    assert await _settings_view(scope) == (0, set())


@pytest.mark.integration
@pytest.mark.asyncio
async def test_warm_context_and_memory_search_skip_what_was_forgotten(
    scope_graph, stub_graphiti_client, hit_tasks
) -> None:
    driver, scope = scope_graph
    forgotten_episode, forgotten = await ingest_facts(
        driver, scope, stub_graphiti_client, [ALICE]
    )
    _, kept = await ingest_facts(driver, scope, stub_graphiti_client, [CAROL])
    await retract(scope, [forgotten[ALICE[2]]])

    warm = await context._fetch(scope, "Atlas")
    await asyncio.gather(*context._pending_hit_tasks)

    assert warm is not None
    assert CAROL[2] in warm
    assert ALICE[2] not in warm, "a forgotten fact reached warm context"
    session = ChatSession.new(scope.owner_user_id, dry_run=False)
    found = await graphiti_search.MemorySearchTool()._execute(
        scope.owner_user_id, session, query="Atlas"
    )
    await asyncio.gather(*hit_tasks)
    assert isinstance(found, MemorySearchResponse)
    assert any(CAROL[2] in fact for fact in found.facts)
    assert not any(ALICE[2] in fact for fact in found.facts)
    assert not any(
        ALICE[2] in episode for episode in found.recent_episodes
    ), "the forgotten fact's episode text resurfaced"
    assert forgotten_episode not in await recalled_episodes(scope)
    assert kept[CAROL[2]] in await recalled_facts(scope)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_memory_search_records_a_hit_on_what_it_returns(
    scope_graph, stub_graphiti_client, hit_tasks
) -> None:
    driver, scope = scope_graph
    _, edges = await ingest_facts(
        driver, scope, stub_graphiti_client, [CAROL], status="tentative"
    )
    session = ChatSession.new(scope.owner_user_id, dry_run=False)

    await graphiti_search.MemorySearchTool()._execute(
        scope.owner_user_id, session, query="Borealis"
    )
    await asyncio.gather(*hit_tasks)

    assert await get_hit_count(scope, edges[CAROL[2]]) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_superseded_fact_is_not_recalled(
    scope_graph, stub_graphiti_client
) -> None:
    driver, scope = scope_graph
    _, edges = await ingest_facts(driver, scope, stub_graphiti_client, [ALICE, BOB])
    alice, bob = edges[ALICE[2]], edges[BOB[2]]

    outcomes = await supersede_unless_recalled(
        driver,
        [alice],
        reason="stale_fact",
        new_status="superseded",
        group_id=scope.group_id,
        protection=RecallProtection(),
        user_id=scope.owner_user_id,
    )

    assert outcomes == [WriteOutcome.CHANGED]
    assert await recalled_facts(scope) == {bob}
    assert await _settings_view(scope) == (1, {bob})


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retracted_fact_is_neither_gathered_nor_ratified_by_the_dream(
    scope_graph, stub_graphiti_client
) -> None:
    driver, scope = scope_graph
    active_episode, active = await ingest_facts(
        driver, scope, stub_graphiti_client, [ALICE]
    )
    tentative_episode, tentative = await ingest_facts(
        driver, scope, stub_graphiti_client, [CAROL], status="tentative"
    )
    await retract(scope, [active[ALICE[2]], tentative[CAROL[2]]])

    facts = await _fetch_active_facts(driver, scope.group_id, 50)
    window_start = datetime.now(timezone.utc) - timedelta(days=14)
    episodes = await _fetch_recent_episodes(driver, scope.group_id, window_start, 50)

    assert facts == []
    assert {e.uuid for e in episodes}.isdisjoint({active_episode, tentative_episode})
    assert await try_ratify_on_hit(scope, [tentative[CAROL[2]]]) == 0
    assert (await edge_row(driver, tentative[CAROL[2]]))["status"] == "retracted"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_the_dream_leaves_out_the_sessions_a_forget_hid(
    scope_graph, stub_graphiti_client
) -> None:
    driver, scope = scope_graph
    _, forgotten = await ingest_facts(
        driver, scope, stub_graphiti_client, [ALICE], session_id="s-forgotten"
    )
    await ingest_facts(
        driver, scope, stub_graphiti_client, [CAROL], session_id="s-kept"
    )
    assert await hidden_session_ids(driver, scope.group_id) == set()

    await retract(scope, [forgotten[ALICE[2]]])

    assert await hidden_session_ids(driver, scope.group_id) == {"s-forgotten"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_the_forgotten_fact_cypher_and_python_agree(
    scope_graph, stub_graphiti_client
) -> None:
    driver, scope = scope_graph
    facts: list[Fact] = [
        (f"Person{i}", "Atlas", f"Person{i} works on Atlas")
        for i in range(len(_EDGE_SHAPES))
    ]
    _, edges = await ingest_facts(driver, scope, stub_graphiti_client, facts)
    uuids = [edges[sentence] for _, _, sentence in facts]
    for edge_uuid, shape in zip(uuids, _EDGE_SHAPES):
        status, reason, expired, invalid, forgotten = shape
        await driver.execute_query(
            "MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() SET e.status = $status, "
            "e.expiration_reason = $reason, e.expired_at = $expired, "
            "e.invalid_at = $invalid, e.forgotten_at = $forgotten",
            uuid=edge_uuid,
            status=status,
            reason=reason,
            expired=expired,
            invalid=invalid,
            forgotten=forgotten,
        )

    found = await rows(
        driver,
        "MATCH ()-[e:RELATES_TO]->() RETURN e.uuid AS uuid, "
        f"CASE WHEN {forgotten_fact_predicate('e')} THEN true ELSE false END AS f",
    )
    in_cypher = {row["uuid"]: row["f"] for row in found}
    read_back = await EntityEdge.get_by_uuids(driver, uuids)
    assert in_cypher == {edge.uuid: is_forgotten(edge) for edge in read_back}
    assert [in_cypher[edge_uuid] for edge_uuid in uuids] == list(_EDGE_SHAPES.values())


@pytest.mark.integration
@pytest.mark.asyncio
async def test_forgetting_in_a_scope_without_a_graph_creates_none(
    falkordb_available,
) -> None:
    scope = MemoryScope.for_user(f"test-{uuid.uuid4().hex[:16]}")

    result = await retract(scope, ["nonexistent"])

    assert result.deleted == []
    assert [f.code for f in result.failures] == [MemoryForgetFailureCode.NO_MATCH]
    driver = open_driver(scope)
    try:
        assert scope.group_id not in await driver.client.list_graphs()
    finally:
        await driver.close()
