"""The recall policy against a live FalkorDB: a forgotten or superseded fact,
and the episode text it came from, can no longer be recalled.

Facts go in through graphiti's real ``add_episode`` (only the LLM boundary is
scripted, as in ``dream_ratification_integration_test.py``) and are read back
through the same ``recall`` functions warm context, ``memory_search``,
``memory_forget_search`` and the settings page use. The unit siblings
(``recall_test.py``, ``recall_forget_test.py``) pin the Cypher; this file
proves it does what it says on FalkorDB.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest backend/copilot/graphiti/recall_integration_test.py
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from graphiti_core.nodes import EpisodeType

from backend.api.features.memory import routes as memory_routes
from backend.copilot.dream.fetch import _fetch_active_facts, _fetch_recent_episodes
from backend.copilot.dream.ratification import try_ratify_on_hit
from backend.copilot.dream.ratification_hits import get_hit_count
from backend.copilot.model import ChatSession
from backend.copilot.tools import graphiti_search
from backend.copilot.tools.graphiti_forget import mark_edges_superseded
from backend.copilot.tools.models import MemorySearchResponse

from . import context
from . import recall as recall_mod
from .falkordb_driver import AutoGPTFalkorDriver, open_driver
from .memory_model import MemoryForgetFailureCode
from .recall import recent_episodes, search_facts
from .recall_forget import retract
from .scope import MemoryScope
from .types import EDGE_TYPE_MAP, EDGE_TYPES, ENTITY_TYPES

ALICE = ("Alice", "Atlas", "Alice works on Atlas")
BOB = ("Bob", "Atlas", "Bob leads Atlas")
CAROL = ("Carol", "Borealis", "Carol owns Borealis")


class _FakeRedis:
    """Honours SET NX like Redis, so ``record_memory_hit`` counts correctly."""

    def __init__(self) -> None:
        self.values: dict[str, int] = {}

    async def get(self, key: str) -> bytes | None:
        return str(self.values[key]).encode() if key in self.values else None

    async def set(self, key: str, value: int, **kwargs: object) -> bool:
        if kwargs.get("nx") and key in self.values:
            return False
        self.values[key] = int(value)
        return True

    async def incr(self, key: str) -> int:
        self.values[key] = self.values.get(key, 0) + 1
        return self.values[key]

    async def expire(self, key: str, ttl_seconds: int) -> bool:
        return True


@pytest.fixture(autouse=True)
def boundaries(mocker, scope_graph, stub_graphiti_client):
    """Search through a stubbed-LLM Graphiti on the test graph; count hits in
    memory; report memory as enabled for the settings routes and tools."""
    driver, _scope = scope_graph
    mocker.patch.object(
        recall_mod,
        "get_graphiti_client",
        mocker.AsyncMock(return_value=stub_graphiti_client(driver, {})),
    )
    mocker.patch(
        "backend.data.redis_client.get_redis_async",
        mocker.AsyncMock(return_value=_FakeRedis()),
    )
    enabled = mocker.AsyncMock(return_value=True)
    mocker.patch.object(memory_routes, "is_enabled_for_user", enabled)
    mocker.patch.object(graphiti_search, "is_enabled_for_user", enabled)


@pytest.fixture
def hit_tasks(mocker) -> list[asyncio.Task]:
    """The hit-recording tasks ``memory_search`` detaches, kept so a test
    can await them instead of racing them."""
    spawned: list[asyncio.Task] = []

    def _spawn(coro, *, name: str) -> asyncio.Task:
        task = asyncio.create_task(coro, name=name)
        spawned.append(task)
        return task

    mocker.patch.object(graphiti_search, "spawn_background_task", _spawn)
    return spawned


async def _ingest(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    stub_graphiti_client,
    facts: list[tuple[str, str, str]],
    *,
    status: str = "active",
) -> tuple[str, dict[str, str]]:
    """One episode through graphiti's ``add_episode``, the LLM scripted to
    extract ``facts`` as ``(source, target, sentence)``.

    Returns the episode uuid and each fact sentence's edge uuid.
    """
    names = list(dict.fromkeys(name for s, t, _ in facts for name in (s, t)))
    responses = {
        "ExtractedEntities": {
            "extracted_entities": [{"name": n, "entity_type_id": 0} for n in names]
        },
        "NodeResolutions": {
            "entity_resolutions": [
                {"id": i, "name": n, "duplicate_candidate_id": -1}
                for i, n in enumerate(names)
            ]
        },
        "ExtractedEdges": {
            "edges": [
                {
                    "source_entity_name": source,
                    "target_entity_name": target,
                    "relation_type": "MemoryFact",
                    "fact": sentence,
                    "valid_at": None,
                    "invalid_at": None,
                }
                for source, target, sentence in facts
            ]
        },
        "MemoryFact": {"status": status, "scope": "real:global"},
        "EdgeDuplicate": {"duplicate_facts": [], "contradicted_facts": []},
    }
    result = await stub_graphiti_client(driver, responses).add_episode(
        name=f"episode-{len(facts)}-{facts[0][2]}",
        episode_body=". ".join(sentence for _, _, sentence in facts),
        source=EpisodeType.text,
        source_description="recall integration test",
        reference_time=datetime.now(timezone.utc) - timedelta(minutes=1),
        group_id=scope.group_id,
        entity_types=ENTITY_TYPES,
        edge_types=EDGE_TYPES,
        edge_type_map=EDGE_TYPE_MAP,
    )
    edges = {edge.fact: edge.uuid for edge in result.edges}
    assert set(edges) == {sentence for _, _, sentence in facts}, (
        "add_episode did not persist the scripted facts; re-run with "
        "--log-cli-level=WARNING to see why"
    )
    return result.episode.uuid, edges


async def _recalled_facts(scope: MemoryScope) -> set[str]:
    return {fact.uuid for fact in await search_facts(scope, "Atlas", limit=20)}


async def _recalled_episodes(scope: MemoryScope) -> set[str]:
    return {episode.uuid for episode in await recent_episodes(scope, 5)}


async def _settings_view(scope: MemoryScope) -> tuple[int, set[str]]:
    """The settings page's fact count and fact list for the scope."""
    overview = await memory_routes._get_overview_impl(scope.owner_user_id, None)
    listing = await memory_routes._list_facts_impl(scope.owner_user_id, None, 50)
    return overview.facts, {item.uuid for item in listing.items}


async def _edge_row(driver: AutoGPTFalkorDriver, edge_uuid: str) -> dict[str, Any]:
    records, _, _ = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO {uuid: $uuid}]->()
        RETURN e.status AS status, e.expired_at AS expired_at,
               e.invalid_at AS invalid_at, e.expiration_reason AS reason
        """,
        uuid=edge_uuid,
    )
    return records[0] if records else {}


async def _count(driver: AutoGPTFalkorDriver, query: str) -> int:
    records, _, _ = await driver.execute_query(query)
    return int(records[0]["c"])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retracted_fact_leaves_recall_and_its_episode_follows_last_fact(
    scope_graph, stub_graphiti_client
) -> None:
    driver, scope = scope_graph
    episode, edges = await _ingest(driver, scope, stub_graphiti_client, [ALICE, BOB])
    alice, bob = edges[ALICE[2]], edges[BOB[2]]

    assert await _recalled_facts(scope) == {alice, bob}
    assert await _recalled_episodes(scope) == {episode}
    assert await _settings_view(scope) == (2, {alice, bob})

    first = await retract(scope, [alice])

    assert first.deleted == [alice] and first.failures == []
    row = await _edge_row(driver, alice)
    assert row["status"] == "retracted"
    assert row["reason"] == "user_signal"
    assert row["expired_at"] is not None
    assert row["invalid_at"] is None, "a forget is not a world change (Snodgrass)"
    assert await _recalled_facts(scope) == {bob}
    assert await _settings_view(scope) == (1, {bob})
    # Bob's fact still comes from this episode, so its text stays recallable.
    assert first.redacted_episodes == []
    assert await _recalled_episodes(scope) == {episode}

    second = await retract(scope, [bob])

    assert second.redacted_episodes == [episode]
    assert await _recalled_facts(scope) == set()
    assert await _recalled_episodes(scope) == set()
    assert await _settings_view(scope) == (0, set())


@pytest.mark.integration
@pytest.mark.asyncio
async def test_warm_context_and_memory_search_skip_what_was_forgotten(
    scope_graph, stub_graphiti_client, hit_tasks
) -> None:
    driver, scope = scope_graph
    forgotten_episode, forgotten = await _ingest(
        driver, scope, stub_graphiti_client, [ALICE]
    )
    _, kept = await _ingest(driver, scope, stub_graphiti_client, [CAROL])
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
    assert forgotten_episode not in await _recalled_episodes(scope)
    assert kept[CAROL[2]] in await _recalled_facts(scope)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_memory_search_records_a_hit_on_what_it_returns(
    scope_graph, stub_graphiti_client, hit_tasks
) -> None:
    driver, scope = scope_graph
    _, edges = await _ingest(
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
async def test_hard_retract_removes_edge_orphaned_episode_and_entities(
    scope_graph, stub_graphiti_client
) -> None:
    driver, scope = scope_graph
    episode, edges = await _ingest(driver, scope, stub_graphiti_client, [ALICE, BOB])
    alice, bob = edges[ALICE[2]], edges[BOB[2]]

    first = await retract(scope, [alice], hard=True)

    # Bob's fact keeps the episode, and the episode's mentions keep every
    # entity it names.
    assert first.deleted == [alice]
    assert first.deleted_episodes == [] and first.deleted_entities == []
    assert await _edge_row(driver, alice) == {}
    records, _, _ = await driver.execute_query(
        "MATCH (ep:Episodic {uuid: $uuid}) RETURN ep.entity_edges AS edges",
        uuid=episode,
    )
    assert records[0]["edges"] == [bob]

    second = await retract(scope, [bob], hard=True)

    assert second.deleted_episodes == [episode]
    assert len(second.deleted_entities) == 3  # Alice, Bob, Atlas
    assert await _count(driver, "MATCH (n:Episodic) RETURN count(n) AS c") == 0
    assert await _count(driver, "MATCH (n:Entity) RETURN count(n) AS c") == 0
    assert await _count(driver, "MATCH ()-[e]->() RETURN count(e) AS c") == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_superseded_fact_is_not_recalled(
    scope_graph, stub_graphiti_client
) -> None:
    driver, scope = scope_graph
    _, edges = await _ingest(driver, scope, stub_graphiti_client, [ALICE, BOB])
    alice, bob = edges[ALICE[2]], edges[BOB[2]]

    demoted, failed = await mark_edges_superseded(
        driver,
        [alice],
        reason="stale_fact",
        user_id=scope.owner_user_id,
        group_id=scope.group_id,
    )

    assert (demoted, failed) == ([alice], [])
    assert await _recalled_facts(scope) == {bob}
    assert await _settings_view(scope) == (1, {bob})


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retracted_fact_is_neither_gathered_nor_ratified_by_the_dream(
    scope_graph, stub_graphiti_client
) -> None:
    driver, scope = scope_graph
    active_episode, active = await _ingest(driver, scope, stub_graphiti_client, [ALICE])
    tentative_episode, tentative = await _ingest(
        driver, scope, stub_graphiti_client, [CAROL], status="tentative"
    )
    await retract(scope, [active[ALICE[2]], tentative[CAROL[2]]])

    facts = await _fetch_active_facts(driver, scope.group_id, 50)
    window_start = datetime.now(timezone.utc) - timedelta(days=14)
    episodes = await _fetch_recent_episodes(driver, scope.group_id, window_start, 50)

    assert facts == []
    assert {e.uuid for e in episodes}.isdisjoint({active_episode, tentative_episode})
    assert await try_ratify_on_hit(scope, [tentative[CAROL[2]]]) == 0
    assert (await _edge_row(driver, tentative[CAROL[2]]))["status"] == "retracted"


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
