"""A hard forget against a live FalkorDB: what it deletes, what it keeps as a
tombstone, and that a failed clean-up finishes when the forget is repeated.

An episode no remaining edge cites is emptied into a tombstone, never
deleted: its name, source description and envelope provenance say which
chat session it came from, and the dream leaves that session out. The edge
itself is deleted last, in one query with what it alone kept, so a forget
that failed part-way finds it again. Each case was reproduced against the
previous commit by an independent validation (``r2-dream-atomicity.py``,
``r2-failure.py``).

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_hard_forget_integration_test.py
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest_mock import MockerFixture

from backend.api.features.admin import memory_admin_routes
from backend.api.features.admin.memory_admin_routes import (
    FactListResponse,
    GraphResponse,
)
from backend.copilot.dream import fetch
from backend.copilot.dream.hidden_sessions import hidden_session_ids
from backend.copilot.dream.prompts import build_consolidate_prompt

from . import recall_hide, recall_orphans
from .falkordb_driver import AutoGPTFalkorDriver
from .memory_model import MemoryEnvelope, MemoryForgetFailureCode
from .recall_forget import retract
from .recall_integration_fixtures import (
    ALICE,
    BOB,
    CAROL,
    count,
    edge_row,
    episode_row,
    ingest_facts,
    patch_recall_boundaries,
    recalled_episodes,
    recalled_facts,
    rows,
)
from .scope import MemoryScope

_STEPS = {
    "scrub": recall_hide.SCRUB_FACTS_QUERY,
    "redaction": recall_hide.REDACT_EPISODES_QUERY,
    "citing-read": recall_orphans._CITING_EPISODES_QUERY,
    "tombstone": recall_orphans._TOMBSTONE_QUERY,
    "edge-delete": recall_orphans._DELETE_EDGE_QUERY,
}


@pytest.fixture(autouse=True)
def boundaries(mocker, scope_graph, stub_graphiti_client):
    patch_recall_boundaries(mocker, scope_graph[0], stub_graphiti_client)


async def _dangling(driver: AutoGPTFalkorDriver) -> list[str]:
    """Episode references to a deleted edge, and edge references to a
    deleted episode."""
    to_edges = await rows(
        driver,
        """
        MATCH (ep:Episodic)
        UNWIND coalesce(ep.entity_edges, []) AS ref
        OPTIONAL MATCH ()-[e:RELATES_TO {uuid: ref}]->()
        WITH ep, ref, e WHERE e IS NULL
        RETURN ep.uuid + ' -> ' + ref AS dangling
        """,
    )
    to_episodes = await rows(
        driver,
        """
        MATCH ()-[e:RELATES_TO]->()
        UNWIND coalesce(e.episodes, []) AS ref
        OPTIONAL MATCH (ep:Episodic {uuid: ref})
        WITH e, ref, ep WHERE ep IS NULL
        RETURN e.uuid + ' -> ' + ref AS dangling
        """,
    )
    return [row["dangling"] for row in to_edges + to_episodes]


async def _admin_views(
    mocker: MockerFixture, scope: MemoryScope
) -> tuple[FactListResponse, GraphResponse]:
    """The admin facts list (any status) and graph view (episodes shown)."""
    mocker.patch.object(
        memory_admin_routes,
        "_resolve_and_audit_memory_scope",
        AsyncMock(return_value=scope),
    )
    common = {
        "request": MagicMock(),
        "user_id": scope.owner_user_id,
        "caller_id": "admin",
        "jwt_payload": {},
        "expert_id": None,
    }
    facts = await memory_admin_routes._list_facts_impl(
        **common, limit=50, status="any", scope=None
    )
    graph = await memory_admin_routes._get_graph_impl(
        **common,
        node_limit=100,
        edge_limit=100,
        include_episodes=True,
        include_communities=False,
    )
    return facts, graph


def _chat_store(*session_ids: str) -> SimpleNamespace:
    message = SimpleNamespace(role="user", content="Whole session text")
    return SimpleNamespace(
        get_user_chat_sessions=AsyncMock(
            return_value=[SimpleNamespace(session_id=s, title=s) for s in session_ids]
        ),
        get_chat_messages_paginated=AsyncMock(
            return_value=SimpleNamespace(messages=[message])
        ),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hard_forget_tombstones_the_last_episode_and_deletes_orphans(
    scope_graph, stub_graphiti_client
) -> None:
    driver, scope = scope_graph
    episode, edges = await ingest_facts(
        driver, scope, stub_graphiti_client, [ALICE, BOB]
    )
    alice, bob = edges[ALICE[2]], edges[BOB[2]]

    first = await retract(scope, [alice], hard=True)

    # Bob's fact keeps the episode, hidden because it holds Alice's sentence,
    # and the episode's mentions keep every entity it names.
    assert first.deleted == [alice] and first.failures == []
    assert first.tombstoned_episodes == [] and first.deleted_entities == []
    assert first.redacted_episodes == [episode]
    assert await edge_row(driver, alice) == {}
    assert (await episode_row(driver, episode))["entity_edges"] == [bob]

    second = await retract(scope, [bob], hard=True)

    assert second.tombstoned_episodes == [episode]
    assert len(second.deleted_entities) == 3  # Alice, Bob, Atlas
    tombstone = await episode_row(driver, episode)
    assert (tombstone["content"], tombstone["entity_edges"]) == ("", [])
    assert tombstone["hard_deleted_at"] and tombstone["redacted_at"]
    assert tombstone["name"], "a tombstone keeps its name"
    assert await count(driver, "MATCH (n:Episodic) RETURN count(n) AS c") == 1
    assert await count(driver, "MATCH (n:Entity) RETURN count(n) AS c") == 0
    assert await count(driver, "MATCH ()-[e]->() RETURN count(e) AS c") == 0
    assert await recalled_episodes(scope) == set()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("named_by", ["chat-turn", "stored-memory"])
async def test_hard_forget_keeps_the_session_out_of_the_dream(
    scope_graph, stub_graphiti_client, mocker, named_by: str
) -> None:
    """The episode's name says its session, or only its stored memory's
    envelope does: either way the tombstone still says it."""
    driver, scope = scope_graph
    if named_by == "chat-turn":
        forgotten, edges = await ingest_facts(
            driver, scope, stub_graphiti_client, [ALICE], session_id="s-forgotten"
        )
    else:
        envelope = MemoryEnvelope(
            content=ALICE[2], provenance="session:s-forgotten#msg:7"
        )
        forgotten, edges = await ingest_facts(
            driver,
            scope,
            stub_graphiti_client,
            [ALICE],
            body=envelope.model_dump_json(),
        )
    await ingest_facts(
        driver, scope, stub_graphiti_client, [CAROL], session_id="s-kept"
    )

    result = await retract(scope, [edges[ALICE[2]]], hard=True)

    assert result.tombstoned_episodes == [forgotten]
    assert await hidden_session_ids(driver, scope.group_id) == {"s-forgotten"}
    store = _chat_store("s-forgotten", "s-kept")
    mocker.patch.object(fetch, "chat_db", return_value=store)
    bundle = await fetch.gather_dream_input(scope)
    assert [session.session_id for session in bundle.recent_sessions] == ["s-kept"]
    read = store.get_chat_messages_paginated.await_args_list
    assert [call.kwargs["session_id"] for call in read] == ["s-kept"]
    prompt = str(build_consolidate_prompt(bundle))
    assert CAROL[2] in prompt and ALICE[2] not in prompt


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("step", list(_STEPS))
async def test_a_failed_hard_forget_finishes_when_repeated(
    scope_graph, stub_graphiti_client, step: str
) -> None:
    driver, scope = scope_graph
    episode, edges = await ingest_facts(
        driver, scope, stub_graphiti_client, [ALICE], session_id="s-forgotten"
    )
    alice = edges[ALICE[2]]
    original = AutoGPTFalkorDriver.execute_query

    async def lose_one_step(self, cypher_query_, **params):
        if cypher_query_ == _STEPS[step]:
            raise RuntimeError(f"{step} lost")
        return await original(self, cypher_query_, **params)

    with patch.object(AutoGPTFalkorDriver, "execute_query", lose_one_step):
        failed = await retract(scope, [alice], hard=True)

    assert failed.deleted == []
    assert [(f.uuid, f.code) for f in failed.failures] == [
        (alice, MemoryForgetFailureCode.CLEANUP_ERROR)
    ]
    assert await edge_row(driver, alice) != {}, "kept, so a retry finds it"
    assert alice not in await recalled_facts(scope)
    assert episode not in await recalled_episodes(scope)

    retry = await retract(scope, [alice], hard=True)

    assert (retry.deleted, retry.failures) == ([alice], [])
    assert await edge_row(driver, alice) == {}
    tombstone = await episode_row(driver, episode)
    assert (tombstone["content"], tombstone["entity_edges"]) == ("", [])
    assert tombstone["hard_deleted_at"] is not None
    assert await hidden_session_ids(driver, scope.group_id) == {"s-forgotten"}
    assert await _dangling(driver) == []
    assert await count(driver, "MATCH (n:Entity) RETURN count(n) AS c") == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_the_admin_audit_views_keep_what_a_forget_hid(
    scope_graph, stub_graphiti_client, mocker
) -> None:
    """The facts list and the graph view show a forgotten fact's own text and
    a tombstone's stamp, never its text."""
    driver, scope = scope_graph
    kept_episode, kept = await ingest_facts(
        driver, scope, stub_graphiti_client, [CAROL], session_id="s-soft"
    )
    gone_episode, gone = await ingest_facts(
        driver, scope, stub_graphiti_client, [ALICE], session_id="s-hard"
    )
    await retract(scope, [kept[CAROL[2]]])
    await retract(scope, [gone[ALICE[2]]], hard=True)

    facts, graph = await _admin_views(mocker, scope)

    [fact] = facts.items
    assert (fact.uuid, fact.status) == (kept[CAROL[2]], "retracted")
    assert (fact.name, fact.fact) == ("MemoryFact", CAROL[2])
    episodes = {node.uuid: node for node in graph.nodes if node.label == "Episodic"}
    assert episodes[gone_episode].hard_deleted_at is not None
    assert episodes[kept_episode].hard_deleted_at is None
    [edge] = [edge for edge in graph.edges if edge.label == "RELATES_TO"]
    assert edge.fact == CAROL[2]
