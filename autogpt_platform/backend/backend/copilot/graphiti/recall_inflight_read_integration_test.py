"""A read already under way when a forget answers, on a live FalkorDB.

Warm context is paused at graphiti's cross-encoder, after its edge search
has read the fact and its episode read has returned; ``memory_search`` is
paused as graphiti's search returns. A forget answers meanwhile. When the
read resumes, its last graph read before rendering (``recall.live_now``,
``recall_recheck.recheck``) no longer finds the fact or its episode, so the
response shows neither, while a live fact and its episode still show.
Reproduced first by an independent validation
(``r5-quality-delayed-operations.py``, ``stale-warm-crossencoder`` and
``stale-memory-search``). A forget that answers once warm context's fact
check has passed is caught too: the recheck checks facts and episodes again
in one statement, so no forget lands between two halves of it
(``r6-read-split-check.py``).

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_inflight_read_integration_test.py
"""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any
from unittest.mock import patch

import pytest
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodicNode

from backend.copilot.model import ChatSession
from backend.copilot.tools.graphiti_search import MemorySearchTool

from . import context, recall
from .falkordb_driver import AutoGPTFalkorDriver
from .recall_forget import retract
from .recall_integration_fixtures import (
    ALICE,
    BOB,
    BuildClient,
    capture_spawned_tasks,
    ingest_facts,
    patch_recall_boundaries,
)
from .scope import MemoryScope


@pytest.fixture(autouse=True)
def boundaries(mocker, scope_graph, stub_graphiti_client):
    patch_recall_boundaries(mocker, scope_graph[0], stub_graphiti_client)
    mocker.patch.object(context, "_spawn_ratification_hits")
    capture_spawned_tasks(mocker)


async def _alice_and_bob(
    driver: AutoGPTFalkorDriver, scope: MemoryScope, build: BuildClient
) -> str:
    """Alice's and Bob's facts, each from its own chat turn: Alice's uuid."""
    _, alice = await ingest_facts(driver, scope, build, [ALICE], session_id="s-1")
    await ingest_facts(driver, scope, build, [BOB], session_id="s-2")
    return alice[ALICE[2]]


async def _forget_while_paused(
    scope: MemoryScope,
    alice: str,
    read: Callable[[], Coroutine[Any, Any, Any]],
    pause: Callable[[asyncio.Event, asyncio.Event], Any],
) -> Any:
    """Start ``read``, forget Alice once it reaches ``pause``, then let it
    finish: what it returns."""
    reached, release = asyncio.Event(), asyncio.Event()
    with pause(reached, release):
        task = asyncio.create_task(read())
        await asyncio.wait_for(reached.wait(), 20)
        forgot = await retract(scope, [alice])
        release.set()
        returned = await asyncio.wait_for(task, 20)
    assert (forgot.deleted, forgot.failures) == ([alice], [])
    return returned


@pytest.mark.integration
@pytest.mark.asyncio
async def test_warm_context_paused_at_its_reranker_shows_nothing_forgotten(
    scope_graph, stub_graphiti_client
) -> None:
    driver, scope = scope_graph
    alice = await _alice_and_bob(driver, scope, stub_graphiti_client)
    client = await recall.get_graphiti_client(scope.group_id)
    rank, recent = client.cross_encoder.rank, context.recent_episodes
    reranked: list[str] = []
    read_back: list[EpisodicNode] = []

    async def episodes(scope: MemoryScope, n: int) -> list[EpisodicNode]:
        read_back.extend(await recent(scope, n))
        return read_back

    def pause(reached: asyncio.Event, release: asyncio.Event):
        async def paused(query: str, passages: list[str]) -> list[tuple[str, float]]:
            reranked.extend(passages)
            reached.set()
            await release.wait()
            return await rank(query, passages)

        return patch.object(client.cross_encoder, "rank", paused)

    with patch.object(context, "recent_episodes", episodes):
        shown = await _forget_while_paused(
            scope, alice, lambda: context._fetch(scope, "Alice Bob Atlas"), pause
        )

    assert ALICE[2] in reranked, "the search read the fact before the forget"
    assert any(ALICE[2] in (ep.content or "") for ep in read_back)
    assert shown is not None and ALICE[2] not in shown
    assert BOB[2] in shown.split("<RECENT_EPISODES>")[0], "the live fact"
    assert BOB[2] in shown.split("<RECENT_EPISODES>")[1], "its episode"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_warm_context_after_its_fact_check_passed_shows_nothing_forgotten(
    scope_graph, stub_graphiti_client
) -> None:
    """The fact check has kept Alice's fact, a forget answers, then the rest
    of the read runs: its one last statement finds neither her fact nor her
    episode."""
    driver, scope = scope_graph
    alice = await _alice_and_bob(driver, scope, stub_graphiti_client)
    live_now = recall.live_now
    passed: list[str] = []

    def pause(reached: asyncio.Event, release: asyncio.Event):
        async def checked(
            driver: GraphDriver, facts: list[EntityEdge], **kwargs: Any
        ) -> list[EntityEdge]:
            kept = await live_now(driver, facts, **kwargs)
            passed.extend(fact.fact for fact in kept)
            reached.set()
            await release.wait()
            return kept

        return patch.object(recall, "live_now", checked)

    shown = await _forget_while_paused(
        scope, alice, lambda: context._fetch(scope, "Alice Bob Atlas"), pause
    )

    assert ALICE[2] in passed, "the fact check passed before the forget"
    assert shown is not None and ALICE[2] not in shown
    assert BOB[2] in shown.split("<RECENT_EPISODES>")[0], "the live fact"
    assert BOB[2] in shown.split("<RECENT_EPISODES>")[1], "its episode"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_memory_search_paused_as_its_search_returns_shows_nothing_forgotten(
    scope_graph, stub_graphiti_client
) -> None:
    driver, scope = scope_graph
    alice = await _alice_and_bob(driver, scope, stub_graphiti_client)
    client = await recall.get_graphiti_client(scope.group_id)
    search = client.search_
    searched: list[str] = []

    def pause(reached: asyncio.Event, release: asyncio.Event):
        async def paused(*args: Any, **kwargs: Any):
            results = await search(*args, **kwargs)
            searched.extend(edge.fact for edge in results.edges)
            reached.set()
            await release.wait()
            return results

        return patch.object(client, "search_", paused)

    async def read():
        session = ChatSession.new(scope.owner_user_id, dry_run=False)
        tool = MemorySearchTool()
        return await tool._execute(scope.owner_user_id, session, query="Atlas")

    shown = (await _forget_while_paused(scope, alice, read, pause)).model_dump_json()

    assert ALICE[2] in searched, "the search read the fact before the forget"
    assert ALICE[2] not in shown
    assert BOB[2] in shown
