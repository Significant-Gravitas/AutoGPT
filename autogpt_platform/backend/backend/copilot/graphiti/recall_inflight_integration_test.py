"""A forget that lands while an ingestion runs, through the production worker
against a live FalkorDB.

graphiti's ``add_episode`` saves the edges and entities it read when it
started (``SET r = edge``, ``SET n = node``), so a forget landing mid-episode
used to be overwritten: the fact came back live, sentence and all. A forget
now stashes what it set before touching the graph (``recall_stash.py``), and
the worker applies it again once ``add_episode`` is done
(``recall_ingest.py``), also when the forget ran in another process: one
case runs it on its own event loop and Redis connection. Reproduced first by
an independent validation (``r3-ingestion-independent.py``, ``mid_add_race``).

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_inflight_integration_test.py
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from graphiti_core.driver.falkordb_driver import FalkorDriverSession
from pytest_mock import MockerFixture
from redis.asyncio import Redis

from backend.copilot.dream.hidden_sessions import hidden_session_ids

from . import recall_stash
from .config import graphiti_config
from .falkordb_driver import AutoGPTFalkorDriver
from .recall import FORGOTTEN_FACT
from .recall_forget import retract
from .recall_integration_fixtures import (
    ALICE,
    BuildClient,
    count,
    edge_row,
    episode_row,
    ingest_through_the_worker,
    live_facts,
    patch_recall_boundaries,
    recalled_episodes,
    recalled_facts,
    scripted_responses,
    sentence_properties,
    stop_ingestion_workers,
)
from .scope import MemoryScope, forget_stash_key

Forget = Callable[[], Awaitable[object]]


@pytest.fixture(autouse=True)
def boundaries(mocker, scope_graph, stub_graphiti_client):
    patch_recall_boundaries(mocker, scope_graph[0], stub_graphiti_client)


@pytest_asyncio.fixture(loop_scope="function")
async def ingest_worker_cleanup(mocker: MockerFixture) -> AsyncIterator[None]:
    mocker.patch(
        "backend.copilot.dream.scheduling.ensure_dream_system_scheduled",
        AsyncMock(return_value=None),
    )
    yield
    await stop_ingestion_workers()


class SharedRedis:
    """The FalkorDB server's own Redis, one client per event loop like
    ``get_redis_async``: two loops share it as two processes would."""

    def __init__(self) -> None:
        self.clients: dict[asyncio.AbstractEventLoop, Redis] = {}

    async def __call__(self) -> Redis:
        loop = asyncio.get_running_loop()
        if loop not in self.clients:
            self.clients[loop] = Redis(
                host=graphiti_config.falkordb_host,
                port=graphiti_config.falkordb_port,
                password=graphiti_config.falkordb_password or None,
                decode_responses=True,
            )
        return self.clients[loop]

    async def close_here(self) -> None:
        client = self.clients.pop(asyncio.get_running_loop(), None)
        if client is not None:
            await client.aclose()


@contextmanager
def _forget_during(marker: str, forget: Forget) -> Iterator[list[bool]]:
    """Run ``forget`` right before graphiti's first query containing
    ``marker``, then let that query go ahead."""
    original = FalkorDriverSession.run
    fired: list[bool] = []

    async def run(session: FalkorDriverSession, query: Any, **params: Any) -> Any:
        text = query if isinstance(query, str) else " ".join(q for q, _ in query)
        if not fired and marker in text:
            fired.append(True)
            await forget()
        return await original(session, query, **params)

    with patch.object(FalkorDriverSession, "run", run):
        yield fired


async def _learn(
    driver: AutoGPTFalkorDriver, scope: MemoryScope, build: BuildClient
) -> str:
    client = build(driver, scripted_responses([ALICE]))
    await ingest_through_the_worker(
        driver, scope, client, [ALICE], session_id="s-first"
    )
    [alice] = await live_facts(driver)
    return alice


async def _say_again(
    driver: AutoGPTFalkorDriver, scope: MemoryScope, build: BuildClient
) -> str:
    client = build(driver, scripted_responses([ALICE]))
    return await ingest_through_the_worker(
        driver, scope, client, [ALICE], session_id="s-again"
    )


async def _assert_forgotten(
    driver: AutoGPTFalkorDriver, scope: MemoryScope, alice: str
) -> None:
    row = await edge_row(driver, alice)
    assert row["forgotten_at"] is not None, "the forget's marker is back"
    assert (row["status"], row["reason"]) == ("retracted", "user_signal")
    assert (row["fact"], row["fact_redacted"]) == (FORGOTTEN_FACT, ALICE[2])
    assert row["name_redacted"] == "MemoryFact"
    assert await live_facts(driver) == {}
    assert await recalled_facts(scope) == set()
    assert await recalled_episodes(scope) == set(), "both episodes hidden"
    assert await hidden_session_ids(driver, scope.group_id) == {"s-first", "s-again"}
    assert await sentence_properties(driver, ALICE[2]) == {
        "Episodic.content",
        "RELATES_TO.fact_redacted",
    }, "an entity kept what graphiti read out of the sentence"


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "marker", ["SET r = edge", "SET n = node"], ids=["edge-save", "entity-save"]
)
async def test_a_forget_landing_while_graphiti_saves_is_applied_again(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup, marker: str
) -> None:
    """Before the edge save, graphiti writes its older copy of the edge back;
    before the entity save, also the summaries it built from the sentence.
    The forget applied again keeps the first one's times."""
    driver, scope = scope_graph
    alice = await _learn(driver, scope, stub_graphiti_client)
    stamped: list[dict[str, Any]] = []

    async def forget() -> None:
        await retract(scope, [alice])
        stamped.append(await edge_row(driver, alice))

    with _forget_during(marker, forget) as fired:
        await _say_again(driver, scope, stub_graphiti_client)

    assert fired, "the save was not reached"
    await _assert_forgotten(driver, scope, alice)
    row, [first] = await edge_row(driver, alice), stamped
    assert (row["forgotten_at"], row["expired_at"]) == (
        first["forgotten_at"],
        first["expired_at"],
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_an_unrelated_episode_saved_after_a_forget_loses_what_it_read(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup
) -> None:
    """The episode states another fact about Alice: graphiti saves her with
    the summary it read before the forget, sentence included."""
    driver, scope = scope_graph
    alice = await _learn(driver, scope, stub_graphiti_client)
    visit = ("Alice", "Borealis", "Alice visits Borealis")
    client = stub_graphiti_client(driver, scripted_responses([visit]))

    with _forget_during("SET n = node", lambda: retract(scope, [alice])) as fired:
        await ingest_through_the_worker(
            driver, scope, client, [visit], session_id="s-other"
        )

    assert fired, "the save was not reached"
    assert list((await live_facts(driver)).values()) == [visit[2]]
    assert await sentence_properties(driver, ALICE[2]) == {
        "Episodic.content",
        "RELATES_TO.fact_redacted",
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_forget_from_another_process_is_applied_again(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup, mocker
) -> None:
    """The forget runs on its own event loop and Redis connection; only
    Redis and the graph are shared, as between two processes."""
    driver, scope = scope_graph
    shared = SharedRedis()
    mocker.patch.object(recall_stash, "get_redis_async", shared)
    alice = await _learn(driver, scope, stub_graphiti_client)

    async def elsewhere() -> None:
        try:
            await retract(scope, [alice])
        finally:
            await shared.close_here()

    def forget() -> Awaitable[object]:
        return asyncio.to_thread(asyncio.run, elsewhere())

    try:
        with _forget_during("SET r = edge", forget) as fired:
            await _say_again(driver, scope, stub_graphiti_client)
        assert fired, "the save was not reached"
        await _assert_forgotten(driver, scope, alice)
    finally:
        redis = await shared()
        await redis.delete(forget_stash_key(scope.group_id))
        await shared.close_here()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_hard_forget_landing_before_the_entities_are_saved_is_redone(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup
) -> None:
    """graphiti then saves the entities it read back into existence, and the
    deleted edge with them; the hard forget runs again and no edge is left
    citing an episode it emptied."""
    driver, scope = scope_graph
    alice = await _learn(driver, scope, stub_graphiti_client)

    async def forget() -> object:
        return await retract(scope, [alice], hard=True)

    with _forget_during("SET n = node", forget) as fired:
        again = await _say_again(driver, scope, stub_graphiti_client)

    assert fired, "the save was not reached"
    assert await edge_row(driver, alice) == {}
    assert await live_facts(driver) == {}
    assert await count(driver, "MATCH ()-[e]->() RETURN count(e) AS c") == 0
    assert await count(driver, "MATCH (n:Entity) RETURN count(n) AS c") == 0
    assert (await episode_row(driver, again))["content"] == ""
    assert await recalled_episodes(scope) == set()
    assert await hidden_session_ids(driver, scope.group_id) == {"s-first", "s-again"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_forget_landing_before_graphiti_resolves_the_fact_covers_it(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup
) -> None:
    """graphiti then reads the placeholder and saves the sentence as a new
    edge; it was said before the forget, so the forget covers it too."""
    driver, scope = scope_graph
    alice = await _learn(driver, scope, stub_graphiti_client)
    client = stub_graphiti_client(driver, scripted_responses([ALICE]))
    answer = client.llm_client._generate_response
    fired: list[bool] = []

    async def extract(messages, response_model=None, *args: Any, **kwargs: Any):
        if response_model is not None and response_model.__name__ == "ExtractedEdges":
            if not fired:
                fired.append(True)
                await retract(scope, [alice])
        return await answer(messages, response_model, *args, **kwargs)

    with patch.object(client.llm_client, "_generate_response", side_effect=extract):
        await ingest_through_the_worker(
            driver, scope, client, [ALICE], session_id="s-again"
        )

    assert fired, "edge extraction was not reached"
    await _assert_forgotten(driver, scope, alice)
    total = "MATCH ()-[e:RELATES_TO]->() RETURN count(e) AS c"
    assert await count(driver, total) == 2, "the statement's own edge, forgotten"
