"""A forget and an ingestion at the same time, through the production worker
against a live FalkorDB whose own Redis holds the graph's write lock.

graphiti's ``add_episode`` saves the edges and entities it read when it
began (``SET r = edge``, ``SET n = node``), so a forget landing mid-episode
used to be written over. Now the two never overlap (``scope_lock.py``): a
forget that arrives while an episode is being written waits for it, then
applies to what it wrote, and search never shows the forgotten sentence once
the forget has answered (``scope_lock_integration_test.py`` covers a lock
held past a writer's wait). Reproduced first by an independent validation
(``r3-ingestion-independent.py``, ``mid_add_race``;
``r4-stash-probes.py``, ``healthy_redis_pre_repair_read``).

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

from . import recall, scope_lock
from .config import graphiti_config
from .falkordb_driver import AutoGPTFalkorDriver
from .memory_model import ForgetResult
from .recall import FORGOTTEN_FACT
from .recall_forget import retract
from .recall_integration_fixtures import (
    ALICE,
    BOB,
    BuildClient,
    count,
    edge_row,
    ingest_through_the_worker,
    live_facts,
    patch_recall_boundaries,
    recalled_episodes,
    rows,
    scripted_responses,
    sentence_properties,
    stop_ingestion_workers,
)
from .scope import MemoryScope

Forget = Callable[[], Awaitable[ForgetResult]]


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


async def _learn(driver: AutoGPTFalkorDriver, scope: MemoryScope, build: BuildClient):
    client = build(driver, scripted_responses([ALICE]))
    await ingest_through_the_worker(driver, scope, client, [ALICE], session_id="s-1")
    [alice] = await live_facts(driver)
    return alice


async def _say(driver: AutoGPTFalkorDriver, scope: MemoryScope, build: BuildClient):
    client = build(driver, scripted_responses([ALICE]))
    await ingest_through_the_worker(driver, scope, client, [ALICE], session_id="s-2")


@contextmanager
def _forget_arrives_at(
    marker: str, forget: Forget, answered: asyncio.Event | None = None
) -> Iterator[tuple[list[asyncio.Future], list[bool]]]:
    """Start ``forget`` right before graphiti's first query containing
    ``marker``, note whether it is still waiting for the lock 0.6 s later,
    then let the query go (a failed check here would abort graphiti's save
    and hide the overwrite it is meant to catch)."""
    original = FalkorDriverSession.run
    started: list[asyncio.Future] = []
    waited: list[bool] = []

    async def run(session: FalkorDriverSession, query: Any, **params: Any) -> Any:
        text = query if isinstance(query, str) else " ".join(q for q, _ in query)
        if not started and marker in text:
            started.append(asyncio.ensure_future(forget()))
            if answered is not None:
                event = answered
                started[0].add_done_callback(lambda _: event.set())
            await asyncio.sleep(0.6)
            waited.append(not started[0].done())
        return await original(session, query, **params)

    with patch.object(FalkorDriverSession, "run", run):
        yield started, waited


async def _watch(scope: MemoryScope, answered: asyncio.Event, seen: list[bool]):
    """Search over and over; after the forget answered, note each time
    whether the sentence came back."""
    while True:
        after = answered.is_set()
        facts = await recall.search_facts(scope, "Alice Atlas", limit=10)
        if after:
            seen.append(any(ALICE[2] in fact.fact for fact in facts))
        await asyncio.sleep(0.05)


async def _searched(seen: list[bool], times: int) -> None:
    """Wait until the watcher has searched ``times`` times since the
    forget answered."""
    async with asyncio.timeout(20):
        while len(seen) < times:
            await asyncio.sleep(0.05)


async def _assert_forgotten(driver: AutoGPTFalkorDriver, scope: MemoryScope, alice):
    row = await edge_row(driver, alice)
    assert row["forgotten_at"] is not None
    assert (row["status"], row["reason"]) == ("retracted", "user_signal")
    assert (row["fact"], row["fact_redacted"]) == (FORGOTTEN_FACT, ALICE[2])
    assert ALICE[2] not in (await live_facts(driver)).values()
    said = await rows(
        driver,
        "MATCH (ep:Episodic) WHERE ep.name IN $names RETURN ep.uuid AS uuid",
        names=["conversation_s-1", "conversation_s-2"],
    )
    assert len(said) == 2
    hidden = {row["uuid"] for row in said}
    assert not hidden & await recalled_episodes(scope), "both episodes hidden"
    assert await hidden_session_ids(driver, scope.group_id) == {"s-1", "s-2"}
    assert await sentence_properties(driver, ALICE[2]) == {
        "Episodic.content",
        "RELATES_TO.fact_redacted",
    }


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("marker", ["SET r = edge", "SET n = node"])
async def test_a_forget_arriving_mid_ingestion_waits_then_applies(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup, marker: str
) -> None:
    """Before the forget answers it has not been applied; from the moment
    it answers, through a later ingestion, search never shows the sentence."""
    driver, scope = scope_graph
    alice = await _learn(driver, scope, stub_graphiti_client)
    answered, seen = asyncio.Event(), []
    watcher = asyncio.ensure_future(_watch(scope, answered, seen))

    forget = _forget_arrives_at(marker, lambda: retract(scope, [alice]), answered)
    with forget as (started, waited):
        await _say(driver, scope, stub_graphiti_client)
    result = await asyncio.wait_for(started[0], 30)
    await _searched(seen, 3)
    later = stub_graphiti_client(driver, scripted_responses([BOB]))
    await ingest_through_the_worker(driver, scope, later, [BOB], session_id="s-3")
    await _searched(seen, len(seen) + 3)
    watcher.cancel()

    assert (result.deleted, result.failures) == ([alice], [])
    assert not any(seen), "search showed the sentence after the forget"
    await _assert_forgotten(driver, scope, alice)
    assert waited == [True], "the forget did not wait for the ingestion"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_hard_forget_arriving_mid_ingestion_deletes_what_it_wrote(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup
) -> None:
    driver, scope = scope_graph
    alice = await _learn(driver, scope, stub_graphiti_client)

    async def forget() -> ForgetResult:
        return await retract(scope, [alice], hard=True)

    with _forget_arrives_at("SET r = edge", forget) as (started, waited):
        await _say(driver, scope, stub_graphiti_client)
    result = await asyncio.wait_for(started[0], 30)

    assert (result.deleted, result.failures) == ([alice], [])
    assert await count(driver, "MATCH ()-[e:RELATES_TO]->() RETURN count(e) AS c") == 0
    assert await recalled_episodes(scope) == set()
    assert await hidden_session_ids(driver, scope.group_id) == {"s-1", "s-2"}
    assert waited == [True], "the forget did not wait for the ingestion"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_forget_from_another_process_waits_for_the_ingestion(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup, mocker
) -> None:
    """The forget runs on its own event loop and Redis connection; only
    Redis and the graph are shared, as between two processes."""
    driver, scope = scope_graph
    clients: dict[asyncio.AbstractEventLoop, Redis] = {}

    async def per_loop() -> Redis:
        return clients.setdefault(asyncio.get_running_loop(), _server_redis())

    mocker.patch.object(scope_lock, "get_redis_async", per_loop)
    alice = await _learn(driver, scope, stub_graphiti_client)

    async def elsewhere() -> ForgetResult:
        try:
            return await retract(scope, [alice])
        finally:
            await _close(clients.pop(asyncio.get_running_loop(), None))

    def forget() -> Awaitable[ForgetResult]:
        return asyncio.to_thread(asyncio.run, elsewhere())

    with _forget_arrives_at("SET r = edge", forget) as (started, waited):
        await _say(driver, scope, stub_graphiti_client)
    result = await asyncio.wait_for(started[0], 30)
    await _close(clients.pop(asyncio.get_running_loop(), None))

    assert (result.deleted, result.failures) == ([alice], [])
    await _assert_forgotten(driver, scope, alice)
    assert waited == [True], "the forget did not wait for the ingestion"


async def _close(redis: Redis | None) -> None:
    if redis is not None:
        await redis.aclose()


def _server_redis() -> Redis:
    return Redis(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
        decode_responses=True,
    )
