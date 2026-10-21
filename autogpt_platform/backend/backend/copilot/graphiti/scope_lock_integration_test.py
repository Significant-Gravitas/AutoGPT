"""The graph's write lock as its writers meet it, through the production
worker against a live FalkorDB whose own Redis holds the lock
(``scope_lock.py``): a forget the lock keeps waiting fails as busy and writes
nothing (the chat tool tries once more), an ingestion it keeps waiting goes
to the back of the queue, as often as it takes, and without Redis both go
ahead.
``recall_inflight_integration_test.py``
covers a forget and an ingestion meeting mid-episode.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/scope_lock_integration_test.py
"""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture
from redis.asyncio import Redis

from backend.copilot.model import ChatSession
from backend.copilot.tools import graphiti_forget
from backend.copilot.tools.graphiti_forget import MemoryForgetConfirmTool
from backend.copilot.tools.models import MemoryForgetConfirmResponse

from . import ingest, recall_forget, scope_lock
from .falkordb_driver import AutoGPTFalkorDriver
from .memory_model import ForgetResult, MemoryForgetFailureCode
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
    scripted_responses,
    stop_ingestion_workers,
)
from .scope import MemoryScope, write_lock_key


@pytest.fixture(autouse=True)
def boundaries(mocker, scope_graph, stub_graphiti_client):
    patch_recall_boundaries(mocker, scope_graph[0], stub_graphiti_client)


@pytest_asyncio.fixture(loop_scope="function")
async def ingest_worker_cleanup(mocker: MockerFixture) -> AsyncIterator[None]:
    mocker.patch(
        "backend.copilot.graphiti.ingest.ensure_scope_scheduled",
        AsyncMock(return_value=None),
    )
    yield
    await stop_ingestion_workers()


async def _learn(driver: AutoGPTFalkorDriver, scope: MemoryScope, build: BuildClient):
    client = build(driver, scripted_responses([ALICE]))
    await ingest_through_the_worker(driver, scope, client, [ALICE], session_id="s-1")
    [alice] = await live_facts(driver)
    return alice


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_forget_the_lock_keeps_waiting_is_busy_and_writes_nothing(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup, live_lock: Redis
) -> None:
    driver, scope = scope_graph
    alice = await _learn(driver, scope, stub_graphiti_client)
    before = await edge_row(driver, alice)
    await live_lock.set(write_lock_key(scope.group_id), "a long ingestion", px=60000)

    with patch.object(recall_forget, "FORGET_LOCK_WAIT_SECONDS", 0.5):
        result = await retract(scope, [alice])

    assert result.deleted == []
    assert [f.code for f in result.failures] == [MemoryForgetFailureCode.BUSY]
    assert await edge_row(driver, alice) == before, "nothing written"
    await live_lock.delete(write_lock_key(scope.group_id))
    assert (await retract(scope, [alice])).deleted == [alice], "a retry works"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_the_chat_tool_tries_a_busy_forget_once_more(
    scope_graph, stub_graphiti_client, ingest_worker_cleanup, live_lock: Redis
) -> None:
    """The first try finds an ingestion holding the lock and writes nothing;
    the ingestion finishes while the tool waits, and the second try forgets."""
    driver, scope = scope_graph
    alice = await _learn(driver, scope, stub_graphiti_client)
    key = write_lock_key(scope.group_id)
    await live_lock.set(key, "a long ingestion", px=60000)
    tries: list[ForgetResult] = []

    async def then_the_ingestion_finishes(*args: Any, **kwargs: Any) -> ForgetResult:
        tries.append(await retract(*args, **kwargs))
        await live_lock.delete(key)
        return tries[-1]

    session = ChatSession.new(scope.owner_user_id, dry_run=False)
    with (
        patch.object(recall_forget, "FORGET_LOCK_WAIT_SECONDS", 0.3),
        patch.object(graphiti_forget, "_BUSY_RETRY_SECONDS", 0.1),
        patch.object(graphiti_forget, "retract", then_the_ingestion_finishes),
    ):
        response = await MemoryForgetConfirmTool()._execute(
            scope.owner_user_id, session, uuids=[alice]
        )

    assert [[f.code for f in t.failures] for t in tries] == [["busy"], []]
    assert isinstance(response, MemoryForgetConfirmResponse)
    assert (response.deleted_uuids, response.failed_uuids) == ([alice], [])
    assert await live_facts(driver) == {}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_worker_a_forget_keeps_waiting_requeues_the_episode(
    scope_graph,
    stub_graphiti_client,
    ingest_worker_cleanup,
    live_lock: Redis,
) -> None:
    """The first wait runs out and writes nothing; the forget finishes while
    the episode pauses, and the episode, back at the end of the queue, is
    written."""
    driver, scope = scope_graph
    key = write_lock_key(scope.group_id)
    await live_lock.set(key, "a long forget", px=60000)

    async def forget_finishes(waits: int) -> None:
        await live_lock.delete(key)

    client = stub_graphiti_client(driver, scripted_responses([BOB]))
    with (
        patch.object(ingest, "INGEST_LOCK_WAIT_SECONDS", 0.5),
        patch.object(ingest, "_pause_before_retry", forget_finishes),
    ):
        await ingest_through_the_worker(driver, scope, client, [BOB], session_id="s-q")

    assert list((await live_facts(driver)).values()) == [BOB[2]]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_worker_locked_out_wait_after_wait_keeps_the_episode(
    scope_graph,
    stub_graphiti_client,
    ingest_worker_cleanup,
    live_lock: Redis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A stuck forget keeps the lock through four waits: the episode is never
    dropped, a warning names it at the third, and it is written once the
    lock frees."""
    driver, scope = scope_graph
    key = write_lock_key(scope.group_id)
    await live_lock.set(key, "a stuck forget", px=60000)
    client = stub_graphiti_client(driver, scripted_responses([BOB]))
    completion = ingest.IngestionCompletion()

    async def pause(waits: int) -> None:
        if waits == 4:
            await live_lock.delete(key)

    with (
        patch.object(ingest, "INGEST_LOCK_WAIT_SECONDS", 0.3),
        patch.object(ingest, "_pause_before_retry", pause),
        patch.object(ingest, "get_graphiti_client", AsyncMock(return_value=client)),
    ):
        assert await ingest.enqueue_episode(
            scope,
            "s-d",
            name="conversation_s-d",
            episode_body=BOB[2],
            source_description="User message in session s-d",
            completion=completion,
        )
        completion.register()
        assert await ingest.wait_for_ingestion(completion, 20), "never completed"

    assert (completion.written, completion.pending) == (1, 0)
    assert "'conversation_s-d' is still queued" in caplog.text
    assert await count(driver, "MATCH (ep:Episodic) RETURN count(ep) AS c") == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_without_redis_both_writers_go_ahead_and_warn(
    scope_graph,
    stub_graphiti_client,
    ingest_worker_cleanup,
    caplog: pytest.LogCaptureFixture,
) -> None:
    driver, scope = scope_graph
    down = AsyncMock(side_effect=ConnectionError("redis down"))

    with patch.object(scope_lock, "get_redis_async", down):
        alice = await _learn(driver, scope, stub_graphiti_client)
        result = await retract(scope, [alice])

    assert (result.deleted, result.failures) == ([alice], [])
    assert caplog.text.count("write lock unavailable") >= 2
