import asyncio
import json
from collections.abc import Callable
from threading import Event

import httpx
import pytest
from mem0 import MemoryClient

from backend.blocks.mem0 import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    AddMemoryBlock,
    GetAllMemoriesBlock,
    GetLatestMemoryBlock,
    SearchMemoryBlock,
)
from backend.data.execution import ExecutionContext
from backend.util.exceptions import BlockExecutionError

MEMORIES = [{"id": "memory-1", "memory": "I am a vegetarian"}]
ADDED_MEMORY = {"event": "ADD", "memory": "I am a vegetarian"}


@pytest.fixture
def mem0_transport(monkeypatch: pytest.MonkeyPatch):
    clients: list[httpx.Client] = []

    def install(
        handler: Callable[[httpx.Request], httpx.Response],
        before_request: Callable[[httpx.Request], None] | None = None,
    ):
        def handle(request: httpx.Request) -> httpx.Response:
            if before_request:
                before_request(request)
            assert request.headers["Authorization"] == "Token mock-mem0-api-key"
            if request.url.path == "/v1/ping/":
                return httpx.Response(
                    200,
                    json={
                        "user_email": "test@example.com",
                        "org_id": "org-1",
                        "project_id": "project-1",
                    },
                )
            return handler(request)

        def create_client(**kwargs):
            client = httpx.Client(transport=httpx.MockTransport(handle))
            clients.append(client)
            return MemoryClient(client=client, **kwargs)

        monkeypatch.setattr("mem0.MemoryClient", create_client)

    yield install
    for client in clients:
        client.close()


@pytest.mark.parametrize("conversation", [False, True])
@pytest.mark.parametrize("results", [[ADDED_MEMORY], []])
async def test_add_memory_outputs(mem0_transport, conversation: bool, results):
    messages = [{"role": "user", "content": "I am a vegetarian"}]

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/memories/"
        payload = json.loads(request.content)
        assert payload["messages"] == messages
        assert payload["user_id"] == "user-1"
        assert payload["agent_id"] == "workflow-1"
        assert payload["run_id"] == "run-1"
        assert payload["metadata"] == {"topic": "food"}
        assert payload["output_format"] == "v1.1"
        assert payload.get("async_mode") is False
        return httpx.Response(200, json={"results": results})

    mem0_transport(handle)
    content = (
        {"discriminator": "conversation", "messages": messages}
        if conversation
        else {"discriminator": "content", "content": "I am a vegetarian"}
    )
    outputs = [
        item
        async for item in AddMemoryBlock().run(
            AddMemoryBlock.Input(
                credentials=TEST_CREDENTIALS_INPUT,
                content=content,
                metadata={"topic": "food"},
                limit_memory_to_run=True,
            ),
            credentials=TEST_CREDENTIALS,
            user_id="user-1",
            graph_id="workflow-1",
            graph_exec_id="run-1",
        )
    ]

    assert outputs == [("results", results)] + (
        [("action", "ADD"), ("memory", "I am a vegetarian")]
        if results
        else [("action", "NO_CHANGE")]
    )


async def test_add_memory_outputs_processed_memory_with_nested_values(
    mem0_transport,
):
    processed = {
        "id": "memory-1",
        "event": "ADD",
        "memory": "I am a vegetarian",
        "structured_attributes": {"day": 13, "is_weekend": False},
    }
    mem0_transport(lambda _: httpx.Response(200, json={"results": [processed]}))

    outputs = [
        item
        async for item in AddMemoryBlock().execute(
            {"credentials": TEST_CREDENTIALS_INPUT},
            execution_context=ExecutionContext(),
            credentials=TEST_CREDENTIALS,
            user_id="user-1",
            graph_id="workflow-1",
            graph_exec_id="run-1",
        )
    ]

    assert outputs == [
        ("results", [processed]),
        ("action", "ADD"),
        ("memory", "I am a vegetarian"),
    ]


@pytest.mark.parametrize(
    "block_type",
    [AddMemoryBlock, SearchMemoryBlock, GetAllMemoriesBlock, GetLatestMemoryBlock],
)
@pytest.mark.parametrize("block_validation", [False, True])
async def test_memory_keeps_event_loop_responsive(
    mem0_transport, block_type, block_validation: bool
):
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = Event()
    results = [{"event": "ADD", "memory": "I am a vegetarian"}]

    def pause_request(request: httpx.Request) -> None:
        if (request.url.path == "/v1/ping/") == block_validation:
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(timeout=10):
                raise TimeoutError("Event loop could not release the Mem0 request")

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": results})

    mem0_transport(handle, before_request=pause_request)
    block = block_type()
    inputs = {"credentials": TEST_CREDENTIALS_INPUT}
    if block_type is SearchMemoryBlock:
        inputs["query"] = "food preferences"

    async def collect_outputs():
        return [
            item
            async for item in block.run(
                block.input_schema(**inputs),
                credentials=TEST_CREDENTIALS,
                user_id="user-1",
                graph_id="workflow-1",
                graph_exec_id="run-1",
            )
        ]

    task = asyncio.create_task(collect_outputs())
    try:
        await asyncio.wait_for(entered.wait(), timeout=15)
        assert not task.done()
    finally:
        release.set()
        outputs = await task

    if block_type is AddMemoryBlock:
        assert outputs == [
            ("results", results),
            ("action", "ADD"),
            ("memory", "I am a vegetarian"),
        ]
    elif block_type is GetLatestMemoryBlock:
        assert outputs == [("memory", results[0]), ("found", True)]
    else:
        assert outputs == [("memories", results)]


@pytest.mark.parametrize(
    "block_type", [SearchMemoryBlock, GetAllMemoriesBlock, GetLatestMemoryBlock]
)
@pytest.mark.parametrize("memories", [MEMORIES, []])
@pytest.mark.parametrize("envelope", [False, True])
async def test_read_memory_preserves_outputs_and_scope(
    mem0_transport, block_type, memories, envelope: bool
):
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        payload = json.loads(request.content)
        assert "version" not in payload
        assert payload["filters"] == {
            "AND": [
                {"user_id": "user-1"},
                {"run_id": "run-1"},
                {"agent_id": "workflow-1"},
                {"metadata": {"topic": "food"}},
            ]
        }
        if block_type is SearchMemoryBlock:
            assert request.url.path == "/v2/memories/search/"
            assert payload["query"] == "food preferences"
        else:
            assert request.url.path == "/v2/memories/"
        return httpx.Response(200, json={"results": memories} if envelope else memories)

    mem0_transport(handle)
    block = block_type()
    inputs = {
        "credentials": TEST_CREDENTIALS_INPUT,
        "metadata_filter": {"topic": "food"},
        "limit_memory_to_run": True,
    }
    if block_type is SearchMemoryBlock:
        inputs["query"] = "food preferences"
    outputs = [
        item
        async for item in block.run(
            block.input_schema(**inputs),
            credentials=TEST_CREDENTIALS,
            user_id="user-1",
            graph_id="workflow-1",
            graph_exec_id="run-1",
        )
    ]

    if block_type is GetLatestMemoryBlock:
        assert outputs == [
            ("memory", memories[0] if memories else None),
            ("found", bool(memories)),
        ]
    else:
        assert outputs == [("memories", memories)]


@pytest.mark.parametrize("interruption", ["cancel", "cancel_twice", "timeout"])
@pytest.mark.parametrize("accepted", [False, True])
async def test_interrupted_add_settles_once_before_reporting_failure(
    mem0_transport, interruption: str, accepted: bool
):
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = Event()
    deadline = asyncio.timeout(None)
    results = [{"event": "ADD", "memory": "I am a vegetarian"}]
    persisted = []
    outputs = []
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.path == "/v1/memories/"
        assert json.loads(request.content)["async_mode"] is False
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(timeout=10):
            raise TimeoutError("Mem0 request was never released")
        if not accepted:
            return httpx.Response(400, json={"detail": "Memory rejected"})
        persisted.extend(results)
        return httpx.Response(200, json={"results": results})

    mem0_transport(handle)

    async def collect_outputs():
        async with deadline:
            async for item in AddMemoryBlock().execute(
                {"credentials": TEST_CREDENTIALS_INPUT},
                execution_context=ExecutionContext(),
                credentials=TEST_CREDENTIALS,
                user_id="user-1",
                graph_id="workflow-1",
                graph_exec_id="run-1",
            ):
                outputs.append(item)

    task = asyncio.create_task(collect_outputs())
    try:
        await asyncio.wait_for(entered.wait(), timeout=15)
        if interruption == "timeout":
            deadline.reschedule(loop.time())
            while not deadline.expired():
                await asyncio.sleep(0)
        else:
            task.cancel()
        await asyncio.sleep(0)
        if interruption == "cancel_twice":
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done()
        assert outputs == []
        assert persisted == []
    finally:
        release.set()
        outcome = (await asyncio.gather(task, return_exceptions=True))[0]

    assert calls == 1
    if accepted:
        assert isinstance(
            outcome,
            TimeoutError if interruption == "timeout" else asyncio.CancelledError,
        )
        assert persisted == results
        assert outputs == [
            ("results", results),
            ("action", "ADD"),
            ("memory", "I am a vegetarian"),
        ]
    else:
        assert persisted == []
        assert outputs == []
        assert isinstance(outcome, BlockExecutionError)
        assert "Memory rejected" in str(outcome)
