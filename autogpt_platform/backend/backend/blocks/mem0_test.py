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

MEMORIES = [{"id": "memory-1", "memory": "I am a vegetarian"}]


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
async def test_add_memory_returns_completed_results(mem0_transport, conversation: bool):
    messages = [{"role": "user", "content": "I am a vegetarian"}]
    results = [{"event": "ADD", "memory": "I am a vegetarian"}]

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

    assert outputs == [
        ("results", results),
        ("action", "ADD"),
        ("memory", "I am a vegetarian"),
    ]


@pytest.mark.parametrize("blocked_path", ["/v1/ping/", "/v1/memories/"])
async def test_add_memory_keeps_event_loop_responsive(
    mem0_transport, blocked_path: str
):
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = Event()
    results = [{"event": "ADD", "memory": "I am a vegetarian"}]

    def pause_request(request: httpx.Request) -> None:
        if request.url.path == blocked_path:
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(timeout=10):
                raise TimeoutError("Event loop could not release the Mem0 request")

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/memories/"
        assert json.loads(request.content)["async_mode"] is False
        return httpx.Response(200, json={"results": results})

    mem0_transport(handle, before_request=pause_request)

    async def collect_outputs():
        return [
            item
            async for item in AddMemoryBlock().run(
                AddMemoryBlock.Input(credentials=TEST_CREDENTIALS_INPUT),
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

    assert outputs == [
        ("results", results),
        ("action", "ADD"),
        ("memory", "I am a vegetarian"),
    ]


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
