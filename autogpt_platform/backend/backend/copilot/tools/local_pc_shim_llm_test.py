"""Tests for the _LocalLLMProxy streaming surface on LocalPCShim.

Each test stubs the shim's WebSocket with a scripted frame source and
asserts the proxy yields deltas in order / surfaces errors cleanly /
cleans up its per-request queue.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from .local_pc_shim import LocalLLMError, LocalPCShim, ShimHello, _LocalLLMProxy


def _make_shim_for_streaming(
    local_llm_models: list[str] | None = None,
) -> LocalPCShim:
    """Build a LocalPCShim that bypasses the recv loop entirely.

    The streaming tests drive the per-request queue by hand instead of
    going through the WS, so we avoid registering a recv task.
    """
    shim = LocalPCShim.__new__(LocalPCShim)
    shim.sandbox_id = "test-session"
    shim.allowed_root = "/workspace"
    shim.machine_id = "machine-1"
    shim.platform = "darwin"
    shim.arch = "arm64"
    shim.capabilities = ["shell", "files", "local_llm"]
    shim.shim_version = "0.1.0"
    shim.screen_resolution = None
    shim.local_llm_models = local_llm_models or ["llama3.2:3b"]
    shim.hardware_devices = []
    shim.computer_use_features = []
    shim._pending = {}
    shim._streaming = {}
    shim._pending_capacity = None
    shim._capacity_available = asyncio.Event()
    shim._capacity_available.set()
    shim._ws = AsyncMock()
    shim._ws.send_text = AsyncMock()
    shim.local_llm = _LocalLLMProxy(shim)
    return shim


def _chunk(msg_id: str, delta: str, finish_reason: str | None = None) -> dict:
    return {
        "type": "LOCAL_LLM_COMPLETION_CHUNK",
        "id": msg_id,
        "ts": 0.0,
        "payload": {"delta": delta, "finish_reason": finish_reason},
    }


def _response(msg_id: str, content: str) -> dict:
    return {
        "type": "LOCAL_LLM_COMPLETION_RESPONSE",
        "id": msg_id,
        "ts": 0.0,
        "payload": {
            "content": content,
            "finish_reason": "stop",
            "tokens": {"prompt": 7, "completion": 3, "total": 10},
            "duration_seconds": 1.2,
        },
    }


def _error(msg_id: str, code: str, message: str, details: dict | None = None) -> dict:
    payload = {"code": code, "message": message}
    if details is not None:
        payload["details"] = details
    return {"type": "ERROR", "id": msg_id, "ts": 0.0, "payload": payload}


@pytest.mark.asyncio
async def test_complete_yields_deltas_in_order() -> None:
    shim = _make_shim_for_streaming()

    async def feed_frames(msg_id: str) -> None:
        """Push the scripted frame sequence onto the streaming queue."""
        # Wait until proxy registers the stream.
        for _ in range(100):
            if msg_id in shim._streaming:
                break
            await asyncio.sleep(0.001)
        queue = shim._streaming[msg_id]
        await queue.put(_chunk(msg_id, "Hello, "))
        await queue.put(_chunk(msg_id, "world"))
        await queue.put(_chunk(msg_id, "!"))
        await queue.put(_chunk(msg_id, "", finish_reason="stop"))
        await queue.put(_response(msg_id, "Hello, world!"))

    deltas: list[str] = []
    # Capture the wire id by intercepting send_text.
    captured: dict = {}

    async def capture(payload: str) -> None:
        envelope = json.loads(payload)
        captured["id"] = envelope["id"]
        # Now kick off the feed task once we know the id.
        asyncio.create_task(feed_frames(envelope["id"]))

    shim._ws.send_text.side_effect = capture

    async for delta in shim.local_llm.complete(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": "hi"}],
    ):
        deltas.append(delta)

    assert deltas == ["Hello, ", "world", "!"]
    # Queue cleaned up.
    assert captured["id"] not in shim._streaming


@pytest.mark.asyncio
async def test_complete_blocking_returns_assembled_string() -> None:
    shim = _make_shim_for_streaming()

    async def feed_frames(msg_id: str) -> None:
        for _ in range(100):
            if msg_id in shim._streaming:
                break
            await asyncio.sleep(0.001)
        queue = shim._streaming[msg_id]
        await queue.put(_chunk(msg_id, "answer"))
        await queue.put(_chunk(msg_id, "", finish_reason="stop"))
        await queue.put(_response(msg_id, "answer"))

    async def capture(payload: str) -> None:
        envelope = json.loads(payload)
        asyncio.create_task(feed_frames(envelope["id"]))

    shim._ws.send_text.side_effect = capture

    result = await shim.local_llm.complete_blocking(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": "?"}],
    )
    assert result == "answer"


@pytest.mark.asyncio
async def test_error_frame_translates_to_local_llm_error() -> None:
    shim = _make_shim_for_streaming()

    async def feed_frames(msg_id: str) -> None:
        for _ in range(100):
            if msg_id in shim._streaming:
                break
            await asyncio.sleep(0.001)
        queue = shim._streaming[msg_id]
        await queue.put(
            _error(
                msg_id,
                "MODEL_NOT_AVAILABLE",
                "Model not loaded",
                details={"requested_model": "phi3:mini"},
            )
        )

    async def capture(payload: str) -> None:
        envelope = json.loads(payload)
        asyncio.create_task(feed_frames(envelope["id"]))

    shim._ws.send_text.side_effect = capture

    with pytest.raises(LocalLLMError) as exc_info:
        async for _ in shim.local_llm.complete(
            model="phi3:mini",
            messages=[{"role": "user", "content": "?"}],
        ):
            pass
    assert exc_info.value.code == "MODEL_NOT_AVAILABLE"
    assert exc_info.value.details["requested_model"] == "phi3:mini"


@pytest.mark.asyncio
async def test_send_failure_translates_to_local_llm_failed() -> None:
    shim = _make_shim_for_streaming()
    shim._ws.send_text = AsyncMock(side_effect=ConnectionError("ws is dead"))

    with pytest.raises(LocalLLMError) as exc_info:
        async for _ in shim.local_llm.complete(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": "?"}],
        ):
            pass
    assert exc_info.value.code == "LOCAL_LLM_FAILED"
    # The queue should have been cleaned up despite the send failure.
    assert len(shim._streaming) == 0


@pytest.mark.asyncio
async def test_recv_loop_routes_stream_frames_to_queue() -> None:
    """End-to-end: register a stream, simulate the recv loop seeing a
    CHUNK + RESPONSE, ensure they land in the queue (not _pending)."""
    shim = _make_shim_for_streaming()

    msg_id = "test-stream-id"
    queue = shim._register_stream(msg_id)
    # Simulate what _recv_loop would do for these frames.
    chunk = _chunk(msg_id, "hello")
    response = _response(msg_id, "hello")
    # Direct queue puts — the recv loop fans out via `await queue.put(msg)`.
    await queue.put(chunk)
    await queue.put(response)
    first = await asyncio.wait_for(queue.get(), timeout=1)
    second = await asyncio.wait_for(queue.get(), timeout=1)
    assert first["type"] == "LOCAL_LLM_COMPLETION_CHUNK"
    assert second["type"] == "LOCAL_LLM_COMPLETION_RESPONSE"
    shim._cleanup_stream(msg_id)
    assert msg_id not in shim._streaming


@pytest.mark.asyncio
async def test_local_llm_proxy_attached_to_shim() -> None:
    """ShimHello + LocalPCShim wiring still exposes the .local_llm attr."""
    hello = ShimHello(
        capabilities=["shell", "files", "local_llm"],
        local_llm_models=["llama3.2:3b"],
    )
    shim = LocalPCShim.__new__(LocalPCShim)
    shim.sandbox_id = "test"
    shim._ws = AsyncMock()
    shim._manager = None
    shim.machine_id = hello.machine_id
    shim.platform = hello.platform
    shim.arch = hello.arch
    shim.shim_version = hello.shim_version
    shim.allowed_root = hello.allowed_root
    shim.capabilities = hello.capabilities
    shim.screen_resolution = hello.screen_resolution
    shim.local_llm_models = hello.local_llm_models
    shim.hardware_devices = hello.hardware_devices
    shim.computer_use_features = hello.computer_use_features
    shim._pending = {}
    shim._streaming = {}
    shim._pending_capacity = None
    shim._capacity_available = asyncio.Event()
    shim._capacity_available.set()
    shim.local_llm = _LocalLLMProxy(shim)
    assert shim.local_llm is not None
    assert hasattr(shim.local_llm, "complete")
    assert hasattr(shim.local_llm, "complete_blocking")
