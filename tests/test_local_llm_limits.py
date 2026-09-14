import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from autogpt_local_executor import local_llm_limits
from autogpt_local_executor.audit import AuditWriter
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.daemon import DaemonPreflightError, ShimDaemon
from autogpt_local_executor.handlers import LocalLLMHandler, OllamaBackend
from autogpt_local_executor.protocol import (
    ErrorMessage,
    LocalLLMCompletionMessage,
    LocalLLMCompletionPayload,
    new_id,
    now_ts,
)


def request(**overrides):
    fields = {"model": "test", "messages": [{"role": "user", "content": "hello"}]}
    fields.update(overrides)
    return LocalLLMCompletionMessage(
        id=new_id(), ts=now_ts(), payload=LocalLLMCompletionPayload(**fields)
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"max_tokens": -1},
        {"max_tokens": 0},
        {"max_tokens": 8193},
        {"temperature": float("nan")},
        {"temperature": 3},
        {"top_p": 0},
        {"messages": []},
        {"messages": [{"role": "user", "content": "x"}] * 257},
        {"messages": [{"role": "user", "content": "x" * 262145}]},
        {"messages": [{"role": "user", "content": "x" * 262144}] * 3},
    ],
)
def test_request_limits_reject_unbounded_generation(fields):
    with pytest.raises(ValidationError):
        request(**fields)


@pytest.mark.asyncio
async def test_producer_aborts_oversized_output_and_closes_backend(tmp_path: Path):
    backend = MagicMock(spec=OllamaBackend)
    closed = asyncio.Event()

    async def stream(**kwargs):
        try:
            yield {"message": {"content": "x" * (4 * 1024 * 1024 + 1)}}
        finally:
            closed.set()

    backend.chat_stream = stream
    handler = LocalLLMHandler(ShimConfig(allowed_root=tmp_path), backend=backend)
    send = AsyncMock()
    result = await handler.handle(request(), send=send)
    assert isinstance(result, ErrorMessage)
    assert closed.is_set()
    send.assert_not_awaited()


@pytest.mark.asyncio
async def test_local_model_preview_cannot_start_or_advertise(tmp_path: Path):
    config = ShimConfig(
        enable_local_llm=True,
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
    )
    daemon = ShimDaemon(
        config,
        token_store=object(),
        audit=AuditWriter(config.audit_log_path, audit_key=b"a" * 32),
    )
    daemon._local_llm_handler.probe = AsyncMock(return_value=["test"])
    hello = await daemon._build_hello()
    assert "local_llm" not in hello.payload.capabilities
    assert hello.payload.local_llm_models == []
    daemon._local_llm_handler.probe.assert_not_awaited()
    with pytest.raises(DaemonPreflightError, match="Local model routing"):
        await daemon._preflight_or_raise()


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", ["size", "deadline", "send_failure", "cancel"])
async def test_producer_closes_and_releases_capacity(tmp_path, monkeypatch, stop_reason):
    backend = MagicMock(spec=OllamaBackend)
    started = asyncio.Event()
    closed = asyncio.Event()
    monkeypatch.setattr(local_llm_limits, "MAX_OUTPUT_BYTES", 8)
    if stop_reason == "deadline":
        monkeypatch.setattr(local_llm_limits, "TOTAL_TIMEOUT_SECONDS", 0.01)

    async def stream(**kwargs):
        try:
            started.set()
            if stop_reason in {"deadline", "cancel"}:
                await asyncio.Event().wait()
            for _ in range(10):
                yield {"message": {"content": "test"}}
        finally:
            closed.set()

    backend.chat_stream = stream
    handler = LocalLLMHandler(ShimConfig(allowed_root=tmp_path), backend=backend)
    send = AsyncMock(
        side_effect=ConnectionError("closed") if stop_reason == "send_failure" else None
    )
    task = asyncio.create_task(handler.handle(request(), send=send))
    await started.wait()
    if stop_reason == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert isinstance(await asyncio.wait_for(task, 1), ErrorMessage)
    assert closed.is_set()
    assert not handler._inflight_lock.locked()


@pytest.mark.asyncio
async def test_json_lines_are_bounded_before_parse(monkeypatch):
    monkeypatch.setattr(local_llm_limits, "MAX_LINE_BYTES", 16)

    async def raw_chunks():
        yield b'{"message":'
        yield b'"' + b"x" * 20

    with pytest.raises(ValueError, match="size limit"):
        _ = [chunk async for chunk in local_llm_limits.read_json_lines(raw_chunks())]


@pytest.mark.asyncio
async def test_json_lines_support_partial_multibyte_and_final_line():
    async def raw_chunks():
        yield b'{"text":"\xc3'
        yield b'\xa9"}\n\n{"done":'
        yield b"true}"

    assert [chunk async for chunk in local_llm_limits.read_json_lines(raw_chunks())] == [
        {"text": "é"},
        {"done": True},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [b"nope", b"[]", b'{"x":"\xff"}'])
async def test_json_lines_reject_invalid_response_without_echoing_body(raw):
    async def raw_chunks():
        yield raw

    with pytest.raises(ValueError, match="^Invalid local model JSON response$"):
        _ = [chunk async for chunk in local_llm_limits.read_json_lines(raw_chunks())]
