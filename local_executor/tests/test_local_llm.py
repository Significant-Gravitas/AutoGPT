"""Tests for LocalLLMHandler + OllamaBackend.

Mocks httpx so we never touch a real Ollama. Covers:
  * Streaming dispatch: CHUNK frames emitted in order, terminal RESPONSE
    carries assembled content + token counts + finish_reason.
  * MODEL_NOT_AVAILABLE: pre-check against advertised list AND backend 404.
  * LOCAL_LLM_FAILED: connection refused at chat-stream time.
  * LOCAL_LLM_BUSY: concurrent request while one is in flight.
  * Audit-log entry shape: never carries prompt/response content.
  * Capability detection at HELLO time (probe success vs failure).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.handlers import (
    LocalLLMHandler,
    OllamaBackend,
    OllamaBackendError,
)
from autogpt_local_executor.protocol import (
    ErrorCode,
    ErrorMessage,
    LocalLLMCompletionChunkMessage,
    LocalLLMCompletionMessage,
    LocalLLMCompletionPayload,
    LocalLLMCompletionResponseMessage,
    LocalLLMMessage,
    new_id,
    now_ts,
)


def make_config(tmp_path: Path) -> ShimConfig:
    return ShimConfig(
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "_audit.log",
        platform_url="http://localhost:9999",
        machine_id="test-machine",
        enable_local_llm=True,
        ollama_url="http://localhost:11434",
    )


def _completion_msg(model: str = "llama3.2:3b", **kw) -> LocalLLMCompletionMessage:
    payload = LocalLLMCompletionPayload(
        model=model,
        messages=[
            LocalLLMMessage(role="system", content="be terse"),
            LocalLLMMessage(role="user", content="hi"),
        ],
        **kw,
    )
    return LocalLLMCompletionMessage(id=new_id(), ts=now_ts(), payload=payload)


class FakeStreamResponse:
    """Mimics httpx.AsyncClient.stream() context manager for testing."""

    def __init__(self, status_code: int, lines: list[str] | None = None, body: bytes = b"") -> None:
        self.status_code = status_code
        self._lines = lines or []
        self._body = body

    async def __aenter__(self) -> FakeStreamResponse:
        return self

    async def __aexit__(self, *a: Any) -> None:
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aiter_bytes(self, **kwargs):
        for line in self._lines:
            yield (line + "\n").encode()

    async def aread(self) -> bytes:
        return self._body


class FakeStreamClient:
    """Minimal async-client stand-in that returns one FakeStreamResponse."""

    def __init__(
        self,
        *,
        tags_response: Any | None = None,
        chat_response: FakeStreamResponse | None = None,
        chat_exc: Exception | None = None,
        tags_exc: Exception | None = None,
    ) -> None:
        self._tags_response = tags_response
        self._chat_response = chat_response
        self._chat_exc = chat_exc
        self._tags_exc = tags_exc

    async def __aenter__(self) -> FakeStreamClient:
        return self

    async def __aexit__(self, *a: Any) -> None:
        return None

    async def get(self, url: str) -> Any:
        if self._tags_exc:
            raise self._tags_exc
        return self._tags_response

    def stream(self, method: str, url: str, **kw: Any) -> FakeStreamResponse:
        if self._chat_exc:
            # The exception is raised when entering the context manager.
            class _Raise:
                async def __aenter__(self_inner):  # noqa: N805
                    raise self._chat_exc

                async def __aexit__(self_inner, *a):  # noqa: N805
                    return None

            return _Raise()  # type: ignore[return-value]
        assert self._chat_response is not None
        return self._chat_response


def _fake_async_client_factory(client: FakeStreamClient):
    """Build a factory we can patch httpx.AsyncClient with."""

    def _factory(*args: Any, **kw: Any) -> FakeStreamClient:
        return client

    return _factory


@pytest.mark.asyncio
async def test_chat_stream_waits_for_the_first_token_as_long_as_a_request_may_take() -> None:
    """Ollama sends its response headers with the first token, once it has
    loaded the model and read the prompt, and httpx's read timeout, the
    backend's 60 seconds, ended that wait early."""
    from autogpt_local_executor import local_llm_limits

    timeouts: list[Any] = []
    done = json.dumps({"message": {"content": ""}, "done": True})
    fake = FakeStreamClient(chat_response=FakeStreamResponse(200, lines=[done]))

    def factory(*args: Any, **kw: Any) -> FakeStreamClient:
        timeouts.append(kw.get("timeout"))
        return fake

    backend = OllamaBackend("http://localhost:11434")
    with patch("httpx.AsyncClient", factory):
        stream = backend.chat_stream(
            model="m", messages=[], max_tokens=1, temperature=0.0, top_p=1.0
        )
        _ = [chunk async for chunk in stream]

    (timeout,) = timeouts
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read == local_llm_limits.TOTAL_TIMEOUT_SECONDS
    assert timeout.connect == backend.timeout_seconds


# ── OllamaBackend.list_models ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_models_parses_names() -> None:
    backend = OllamaBackend("http://localhost:11434")
    tags_resp = MagicMock()
    tags_resp.status_code = 200
    tags_resp.json.return_value = {
        "models": [
            {"name": "llama3.2:3b", "size": 1234},
            {"name": "mistral:7b", "size": 5678},
            {"size": 999},  # missing name — skipped
        ]
    }
    fake = FakeStreamClient(tags_response=tags_resp)
    with patch("httpx.AsyncClient", _fake_async_client_factory(fake)):
        models = await backend.list_models()
    assert models == ["llama3.2:3b", "mistral:7b"]


@pytest.mark.asyncio
async def test_list_models_connection_refused_raises() -> None:
    backend = OllamaBackend("http://localhost:11434")
    fake = FakeStreamClient(tags_exc=httpx.ConnectError("connection refused"))
    with patch("httpx.AsyncClient", _fake_async_client_factory(fake)):
        with pytest.raises(OllamaBackendError) as exc_info:
            await backend.list_models()
    assert exc_info.value.code == ErrorCode.LOCAL_LLM_FAILED


@pytest.mark.asyncio
async def test_list_models_http_5xx_raises() -> None:
    backend = OllamaBackend("http://localhost:11434")
    tags_resp = MagicMock()
    tags_resp.status_code = 500
    fake = FakeStreamClient(tags_response=tags_resp)
    with patch("httpx.AsyncClient", _fake_async_client_factory(fake)):
        with pytest.raises(OllamaBackendError) as exc_info:
            await backend.list_models()
    assert exc_info.value.code == ErrorCode.LOCAL_LLM_FAILED


# ── LocalLLMHandler.probe ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_probe_caches_models(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    backend = MagicMock(spec=OllamaBackend)
    backend.list_models = AsyncMock(return_value=["llama3.2:3b", "mistral:7b"])
    handler = LocalLLMHandler(cfg, backend=backend)
    models = await handler.probe()
    assert models == ["llama3.2:3b", "mistral:7b"]
    assert handler.models == ["llama3.2:3b", "mistral:7b"]


@pytest.mark.asyncio
async def test_probe_backend_failure_returns_empty(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    backend = MagicMock(spec=OllamaBackend)
    backend.list_models = AsyncMock(
        side_effect=OllamaBackendError(ErrorCode.LOCAL_LLM_FAILED, "connection refused")
    )
    handler = LocalLLMHandler(cfg, backend=backend)
    models = await handler.probe()
    assert models == []
    assert handler.models == []


# ── Streaming dispatch ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_streaming_dispatch_emits_chunks_and_terminal_response(tmp_path: Path) -> None:
    """Happy path: backend produces 3 deltas + a final done frame; handler
    emits 3 CHUNK frames + 1 terminal CHUNK (delta="", finish_reason=stop)
    + returns a LocalLLMCompletionResponseMessage with the assembled content."""
    cfg = make_config(tmp_path)
    backend = MagicMock(spec=OllamaBackend)

    async def fake_stream(**kw: Any):
        yield {"message": {"content": "Hello, "}, "done": False}
        yield {"message": {"content": "world"}, "done": False}
        yield {"message": {"content": "!"}, "done": False}
        yield {
            "message": {"content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 12,
            "eval_count": 5,
        }

    backend.chat_stream = fake_stream
    handler = LocalLLMHandler(cfg, backend=backend)
    # Cache-empty so the pre-check is a no-op (we want to test the streaming path).
    handler._models = []

    sent_frames: list[Any] = []

    async def send(frame: Any) -> None:
        sent_frames.append(frame)

    response = await handler.handle(_completion_msg(), send=send)
    assert isinstance(response, LocalLLMCompletionResponseMessage)
    assert response.payload.content == ""
    assert response.payload.finish_reason == "stop"
    assert response.payload.tokens.prompt == 12
    assert response.payload.tokens.completion == 5
    assert response.payload.tokens.total == 17
    # 3 deltas + 1 terminal-marker CHUNK
    chunk_msgs = [f for f in sent_frames if isinstance(f, LocalLLMCompletionChunkMessage)]
    assert len(chunk_msgs) == 4
    deltas = [c.payload.delta for c in chunk_msgs]
    assert deltas == ["Hello, ", "world", "!", ""]
    # Non-terminal chunks have finish_reason None; terminal has "stop"
    finishes = [c.payload.finish_reason for c in chunk_msgs]
    assert finishes == [None, None, None, "stop"]


@pytest.mark.asyncio
async def test_non_streaming_emits_no_chunks_only_response(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    backend = MagicMock(spec=OllamaBackend)

    async def fake_stream(**kw: Any):
        yield {"message": {"content": "answer"}, "done": False}
        yield {"message": {"content": ""}, "done": True, "done_reason": "stop"}

    backend.chat_stream = fake_stream
    handler = LocalLLMHandler(cfg, backend=backend)
    handler._models = []
    msg = _completion_msg(stream=False)

    sent_frames: list[Any] = []

    async def send(frame: Any) -> None:
        sent_frames.append(frame)

    response = await handler.handle(msg, send=send)
    assert isinstance(response, LocalLLMCompletionResponseMessage)
    assert response.payload.content == "answer"
    # Stream=False → no intermediate CHUNK frames
    assert sent_frames == []


# ── Pre-check: MODEL_NOT_AVAILABLE against advertised list ──────────────────


@pytest.mark.asyncio
async def test_model_not_in_advertised_list_returns_error(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    backend = MagicMock(spec=OllamaBackend)
    handler = LocalLLMHandler(cfg, backend=backend)
    handler._models = ["llama3.2:3b"]  # simulate prior probe
    msg = _completion_msg(model="phi3:never-loaded")
    response = await handler.handle(msg, send=None)
    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.MODEL_NOT_AVAILABLE
    assert response.payload.details is not None
    assert response.payload.details["requested_model"] == "phi3:never-loaded"
    assert response.payload.details["available_models"] == ["llama3.2:3b"]


# ── Backend errors translate cleanly ────────────────────────────────────────


@pytest.mark.asyncio
async def test_backend_404_returns_model_not_available(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    backend = MagicMock(spec=OllamaBackend)

    async def fake_stream(**kw: Any):
        raise OllamaBackendError(ErrorCode.MODEL_NOT_AVAILABLE, "model 'foo' not found")
        yield {}  # unreachable; makes this an async gen

    backend.chat_stream = fake_stream
    handler = LocalLLMHandler(cfg, backend=backend)
    handler._models = []  # skip pre-check so the backend error path runs
    response = await handler.handle(_completion_msg(), send=None)
    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.MODEL_NOT_AVAILABLE
    assert response.payload.details is not None
    assert "foo" in response.payload.details["backend_error"]


@pytest.mark.asyncio
async def test_backend_connection_refused_returns_local_llm_failed(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    backend = MagicMock(spec=OllamaBackend)

    async def fake_stream(**kw: Any):
        raise OllamaBackendError(ErrorCode.LOCAL_LLM_FAILED, "connection refused")
        yield {}

    backend.chat_stream = fake_stream
    handler = LocalLLMHandler(cfg, backend=backend)
    handler._models = []
    response = await handler.handle(_completion_msg(), send=None)
    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.LOCAL_LLM_FAILED
    assert response.payload.details["backend_error"] == "connection refused"


# ── Concurrency / LOCAL_LLM_BUSY ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_request_returns_local_llm_busy(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    backend = MagicMock(spec=OllamaBackend)

    started = asyncio.Event()
    can_finish = asyncio.Event()

    async def slow_stream(**kw: Any):
        started.set()
        await can_finish.wait()
        yield {"message": {"content": "ok"}, "done": False}
        yield {"message": {"content": ""}, "done": True, "done_reason": "stop"}

    backend.chat_stream = slow_stream
    handler = LocalLLMHandler(cfg, backend=backend)
    handler._models = []

    first = asyncio.create_task(handler.handle(_completion_msg(), send=None))
    await started.wait()
    # While the first one is in flight, a second arrives:
    second = await handler.handle(_completion_msg(), send=None)
    assert isinstance(second, ErrorMessage)
    assert second.payload.code == ErrorCode.LOCAL_LLM_BUSY
    # Now let the first finish so the test exits cleanly.
    can_finish.set()
    first_resp = await first
    assert isinstance(first_resp, LocalLLMCompletionResponseMessage)


# ── Audit log shape ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_audit_record_omits_content(tmp_path: Path) -> None:
    """Per AUDIT_LOG.md the prompt and response content are NEVER logged —
    only sizes, model, finish_reason, and token counts. This test asserts
    the audit record matches that shape."""
    cfg = make_config(tmp_path)
    backend = MagicMock(spec=OllamaBackend)

    async def fake_stream(**kw: Any):
        yield {"message": {"content": "secret answer"}, "done": False}
        yield {
            "message": {"content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 7,
            "eval_count": 2,
        }

    backend.chat_stream = fake_stream

    # Capture audit writes.
    audit = MagicMock()
    audit.write = AsyncMock()

    handler = LocalLLMHandler(cfg, audit=audit, backend=backend)
    handler._models = []
    secret_msg = LocalLLMCompletionMessage(
        id=new_id(),
        ts=now_ts(),
        payload=LocalLLMCompletionPayload(
            model="llama3.2:3b",
            messages=[
                LocalLLMMessage(role="system", content="DO NOT LEAK ME"),
                LocalLLMMessage(role="user", content="DO NOT LEAK EITHER"),
            ],
        ),
    )
    response = await handler.handle(secret_msg, send=None)
    assert isinstance(response, LocalLLMCompletionResponseMessage)
    assert response.payload.content == "secret answer"

    # Exactly one audit write per request.
    assert audit.write.call_count == 1
    args, kwargs = audit.write.call_args
    assert args[0] == "LOCAL_LLM_COMPLETION"
    details = kwargs["details"]
    # Sizes + metadata yes …
    assert details["model"] == "llama3.2:3b"
    assert details["prompt_chars"] == len("DO NOT LEAK ME") + len("DO NOT LEAK EITHER")
    assert details["response_chars"] == len("secret answer")
    assert details["finish_reason"] == "stop"
    assert details["tokens_prompt"] == 7
    assert details["tokens_completion"] == 2
    assert details["tokens_total"] == 9
    # … content never.
    serialized = json.dumps(details)
    assert "DO NOT LEAK ME" not in serialized
    assert "DO NOT LEAK EITHER" not in serialized
    assert "secret answer" not in serialized


@pytest.mark.asyncio
async def test_audit_records_error_codes(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    backend = MagicMock(spec=OllamaBackend)

    async def fake_stream(**kw: Any):
        raise OllamaBackendError(ErrorCode.LOCAL_LLM_FAILED, "backend kapow")
        yield {}

    backend.chat_stream = fake_stream
    audit = MagicMock()
    audit.write = AsyncMock()
    handler = LocalLLMHandler(cfg, audit=audit, backend=backend)
    handler._models = []
    await handler.handle(_completion_msg(), send=None)

    assert audit.write.call_count == 1
    _, kwargs = audit.write.call_args
    result = kwargs["result"]
    assert result["ok"] is False
    assert result["error_code"] == ErrorCode.LOCAL_LLM_FAILED.value


# ── HELLO-time capability/model wiring (daemon-level integration) ────────────


@pytest.mark.asyncio
async def test_build_hello_omits_unfinished_local_model_preview(tmp_path: Path) -> None:
    """The unwired capability cannot be enabled by a successful Ollama probe."""
    from autogpt_local_executor.audit import AuditWriter
    from autogpt_local_executor.daemon import ShimDaemon

    cfg = make_config(tmp_path)
    audit = AuditWriter(path=cfg.audit_log_path, audit_key=b"a" * 32)
    daemon = ShimDaemon(cfg, token_store=object(), audit=audit)
    # Replace the handler's backend with one that returns 2 models.
    backend = MagicMock(spec=OllamaBackend)
    backend.list_models = AsyncMock(return_value=["llama3.2:3b", "mistral:7b"])
    daemon._local_llm_handler._backend = backend

    hello = await daemon._build_hello()
    assert "local_llm" not in hello.payload.capabilities
    assert hello.payload.local_llm_models == []
    backend.list_models.assert_not_awaited()


@pytest.mark.asyncio
async def test_build_hello_strips_capability_when_probe_returns_empty(tmp_path: Path) -> None:
    from autogpt_local_executor.audit import AuditWriter
    from autogpt_local_executor.daemon import ShimDaemon

    cfg = make_config(tmp_path)
    audit = AuditWriter(path=cfg.audit_log_path, audit_key=b"a" * 32)
    daemon = ShimDaemon(cfg, token_store=object(), audit=audit)
    backend = MagicMock(spec=OllamaBackend)
    backend.list_models = AsyncMock(
        side_effect=OllamaBackendError(ErrorCode.LOCAL_LLM_FAILED, "connection refused")
    )
    daemon._local_llm_handler._backend = backend

    hello = await daemon._build_hello()
    assert "local_llm" not in hello.payload.capabilities
    assert hello.payload.local_llm_models == []
