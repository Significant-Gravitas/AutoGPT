"""The attempt log sees every HTTP request a provider path sends (#16173).

Each SDK path hands its SDK a hooked httpx client and closes it after the
call; Ollama adds the hooks to its own client. These run the real SDKs
against ``httpx.MockTransport``.
"""

import contextlib
from collections.abc import Callable, Iterator
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

import anthropic
import groq
import httpx
import openai
import pytest

from backend.util.llm import providers as providers_mod
from backend.util.llm.http_attempts import HttpAttemptLog, current_http_attempts
from backend.util.llm.providers import ProviderResponse, call_provider

_OK_BODIES: dict[str, dict[str, Any]] = {
    "openai": {
        "id": "r",
        "object": "response",
        "created_at": 0,
        "model": "gpt-4.1",
        "status": "completed",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "type": "message",
                "id": "m",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    },
    "anthropic": {
        "id": "msg",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    },
    "chat": {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
}
_HI = [{"role": "user", "content": "hi"}]


def _error(status: int) -> httpx.Response:
    # A 1 ms retry-after keeps the SDK's backoff out of the test's time.
    return httpx.Response(status, json={"error": {}}, headers={"retry-after-ms": "1"})


def _ticking_log() -> HttpAttemptLog:
    ticks = iter(range(100))
    return HttpAttemptLog(clock=lambda: float(next(ticks)))


@contextlib.contextmanager
def _logged(log: HttpAttemptLog) -> Iterator[None]:
    """Record into ``log`` for the duration, as the block-side llm_call does."""
    token = current_http_attempts.set(log)
    try:
        yield
    finally:
        current_http_attempts.reset(token)


def _route_sdk(
    monkeypatch: pytest.MonkeyPatch,
    sdk: ModuleType,
    handler: Callable[[httpx.Request], httpx.Response],
) -> list[httpx.AsyncClient]:
    """Send the SDK's default client to ``handler``; returns the clients made."""
    clients: list[httpx.AsyncClient] = []
    real_factory = sdk.DefaultAsyncHttpxClient

    def mocked_factory(**kwargs: Any) -> httpx.AsyncClient:
        clients.append(real_factory(**kwargs, transport=httpx.MockTransport(handler)))
        return clients[-1]

    monkeypatch.setattr(sdk, "DefaultAsyncHttpxClient", mocked_factory)
    return clients


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider, model, sdk, body",
    [
        pytest.param("openai", "gpt-4.1", openai, "openai", id="openai"),
        pytest.param(
            "open_router",
            "anthropic/claude-sonnet-4.6",
            openai,
            "chat",
            id="openai-compatible",
        ),
        pytest.param(
            "anthropic", "claude-sonnet-4-6", anthropic, "anthropic", id="anthropic"
        ),
        pytest.param("groq", "llama-3.3-70b", groq, "chat", id="groq"),
    ],
)
async def test_every_sdk_attempt_is_logged_and_the_client_closed(
    monkeypatch, provider, model, sdk, body
):
    replies = [_error(500), httpx.Response(200, json=_OK_BODIES[body])]
    clients = _route_sdk(monkeypatch, sdk, lambda request: replies.pop(0))
    log = _ticking_log()

    with _logged(log):
        result = await call_provider(
            provider=provider,
            model=model,
            api_key="sk-test",
            messages=_HI,
            max_tokens=10,
        )

    assert isinstance(result, ProviderResponse) and result.content == "ok"
    assert log.starts == [0.0, 1.0]
    assert log.statuses == [500, 200]
    assert len(clients) == 1 and clients[0].is_closed


@pytest.mark.asyncio
async def test_a_status_stays_with_the_request_it_answered(monkeypatch):
    """A request that got no response at all keeps its place, so the next
    status lands on the request it answered. The SDK retries a dropped
    connection after its default backoff, about half a second."""
    replies = [_error(500), httpx.Response(200, json=_OK_BODIES["openai"])]
    dropped: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if not dropped:
            dropped.append(request)
            raise httpx.ConnectError("connection refused", request=request)
        return replies.pop(0)

    clients = _route_sdk(monkeypatch, openai, handler)
    log = _ticking_log()

    with _logged(log):
        await call_provider(
            provider="openai",
            model="gpt-4.1",
            api_key="sk-test",
            messages=_HI,
            max_tokens=10,
        )

    assert log.starts == [0.0, 1.0, 2.0]
    assert log.statuses == [None, 500, 200]
    assert len(clients) == 1 and clients[0].is_closed


@pytest.mark.asyncio
async def test_ollama_requests_are_logged(monkeypatch):
    host = "http://ollama.example"
    monkeypatch.setattr(providers_mod.settings.config, "ollama_host", host)
    reply = {"model": "llama3", "message": {"role": "assistant", "content": "hi"}}
    real_client = providers_mod.ollama.AsyncClient
    monkeypatch.setattr(
        providers_mod.ollama,
        "AsyncClient",
        lambda **kwargs: real_client(
            **kwargs,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=reply)
            ),
        ),
    )
    log = HttpAttemptLog(clock=lambda: 7.0)

    with (
        _logged(log),
        patch(
            "backend.copilot.config.ChatConfig",
            return_value=SimpleNamespace(base_url=None),
        ),
    ):
        result = await call_provider(
            provider="ollama",
            model="llama3",
            api_key="",
            messages=_HI,
            max_tokens=10,
            ollama_host=host,
            timeout_seconds=10.0,
        )

    assert isinstance(result, ProviderResponse)
    assert log.starts == [7.0]
    assert log.statuses == [200]
