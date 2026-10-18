"""The platform's background calls (dream passes, briefings, consults) follow
the platform route: its own key, its own URL, its own spelling of the model,
and no call at all on a route without an API.

The two endpoint tests run the real provider clients against a local server
standing in for each API route's custom URL.
"""

from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from pydantic import BaseModel

from backend.copilot import usage_activation_fixtures
from backend.copilot.config import ChatConfig
from backend.copilot.dream import batch_callbacks, orchestrator
from backend.copilot.dream.llm import (
    NO_PLATFORM_API,
    DreamLLMError,
    structured_completion,
)
from backend.copilot.transport_routing import routing_kwargs_for_chat_transport
from backend.executor import batch_executor

_CHAT_ENV = (
    "CHAT_PLATFORM_ROUTE",
    "CHAT_OPENROUTER_API_KEY",
    "CHAT_OPENROUTER_BASE_URL",
    "CHAT_ANTHROPIC_API_KEY",
    "CHAT_ANTHROPIC_BASE_URL",
    "CHAT_LOCAL_BASE_URL",
    "CHAT_LOCAL_API_KEY",
    "CHAT_STANDARD_MODEL",
    "CHAT_ADVANCED_MODEL",
)
_MESSAGES = [{"role": "user", "content": "answer"}]


class _Answer(BaseModel):
    ok: bool


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _CHAT_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(ChatConfig.model_config, "env_file", None)


@pytest.fixture
def integration_keys(mocker):
    """Integration keys the deployment holds for other uses."""
    from backend.copilot import config as config_mod

    secrets = config_mod.settings.secrets
    mocker.patch.object(secrets, "anthropic_api_key", "sk-ant-integration")
    mocker.patch.object(secrets, "open_router_api_key", "sk-or-integration")


def _use(monkeypatch: pytest.MonkeyPatch, cfg: ChatConfig) -> None:
    from backend.copilot.sdk import env

    monkeypatch.setattr(env, "config", cfg)


async def _serve(path: str, reply: dict, seen: list[dict]) -> tuple[web.AppRunner, int]:
    async def handle(request: web.Request) -> web.Response:
        seen.append(
            {
                "path": request.path,
                "authorization": request.headers.get("Authorization"),
                "x_api_key": request.headers.get("x-api-key"),
                "model": (await request.json())["model"],
            }
        )
        return web.json_response(reply)

    app = web.Application()
    app.router.add_post(path, handle)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    return runner, runner.addresses[0][1]


_CHAT_COMPLETION = {
    "id": "gen-1",
    "object": "chat.completion",
    "created": 0,
    "model": "served",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": '{"ok": true}'},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}
_MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "served",
    "content": [{"type": "text", "text": '{"ok": true}'}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 2},
}


@pytest.mark.asyncio
async def test_openrouter_calls_go_to_the_routes_url_with_its_key(
    monkeypatch, integration_keys
):
    seen: list[dict] = []
    runner, port = await _serve(
        "/custom/api/v1/chat/completions", _CHAT_COMPLETION, seen
    )
    try:
        _use(
            monkeypatch,
            ChatConfig(
                platform_route="openrouter",
                openrouter_api_key="sk-or-route",
                openrouter_base_url=f"http://127.0.0.1:{port}/custom/api",
            ),
        )
        result = await structured_completion(
            model="anthropic/claude-sonnet-5-5",
            messages=_MESSAGES,
            response_model=_Answer,
        )
    finally:
        await runner.cleanup()

    assert result.value.ok is True
    assert seen == [
        {
            "path": "/custom/api/v1/chat/completions",
            "authorization": "Bearer sk-or-route",
            "x_api_key": None,
            "model": "anthropic/claude-sonnet-5-5",
        }
    ]
    assert result.usage.model == "anthropic/claude-sonnet-5-5"


@pytest.mark.asyncio
async def test_direct_calls_go_to_the_routes_url_with_its_key_and_native_names(
    monkeypatch, integration_keys
):
    # What the Anthropic SDK would otherwise pick up from the process.
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "FAKE-process-token")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")
    seen: list[dict] = []
    runner, port = await _serve("/proxy/v1/messages", _MESSAGE, seen)
    try:
        _use(
            monkeypatch,
            ChatConfig(
                platform_route="direct_anthropic",
                anthropic_api_key="sk-ant-route",
                anthropic_base_url=f"http://127.0.0.1:{port}/proxy",
            ),
        )
        result = await structured_completion(
            model="anthropic/claude-sonnet-5-5",
            messages=_MESSAGES,
            response_model=_Answer,
        )
    finally:
        await runner.cleanup()

    assert result.value.ok is True
    assert seen == [
        {
            "path": "/proxy/v1/messages",
            "authorization": None,
            "x_api_key": "sk-ant-route",
            "model": "claude-sonnet-5-5",
        }
    ]
    assert result.usage.model == "claude-sonnet-5-5"


def _direct(**overrides) -> ChatConfig:
    return ChatConfig(
        platform_route="direct_anthropic", anthropic_api_key="sk-ant-route", **overrides
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides, tier, sent",
    [
        ({}, "standard", "claude-sonnet-5-5"),
        ({}, "advanced", "claude-opus-5-5"),
        (
            {"standard_model": "anthropic/claude-haiku-4-5"},
            "standard",
            "claude-haiku-4-5",
        ),
        (
            {"advanced_model": "anthropic/claude-opus-4.7"},
            "advanced",
            "claude-opus-4-7",
        ),
        ({"standard_model": "claude-sonnet-4.6"}, "standard", "claude-sonnet-4-6"),
    ],
    ids=[
        "default-standard",
        "default-advanced",
        "prefixed",
        "prefixed-dotted",
        "dotted",
    ],
)
async def test_direct_sends_and_attributes_the_tiers_in_native_spelling(
    monkeypatch, mocker, overrides, tier, sent
):
    from backend.util.llm.providers import ProviderResponse

    cfg = _direct(**overrides)
    _use(monkeypatch, cfg)
    dispatch = mocker.patch(
        "backend.copilot.dream.llm.call_provider",
        new=AsyncMock(
            return_value=ProviderResponse(
                content='{"ok": true}', prompt_tokens=1, completion_tokens=1
            )
        ),
    )

    result = await structured_completion(
        model=cfg.job_tier_model(tier), messages=_MESSAGES, response_model=_Answer
    )

    assert dispatch.await_args.kwargs["model"] == sent
    assert dispatch.await_args.kwargs["base_url"] == "https://api.anthropic.com"
    assert result.usage.model == sent


@pytest.mark.asyncio
async def test_direct_refuses_another_vendors_model_before_any_call(
    monkeypatch, mocker
):
    _use(monkeypatch, _direct())
    dispatch = mocker.patch("backend.copilot.dream.llm.call_provider", new=AsyncMock())

    with pytest.raises(DreamLLMError, match="serves Anthropic models only"):
        await structured_completion(
            model="openai/gpt-5.4-mini", messages=_MESSAGES, response_model=_Answer
        )
    dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_subscription_never_hands_an_integration_key_to_a_background_call(
    monkeypatch, mocker, integration_keys
):
    """A Claude login is no platform API. With Anthropic and OpenRouter
    integration keys in the deployment, no background call receives one:
    the dream pass skips, the synchronous dispatch refuses before calling,
    and the Batch path has no key."""
    monkeypatch.setenv("CHAT_PLATFORM_ROUTE", "subscription")
    cfg = ChatConfig()
    assert cfg.platform_route == "subscription"
    _use(monkeypatch, cfg)
    sync_dispatch = mocker.patch(
        "backend.copilot.dream.llm.call_provider", new=AsyncMock()
    )
    batch_dispatch = mocker.patch(
        "backend.copilot.dream.batch_submit.call_provider", new=AsyncMock()
    )
    # The pass must skip before any of this; reaching it fails fast.
    mocker.patch.object(
        orchestrator, "is_feature_enabled", AsyncMock(return_value=True)
    )
    mocker.patch.object(
        orchestrator, "dream_lock", side_effect=AssertionError("took the lock")
    )

    result = await orchestrator._execute_dream_pass_async(
        "user-1", expert_id=None, config=cfg
    )
    assert result.skipped is True
    assert result.skip_reason == "no_platform_api"

    with pytest.raises(DreamLLMError) as raised:
        await structured_completion(
            model=cfg.job_tier_model("standard"),
            messages=_MESSAGES,
            response_model=_Answer,
        )
    assert str(raised.value) == NO_PLATFORM_API

    routing = routing_kwargs_for_chat_transport()
    assert routing.provider is None and routing.api_key == ""
    assert cfg.batch_anthropic_api_key is None
    assert batch_executor._default_api_key_for("anthropic") is None
    assert batch_callbacks._anthropic_api_key() is None
    sync_dispatch.assert_not_awaited()
    batch_dispatch.assert_not_awaited()


usage_snapshot = usage_activation_fixtures.usage_snapshot
