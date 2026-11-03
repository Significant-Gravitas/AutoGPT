"""``resolve_route`` reproduces what each migrated caller (the dream phases,
the briefing lede, consult_teammate, the style judge) did on its own: the chat
transport's provider and platform key, and the model the caller's
``ChatConfig`` field names, in the native spelling on the Anthropic API.

Each case builds the transport's real ``ChatConfig`` and runs the real
``routing_kwargs_for_chat_transport`` (the pattern of
``copilot/transport_routing_test.py``), so a change on either side shows up.
"""

import ast
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from backend.copilot.config import ChatConfig
from backend.copilot.sdk import env as sdk_env
from backend.util import settings as settings_mod

from .context import InferenceError, InferenceJob, InferenceScope
from .routing import anthropic_batch_route, platform_credentials, resolve_route

_ENV_VARS_TO_CLEAR = (
    "CHAT_PLATFORM_ROUTE",
    "CHAT_OPENROUTER_API_KEY",
    "CHAT_OPENROUTER_BASE_URL",
    "CHAT_ANTHROPIC_API_KEY",
    "CHAT_ANTHROPIC_BASE_URL",
    "CHAT_LOCAL_BASE_URL",
    "CHAT_LOCAL_API_KEY",
    "CHAT_STANDARD_MODEL",
    "CHAT_ADVANCED_MODEL",
    "CHAT_AUX_MODEL",
)

_MODELS = {
    "standard_model": "anthropic/claude-sonnet-5",
    "advanced_model": "anthropic/claude-opus-5.5",
}

_SCOPE = InferenceScope(user_id="u1", expert_id="e1")


def _config(**settings: Any) -> ChatConfig:
    """A config built from these settings alone: no developer ``.env``
    (``_env_file=None``) can pick the transport or the models under test."""
    return ChatConfig(_env_file=None, **settings)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)


def _use_transport(
    monkeypatch: pytest.MonkeyPatch, cfg: ChatConfig, anthropic_api_key: str = ""
) -> ChatConfig:
    monkeypatch.setattr(sdk_env, "config", cfg)
    fake = MagicMock()
    fake.secrets.anthropic_api_key = anthropic_api_key
    fake.secrets.open_router_api_key = ""
    monkeypatch.setattr(settings_mod, "Settings", lambda: fake)
    return cfg


def _openrouter(monkeypatch: pytest.MonkeyPatch) -> ChatConfig:
    return _use_transport(
        monkeypatch,
        _config(
            platform_route="openrouter",
            openrouter_api_key="or-key",
            openrouter_base_url="https://openrouter.ai/api",
            **_MODELS,
        ),
    )


def _direct_anthropic(monkeypatch: pytest.MonkeyPatch) -> ChatConfig:
    return _use_transport(
        monkeypatch,
        _config(
            platform_route="direct_anthropic",
            anthropic_api_key="ant-key-direct",
            **_MODELS,
        ),
    )


def _local(monkeypatch: pytest.MonkeyPatch) -> ChatConfig:
    return _use_transport(
        monkeypatch,
        _config(
            platform_route="local",
            local_api_key="",
            local_base_url="http://localhost:11434/v1",
            advanced_model="llama3.1:8b",
            standard_model="llama3.1:8b",
        ),
    )


def _job(tier: str = "standard", **overrides) -> InferenceJob:
    return InferenceJob(
        **{
            "kind": "dream",
            "phase": "consolidate",
            "correlation_id": "pass-1",
            "latency_class": "deferred",
            "tier": tier,
            **overrides,
        }
    )


@pytest.mark.parametrize(
    "transport,tier,provider,model,payer,cost_log_provider",
    [
        # OpenRouter takes the configured slug as is.
        (
            "openrouter",
            "standard",
            "open_router",
            "anthropic/claude-sonnet-5",
            "platform_allowance",
            "open_router",
        ),
        (
            "openrouter",
            "advanced",
            "open_router",
            "anthropic/claude-opus-5.5",
            "platform_allowance",
            "open_router",
        ),
        (
            "openrouter",
            "aux",
            "open_router",
            "anthropic/claude-haiku-5.5",
            "platform_allowance",
            "open_router",
        ),
        # The native Anthropic API takes its own spelling.
        (
            "direct_anthropic",
            "standard",
            "anthropic",
            "claude-sonnet-5",
            "platform_allowance",
            "anthropic",
        ),
        (
            "direct_anthropic",
            "advanced",
            "anthropic",
            "claude-opus-5-5",
            "platform_allowance",
            "anthropic",
        ),
        (
            "direct_anthropic",
            "aux",
            "anthropic",
            "claude-haiku-5-5",
            "platform_allowance",
            "anthropic",
        ),
        # A local backend bills nobody; every tier is the operator's model.
        ("local", "standard", "ollama", "llama3.1:8b", "local", "local"),
        ("local", "advanced", "ollama", "llama3.1:8b", "local", "local"),
        ("local", "aux", "ollama", "llama3.1:8b", "local", "local"),
    ],
)
def test_route_reproduces_todays_provider_model_and_payer(
    monkeypatch, transport, tier, provider, model, payer, cost_log_provider
):
    cfg = {
        "openrouter": _openrouter,
        "direct_anthropic": _direct_anthropic,
        "local": _local,
    }[transport](monkeypatch)

    route = resolve_route(_SCOPE, _job(tier), config=cfg)

    assert route.engine == "provider_sync"
    assert route.auth_provider == "platform"
    assert route.credential_id is None
    assert route.execution_path == "sync_baseline"
    assert (route.provider, route.model, route.payer, route.cost_log_provider) == (
        provider,
        model,
        payer,
        cost_log_provider,
    )


def test_reason_names_the_config_field_the_model_came_from(monkeypatch):
    route = resolve_route(_SCOPE, _job("advanced"), config=_openrouter(monkeypatch))
    assert route.reason == (
        "open_router chat transport, advanced tier (advanced_model)"
    )


def test_a_pinned_model_wins_over_the_tier(monkeypatch):
    cfg = _direct_anthropic(monkeypatch)
    route = resolve_route(
        _SCOPE, _job("aux", pinned_model="anthropic/claude-opus-4.7"), config=cfg
    )
    assert route.model == "claude-opus-4-7"
    assert "pinned" in route.reason


def test_a_model_the_anthropic_api_cannot_take_fails_as_an_inference_error(
    monkeypatch,
):
    """Before the seam this surfaced from the call itself; the callers still
    catch it as a failed call."""
    cfg = _direct_anthropic(monkeypatch)
    with pytest.raises(InferenceError, match="requires an Anthropic model"):
        resolve_route(_SCOPE, _job(pinned_model="openai/gpt-4.1-mini"), config=cfg)


def test_the_batch_route_is_anthropic_at_the_platforms_expense():
    route = anthropic_batch_route("claude-opus-5-5")
    assert route.engine == "provider_batch"
    assert route.execution_path == "anthropic_batch"
    assert (route.provider, route.cost_log_provider) == ("anthropic", "anthropic")
    assert route.payer == "platform_allowance"
    assert route.model == "claude-opus-5-5"


def test_shared_inference_does_not_import_dream() -> None:
    for path in Path(__file__).parent.glob("*.py"):
        module = ast.parse(path.read_text(encoding="utf-8"))
        imports = [
            node.module for node in ast.walk(module) if isinstance(node, ast.ImportFrom)
        ]
        imports.extend(
            alias.name
            for node in ast.walk(module)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any(
            name and name.startswith("backend.copilot.dream") for name in imports
        ), path.name


class TestPlatformCredentials:
    def test_returns_the_transports_key(self, monkeypatch):
        cfg = _direct_anthropic(monkeypatch)
        route = resolve_route(_SCOPE, _job(), config=cfg)
        assert platform_credentials(route).api_key == "ant-key-direct"

    @pytest.mark.parametrize("route", ["subscription", "disabled"])
    def test_route_without_platform_api_refuses_background_calls(
        self, monkeypatch, route
    ):
        cfg = _use_transport(monkeypatch, _config(platform_route=route, **_MODELS))
        with pytest.raises(InferenceError, match="no API for background calls"):
            resolve_route(_SCOPE, _job(), config=cfg)

    @pytest.mark.parametrize(
        "route,field",
        [
            ("direct_anthropic", "CHAT_ANTHROPIC_API_KEY"),
            ("openrouter", "CHAT_OPENROUTER_API_KEY"),
        ],
    )
    def test_missing_key_is_reported_before_an_incompatible_model(
        self, monkeypatch, route, field
    ):
        cfg = _use_transport(monkeypatch, _config(platform_route=route, **_MODELS))
        with pytest.raises(InferenceError, match=field):
            resolve_route(_SCOPE, _job(pinned_model="openai/gpt-4o-mini"), config=cfg)

    def test_a_local_backend_needs_no_key(self, monkeypatch):
        cfg = _local(monkeypatch)
        route = resolve_route(_SCOPE, _job(), config=cfg)
        assert platform_credentials(route).base_url == "http://localhost:11434/v1"

    def test_refuses_a_route_the_transport_does_not_dispatch_to(self, monkeypatch):
        _openrouter(monkeypatch)
        with pytest.raises(InferenceError, match="not the route's provider"):
            platform_credentials(anthropic_batch_route("claude-sonnet-5"))
