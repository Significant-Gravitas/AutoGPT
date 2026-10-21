"""The OpenRouter Broadcast fields ``build_sdk_env`` hands the Claude CLI.

The CLI owns the request body, so the fields that attribute a generation in
PostHog LLM analytics travel in ``CLAUDE_CODE_EXTRA_BODY``, which the CLI
merges into every request it makes.
"""

import json
from unittest.mock import patch

import pytest

from backend.copilot.config import ChatConfig
from backend.copilot.llm_attribution import set_turn_impersonated, set_turn_trace_id
from backend.copilot.llm_attribution import settings as attribution_settings
from backend.copilot.sdk.env import build_sdk_env


def _config(**overrides) -> ChatConfig:
    fields = {
        "use_claude_code_subscription": False,
        "use_openrouter": True,
        "api_key": "sk-or-test-key",
        "base_url": "https://openrouter.ai/api/v1",
        "thinking_standard_model": "anthropic/claude-sonnet-4-6",
        "thinking_advanced_model": "anthropic/claude-opus-4-7",
        "aux_api_key": "or-aux-key",
    }
    fields.update(overrides)
    return ChatConfig(**fields)


@pytest.fixture(autouse=True)
def _outside_a_turn():
    set_turn_impersonated(False)
    set_turn_trace_id(None)
    yield
    set_turn_impersonated(False)
    set_turn_trace_id(None)


SONNET = "anthropic/claude-sonnet-5.5"
ORDER = ["provider-a/global", "provider-b"]


def _extra_body(cfg: ChatConfig | None = None, **kwargs) -> dict:
    with patch("backend.copilot.sdk.env.config", cfg or _config()):
        env = build_sdk_env(**kwargs)
    return json.loads(env["CLAUDE_CODE_EXTRA_BODY"])


def test_openrouter_mode_attributes_every_request_to_the_user_and_chat():
    cfg = _config(openrouter_provider_order=ORDER)

    body = _extra_body(cfg, session_id="sess-1", user_id="user-1", model=SONNET)

    assert body == {
        "user": "user-1",
        "session_id": "sess-1",
        "metadata": {"user_id": "user-1"},
        "trace": {
            "generation_name": "copilot-sdk",
            "environment": attribution_settings.config.app_env.value,
            "impersonated": False,
        },
        "provider": {"order": ORDER, "allow_fallbacks": False},
    }


def test_claude_calls_keep_openrouters_routing_unless_providers_are_listed(
    monkeypatch: pytest.MonkeyPatch,
):
    """A self-hosted install that sets nothing routes as OpenRouter decides."""
    monkeypatch.delenv("CHAT_OPENROUTER_PROVIDER_ORDER", raising=False)

    body = _extra_body(_config(_env_file=None), user_id="user-1", model=SONNET)

    assert "provider" not in body


@pytest.mark.parametrize(
    "model",
    [
        SONNET,
        "claude-sonnet-5-5",
        "anthropic/claude-opus-4-7",
        "anthropic/claude-haiku-4-5",
    ],
)
def test_a_claude_call_asks_for_the_configured_providers_first(model: str):
    cfg = _config(openrouter_provider_order=ORDER)

    body = _extra_body(cfg, session_id="sess-1", user_id="user-1", model=model)

    assert body["provider"] == {"order": ORDER, "allow_fallbacks": False}


@pytest.mark.parametrize("model", ["moonshotai/kimi-k2.6", "google/gemini-3-pro", None])
def test_another_vendors_model_keeps_openrouters_routing(model: str | None):
    cfg = _config(openrouter_provider_order=ORDER)

    body = _extra_body(cfg, session_id="sess-1", user_id="user-1", model=model)

    assert "provider" not in body


@pytest.mark.parametrize(
    "fallback_model, routes_claude",
    [
        ("", True),
        ("anthropic/claude-opus-4-7", True),
        ("claude-haiku-4-5", True),
        ("moonshotai/kimi-k2.6", False),
        ("qwen/qwen3-coder", False),
    ],
)
def test_mixed_vendor_fallback_keeps_openrouters_routing(
    fallback_model: str, routes_claude: bool
):
    cfg = _config(
        openrouter_provider_order=ORDER, claude_agent_fallback_model=fallback_model
    )

    body = _extra_body(cfg, model=SONNET)

    assert ("provider" in body) is routes_claude


@pytest.mark.parametrize(
    "value, order",
    [
        ("google-vertex/global,anthropic", ["google-vertex/global", "anthropic"]),
        (" google-vertex/global , anthropic ", ["google-vertex/global", "anthropic"]),
        (
            '["google-vertex/global", "anthropic"]',
            ["google-vertex/global", "anthropic"],
        ),
        ("anthropic", ["anthropic"]),
        ("", []),
        ("[]", []),
        # The hosted platform's value (the copilot executor's Helm values).
        (
            "google-vertex/global,anthropic,amazon-bedrock,azure/global,claude-on-aws",
            [
                "google-vertex/global",
                "anthropic",
                "amazon-bedrock",
                "azure/global",
                "claude-on-aws",
            ],
        ),
    ],
)
def test_the_provider_order_reads_from_the_environment_in_either_spelling(
    monkeypatch: pytest.MonkeyPatch, value: str, order: list[str]
):
    monkeypatch.setenv("CHAT_OPENROUTER_PROVIDER_ORDER", value)

    assert _config().openrouter_provider_order == order


def test_an_empty_provider_order_leaves_routing_to_openrouter(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("CHAT_OPENROUTER_PROVIDER_ORDER", "")

    body = _extra_body(_config(), session_id="sess-1", user_id="user-1", model=SONNET)

    assert "provider" not in body


def test_openrouter_mode_keeps_the_session_header_broadcast_needs():
    with patch("backend.copilot.sdk.env.config", _config()):
        env = build_sdk_env(session_id="sess-1", user_id="user-1")

    assert "x-session-id: sess-1" in env["ANTHROPIC_CUSTOM_HEADERS"]


def test_an_impersonated_turn_is_marked():
    set_turn_impersonated(True)

    body = _extra_body(session_id="sess-1", user_id="user-1")

    assert body["trace"]["impersonated"] is True


def test_every_request_of_a_turn_carries_its_trace_id():
    """Broadcast maps ``trace.trace_id`` to ``$ai_trace_id``, so the turn's
    main, sub-agent and compaction calls group into one PostHog trace."""
    set_turn_trace_id("turn-1")

    body = _extra_body(session_id="sess-1", user_id="user-1")

    assert body["trace"]["trace_id"] == "turn-1"


def test_no_user_sends_no_user_fields():
    body = _extra_body(session_id="sess-1")

    assert "user" not in body
    assert "metadata" not in body
    assert body["session_id"] == "sess-1"


def test_long_ids_are_cut_to_openrouters_limit():
    body = _extra_body(session_id="s" * 300, user_id="u" * 300)

    assert body["user"] == "u" * 128
    assert body["metadata"] == {"user_id": "u" * 128}
    assert body["session_id"] == "s" * 128


def test_direct_anthropic_mode_sends_no_broadcast_fields():
    """Broadcast never sees a call that skips OpenRouter."""
    with patch("backend.copilot.sdk.env.config", _config(use_openrouter=False)):
        env = build_sdk_env(session_id="sess-1", user_id="user-1")

    assert "CLAUDE_CODE_EXTRA_BODY" not in env


@patch("backend.copilot.sdk.env.validate_subscription")
def test_subscription_mode_sends_no_broadcast_fields(_validate):
    cfg = _config(use_claude_code_subscription=True)
    with patch("backend.copilot.sdk.env.config", cfg):
        env = build_sdk_env(session_id="sess-1", user_id="user-1")

    assert "CLAUDE_CODE_EXTRA_BODY" not in env
