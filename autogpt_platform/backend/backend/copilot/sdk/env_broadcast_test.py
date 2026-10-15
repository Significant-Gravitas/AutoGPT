"""The OpenRouter Broadcast fields ``build_sdk_env`` hands the Claude CLI.

The CLI owns the request body, so the fields that attribute a generation in
PostHog LLM analytics travel in ``CLAUDE_CODE_EXTRA_BODY``, which the CLI
merges into every request it makes.
"""

import json
from unittest.mock import patch

import pytest

from backend.copilot.config import ChatConfig
from backend.copilot.llm_attribution import set_turn_impersonated
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
def _not_impersonated():
    set_turn_impersonated(False)
    yield
    set_turn_impersonated(False)


def _extra_body(**kwargs) -> dict:
    with patch("backend.copilot.sdk.env.config", _config()):
        env = build_sdk_env(**kwargs)
    return json.loads(env["CLAUDE_CODE_EXTRA_BODY"])


def test_openrouter_mode_attributes_every_request_to_the_user_and_chat():
    body = _extra_body(session_id="sess-1", user_id="user-1")

    assert body == {
        "user": "user-1",
        "session_id": "sess-1",
        "metadata": {"user_id": "user-1"},
        "trace": {
            "generation_name": "copilot-sdk",
            "environment": attribution_settings.config.app_env.value,
            "impersonated": False,
        },
    }


def test_openrouter_mode_keeps_the_session_header_broadcast_needs():
    with patch("backend.copilot.sdk.env.config", _config()):
        env = build_sdk_env(session_id="sess-1", user_id="user-1")

    assert "x-session-id: sess-1" in env["ANTHROPIC_CUSTOM_HEADERS"]


def test_an_impersonated_turn_is_marked():
    set_turn_impersonated(True)

    body = _extra_body(session_id="sess-1", user_id="user-1")

    assert body["trace"]["impersonated"] is True


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
