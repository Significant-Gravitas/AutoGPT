"""The OpenRouter Broadcast fields ``route_launch_env`` hands the Claude CLI.

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
from backend.copilot.route_profile import (
    RouteProfile,
    platform_profile,
    user_key_profile,
)
from backend.copilot.route_table import is_openrouter_url
from backend.copilot.sdk.env import route_launch_env


def _openrouter() -> RouteProfile:
    cfg = ChatConfig(platform_route="openrouter", openrouter_api_key="sk-or-test-key")
    return platform_profile(cfg, "anthropic/claude-sonnet-4-6", "env")


@pytest.fixture(autouse=True)
def _not_impersonated():
    set_turn_impersonated(False)
    yield
    set_turn_impersonated(False)


def _extra_body(**kwargs) -> dict:
    env = route_launch_env(_openrouter(), **kwargs)
    return json.loads(env["CLAUDE_CODE_EXTRA_BODY"])


def test_openrouter_route_attributes_every_request_to_the_user_and_chat():
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


def test_openrouter_route_keeps_the_session_header_broadcast_needs():
    env = route_launch_env(_openrouter(), session_id="sess-1", user_id="user-1")

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


def test_a_launch_on_a_users_own_openrouter_key_sends_no_broadcast_fields():
    """Only the chat's launches name a session and user; the orchestrator
    block's launch on the user's own key attributes nothing to the platform."""
    profile = user_key_profile("open_router", "sk-or-user-key", None)

    env = route_launch_env(profile)

    assert "CLAUDE_CODE_EXTRA_BODY" not in env


def test_another_server_on_the_openrouter_route_gets_no_broadcast_fields():
    """An Anthropic-compatible server behind the OpenRouter route (a proxy,
    a gateway) rejects body fields it does not know; Broadcast is OpenRouter's
    own. The trace headers stay: a server ignores headers it does not know."""
    cfg = ChatConfig(
        platform_route="openrouter",
        openrouter_api_key="sk-or-test-key",
        openrouter_base_url="https://gateway.example.test/api",
    )
    profile = platform_profile(cfg, "claude-sonnet-4-6", "env")

    env = route_launch_env(profile, session_id="sess-1", user_id="user-1")

    assert "CLAUDE_CODE_EXTRA_BODY" not in env
    assert "x-session-id: sess-1" in env["ANTHROPIC_CUSTOM_HEADERS"]


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://openrouter.ai/api", True),
        ("https://openrouter.ai/api/v1", True),
        ("https://eu.openrouter.ai/api", True),
        ("https://gateway.example.test/api", False),
        ("https://openrouter.ai.example.test/api", False),
        ("http://localhost:8317", False),
        ("", False),
    ],
)
def test_only_openrouters_own_host_counts_as_openrouter(url, expected):
    assert is_openrouter_url(url) is expected


def test_direct_anthropic_route_sends_no_broadcast_fields():
    """Broadcast never sees a call that skips OpenRouter."""
    cfg = ChatConfig(platform_route="direct_anthropic", anthropic_api_key="sk-ant-test")
    profile = platform_profile(cfg, "claude-sonnet-4-6", "env")

    env = route_launch_env(profile, session_id="sess-1", user_id="user-1")

    assert "CLAUDE_CODE_EXTRA_BODY" not in env


@patch("backend.copilot.sdk.env.validate_subscription")
def test_subscription_route_sends_no_broadcast_fields(_validate):
    profile = platform_profile(
        ChatConfig(platform_route="subscription"), "claude-sonnet-4-6", "env"
    )

    env = route_launch_env(profile, session_id="sess-1", user_id="user-1")

    assert "CLAUDE_CODE_EXTRA_BODY" not in env
