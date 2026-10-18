"""The startup check on the chat settings (``chat_env.py``)."""

from pathlib import Path

import pytest

from backend.copilot.chat_env import (
    ACCEPTED_CHAT_SETTINGS,
    CHAT_SETTINGS_READ_ELSEWHERE,
    MIN_LOCAL_CONTEXT_TOKENS,
    PLATFORM_ROUTES,
    RETIRED_CHAT_SETTINGS,
    ChatEnvironmentError,
    chat_environment_problems,
    check_chat_environment,
)
from backend.copilot.config import ChatConfig
from backend.copilot.route_table import PlatformRoute

_SECRET = "sk-must-never-be-printed"
_OPENROUTER = {
    "CHAT_PLATFORM_ROUTE": "openrouter",
    "CHAT_OPENROUTER_API_KEY": _SECRET,
}


def _problems(env: dict[str, str], dotenv: list[Path] | None = None) -> list[str]:
    return chat_environment_problems(env, dotenv or [])[0]


@pytest.mark.parametrize("name", sorted(RETIRED_CHAT_SETTINGS))
@pytest.mark.parametrize("value", ["", "false", _SECRET])
def test_every_retired_setting_stops_startup_naming_its_replacement(name, value):
    with pytest.raises(ChatEnvironmentError) as raised:
        check_chat_environment({**_OPENROUTER, name: value}, [])
    (problem,) = raised.value.problems
    assert problem == f"{name} is no longer read: {RETIRED_CHAT_SETTINGS[name]}"
    assert _SECRET not in str(raised.value)


def test_the_retired_and_accepted_registries_do_not_overlap():
    assert not set(RETIRED_CHAT_SETTINGS) & ACCEPTED_CHAT_SETTINGS
    assert not set(RETIRED_CHAT_SETTINGS) & CHAT_SETTINGS_READ_ELSEWHERE


def test_the_accepted_registry_is_every_setting_chatconfig_reads():
    assert ACCEPTED_CHAT_SETTINGS == {
        f"CHAT_{name.upper()}" for name in ChatConfig.model_fields
    }


def test_the_route_list_matches_the_route_table():
    from typing import get_args

    assert set(PLATFORM_ROUTES) == set(get_args(PlatformRoute))


def test_old_names_only_fails_once_naming_every_replacement():
    """A deployment still on the old format: one error, every name in it."""
    old = {
        "CHAT_USE_OPENROUTER": "true",
        "CHAT_API_KEY": _SECRET,
        "CHAT_BASE_URL": "https://openrouter.ai/api/v1",
        "CHAT_THINKING_STANDARD_MODEL": "anthropic/claude-sonnet-5-5",
        "CHAT_CLAUDE_AGENT_CONTEXT_WINDOW": "200000",
    }
    with pytest.raises(ChatEnvironmentError) as raised:
        check_chat_environment(old, [])
    problems = raised.value.problems
    for name in old:
        assert any(p.startswith(f"{name} is no longer read: ") for p in problems)
    assert problems[-1].startswith("CHAT_PLATFORM_ROUTE is not set")
    assert len(problems) == len(old) + 1
    assert _SECRET not in str(raised.value)


def test_a_retired_name_in_a_dotenv_file_counts(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("CHAT_USE_LOCAL=true\nCHAT_TITLE_MODEL=\n", encoding="utf-8")
    problems = _problems(dict(_OPENROUTER), [env_file])
    assert problems == [
        f"CHAT_TITLE_MODEL is no longer read: {RETIRED_CHAT_SETTINGS['CHAT_TITLE_MODEL']}",
        f"CHAT_USE_LOCAL is no longer read: {RETIRED_CHAT_SETTINGS['CHAT_USE_LOCAL']}",
    ]


def test_the_route_may_come_from_a_dotenv_file(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "CHAT_PLATFORM_ROUTE=openrouter\nCHAT_OPENROUTER_API_KEY=k\n", encoding="utf-8"
    )
    assert _problems({}, [env_file]) == []


def test_shared_names_keep_working():
    env = {
        **_OPENROUTER,
        "CHAT_OPENAI_API_KEY": "embeddings",
        "OPEN_ROUTER_API_KEY": "integration",
        "ANTHROPIC_API_KEY": "batch",
        "OPENAI_API_KEY": "voice",
    }
    assert chat_environment_problems(env, []) == ([], [])


def test_an_unknown_chat_name_is_a_warning_not_an_error():
    problems, warnings = chat_environment_problems(
        {**_OPENROUTER, "CHAT_OPENROUTER_KEY": "typo"}, []
    )
    assert problems == []
    assert warnings == ["CHAT_OPENROUTER_KEY is not a setting anything reads"]


class TestRoute:
    def test_missing_route_is_an_error_never_inferred_from_keys(self):
        assert _problems({"CHAT_OPENROUTER_API_KEY": "k"})[0].startswith(
            "CHAT_PLATFORM_ROUTE is not set"
        )

    def test_an_unknown_route_is_named_without_its_value(self):
        (problem,) = _problems({"CHAT_PLATFORM_ROUTE": _SECRET})
        assert problem.startswith("CHAT_PLATFORM_ROUTE is not a route")
        assert _SECRET not in problem

    def test_disabled_needs_nothing(self):
        assert _problems({"CHAT_PLATFORM_ROUTE": "disabled"}) == []

    def test_openrouter_needs_its_key(self):
        assert _problems({"CHAT_PLATFORM_ROUTE": "openrouter"}) == [
            "CHAT_PLATFORM_ROUTE=openrouter needs CHAT_OPENROUTER_API_KEY"
        ]

    def test_a_generic_url_override_needs_an_explicit_chat_url(self):
        (problem,) = _problems({**_OPENROUTER, "OPENAI_BASE_URL": "https://proxy"})
        assert problem.startswith("OPENAI_BASE_URL no longer point the chat")
        assert (
            _problems(
                {
                    **_OPENROUTER,
                    "OPENAI_BASE_URL": "https://proxy",
                    "CHAT_OPENROUTER_BASE_URL": "https://openrouter.ai/api",
                }
            )
            == []
        )

    def test_direct_needs_its_key_and_anthropic_tiers(self):
        assert _problems(
            {
                "CHAT_PLATFORM_ROUTE": "direct_anthropic",
                "CHAT_STANDARD_MODEL": "moonshotai/kimi-k2.6",
            }
        ) == [
            "CHAT_PLATFORM_ROUTE=direct_anthropic needs CHAT_ANTHROPIC_API_KEY",
            "CHAT_STANDARD_MODEL names a model the direct_anthropic route cannot "
            "serve: use an anthropic/* or claude-* model",
        ]

    def test_subscription_is_refused_on_cloud(self):
        (problem,) = _problems(
            {"CHAT_PLATFORM_ROUTE": "subscription", "BEHAVE_AS": "cloud"}
        )
        assert "BEHAVE_AS=cloud" in problem

    def test_local_needs_url_adapter_window_and_both_tiers(self):
        assert _problems({"CHAT_PLATFORM_ROUTE": "local"}) == [
            "CHAT_PLATFORM_ROUTE=local needs CHAT_LOCAL_BASE_URL",
            "CHAT_PLATFORM_ROUTE=local needs CHAT_STANDARD_MODEL",
            "CHAT_PLATFORM_ROUTE=local needs CHAT_ADVANCED_MODEL",
            "CHAT_PLATFORM_ROUTE=local needs CHAT_LOCAL_ADAPTER",
            "CHAT_PLATFORM_ROUTE=local needs CHAT_LOCAL_CONTEXT_TOKENS, the model's "
            "context window as the server has it loaded",
        ]

    @pytest.mark.parametrize(
        ("adapter", "window", "expected"),
        [
            ("openai_chat", str(MIN_LOCAL_CONTEXT_TOKENS), []),
            ("anthropic_passthrough", "262144", []),
            ("ollama", "131072", ["CHAT_LOCAL_ADAPTER is not an adapter"]),
            ("openai_chat", "32768", ["CHAT_LOCAL_CONTEXT_TOKENS is below 65536"]),
            ("openai_chat", "64k", ["CHAT_LOCAL_CONTEXT_TOKENS must be"]),
        ],
    )
    def test_local_adapter_and_window_are_validated(self, adapter, window, expected):
        env = {
            "CHAT_PLATFORM_ROUTE": "local",
            "CHAT_LOCAL_BASE_URL": "http://ollama:11434/v1",
            "CHAT_LOCAL_ADAPTER": adapter,
            "CHAT_LOCAL_CONTEXT_TOKENS": window,
            "CHAT_STANDARD_MODEL": "qwen3:8b",
            "CHAT_ADVANCED_MODEL": "qwen3:32b",
        }
        problems = _problems(env)
        assert len(problems) == len(expected)
        for problem, start in zip(problems, expected):
            assert problem.startswith(start)

    def test_test_mode_needs_no_credential(self):
        assert (
            _problems({"CHAT_PLATFORM_ROUTE": "openrouter", "CHAT_TEST_MODE": "true"})
            == []
        )

    def test_a_disabled_route_names_the_integration_keys_it_ignores(self):
        # They used to drive the chat as a fallback; now nothing does.
        _, warnings = chat_environment_problems(
            {
                "CHAT_PLATFORM_ROUTE": "disabled",
                "OPEN_ROUTER_API_KEY": "sk-or-blocks",
                "ANTHROPIC_API_KEY": "sk-ant-blocks",
            },
            [],
        )
        assert len(warnings) == 1
        assert warnings[0].startswith(
            "OPEN_ROUTER_API_KEY and ANTHROPIC_API_KEY are set, but "
            "CHAT_PLATFORM_ROUTE is disabled"
        )
        assert "sk-" not in warnings[0]

    def test_a_selected_route_does_not_warn_about_integration_keys(self):
        _, warnings = chat_environment_problems(
            {**_OPENROUTER, "OPEN_ROUTER_API_KEY": "sk-or-blocks"}, []
        )
        assert warnings == []

    def test_the_anthropic_key_is_never_stray(self):
        # The Batch API's key for the platform's background jobs, on any
        # route.
        _, warnings = chat_environment_problems(
            {**_OPENROUTER, "CHAT_ANTHROPIC_API_KEY": "sk-ant-batch"}, []
        )
        assert warnings == []

    def test_a_key_for_a_route_nobody_selected_is_a_warning(self):
        problems, warnings = chat_environment_problems(
            {**_OPENROUTER, "CHAT_LOCAL_BASE_URL": "http://ollama:11434/v1"}, []
        )
        assert problems == []
        assert warnings == [
            "CHAT_LOCAL_BASE_URL is set, but CHAT_PLATFORM_ROUTE is openrouter: "
            "the local route is not used"
        ]
