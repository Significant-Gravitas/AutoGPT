"""Legacy chat migration and names-only startup diagnostics."""

import os
from unittest.mock import MagicMock

import pytest

from backend.copilot import chat_env
from backend.copilot.chat_env import (
    ACCEPTED_CHAT_SETTINGS,
    CHAT_SETTINGS_READ_ELSEWHERE,
    MIN_LOCAL_CONTEXT_TOKENS,
    PLATFORM_ROUTES,
    RETIRED_CHAT_SETTINGS,
    ChatEnvironmentError,
    chat_environment_problems,
    check_chat_environment,
    migrate_chat_settings,
    read_chat_settings,
)
from backend.copilot.config import ChatConfig


def resolved(env):
    supplied, notes = migrate_chat_settings(env)
    return {**env, **supplied}, notes


@pytest.mark.parametrize("name", sorted(RETIRED_CHAT_SETTINGS))
@pytest.mark.parametrize("value", ["", "false", "synthetic-private-value"])
def test_retired_names_migrate_or_warn_without_stopping(name, value):
    notes = check_chat_environment({"CHAT_PLATFORM_ROUTE": "disabled", name: value}, [])
    assert sum(note.startswith(name + " ") for note in notes) == 1
    assert "synthetic-private-value" not in str(notes)


def test_registries_match_config():
    from typing import get_args

    assert not set(RETIRED_CHAT_SETTINGS) & ACCEPTED_CHAT_SETTINGS
    assert not set(RETIRED_CHAT_SETTINGS) & CHAT_SETTINGS_READ_ELSEWHERE
    assert ACCEPTED_CHAT_SETTINGS == {
        f"CHAT_{name.upper()}" for name in ChatConfig.model_fields
    }
    from backend.copilot.route_table import PlatformRoute

    assert set(PLATFORM_ROUTES) == set(get_args(PlatformRoute))


@pytest.mark.parametrize(
    "env,route",
    [
        (
            {"CHAT_USE_CLAUDE_CODE_SUBSCRIPTION": "true", "CHAT_USE_LOCAL": "true"},
            "local",
        ),
        (
            {
                "CHAT_USE_LOCAL": "true",
                "CHAT_BASE_URL": "http://local/v1",
                "CHAT_USE_OPENROUTER": "false",
            },
            "local",
        ),
        (
            {"CHAT_USE_OPENROUTER": "false", "ANTHROPIC_API_KEY": "ant"},
            "direct_anthropic",
        ),
        ({"CHAT_USE_OPENROUTER": "true", "CHAT_API_KEY": "chat"}, "openrouter"),
        ({"CHAT_API_KEY": "chat"}, "openrouter"),
        ({}, "disabled"),
        ({"CHAT_USE_OPENROUTER": "false"}, "disabled"),
        ({"CHAT_PLATFORM_ROUTE": "disabled", "CHAT_USE_LOCAL": "true"}, "disabled"),
    ],
)
def test_route_selection(env, route):
    assert resolved(env)[0]["CHAT_PLATFORM_ROUTE"] == route
    assert chat_environment_problems(env)[0] == []


@pytest.mark.parametrize(
    "old,new,route,value,expected",
    [
        ("CHAT_API_KEY", "CHAT_OPENROUTER_API_KEY", "openrouter", "chat", "chat"),
        ("CHAT_API_KEY", "CHAT_LOCAL_API_KEY", "local", "local-key", "local-key"),
        (
            "CHAT_DIRECT_ANTHROPIC_API_KEY",
            "CHAT_ANTHROPIC_API_KEY",
            "openrouter",
            "ant",
            "ant",
        ),
        (
            "CHAT_BASE_URL",
            "CHAT_OPENROUTER_BASE_URL",
            "openrouter",
            "https://proxy/api/v1/",
            "https://proxy/api",
        ),
        (
            "CHAT_BASE_URL",
            "CHAT_LOCAL_BASE_URL",
            "local",
            "http://local/v1",
            "http://local/v1",
        ),
        (
            "CHAT_THINKING_STANDARD_MODEL",
            "CHAT_STANDARD_MODEL",
            "openrouter",
            "sdk-model",
            "sdk-model",
        ),
        ("CHAT_MODEL", "CHAT_STANDARD_MODEL", "openrouter", "sdk-model", "sdk-model"),
        (
            "CHAT_THINKING_ADVANCED_MODEL",
            "CHAT_ADVANCED_MODEL",
            "openrouter",
            "sdk-advanced",
            "sdk-advanced",
        ),
        (
            "CHAT_FAST_STANDARD_MODEL",
            "CHAT_STANDARD_MODEL",
            "local",
            "local-model",
            "local-model",
        ),
        (
            "CHAT_FAST_MODEL",
            "CHAT_STANDARD_MODEL",
            "local",
            "local-model",
            "local-model",
        ),
        (
            "CHAT_FAST_ADVANCED_MODEL",
            "CHAT_ADVANCED_MODEL",
            "local",
            "local-advanced",
            "local-advanced",
        ),
        (
            "CHAT_CLAUDE_AGENT_MODEL",
            "CHAT_STANDARD_MODEL",
            "openrouter",
            "claude-override",
            "claude-override",
        ),
        (
            "CHAT_CLAUDE_AGENT_MODEL",
            "CHAT_ADVANCED_MODEL",
            "openrouter",
            "claude-override",
            "claude-override",
        ),
        (
            "CHAT_CLAUDE_AGENT_MAX_TURNS",
            "CHAT_AGENT_MAX_TURNS",
            "openrouter",
            "17",
            "17",
        ),
        (
            "CHAT_CLAUDE_AGENT_CONTEXT_WINDOW",
            "CHAT_LOCAL_CONTEXT_TOKENS",
            "local",
            "131072",
            "131072",
        ),
    ],
)
def test_mapping_rows(old, new, route, value, expected):
    env = {"CHAT_PLATFORM_ROUTE": route, old: value}
    result, notes = resolved(env)
    assert result[new] == expected
    assert any(note.startswith(old + " ") for note in notes)


@pytest.mark.parametrize(
    "source,target,extra",
    [
        ("OPEN_ROUTER_API_KEY", "CHAT_OPENROUTER_API_KEY", {}),
        ("OPENAI_API_KEY", "CHAT_OPENROUTER_API_KEY", {}),
        (
            "ANTHROPIC_API_KEY",
            "CHAT_ANTHROPIC_API_KEY",
            {"CHAT_USE_OPENROUTER": "false"},
        ),
        (
            "ANTHROPIC_BASE_URL",
            "CHAT_ANTHROPIC_BASE_URL",
            {"CHAT_USE_OPENROUTER": "false", "ANTHROPIC_API_KEY": "ant"},
        ),
        ("OPENROUTER_BASE_URL", "CHAT_OPENROUTER_BASE_URL", {"CHAT_API_KEY": "chat"}),
        ("OPENAI_BASE_URL", "CHAT_OPENROUTER_BASE_URL", {"CHAT_API_KEY": "chat"}),
    ],
)
def test_legacy_integration_fallbacks(source, target, extra):
    value = "https://fallback.example" if "URL" in source else "fallback"
    result, _ = resolved({**extra, source: value})
    assert result[target] == value
    explicit, _ = resolved(
        {**extra, source: value, "CHAT_PLATFORM_ROUTE": "openrouter"}
    )
    assert target not in explicit


def test_key_fallback_priority_and_removal():
    env = {
        "CHAT_API_KEY": "chat",
        "OPEN_ROUTER_API_KEY": "router",
        "OPENAI_API_KEY": "openai",
        "CHAT_DIRECT_ANTHROPIC_API_KEY": "direct",
        "ANTHROPIC_API_KEY": "batch",
    }
    for removed, expected in [
        (None, "chat"),
        ("CHAT_API_KEY", "router"),
        ("OPEN_ROUTER_API_KEY", "openai"),
    ]:
        if removed:
            env.pop(removed)
        actual, _ = resolved(env)
        assert actual["CHAT_OPENROUTER_API_KEY"] == expected
        assert actual["CHAT_ANTHROPIC_API_KEY"] == "direct"
    env.pop("CHAT_DIRECT_ANTHROPIC_API_KEY")
    assert resolved(env)[0]["CHAT_ANTHROPIC_API_KEY"] == "batch"


def test_local_defaults_and_other_engine_ignored():
    local = {
        "CHAT_USE_LOCAL": "true",
        "CHAT_BASE_URL": "http://local/v1",
        "CHAT_FAST_STANDARD_MODEL": "qwen:8b",
        "CHAT_FAST_ADVANCED_MODEL": "anthropic/old",
        "CHAT_THINKING_STANDARD_MODEL": "ignored",
    }
    result, notes = resolved(local)
    assert result["CHAT_STANDARD_MODEL"] == result["CHAT_ADVANCED_MODEL"] == "qwen:8b"
    assert result["CHAT_LOCAL_ADAPTER"] == "openai_chat"
    assert result["CHAT_LOCAL_CONTEXT_TOKENS"] == str(MIN_LOCAL_CONTEXT_TOKENS)
    assert any(
        note.startswith("CHAT_THINKING_STANDARD_MODEL ") and "ignored" in note
        for note in notes
    )
    assert (
        resolved({"CHAT_API_KEY": "chat", "CHAT_FAST_MODEL": "ignored"})[0].get(
            "CHAT_STANDARD_MODEL"
        )
        is None
    )


@pytest.mark.parametrize("bad_number", ["bad", "²", "9" * 5000])
def test_invalid_legacy_numeric_settings_do_not_stop(bad_number):
    env = {
        "CHAT_USE_LOCAL": "true",
        "CHAT_BASE_URL": "http://local",
        "CHAT_CLAUDE_AGENT_CONTEXT_WINDOW": bad_number,
        "CHAT_CLAUDE_AGENT_MAX_TURNS": bad_number,
    }
    result, _ = resolved(env)
    assert result["CHAT_LOCAL_CONTEXT_TOKENS"] == str(MIN_LOCAL_CONTEXT_TOKENS)
    assert "CHAT_AGENT_MAX_TURNS" not in result
    assert check_chat_environment(env, [])


def test_new_names_win_even_from_dotenv(tmp_path):
    envfile = tmp_path / ".env"
    envfile.write_text("CHAT_OPENROUTER_API_KEY=new-key\n")
    env = {"CHAT_API_KEY": "old-key"}
    result, notes = resolved(read_chat_settings(env, [envfile]))
    assert result["CHAT_OPENROUTER_API_KEY"] == "new-key"
    assert any("conflicts with CHAT_OPENROUTER_API_KEY" in note for note in notes)
    assert "old-key" not in str(notes)
    assert "new-key" not in str(notes)


@pytest.mark.parametrize(
    "env,name",
    [
        ({"CHAT_PLATFORM_ROUTE": "bad"}, "CHAT_PLATFORM_ROUTE"),
        ({"CHAT_PLATFORM_ROUTE": ""}, "CHAT_PLATFORM_ROUTE"),
        ({"CHAT_LOCAL_ADAPTER": "bad"}, "CHAT_LOCAL_ADAPTER"),
        ({"CHAT_LOCAL_CONTEXT_TOKENS": "not-a-number"}, "CHAT_LOCAL_CONTEXT_TOKENS"),
        ({"CHAT_LOCAL_CONTEXT_TOKENS": "32768"}, "CHAT_LOCAL_CONTEXT_TOKENS"),
        (
            {"CHAT_PLATFORM_ROUTE": "openrouter", "OPEN_ROUTER_API_KEY": "integration"},
            "CHAT_OPENROUTER_API_KEY",
        ),
        (
            {
                "CHAT_PLATFORM_ROUTE": "direct_anthropic",
                "ANTHROPIC_API_KEY": "integration",
            },
            "CHAT_ANTHROPIC_API_KEY",
        ),
        ({"CHAT_PLATFORM_ROUTE": "subscription", "BEHAVE_AS": "cloud"}, "BEHAVE_AS"),
        (
            {"CHAT_USE_CLAUDE_CODE_SUBSCRIPTION": "true", "BEHAVE_AS": "cloud"},
            "BEHAVE_AS",
        ),
    ],
)
def test_stop_cases(env, name):
    with pytest.raises(ChatEnvironmentError) as exc:
        check_chat_environment(env, [])
    assert name in str(exc.value)
    assert "integration" not in str(exc.value)


@pytest.fixture
def clean_settings(monkeypatch):
    for name in list(os.environ):
        if name.startswith("CHAT_") or name in chat_env._OTHER_SETTINGS:
            monkeypatch.delenv(name)
    monkeypatch.setitem(ChatConfig.model_config, "env_file", None)
    monkeypatch.setattr(chat_env, "backend_dotenv_paths", lambda: [])
    yield
    for name in list(os.environ):
        if name.startswith("CHAT_") or name in chat_env._OTHER_SETTINGS:
            os.environ.pop(name, None)


def test_hosted_environment_starts_with_chat_and_batch_keys(
    clean_settings, monkeypatch
):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    from backend.app import run_processes

    env = {
        "BEHAVE_AS": "cloud",
        "CHAT_API_KEY": "hosted-chat",
        "CHAT_OPENAI_API_KEY": "embeddings",
        "ANTHROPIC_API_KEY": "hosted-batch",
        "OPEN_ROUTER_API_KEY": "integration-router",
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    configs = []
    process = MagicMock()

    def start(**kwargs):
        cfg = ChatConfig()
        configs.append(cfg)
        assert cfg.platform_route == "openrouter"
        assert cfg.openrouter_api_key == "hosted-chat"
        assert cfg.anthropic_api_key == cfg.batch_anthropic_api_key == "hosted-batch"
        assert os.environ["CHAT_PLATFORM_ROUTE"] == "openrouter"

    process.start.side_effect = start
    # Standalone CLI construction must work before run_processes exports names.
    cfg = ChatConfig()
    assert cfg.openrouter_api_key == "hosted-chat"
    assert cfg.anthropic_api_key == "hosted-batch"
    run_processes(process)
    assert len(configs) == 1
    process.stop.assert_called_once()


@pytest.mark.parametrize("local", [False, True])
def test_full_dev_dotenv_config(clean_settings, tmp_path, local):
    envfile = tmp_path / ".env"
    text = (
        "CHAT_USE_LOCAL=true\nCHAT_BASE_URL=http://localhost:11434/v1\n"
        "CHAT_FAST_MODEL=qwen:8b\nCHAT_CLAUDE_AGENT_CONTEXT_WINDOW=131072\n"
        if local
        else "CHAT_USE_OPENROUTER=false\nANTHROPIC_API_KEY=self-host-ant\n"
        "ANTHROPIC_BASE_URL=https://deployment.example\nCHAT_MODEL=claude-sonnet-5-5\n"
    )
    envfile.write_text(text)
    cfg = ChatConfig(_env_file=envfile)
    assert cfg.platform_route == ("local" if local else "direct_anthropic")
    if local:
        assert cfg.local_adapter == "openai_chat"
        assert cfg.tier_model("standard") == cfg.tier_model("advanced") == "qwen:8b"
        assert cfg.local_context_tokens == 131072
    else:
        assert cfg.anthropic_api_key == "self-host-ant"
        assert cfg.anthropic_base_url == "https://deployment.example"


def test_logs_once_per_name_and_error_when_disabled(monkeypatch, caplog):
    monkeypatch.setattr(chat_env, "_logged_names", set())
    _, notes = migrate_chat_settings({"CHAT_USE_OPENROUTER": "false"})
    for _ in range(2):
        chat_env.log_chat_notes(notes)
    assert len(caplog.records) == len(notes)
    assert any(
        r.levelname == "ERROR" and r.message.startswith("CHAT_PLATFORM_ROUTE ")
        for r in caplog.records
    )


@pytest.mark.parametrize(
    "env,route",
    [
        ({"ANTHROPIC_API_KEY": "ant"}, "direct_anthropic"),
        ({"CHAT_DIRECT_ANTHROPIC_API_KEY": "ant"}, "direct_anthropic"),
        ({"CHAT_API_KEY": "router", "ANTHROPIC_API_KEY": "ant"}, "openrouter"),
        (
            {
                "CHAT_API_KEY": "router",
                "CHAT_BASE_URL": "ftp://wrong",
                "ANTHROPIC_API_KEY": "ant",
            },
            "direct_anthropic",
        ),
        (
            {
                "CHAT_API_KEY": "router",
                "CHAT_BASE_URL": "not-http",
                "ANTHROPIC_API_KEY": "ant",
            },
            "direct_anthropic",
        ),
        ({"CHAT_API_KEY": "router", "CHAT_BASE_URL": "ftp://wrong"}, "disabled"),
        (
            {
                "CHAT_USE_OPENROUTER": "false",
                "CHAT_API_KEY": "router",
                "ANTHROPIC_API_KEY": "ant",
            },
            "direct_anthropic",
        ),
    ],
)
def test_dev_fallback_ladder(env, route):
    assert resolved(env)[0]["CHAT_PLATFORM_ROUTE"] == route


@pytest.mark.parametrize(
    "source,target,env",
    [
        ("OPEN_ROUTER_API_KEY", "CHAT_OPENROUTER_API_KEY", {}),
        ("OPENAI_API_KEY", "CHAT_OPENROUTER_API_KEY", {}),
        ("ANTHROPIC_API_KEY", "CHAT_ANTHROPIC_API_KEY", {}),
        ("OPENROUTER_BASE_URL", "CHAT_OPENROUTER_BASE_URL", {"CHAT_API_KEY": "chat"}),
        ("OPENAI_BASE_URL", "CHAT_OPENROUTER_BASE_URL", {"CHAT_API_KEY": "chat"}),
        ("ANTHROPIC_BASE_URL", "CHAT_ANTHROPIC_BASE_URL", {"ANTHROPIC_API_KEY": "ant"}),
    ],
)
def test_integration_fallback_is_not_a_deprecation(source, target, env):
    value = "https://endpoint.example" if "URL" in source else "integration-key"
    _, notes = resolved({**env, source: value})
    matching = [note for note in notes if source in note]
    assert len(matching) == 1
    assert matching[0].startswith(target + " is not set")
    assert "as dev did" in matching[0]
    assert "deprecated" not in matching[0]
    assert value not in str(notes)
    _, unused = resolved({**env, source: value, "CHAT_PLATFORM_ROUTE": "disabled"})
    assert not any(source in note for note in unused)


@pytest.mark.parametrize("missing", ["CHAT_LOCAL_BASE_URL", "CHAT_STANDARD_MODEL"])
def test_explicit_local_requirements(missing):
    env = {
        "CHAT_PLATFORM_ROUTE": "local",
        "CHAT_LOCAL_BASE_URL": "http://local/v1",
        "CHAT_STANDARD_MODEL": "local-model",
    }
    env.pop(missing)
    assert any(missing in problem for problem in chat_environment_problems(env)[0])
    legacy = {"CHAT_USE_LOCAL": "true"}
    problems, notes = chat_environment_problems(legacy)
    assert not problems
    assert any("CHAT_LOCAL_BASE_URL" in note for note in notes)


@pytest.mark.parametrize(
    "tier,old",
    [
        ("CHAT_STANDARD_MODEL", "CHAT_THINKING_STANDARD_MODEL"),
        ("CHAT_ADVANCED_MODEL", "CHAT_THINKING_ADVANCED_MODEL"),
    ],
)
@pytest.mark.parametrize("route", ["direct_anthropic", "subscription"])
def test_anthropic_model_requirements_depend_on_origin(tier, old, route):
    env = {"CHAT_PLATFORM_ROUTE": route, "CHAT_ANTHROPIC_API_KEY": "ant"}
    problems, _ = chat_environment_problems({**env, tier: "other-vendor/model"})
    assert any(tier in problem for problem in problems)
    problems, notes = chat_environment_problems({**env, old: "other-vendor/model"})
    assert not problems
    assert any(tier in note and "Anthropic" in note for note in notes)


def test_migration_cached_and_no_extra_dotenv_read(
    clean_settings, monkeypatch, tmp_path
):
    from unittest.mock import Mock

    chat_env._cached_chat_migration.cache_clear()
    file = tmp_path / ".env"
    file.write_text("CHAT_API_KEY=chat-one\n")
    spy = Mock(wraps=chat_env.migrate_chat_settings)
    monkeypatch.setattr(chat_env, "migrate_chat_settings", spy)
    monkeypatch.setattr(
        chat_env,
        "read_chat_settings",
        Mock(side_effect=AssertionError("extra dotenv read")),
    )
    monkeypatch.setattr(
        chat_env,
        "check_chat_environment",
        Mock(side_effect=AssertionError("constructor preflight")),
    )
    assert ChatConfig(_env_file=file).openrouter_api_key == "chat-one"
    assert ChatConfig(_env_file=file).openrouter_api_key == "chat-one"
    assert spy.call_count == 1
    monkeypatch.setenv("CHAT_SESSION_TTL", "123")
    assert ChatConfig(_env_file=file).openrouter_api_key == "chat-one"
    assert spy.call_count == 1
    file.write_text("CHAT_API_KEY=chat-two\n")
    assert ChatConfig(_env_file=file).openrouter_api_key == "chat-two"
    assert spy.call_count == 2
    chat_env._cached_chat_migration.cache_clear()


def test_constructor_does_not_run_preflight(clean_settings, monkeypatch):
    monkeypatch.setenv("CHAT_PLATFORM_ROUTE", "not-a-route")
    assert ChatConfig().platform_route == "disabled"
    with pytest.raises(ChatEnvironmentError):
        check_chat_environment(dict(os.environ), [])
