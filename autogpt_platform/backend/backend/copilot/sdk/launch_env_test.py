"""A CLI launch keeps its route's credential, the CLI's own configuration and
the runtime the OS and the network need from the backend's environment, and
nothing else.

The parent is seeded with the kinds of setting a backend holds, all fake;
each launch's environment is read off the real SDK transport as it spawns the
CLI.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch

import anyio
import pytest
from claude_agent_sdk import ClaudeAgentOptions, CLIConnectionError
from claude_agent_sdk._internal.transport import subprocess_cli

from backend.copilot.config import ChatConfig
from backend.copilot.executor import processor

from . import cli_version, subscription
from .env import build_sdk_env
from .launch_env import (
    SDK_MANAGED_NAMES,
    SKIP_SDK_VERSION_CHECK,
    inherited,
    kept_by_launch,
    launch_options,
)

# The platform's own Anthropic key and the host's Claude login: each is the
# credential of exactly one route below and must reach no other.
_PLATFORM_ANTHROPIC_KEY = "sk-ant-FAKE-platform"
_HOST_LOGIN = "sk-ant-oat01-FAKE-host-login"
_PARENT_SECRETS = {
    "DATABASE_URL": "postgresql://app:FAKE-db-password@db:5432/app",
    "REDIS_PASSWORD": "FAKE-redis-password",
    "ENCRYPTION_KEY": "FAKE-encryption-key",
    "SUPABASE_SERVICE_ROLE_KEY": "FAKE-service-role",
    "CHAT_API_KEY": "sk-or-FAKE-chat-route",
    "OPEN_ROUTER_API_KEY": "sk-or-FAKE-integration",
    "OPENAI_API_KEY": "sk-FAKE-integration",
    "E2B_API_KEY": "e2b_FAKE",
    "LANGFUSE_SECRET_KEY": "sk-lf-FAKE",
    "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Basic FAKE-otel",
    "SENTRY_DSN": "https://FAKE-sentry@sentry.example/1",
    "ANTHROPIC_API_KEY": _PLATFORM_ANTHROPIC_KEY,
    "CLAUDE_CODE_OAUTH_TOKEN": _HOST_LOGIN,
}
_RUNTIME = {
    "HTTPS_PROXY": "http://proxy.example:3128",
    "NODE_EXTRA_CA_CERTS": "/etc/ssl/corp-ca.pem",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
}
# An operator's setting for the CLI, which every launch keeps.
_CLI_SETTING = ("CLAUDE_CODE_MAX_RETRIES", "7")
_ROUTE_KEY = "sk-FAKE-the-routes-own-key"


class _Spawned(Exception):
    pass


@pytest.fixture
def parent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name, value in {**_PARENT_SECRETS, **_RUNTIME}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(*_CLI_SETTING)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    # Set as a launch sets it, so a launch's setting is undone after each test.
    monkeypatch.setenv(SKIP_SDK_VERSION_CHECK, "1")


async def _spawned_env(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> dict[str, str]:
    """The environment the SDK hands the CLI for a launch with *env*."""
    seen: dict[str, str] = {}

    async def open_process(*_args, **kwargs):
        seen.update(kwargs["env"])
        raise _Spawned

    # The SDK's transport spawns through the anyio module.
    monkeypatch.setattr(anyio, "open_process", open_process)
    options = launch_options(ClaudeAgentOptions(env=env, cli_path="claude-under-test"))
    transport = subprocess_cli.SubprocessCLITransport(prompt="", options=options)
    # The SDK reports the stopped spawn as a failed start.
    with pytest.raises(CLIConnectionError):
        await transport.connect()
    assert seen, "The SDK never spawned the CLI"
    return seen


def _assert_runtime_only(env: dict[str, str]) -> None:
    """*env* holds the runtime the parent was given and nothing else of it."""
    assert all(inherited(name) for name in env)
    assert not set(_PARENT_SECRETS.values()) & set(env.values())
    for name, value in _RUNTIME.items():
        assert env[name] == value


def _config(**overrides) -> ChatConfig:
    """A route's config, with every field the route reads set explicitly."""
    settings = {
        "use_claude_code_subscription": False,
        "use_openrouter": False,
        "api_key": None,
        "base_url": None,
        "thinking_standard_model": "anthropic/claude-sonnet-4-6",
        "thinking_advanced_model": "anthropic/claude-opus-4-7",
        "aux_api_key": "or-aux-key",
        **overrides,
    }
    return ChatConfig(**settings)


def _copilot_env(**overrides) -> dict[str, str]:
    with (
        patch("backend.copilot.sdk.env.config", _config(**overrides)),
        patch("backend.copilot.sdk.env.validate_subscription"),
    ):
        return build_sdk_env(session_id="session-1", user_id="user-1")


def _codex_env() -> dict[str, str]:
    with patch("backend.copilot.sdk.env.config", _config()):
        return build_sdk_env(
            codex_gateway_url="http://127.0.0.1:1", codex_gateway_token=_ROUTE_KEY
        )


# Each route's environment, the variable its credential travels in, and that
# credential.
_LAUNCHES = {
    "openrouter": (
        lambda: _copilot_env(
            use_openrouter=True,
            api_key=_ROUTE_KEY,
            base_url="https://openrouter.ai/api/v1",
        ),
        "ANTHROPIC_AUTH_TOKEN",
        _ROUTE_KEY,
    ),
    # The direct route's key comes from the parent, through the CLI's own
    # configuration.
    "direct_anthropic": (
        lambda: _copilot_env(),
        "ANTHROPIC_API_KEY",
        _PLATFORM_ANTHROPIC_KEY,
    ),
    "subscription": (
        lambda: _copilot_env(use_claude_code_subscription=True),
        "CLAUDE_CODE_OAUTH_TOKEN",
        _HOST_LOGIN,
    ),
    "codex_gateway": (_codex_env, "ANTHROPIC_AUTH_TOKEN", _ROUTE_KEY),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("launch", sorted(_LAUNCHES))
async def test_a_launch_keeps_only_its_credential_the_cli_config_and_the_runtime(
    launch: str, parent: None, monkeypatch: pytest.MonkeyPatch
):
    build, credential_name, credential = _LAUNCHES[launch]
    env = build()
    spawned = await _spawned_env(monkeypatch, env)

    others = {v for v in _PARENT_SECRETS.values() if v != credential}
    assert not {
        name for name, value in spawned.items() if any(s in value for s in others)
    }, f"{launch} kept another setting of the parent"
    assert spawned[credential_name] == credential
    unexpected = {
        name
        for name, value in spawned.items()
        if value
        and name not in env
        and not kept_by_launch(name)
        and name.upper() not in SDK_MANAGED_NAMES
    }
    assert not unexpected, f"{launch} kept {sorted(unexpected)}"
    for name, value in _RUNTIME.items():
        assert spawned[name] == value
    name, value = _CLI_SETTING
    assert spawned[name] == value


@pytest.mark.asyncio
async def test_a_variable_the_backend_gains_later_is_blanked_too(
    parent: None, monkeypatch: pytest.MonkeyPatch
):
    """Worked out at each launch, not from a list of names."""
    env = _copilot_env(
        use_openrouter=True, api_key=_ROUTE_KEY, base_url="https://openrouter.ai/api"
    )
    monkeypatch.setenv("SOME_SERVICE_TOKEN_ADDED_LATER", "FAKE-later")

    spawned = await _spawned_env(monkeypatch, env)

    assert spawned["SOME_SERVICE_TOKEN_ADDED_LATER"] == ""


@pytest.mark.asyncio
async def test_a_launch_turns_off_the_sdks_own_version_check(
    parent: None, monkeypatch: pytest.MonkeyPatch
):
    """The SDK's check would start the CLI on the parent's whole environment
    before the launch."""
    monkeypatch.delenv(SKIP_SDK_VERSION_CHECK)
    spawned: list[dict[str, str] | None] = []

    async def open_process(*_args, **kwargs):
        spawned.append(kwargs.get("env"))
        raise _Spawned

    monkeypatch.setattr(anyio, "open_process", open_process)
    options = launch_options(ClaudeAgentOptions(cli_path="claude-under-test"))
    transport = subprocess_cli.SubprocessCLITransport(prompt="", options=options)
    with pytest.raises(CLIConnectionError):
        await transport.connect()

    assert None not in spawned, "the SDK's version check started the CLI"
    [env] = spawned
    assert env["DATABASE_URL"] == ""


def test_the_worker_warm_up_starts_the_cli_from_the_runtime_only(
    parent: None, monkeypatch: pytest.MonkeyPatch
):
    runs: list[dict[str, str]] = []

    def run(args, **kwargs):
        runs.append(kwargs["env"])
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(processor.subprocess, "run", run)
    # The warm-up uses nothing of the processor, so it is not started.
    warmer = processor.CoPilotProcessor.__new__(processor.CoPilotProcessor)

    warmer._prewarm_cli(cli_path="claude-under-test")

    [env] = runs
    _assert_runtime_only(env)


def test_the_version_check_starts_the_cli_from_the_runtime_only(
    parent: None, monkeypatch: pytest.MonkeyPatch
):
    runs: list[dict[str, str]] = []

    def run(args, **kwargs):
        runs.append(kwargs["env"])
        return subprocess.CompletedProcess(args, 0, stdout="2.1.284 (Claude Code)")

    monkeypatch.setattr(cli_version.subprocess, "run", run)

    assert cli_version._probe_version("claude-under-test") == (2, 1, 284)

    [env] = runs
    _assert_runtime_only(env)


def test_the_subscription_check_keeps_only_the_host_login(
    parent: None, monkeypatch: pytest.MonkeyPatch
):
    runs: list[dict[str, str]] = []

    def run(args, **kwargs):
        runs.append(kwargs["env"])
        stdout = '{"loggedIn": true}' if "auth" in args else "2.1.284"
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subscription, "find_bundled_cli", lambda: "claude-under-test")
    monkeypatch.setattr(subscription, "provision_credentials_file", lambda: None)
    monkeypatch.setattr(subscription.subprocess, "run", run)

    subscription.validate_subscription.__wrapped__()

    version_env, auth_env = runs
    _assert_runtime_only(version_env)
    assert set(auth_env.values()) & set(_PARENT_SECRETS.values()) == {_HOST_LOGIN}
    assert auth_env["ANTHROPIC_API_KEY"] == ""
    for name, value in _RUNTIME.items():
        assert auth_env[name] == value
