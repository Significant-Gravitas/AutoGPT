"""A CLI launch keeps only its own credential, plus the runtime the OS and the
network need, from the backend's environment.

The parent is seeded with the kinds of secret a backend holds, all fake; each
launch's environment is read off the real SDK transport as it spawns the CLI.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import ClassVar, Self
from unittest.mock import MagicMock, patch

import anyio
import claude_agent_sdk
import pytest
from claude_agent_sdk import ClaudeAgentOptions, CLIConnectionError
from claude_agent_sdk._internal.transport import subprocess_cli
from pydantic import SecretStr

from backend.blocks.orchestrator import OrchestratorBlock
from backend.copilot.config import ChatConfig
from backend.copilot.executor.processor import CoPilotProcessor
from backend.data.model import APIKeyCredentials

from . import cli_version, subscription
from .env import build_sdk_env
from .launch_env import SDK_MANAGED_NAMES, inherited, launch_options

_HOST_LOGIN = "sk-ant-oat01-FAKE-host-login"
# The deployment's own Anthropic key: the direct-Anthropic route's credential,
# and a platform key no user's launch may hold.
_DEPLOYMENT_KEY = "sk-ant-FAKE-deployment"
_PARENT_SECRETS = {
    "CHAT_API_KEY": "sk-or-FAKE-chat-route",
    "OPEN_ROUTER_API_KEY": "sk-or-FAKE-integration",
    "ANTHROPIC_API_KEY": _DEPLOYMENT_KEY,
    "OPENAI_API_KEY": "sk-FAKE-integration",
    "E2B_API_KEY": "e2b_FAKE",
    "DATABASE_URL": "postgresql://app:FAKE-db-password@db:5432/app",
    "REDIS_PASSWORD": "FAKE-redis-password",
    "SUPABASE_SERVICE_ROLE_KEY": "FAKE-service-role",
    "ENCRYPTION_KEY": "FAKE-encryption-key",
    "STRIPE_API_KEY": "sk_live_FAKE",
    "LANGFUSE_SECRET_KEY": "sk-lf-FAKE",
    "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Bearer FAKE-otel",
    "CLAUDE_CODE_OAUTH_TOKEN": _HOST_LOGIN,
}
# What the OS and the network need, kept as configured: a proxy's credentials
# too, since the CLI cannot reach its model without them.
_RUNTIME = {
    "HTTPS_PROXY": "http://proxy-user:FAKE-proxy-password@proxy.example:3128",
    "NODE_EXTRA_CA_CERTS": "/etc/ssl/corp-ca.pem",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
}
_ROUTE_KEY = "sk-FAKE-the-routes-own-key"
_CLI = "claude-under-test"


class _Spawned(Exception):
    pass


@pytest.fixture
def parent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name, value in {**_PARENT_SECRETS, **_RUNTIME}.items():
        monkeypatch.setenv(name, value)
    for name in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setattr("backend.copilot.sdk.env.validate_subscription", lambda: None)


async def _spawned_env(
    monkeypatch: pytest.MonkeyPatch, options: ClaudeAgentOptions
) -> dict[str, str]:
    """The environment the SDK hands the CLI it starts with *options*, which
    is the only process it starts: one started without an environment of its
    own, like the SDK's own version check, gets the backend's whole one."""
    envs: list[dict[str, str] | None] = []

    async def open_process(*_args, **kwargs):
        envs.append(kwargs.get("env"))
        raise _Spawned

    # The SDK's transport spawns through the anyio module.
    monkeypatch.setattr(anyio, "open_process", open_process)
    transport = subprocess_cli.SubprocessCLITransport(prompt="", options=options)
    with pytest.raises(CLIConnectionError):
        await transport.connect()
    [env] = envs
    assert env is not None, "the SDK started the CLI with the whole environment"
    return env


def _config(**settings) -> ChatConfig:
    defaults = {
        "use_local": False,
        "use_claude_code_subscription": False,
        "use_openrouter": False,
        "thinking_standard_model": "anthropic/claude-sonnet-4-6",
        "thinking_advanced_model": "anthropic/claude-opus-4-7",
        "aux_api_key": "or-aux-key",
    }
    return ChatConfig(**{**defaults, **settings})


# Each copilot route: the deployment settings, the ``build_sdk_env``
# arguments, and the credential the launch must keep.
_LAUNCHES = {
    "openrouter": (
        {
            "use_openrouter": True,
            "api_key": _ROUTE_KEY,
            "base_url": "https://openrouter.ai/api/v1",
        },
        {},
        "ANTHROPIC_AUTH_TOKEN",
        _ROUTE_KEY,
    ),
    "direct_anthropic": ({}, {}, "ANTHROPIC_API_KEY", _DEPLOYMENT_KEY),
    "subscription": (
        {"use_claude_code_subscription": True},
        {},
        "CLAUDE_CODE_OAUTH_TOKEN",
        _HOST_LOGIN,
    ),
    "codex_gateway": (
        {},
        {"codex_gateway_url": "http://127.0.0.1:1", "codex_gateway_token": _ROUTE_KEY},
        "ANTHROPIC_AUTH_TOKEN",
        _ROUTE_KEY,
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("launch", sorted(_LAUNCHES))
async def test_a_copilot_launch_keeps_only_its_credential_and_the_runtime(
    launch: str, parent: None, monkeypatch: pytest.MonkeyPatch
):
    settings, arguments, credential_name, credential = _LAUNCHES[launch]
    monkeypatch.setattr("backend.copilot.sdk.env.config", _config(**settings))
    env = build_sdk_env(**arguments)
    options = launch_options(ClaudeAgentOptions(env=env, cli_path=_CLI))

    spawned = await _spawned_env(monkeypatch, options)

    _assert_keeps_only(spawned, env, credential)
    assert spawned[credential_name] == credential


@pytest.mark.asyncio
async def test_a_variable_the_backend_gains_later_is_blanked_too(
    parent: None, monkeypatch: pytest.MonkeyPatch
):
    """Worked out at each launch, not from a list of names."""
    monkeypatch.setattr("backend.copilot.sdk.env.config", _config())
    env = build_sdk_env()
    monkeypatch.setenv("SOME_SERVICE_TOKEN_ADDED_LATER", "FAKE-later")

    spawned = await _spawned_env(
        monkeypatch, launch_options(ClaudeAgentOptions(env=env, cli_path=_CLI))
    )

    assert spawned["SOME_SERVICE_TOKEN_ADDED_LATER"] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider, credential_name",
    [("anthropic", "ANTHROPIC_API_KEY"), ("open_router", "ANTHROPIC_AUTH_TOKEN")],
)
async def test_an_orchestrator_launch_keeps_only_the_users_key_and_the_runtime(
    provider: str, credential_name: str, parent: None, monkeypatch: pytest.MonkeyPatch
):
    """Read off the options the block starts its CLI with, so a launch that
    skips the allowlist fails here."""
    options = await _orchestrator_launch(provider)

    spawned = await _spawned_env(monkeypatch, options)

    _assert_keeps_only(spawned, options.env, _ROUTE_KEY)
    assert spawned[credential_name] == _ROUTE_KEY


@pytest.mark.asyncio
async def test_an_orchestrator_launch_sends_the_users_key_where_anthropic_is_reached(
    parent: None, monkeypatch: pytest.MonkeyPatch
):
    """A base URL the deployment points Anthropic traffic at still applies,
    without a password embedded in it or the deployment's auth token."""
    monkeypatch.setenv(
        "ANTHROPIC_BASE_URL",
        "https://svc:FAKE-gateway-password@anthropic-proxy.example:8443/v",
    )
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "FAKE-deployment-bearer")
    options = await _orchestrator_launch("anthropic")

    spawned = await _spawned_env(monkeypatch, options)

    assert spawned["ANTHROPIC_BASE_URL"] == "https://anthropic-proxy.example:8443/v"
    assert spawned["ANTHROPIC_API_KEY"] == _ROUTE_KEY
    assert spawned["ANTHROPIC_AUTH_TOKEN"] == ""
    assert not any("FAKE-gateway-password" in value for value in spawned.values())


@pytest.mark.parametrize("preset", [None, ""])
def test_loading_the_launch_code_turns_the_sdks_own_version_check_off(
    preset: str | None,
):
    """The SDK version-checks the CLI by starting it with no environment of
    its own, the backend's whole one, before each launch. It reads the switch
    for truthiness, so an empty value a deployment sets must not leave it on.

    The child gets its environment explicitly: on Windows an empty value set
    through ``os.environ`` removes the variable instead of passing it on."""
    name = "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK"
    env = {key: value for key, value in os.environ.items() if key != name}
    if preset is not None:
        env[name] = preset
    check = (
        f"import os; print(repr(os.environ.get({name!r}))); "
        f"import backend.copilot.sdk.env; print(repr(os.environ.get({name!r})))"
    )

    result = subprocess.run(
        [sys.executable, "-c", check],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    lines = result.stdout.splitlines()
    assert lines[:1] == [repr(preset)], result.stderr[-2000:]
    assert lines[-1:] == ["'1'"], result.stderr[-2000:]


def test_the_worker_warm_up_starts_the_cli_from_the_allowlist(
    parent: None, monkeypatch: pytest.MonkeyPatch
):
    envs = _record_runs(monkeypatch, "2.1.284")

    CoPilotProcessor.__new__(CoPilotProcessor)._prewarm_cli(cli_path=_CLI)

    [env] = envs
    assert all(inherited(name) for name in env)
    assert not set(_PARENT_SECRETS.values()) & set(env.values())


def test_the_version_check_starts_the_cli_from_the_allowlist(
    parent: None, monkeypatch: pytest.MonkeyPatch
):
    envs = _record_runs(monkeypatch, "2.1.284 (Claude Code)")

    cli_version._probe_version(_CLI)

    [env] = envs
    assert all(inherited(name) for name in env)
    assert not set(_PARENT_SECRETS.values()) & set(env.values())


def test_the_subscription_check_starts_the_cli_from_the_allowlist(
    parent: None, monkeypatch: pytest.MonkeyPatch
):
    envs = _record_runs(monkeypatch, "2.1.284", auth='{"loggedIn": true}')
    monkeypatch.setattr(subscription, "find_bundled_cli", lambda: _CLI)
    monkeypatch.setattr(subscription, "provision_credentials_file", lambda: None)

    subscription.validate_subscription.__wrapped__()

    version_env, auth_env = envs
    secrets = set(_PARENT_SECRETS.values())
    assert not secrets & set(version_env.values())
    assert set(auth_env.values()) & secrets == {_HOST_LOGIN}
    assert all(inherited(name) for name in version_env)


def _assert_keeps_only(
    spawned: dict[str, str], env: dict[str, str], credential: str
) -> None:
    others = {value for value in _PARENT_SECRETS.values() if value != credential}
    assert not {
        name for name, value in spawned.items() if any(s in value for s in others)
    }, "the launch kept another secret of the parent"
    unexpected = {
        name
        for name, value in spawned.items()
        if value
        and name not in env
        and not inherited(name)
        and name.upper() not in SDK_MANAGED_NAMES
    }
    assert not unexpected, f"the launch kept {sorted(unexpected)}"
    for name, value in _RUNTIME.items():
        assert spawned[name] == value


class _Started(Exception):
    pass


class _StartingClient:
    """Stands in for ``ClaudeSDKClient``: keeps the options the CLI would
    start with, then stops the run."""

    seen: ClassVar[list[ClaudeAgentOptions]] = []

    def __init__(self, options: ClaudeAgentOptions) -> None:
        self.seen.append(options)

    async def __aenter__(self) -> Self:
        raise _Started

    async def __aexit__(self, *_: object) -> None:
        return None


async def _orchestrator_launch(provider: str) -> ClaudeAgentOptions:
    """The options the orchestrator block starts its CLI with, on a user's
    *provider* key."""
    input_data = MagicMock(sys_prompt="", prompt="hello")
    input_data.model.value = "claude-sonnet-4-6"
    credentials = APIKeyCredentials(
        id="test-user-key",
        provider=provider,
        api_key=SecretStr(_ROUTE_KEY),
        title="The user's key",
        expires_at=None,
    )
    with (
        patch.object(_StartingClient, "seen", []) as seen,
        patch.object(claude_agent_sdk, "ClaudeSDKClient", _StartingClient),
        patch(
            "backend.blocks.orchestrator.require_verbatim_prompts_cli",
            return_value=_CLI,
        ),
    ):
        async for _ in OrchestratorBlock()._execute_tools_sdk_mode(
            input_data=input_data,
            credentials=credentials,
            tool_functions=[],
            prompt=[{"role": "user", "content": "hello"}],
            execution_params=MagicMock(graph_exec_id="graph-exec-1"),
            execution_processor=MagicMock(),
        ):
            pass
    [options] = seen
    return options


def _record_runs(
    monkeypatch: pytest.MonkeyPatch, version: str, *, auth: str = ""
) -> list[dict[str, str]]:
    """Stand in for ``subprocess.run``; return the environment of each run."""
    envs: list[dict[str, str]] = []

    def run(args, **kwargs):
        envs.append(kwargs["env"])
        stdout = auth if "auth" in args else version
        if not kwargs.get("text") and not kwargs.get("encoding"):
            return subprocess.CompletedProcess(args, 0, stdout.encode(), b"")
        return subprocess.CompletedProcess(args, 0, stdout, "")

    monkeypatch.setattr(subprocess, "run", run)
    return envs
