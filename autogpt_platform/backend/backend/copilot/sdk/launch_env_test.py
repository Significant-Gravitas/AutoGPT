"""A CLI launch keeps only its route's credential, plus the runtime the OS and
the network need, from the backend's environment.

The parent is seeded with the kinds of secret a backend holds, all fake; each
launch's environment is read off the real SDK transport as it spawns the CLI.
"""

import subprocess
from pathlib import Path

import anyio
import pytest
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk._internal.transport import subprocess_cli

from backend.copilot.config import ChatConfig
from backend.copilot.route_profile import (
    codex_profile,
    platform_profile,
    user_key_profile,
)

from .env import GatewayEndpoint, route_launch_env
from .launch_env import SDK_MANAGED_NAMES, inherited, launch_options

_HOST_LOGIN = "sk-ant-oat01-FAKE-host-login"
_PARENT_SECRETS = {
    "CHAT_OPENROUTER_API_KEY": "sk-or-FAKE-chat-route",
    "CHAT_ANTHROPIC_API_KEY": "sk-ant-FAKE-chat-route",
    "CHAT_LOCAL_API_KEY": "FAKE-local-server-key",
    "OPEN_ROUTER_API_KEY": "sk-or-FAKE-integration",
    "ANTHROPIC_API_KEY": "sk-ant-FAKE-integration",
    "OPENAI_API_KEY": "sk-FAKE-integration",
    "E2B_API_KEY": "e2b_FAKE",
    "DATABASE_URL": "postgresql://app:FAKE-db-password@db:5432/app",
    "REDIS_PASSWORD": "FAKE-redis-password",
    "SUPABASE_SERVICE_ROLE_KEY": "FAKE-service-role",
    "LANGFUSE_SECRET_KEY": "sk-lf-FAKE",
    "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Bearer FAKE-otel",
    "SENTRY_DSN": "https://FAKE-sentry@sentry.example/1",
    "CLAUDE_CODE_OAUTH_TOKEN": _HOST_LOGIN,
}
_RUNTIME = {
    "HTTPS_PROXY": "http://proxy.example:3128",
    "NODE_EXTRA_CA_CERTS": "/etc/ssl/corp-ca.pem",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
}
_ROUTE_KEY = "sk-FAKE-the-routes-own-key"
_MODEL = "claude-sonnet-4-6"


class _Spawned(Exception):
    pass


@pytest.fixture
def parent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name, value in {**_PARENT_SECRETS, **_RUNTIME}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    # The SDK's version probe spawns the CLI too; this test reads the launch.
    monkeypatch.setenv("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")
    monkeypatch.setattr("backend.copilot.sdk.env.validate_subscription", lambda: None)


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
    with pytest.raises(Exception):
        await transport.connect()
    assert seen, "The SDK never spawned the CLI"
    return seen


def _platform(route: str, **settings) -> dict[str, str]:
    cfg = ChatConfig(platform_route=route, **settings)
    return route_launch_env(platform_profile(cfg, _MODEL, "env"))


_LAUNCHES = {
    "openrouter": (
        lambda: _platform("openrouter", openrouter_api_key=_ROUTE_KEY),
        "ANTHROPIC_AUTH_TOKEN",
        _ROUTE_KEY,
    ),
    "direct_anthropic": (
        lambda: _platform("direct_anthropic", anthropic_api_key=_ROUTE_KEY),
        "ANTHROPIC_API_KEY",
        _ROUTE_KEY,
    ),
    "subscription": (
        lambda: _platform("subscription"),
        "CLAUDE_CODE_OAUTH_TOKEN",
        _HOST_LOGIN,
    ),
    "codex_gateway": (
        lambda: route_launch_env(
            codex_profile(_MODEL, effort=None, routing_source="env", codex_engine=None),
            gateway=GatewayEndpoint(
                base_url="http://127.0.0.1:1", auth_token=_ROUTE_KEY
            ),
        ),
        "ANTHROPIC_AUTH_TOKEN",
        _ROUTE_KEY,
    ),
    "user_anthropic_key": (
        lambda: route_launch_env(user_key_profile("anthropic", _ROUTE_KEY, _MODEL)),
        "ANTHROPIC_API_KEY",
        _ROUTE_KEY,
    ),
    "user_openrouter_key": (
        lambda: route_launch_env(user_key_profile("open_router", _ROUTE_KEY, _MODEL)),
        "ANTHROPIC_AUTH_TOKEN",
        _ROUTE_KEY,
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("launch", sorted(_LAUNCHES))
async def test_a_launch_keeps_only_its_credential_and_the_runtime(
    launch: str, parent: None, monkeypatch: pytest.MonkeyPatch
):
    build, credential_name, credential = _LAUNCHES[launch]
    env = build()
    spawned = await _spawned_env(monkeypatch, env)

    others = {v for v in _PARENT_SECRETS.values() if v != credential}
    assert not {
        name for name, value in spawned.items() if any(s in value for s in others)
    }, f"{launch} kept another secret of the parent"
    assert spawned[credential_name] == credential
    unexpected = {
        name
        for name, value in spawned.items()
        if value
        and name not in env
        and not inherited(name)
        and name.upper() not in SDK_MANAGED_NAMES
    }
    assert not unexpected, f"{launch} kept {sorted(unexpected)}"
    for name, value in _RUNTIME.items():
        assert spawned[name] == value


@pytest.mark.asyncio
async def test_a_variable_the_backend_gains_later_is_blanked_too(
    parent: None, monkeypatch: pytest.MonkeyPatch
):
    """Worked out at each launch, not from a list of names."""
    env = _platform("openrouter", openrouter_api_key=_ROUTE_KEY)
    monkeypatch.setenv("SOME_SERVICE_TOKEN_ADDED_LATER", "FAKE-later")

    spawned = await _spawned_env(monkeypatch, env)

    assert spawned["SOME_SERVICE_TOKEN_ADDED_LATER"] == ""


def test_the_worker_warm_up_starts_the_cli_from_the_allowlist(
    parent: None, monkeypatch: pytest.MonkeyPatch
):
    from backend.copilot.executor import processor

    runs: list[dict[str, str]] = []

    def run(args, **kwargs):
        runs.append(kwargs["env"])
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(processor.subprocess, "run", run)

    processor.CoPilotProcessor._prewarm_cli(
        None, cli_path="claude-under-test"  # type: ignore[arg-type]
    )

    [env] = runs
    assert all(inherited(name) for name in env)
    assert not set(_PARENT_SECRETS.values()) & set(env.values())


def test_the_subscription_check_starts_the_cli_from_the_allowlist(
    parent: None, monkeypatch: pytest.MonkeyPatch
):
    from . import subscription

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
    secrets = set(_PARENT_SECRETS.values())
    assert not secrets & set(version_env.values())
    assert set(auth_env.values()) & secrets == {_HOST_LOGIN}
    assert all(inherited(name) for name in version_env)
