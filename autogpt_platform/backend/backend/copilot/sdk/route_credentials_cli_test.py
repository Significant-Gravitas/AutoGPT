"""Wire test: a key-based launch never puts a Claude OAuth credential on the wire.

The host is set up the way a self-hosted box with a Claude login looks: a
``claude login`` credentials file in the CLI's config directory, and
``CLAUDE_CODE_OAUTH_TOKEN`` plus a stray key, token and base URL in the
backend's own environment, which the SDK merges under each launch's ``env``.
Every credential is fake.  Each case runs the real Claude Code CLI against a
local stub of the Anthropic Messages API with the environment the route
builds, and reads the credential off the request.

Skipped only where no CLI binary exists (the SDK's Windows wheel bundles
none); ``CLAUDE_AGENT_CLI_PATH`` points it at one.
"""

import asyncio
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

from backend.copilot.config import ChatConfig
from backend.copilot.route_profile import (
    codex_profile,
    platform_profile,
    user_key_profile,
)

from .cli_openrouter_compat_test import (
    _CapturedRequest,
    _resolve_cli_path,
    _start_fake_anthropic_server,
    assert_pinned_cli,
)
from .env import GatewayEndpoint, route_launch_env
from .launch_env import launch_options

_FAKE_OAUTH = "sk-ant-oat01-FAKE-not-a-real-credential-0000000000000000"
_PARENT_KEY = "sk-ant-FAKE-parent-environment-key"
_PARENT_TOKEN = "FAKE-parent-environment-token"
_ROUTE_KEY = "sk-FAKE-the-key-this-route-was-given"
_BILLING_HEADER = "x-anthropic-billing-header"
_MODEL = "claude-sonnet-4-6"
_TURN_TIMEOUT_SECONDS = 120

EnvBuilder = Callable[[str, Path], dict[str, str]]


@pytest.fixture
def cli_path() -> Path:
    path = _resolve_cli_path()
    if path is None:
        pytest.skip(
            "No Claude Code CLI binary: none bundled with claude-agent-sdk and "
            "CLAUDE_AGENT_CLI_PATH is unset."
        )
    return path


@pytest.fixture
def host_with_claude_login(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A host holding a Claude login; returns the launch's working directory."""
    # A run inside Claude Code inherits its CLAUDE_* flags, which change the
    # CLI's behaviour.
    for key in [k for k in os.environ if k.startswith("CLAUDE")]:
        monkeypatch.delenv(key)
    config_dir = tmp_path / "claude-config"
    config_dir.mkdir()
    (config_dir / ".credentials.json").write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": _FAKE_OAUTH,
                    "refreshToken": "",
                    "expiresAt": int((time.time() + 86400) * 1000),
                    "scopes": ["user:inference", "user:profile"],
                    "subscriptionType": "max",
                }
            }
        )
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    monkeypatch.setenv("DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _FAKE_OAUTH)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _PARENT_KEY)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", _PARENT_TOKEN)
    # A closed port: a launch that inherits it never reaches the stub.
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")
    work = tmp_path / "work"
    work.mkdir()
    return work


def _openrouter_env(stub_url: str, work: Path) -> dict[str, str]:
    cfg = ChatConfig(
        platform_route="openrouter",
        openrouter_api_key=_ROUTE_KEY,
        openrouter_base_url=stub_url,
    )
    return route_launch_env(platform_profile(cfg, _MODEL, "env"), sdk_cwd=str(work))


def _direct_env(stub_url: str, work: Path) -> dict[str, str]:
    cfg = ChatConfig(
        platform_route="direct_anthropic",
        anthropic_api_key=_ROUTE_KEY,
        anthropic_base_url=stub_url,
    )
    return route_launch_env(platform_profile(cfg, _MODEL, "env"), sdk_cwd=str(work))


def _codex_env(stub_url: str, work: Path) -> dict[str, str]:
    profile = codex_profile(
        _MODEL, effort=None, routing_source="env", codex_engine=None
    )
    return route_launch_env(
        profile,
        sdk_cwd=str(work),
        gateway=GatewayEndpoint(base_url=stub_url, auth_token=_ROUTE_KEY),
    )


def _user_key_env(provider: str) -> EnvBuilder:
    def build(stub_url: str, work: Path) -> dict[str, str]:
        env = route_launch_env(
            user_key_profile(provider, _ROUTE_KEY, _MODEL), sdk_cwd=str(work)
        )
        # A user's key goes to OpenRouter or Anthropic itself; only the test
        # points it at the stub.
        return {**env, "ANTHROPIC_BASE_URL": stub_url}

    return build


# Each launch's builder, the header its key travels in, and whether the
# CLI's billing-header system block stays (only where the CLI talks to
# Anthropic first-party).
_KEY_BASED_LAUNCHES: dict[str, tuple[EnvBuilder, str, bool]] = {
    "openrouter": (_openrouter_env, "bearer", False),
    "direct_anthropic": (_direct_env, "x-api-key", True),
    "codex_gateway": (_codex_env, "bearer", False),
    "orchestrator_openrouter_key": (_user_key_env("open_router"), "bearer", False),
    "orchestrator_anthropic_key": (_user_key_env("anthropic"), "x-api-key", True),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("launch", sorted(_KEY_BASED_LAUNCHES))
async def test_key_based_launch_authenticates_with_its_own_key(
    launch: str,
    cli_path: Path,
    host_with_claude_login: Path,
):
    builder, carrier, _ = _KEY_BASED_LAUNCHES[launch]
    requests = await _run_turn(cli_path, host_with_claude_login, builder)

    for request in requests:
        sent = _credentials(request)
        assert _FAKE_OAUTH not in sent.values(), f"{launch} sent the OAuth token"
        assert sent[carrier] == _ROUTE_KEY, f"{launch} did not send its own key"
        assert set(sent.values()) <= {"", _ROUTE_KEY}, f"{launch} sent {sent}"
        assert "oauth" not in _headers(request).get("anthropic-beta", "")


@pytest.mark.asyncio
async def test_without_an_explicit_key_the_cli_sends_the_oauth_token(
    cli_path: Path, host_with_claude_login: Path
):
    """The hazard the guards exist for: left to itself on this host, the CLI
    sends the OAuth bearer to whatever base URL it is given."""

    def no_key(stub_url: str, work: Path) -> dict[str, str]:
        return {
            "ANTHROPIC_BASE_URL": stub_url,
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_AUTH_TOKEN": "",
            # The launch keeps none of the parent's own CLI flags.
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }

    requests = await _run_turn(cli_path, host_with_claude_login, no_key)

    assert all(_credentials(r)["bearer"] == _FAKE_OAUTH for r in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("launch", sorted(_KEY_BASED_LAUNCHES))
async def test_the_route_decides_the_billing_header_block(
    launch: str, cli_path: Path, host_with_claude_login: Path
):
    builder, _, keeps_block = _KEY_BASED_LAUNCHES[launch]
    requests = await _run_turn(cli_path, host_with_claude_login, builder)

    for request in requests:
        has_block = any(_BILLING_HEADER in text for text in _system_texts(request))
        assert has_block is keeps_block, launch


async def _run_turn(
    cli_path: Path, work: Path, build_env: EnvBuilder
) -> list[_CapturedRequest]:
    """One CLI turn against a stub; returns its ``/v1/messages`` requests."""
    captured: list[_CapturedRequest] = []
    runner, port = await _start_fake_anthropic_server(captured)
    try:
        options = ClaudeAgentOptions(
            system_prompt="You are a test assistant.",
            setting_sources=[],
            extra_args={"strict-mcp-config": None},
            cwd=str(work),
            env=build_env(f"http://127.0.0.1:{port}", work),
            cli_path=str(cli_path),
        )
        await asyncio.wait_for(_one_turn(options), timeout=_TURN_TIMEOUT_SECONDS)
    finally:
        await runner.cleanup()
    assert captured, "The CLI made no request to the stub"
    assert_pinned_cli([r.headers.get("User-Agent", "") for r in captured], cli_path)
    return captured


async def _one_turn(options: ClaudeAgentOptions) -> None:
    # As the chat and the orchestrator launch: from the allowlisted parent.
    async with ClaudeSDKClient(launch_options(options)) as client:
        await client.query("hello")
        async for _ in client.receive_response():
            pass


def _headers(request: _CapturedRequest) -> dict[str, str]:
    return {name.lower(): value for name, value in request.headers.items()}


def _credentials(request: _CapturedRequest) -> dict[str, str]:
    headers = _headers(request)
    scheme, _, token = headers.get("authorization", "").partition(" ")
    return {
        "bearer": token if scheme.lower() == "bearer" else "",
        "x-api-key": headers.get("x-api-key", ""),
    }


def _system_texts(request: _CapturedRequest) -> list[str]:
    system = json.loads(request.body).get("system") or []
    if isinstance(system, str):
        return [system]
    return [block.get("text", "") for block in system]
