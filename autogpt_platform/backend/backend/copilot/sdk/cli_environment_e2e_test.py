"""The Claude Code CLI a run starts holds no backend secret beyond its route's
credential and the network settings it needs.

A copilot turn (``stream_chat_completion_sdk``) and an orchestrator block run in
SDK mode each drive the Claude Code CLI against the billing suite's fake
provider, with secrets in this process's environment. While the CLI waits on
each model request, the provider reads that CLI process's environment as the
operating system holds it, so what is checked is what the CLI was started with.

The Linux SDK wheel bundles the CLI, so without one the tests fail there; on
other platforms, whose wheels ship without it, they skip. CI runs them on Linux.
"""

import platform
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest
from aiohttp import web
from pydantic import SecretStr

from backend.blocks.orchestrator import OrchestratorBlock
from backend.copilot.billing_e2e.invoice import SONNET, Route
from backend.copilot.billing_e2e.ledger import PlatformLedger
from backend.copilot.billing_e2e.provider import FakeProvider
from backend.copilot.billing_e2e.sdk_turns import SdkTurns
from backend.copilot.config import ChatConfig
from backend.copilot.model import ChatMessage, ChatSession
from backend.data.model import APIKeyCredentials

from . import service as sdk_service
from .cli_openrouter_compat_test import _resolve_cli_path

_SECRET_NAME = "SOME_PLATFORM_SECRET"
_SECRET = "sk-FAKE-platform-secret"
# The deployment's own Anthropic key, which no user's run may hold.
_PLATFORM_KEY = "sk-ant-FAKE-platform-key"
_USER_KEY = "sk-ant-FAKE-the-users-own-key"


@pytest.fixture
def ledger(monkeypatch: pytest.MonkeyPatch) -> PlatformLedger:
    """Takes the spend, cost-log and credit writes each turn makes."""
    ledger = PlatformLedger()
    ledger.install(monkeypatch)
    return ledger


@pytest.fixture(autouse=True)
def _memory_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Memory is not under test here, and its ingest needs a database."""
    monkeypatch.setattr(
        sdk_service, "_fetch_graphiti_context", AsyncMock(return_value=(False, ""))
    )


# Its own event loop, like the billing suite: a background task another test
# left on the shared session loop must not run into this one's turns.
@pytest.mark.asyncio(loop_scope="function")
async def test_a_copilot_turns_cli_holds_only_its_routes_credential(
    ledger: PlatformLedger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _require_cli()
    monkeypatch.setenv(_SECRET_NAME, _SECRET)
    async with _CliEnvironments() as provider:
        turns = SdkTurns(tmp_path, monkeypatch, provider, _config())
        session = _session()
        session.messages.append(ChatMessage(role="user", content="hello"))
        await turns(session, "hello", new_pod=False)
        await ledger.settle()

    assert provider.seen, "the CLI made no model request"
    for env in provider.seen:
        assert not _holds(env, _SECRET)
        # The direct-Anthropic route's own key, which it carries over.
        assert env["ANTHROPIC_API_KEY"] == "sk-ant-billing-e2e"


@pytest.mark.asyncio(loop_scope="function")
async def test_an_orchestrator_runs_cli_holds_only_the_users_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cli = _require_cli()
    monkeypatch.setenv(_SECRET_NAME, _SECRET)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _PLATFORM_KEY)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "home" / ".claude"))
    async with _CliEnvironments() as provider:
        # Where the deployment sends Anthropic traffic; the run keeps that.
        monkeypatch.setenv("ANTHROPIC_BASE_URL", f"{provider.url}/anthropic")
        with patch(
            "backend.blocks.orchestrator.require_verbatim_prompts_cli",
            return_value=str(cli),
        ):
            outputs = [
                output
                async for output in OrchestratorBlock()._execute_tools_sdk_mode(
                    input_data=_orchestrator_input(),
                    credentials=_user_key(),
                    tool_functions=[],
                    prompt=[{"role": "user", "content": "hello"}],
                    execution_params=MagicMock(graph_exec_id="graph-exec-e2e"),
                    execution_processor=MagicMock(),
                )
            ]

    assert not [value for name, value in outputs if name == "error"]
    assert provider.seen, "the CLI made no model request"
    for env in provider.seen:
        assert not _holds(env, _SECRET)
        assert not _holds(env, _PLATFORM_KEY)
        assert env["ANTHROPIC_API_KEY"] == _USER_KEY


class _CliEnvironments(FakeProvider):
    """The fake provider, reading the environment of each CLI that asks it
    for a reply while that CLI waits."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[dict[str, str]] = []

    async def _messages(self, request: web.Request, route: Route) -> web.StreamResponse:
        self.seen += _cli_environments()
        return await super()._messages(request, route)


def _cli_environments() -> list[dict[str, str]]:
    """The environment of every Claude Code CLI this process has started that
    is still running: the SDK starts each one with ``stream-json`` I/O."""
    environments = []
    for process in psutil.Process().children(recursive=True):
        try:
            if "stream-json" in process.cmdline():
                environments.append(process.environ())
        except psutil.Error:
            continue
    return environments


def _holds(env: dict[str, str], secret: str) -> bool:
    return any(secret in value for value in env.values())


def _require_cli() -> Path:
    """The Linux SDK wheel bundles the CLI, so there a missing one fails the
    test instead of skipping it."""
    cli = _resolve_cli_path()
    if cli is not None:
        return cli
    if platform.system() == "Linux":
        pytest.fail("No Claude Code CLI binary: the Linux SDK wheel bundles one")
    pytest.skip("No Claude Code CLI binary: this platform's SDK wheel has none")


def _config() -> ChatConfig:
    """A direct-Anthropic deployment, pinned so the deployment's own settings
    cannot change the path a turn takes, on the CLI ``_require_cli`` found."""
    cli = _resolve_cli_path()
    return sdk_service.config.model_copy(
        update={
            "use_local": False,
            "use_claude_code_subscription": False,
            "use_e2b_sandbox": False,
            "use_openrouter": False,
            "api_key": "sk-ant-cli-environment-e2e",
            "claude_agent_model": SONNET,
            "claude_agent_fallback_model": None,
            "claude_agent_cli_path": str(cli) if cli else None,
        }
    )


def _session() -> ChatSession:
    session = ChatSession.new("cli-environment-e2e-user", dry_run=False)
    # A titled chat skips title generation, a separate model call.
    session.title = "CLI environment e2e"
    return session


def _orchestrator_input() -> MagicMock:
    input_data = MagicMock(sys_prompt="", prompt="hello")
    input_data.model.value = SONNET
    return input_data


def _user_key() -> APIKeyCredentials:
    return APIKeyCredentials(
        id="cli-environment-e2e-key",
        provider="anthropic",
        api_key=SecretStr(_USER_KEY),
        title="The user's Anthropic key",
        expires_at=None,
    )
