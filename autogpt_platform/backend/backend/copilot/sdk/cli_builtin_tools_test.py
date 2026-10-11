"""What the real Claude CLI offers the model under the copilot's options.

A claude-agent-sdk bump that ships a new CLI built-in fails here until it is
classified, and a change that would declare a built-in the session may not
call fails here too.
"""

import asyncio
import os
import platform
from pathlib import Path

import claude_agent_sdk
import pytest
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, SystemMessage

from backend.copilot.permissions import CopilotPermissions

from .env import build_sdk_env
from .service import _build_sdk_tool_options
from .tool_adapter import (
    SDK_SUBAGENT_TOOLS,
    get_copilot_tool_names,
    get_sdk_disallowed_tools,
)

_EXTRA_BUILTINS = "backend.copilot.sdk.service.config.claude_agent_extra_builtin_tools"


def _cli_path() -> str:
    """The CLI the copilot would run: ``ChatConfig.claude_agent_cli_path``'s
    env vars, else the binary bundled with the SDK wheel (none on Windows)."""
    override = os.environ.get("CHAT_CLAUDE_AGENT_CLI_PATH") or os.environ.get(
        "CLAUDE_AGENT_CLI_PATH"
    )
    if override:
        return override
    name = "claude.exe" if platform.system() == "Windows" else "claude"
    return str(Path(claude_agent_sdk.__file__).parent / "_bundled" / name)


_CLI_PATH = _cli_path()

pytestmark = pytest.mark.skipif(
    not Path(_CLI_PATH).is_file(),
    reason="No Claude CLI binary: the SDK wheel bundles none here and "
    "CHAT_CLAUDE_AGENT_CLI_PATH / CLAUDE_AGENT_CLI_PATH is unset",
)


@pytest.fixture(autouse=True)
def _clean_claude_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # A run inside Claude Code inherits its CLAUDE_* flags, which change the tool set.
    for key in [k for k in os.environ if k.startswith("CLAUDE")]:
        monkeypatch.delenv(key)


@pytest.mark.parametrize("use_e2b", [False, True])
@pytest.mark.asyncio
async def test_cli_offers_only_allowed_tools(use_e2b: bool, tmp_path: Path):
    allowed = get_copilot_tool_names(use_e2b=use_e2b)
    offered = await asyncio.wait_for(
        _cli_offered_tools(
            allowed, get_sdk_disallowed_tools(use_e2b=use_e2b), None, tmp_path
        ),
        timeout=60,
    )
    assert {"Task", "TodoWrite"} <= set(offered), offered
    # The sub-agent tool is known and deliberately left out of the allowed set.
    unclassified = sorted(set(offered) - set(allowed) - set(SDK_SUBAGENT_TOOLS))
    assert not unclassified, (
        f"The bundled Claude CLI offers {unclassified}, which security_hooks "
        "denies and nothing classifies. Add each to SDK_DISALLOWED_TOOLS in "
        "sdk/tool_adapter.py, or to _SDK_BUILTIN_ALWAYS if the copilot should use it."
    )


@pytest.mark.parametrize(
    "use_e2b, extra_builtins, whitelist, expected",
    [
        (False, [], None, {"Glob", "Grep", "TodoWrite"}),
        (True, [], None, {"TodoWrite"}),
        (True, [], ["web_fetch"], set()),
        # The init message names the sub-agent tool "Task"; the wire calls it "Agent".
        (False, ["Agent"], None, {"Glob", "Grep", "TodoWrite", "Task"}),
        (True, ["Agent"], None, {"TodoWrite", "Task"}),
        # A whitelist naming either alias keeps the re-enabled tool declared.
        (False, ["Agent"], ["Agent", "web_fetch", "TodoWrite"], {"TodoWrite", "Task"}),
        (True, ["Agent"], ["Task", "web_fetch"], {"Task"}),
    ],
)
@pytest.mark.asyncio
async def test_cli_declares_exactly_the_session_builtins(
    use_e2b: bool,
    extra_builtins: list[str],
    whitelist: list[str] | None,
    expected: set[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(_EXTRA_BUILTINS, extra_builtins)
    permissions = (
        None
        if whitelist is None
        else CopilotPermissions(tools=whitelist, tools_exclude=False)
    )
    tool_options = _build_sdk_tool_options(
        permissions, use_e2b=use_e2b, disabled_groups=[]
    )
    offered = await asyncio.wait_for(
        _cli_offered_tools(
            tool_options["allowed_tools"],
            tool_options["disallowed_tools"],
            tool_options["tools"],
            tmp_path,
        ),
        timeout=60,
    )
    assert {name for name in offered if not name.startswith("mcp__")} == expected


async def _cli_offered_tools(
    allowed: list[str],
    disallowed: list[str],
    tools: list[str] | None,
    tmp_path: Path,
) -> list[str]:
    """The tool list in the CLI's ``init`` message, which precedes any API call."""
    sdk_cwd = tmp_path / "copilot-session"
    sdk_cwd.mkdir()
    # A closed loopback port: the CLI only ever retries, so no network or key is needed.
    env = build_sdk_env(
        sdk_cwd=str(sdk_cwd),
        codex_gateway_url="http://127.0.0.1:9",
        codex_gateway_token="unused",
    )
    env["HOME"] = str(tmp_path)
    env["CLAUDE_CONFIG_DIR"] = str(tmp_path / "claude-config")
    options = ClaudeAgentOptions(
        setting_sources=[],
        extra_args={"strict-mcp-config": None},
        tools=tools,
        allowed_tools=allowed,
        disallowed_tools=disallowed,
        cwd=str(sdk_cwd),
        env=env,
        cli_path=_CLI_PATH,
    )
    async with ClaudeSDKClient(options) as client:
        await client.query("hi")
        async for message in client.receive_messages():
            if isinstance(message, SystemMessage) and message.subtype == "init":
                return message.data["tools"]
    raise AssertionError("The CLI ended without an init message")
