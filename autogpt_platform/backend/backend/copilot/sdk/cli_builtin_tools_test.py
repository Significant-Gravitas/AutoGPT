"""A claude-agent-sdk bump that ships a new CLI built-in fails here until it is classified."""

import asyncio
import os
from pathlib import Path

import pytest
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, SystemMessage

from .env import build_sdk_env
from .tool_adapter import get_copilot_tool_names, get_sdk_disallowed_tools


@pytest.mark.parametrize("use_e2b", [False, True])
@pytest.mark.asyncio
async def test_cli_offers_only_allowed_tools(
    use_e2b: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # A run inside Claude Code inherits its CLAUDE_* flags, which change the tool set.
    for key in [k for k in os.environ if k.startswith("CLAUDE")]:
        monkeypatch.delenv(key)
    allowed = get_copilot_tool_names(use_e2b=use_e2b)
    offered = await asyncio.wait_for(
        _cli_offered_tools(
            allowed, get_sdk_disallowed_tools(use_e2b=use_e2b), tmp_path
        ),
        timeout=60,
    )
    assert {"Task", "TodoWrite"} <= set(offered), offered
    unclassified = sorted(set(offered) - set(allowed))
    assert not unclassified, (
        f"The bundled Claude CLI offers {unclassified}, which security_hooks "
        "denies and nothing classifies. Add each to SDK_DISALLOWED_TOOLS in "
        "sdk/tool_adapter.py, or to _SDK_BUILTIN_ALWAYS if the copilot should use it."
    )


async def _cli_offered_tools(
    allowed: list[str], disallowed: list[str], tmp_path: Path
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
    options = ClaudeAgentOptions(
        setting_sources=[],
        extra_args={"strict-mcp-config": None},
        allowed_tools=allowed,
        disallowed_tools=disallowed,
        cwd=str(sdk_cwd),
        env=env,
    )
    async with ClaudeSDKClient(options) as client:
        await client.query("hi")
        async for message in client.receive_messages():
            if isinstance(message, SystemMessage) and message.subtype == "init":
                return message.data["tools"]
    raise AssertionError("The CLI ended without an init message")
