"""The tool lists ``stream_chat_completion_sdk`` hands to ClaudeAgentOptions."""

import pytest
from claude_agent_sdk import ClaudeAgentOptions

from backend.copilot.permissions import CopilotPermissions

from .service import _build_sdk_tool_options
from .tool_adapter import MCP_TOOL_PREFIX, SDK_SUBAGENT_TOOLS

_EXTRA_BUILTINS = "backend.copilot.sdk.service.config.claude_agent_extra_builtin_tools"


@pytest.fixture(autouse=True)
def _no_extra_builtins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_EXTRA_BUILTINS, [])


def _builtins(names: list[str]) -> set[str]:
    return {name for name in names if not name.startswith("mcp__")}


@pytest.mark.parametrize(
    "use_e2b, expected",
    [(False, ["Glob", "Grep", "TodoWrite"]), (True, ["TodoWrite"])],
)
def test_declares_exactly_the_builtins_the_hook_allows(
    use_e2b: bool, expected: list[str]
):
    options = _build_sdk_tool_options(None, use_e2b=use_e2b, disabled_groups=[])

    assert sorted(options["tools"]) == expected
    assert set(options["tools"]) == _builtins(options["allowed_tools"]) - set(
        options["disallowed_tools"]
    )


@pytest.mark.parametrize("use_e2b", [False, True])
def test_every_list_reaches_the_agent_options(use_e2b: bool):
    tool_options = _build_sdk_tool_options(None, use_e2b=use_e2b, disabled_groups=[])

    options = ClaudeAgentOptions(**tool_options)

    assert options.tools == tool_options["tools"]
    assert options.allowed_tools == tool_options["allowed_tools"]
    assert options.disallowed_tools == tool_options["disallowed_tools"]


@pytest.mark.parametrize("use_e2b", [False, True])
def test_operator_setting_declares_the_sub_agent_tool(
    use_e2b: bool, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(_EXTRA_BUILTINS, ["Agent"])

    options = _build_sdk_tool_options(None, use_e2b=use_e2b, disabled_groups=[])

    assert set(SDK_SUBAGENT_TOOLS) <= set(options["allowed_tools"])
    assert set(SDK_SUBAGENT_TOOLS) <= set(options["tools"])


@pytest.mark.parametrize("use_e2b", [False, True])
@pytest.mark.parametrize("named", SDK_SUBAGENT_TOOLS)
def test_operator_setting_reaches_a_whitelisted_turn(
    use_e2b: bool, named: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(_EXTRA_BUILTINS, ["Agent"])
    permissions = CopilotPermissions(tools=[named, "web_fetch"], tools_exclude=False)

    options = _build_sdk_tool_options(permissions, use_e2b=use_e2b, disabled_groups=[])

    assert set(options["tools"]) == set(SDK_SUBAGENT_TOOLS)
    assert not set(SDK_SUBAGENT_TOOLS) & set(options["disallowed_tools"])


@pytest.mark.parametrize("use_e2b", [False, True])
def test_whitelist_narrows_the_declared_tools(use_e2b: bool):
    permissions = CopilotPermissions(tools=["web_fetch"], tools_exclude=False)

    options = _build_sdk_tool_options(permissions, use_e2b=use_e2b, disabled_groups=[])

    assert options["tools"] == []
    assert f"{MCP_TOOL_PREFIX}web_fetch" in options["allowed_tools"]


def test_denied_tools_leave_only_the_allowed_set():
    baseline = _build_sdk_tool_options(None, use_e2b=False, disabled_groups=[])

    options = _build_sdk_tool_options(
        None,
        use_e2b=False,
        disabled_groups=[],
        denied_tools=frozenset({"web_fetch"}),
    )

    assert f"{MCP_TOOL_PREFIX}web_fetch" in baseline["allowed_tools"]
    assert f"{MCP_TOOL_PREFIX}web_fetch" not in options["allowed_tools"]
    assert options["tools"] == baseline["tools"]
    assert options["disallowed_tools"] == baseline["disallowed_tools"]


@pytest.mark.parametrize(
    "permissions",
    [
        pytest.param(None, id="unfiltered"),
        pytest.param(
            CopilotPermissions(tools=["bash_exec"], tools_exclude=True),
            id="blacklist",
        ),
    ],
)
def test_disabled_groups_leave_the_allowed_set(
    permissions: CopilotPermissions | None,
):
    baseline = _build_sdk_tool_options(permissions, use_e2b=False, disabled_groups=[])

    options = _build_sdk_tool_options(
        permissions, use_e2b=False, disabled_groups=["graphiti"]
    )

    assert set(options["allowed_tools"]) < set(baseline["allowed_tools"])
