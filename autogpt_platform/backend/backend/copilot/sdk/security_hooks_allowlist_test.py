import pytest

from .security_hooks import create_security_hooks
from .tool_adapter import get_copilot_tool_names


@pytest.mark.parametrize("use_e2b", [False, True])
@pytest.mark.parametrize("tool_name", ["Glob", "Grep"])
@pytest.mark.parametrize("tool_input", [{}, {"path": "/tmp/copilot-test/src"}])
@pytest.mark.asyncio
async def test_workspace_tool_removed_from_allowed_set_is_denied(
    tool_name, tool_input, use_e2b
):
    allowed = [
        name for name in get_copilot_tool_names(use_e2b=use_e2b) if name != tool_name
    ]
    pre = create_security_hooks(
        "u1", sdk_cwd="/tmp/copilot-test", allowed_tools=allowed
    )["PreToolUse"][0].hooks[0]
    result = await pre(
        {"tool_name": tool_name, "tool_input": tool_input}, tool_use_id=None, context={}
    )
    output = result.get("hookSpecificOutput", {})
    assert output.get("permissionDecision") == "deny"
    assert "not available in this session" in output.get("permissionDecisionReason", "")


@pytest.mark.asyncio
async def test_blocked_read_stays_denied_even_when_allowed():
    pre = create_security_hooks("u1", allowed_tools={"Read"})["PreToolUse"][0].hooks[0]
    result = await pre(
        {"tool_name": "Read", "tool_input": {}}, tool_use_id=None, context={}
    )
    output = result.get("hookSpecificOutput", {})
    assert output.get("permissionDecision") == "deny"
    assert "blocked for security" in output.get("permissionDecisionReason", "")


@pytest.mark.parametrize("tool_name", ["Glob", "Grep"])
@pytest.mark.asyncio
async def test_allowed_workspace_tool_cannot_escape_workspace(tool_name, tmp_path):
    pre = create_security_hooks("u1", sdk_cwd=str(tmp_path), allowed_tools={tool_name})[
        "PreToolUse"
    ][0].hooks[0]
    result = await pre(
        {"tool_name": tool_name, "tool_input": {"path": str(tmp_path.parent)}},
        tool_use_id=None,
        context={},
    )
    output = result.get("hookSpecificOutput", {})
    assert output.get("permissionDecision") == "deny"
    assert "within the workspace" in output.get("permissionDecisionReason", "")
