"""Tests for SDK security hooks — workspace paths, tool access, and deny messages.

These are pure unit tests with no external dependencies (no SDK, no DB, no server).
They validate that the security hooks correctly block unauthorized paths,
tool access, and dangerous input patterns.
"""

import os

import pytest

from backend.copilot.context import _current_project_dir

from .security_hooks import (
    _validate_tool_access,
    _validate_user_isolation,
    create_security_hooks,
)

SDK_CWD = "/tmp/copilot-abc123"


def _sdk_available() -> bool:
    try:
        import claude_agent_sdk  # noqa: F401

        return True
    except ImportError:
        return False


def _is_denied(result: dict) -> bool:
    hook = result.get("hookSpecificOutput", {})
    return hook.get("permissionDecision") == "deny"


def _reason(result: dict) -> str:
    return result.get("hookSpecificOutput", {}).get("permissionDecisionReason", "")


# -- Blocked tools -----------------------------------------------------------


def test_blocked_tools_denied():
    for tool in ("bash", "shell", "exec", "terminal", "command"):
        result = _validate_tool_access(tool, {})
        assert _is_denied(result), f"{tool} should be blocked"


def test_unknown_tool_allowed():
    result = _validate_tool_access("SomeCustomTool", {})
    assert result == {}


# -- Workspace-scoped tools --------------------------------------------------


def test_read_within_workspace_allowed():
    result = _validate_tool_access(
        "Read", {"file_path": f"{SDK_CWD}/file.txt"}, sdk_cwd=SDK_CWD
    )
    assert result == {}


def test_write_within_workspace_allowed():
    result = _validate_tool_access(
        "Write", {"file_path": f"{SDK_CWD}/output.json"}, sdk_cwd=SDK_CWD
    )
    assert result == {}


def test_edit_within_workspace_allowed():
    result = _validate_tool_access(
        "Edit", {"file_path": f"{SDK_CWD}/src/main.py"}, sdk_cwd=SDK_CWD
    )
    assert result == {}


def test_glob_within_workspace_allowed():
    result = _validate_tool_access("Glob", {"path": f"{SDK_CWD}/src"}, sdk_cwd=SDK_CWD)
    assert result == {}


def test_grep_within_workspace_allowed():
    result = _validate_tool_access("Grep", {"path": f"{SDK_CWD}/src"}, sdk_cwd=SDK_CWD)
    assert result == {}


def test_read_outside_workspace_denied():
    result = _validate_tool_access(
        "Read", {"file_path": "/etc/passwd"}, sdk_cwd=SDK_CWD
    )
    assert _is_denied(result)


def test_write_outside_workspace_denied():
    result = _validate_tool_access(
        "Write", {"file_path": "/home/user/secrets.txt"}, sdk_cwd=SDK_CWD
    )
    assert _is_denied(result)


def test_traversal_attack_denied():
    result = _validate_tool_access(
        "Read",
        {"file_path": f"{SDK_CWD}/../../etc/passwd"},
        sdk_cwd=SDK_CWD,
    )
    assert _is_denied(result)


def test_no_path_allowed():
    """Glob/Grep without a path argument defaults to cwd — should pass."""
    result = _validate_tool_access("Glob", {}, sdk_cwd=SDK_CWD)
    assert result == {}


def test_read_no_cwd_denies_absolute():
    """If no sdk_cwd is set, absolute paths are denied."""
    result = _validate_tool_access("Read", {"file_path": "/tmp/anything"})
    assert _is_denied(result)


# -- Tool-results directory --------------------------------------------------


def test_read_tool_results_allowed():
    home = os.path.expanduser("~")
    path = f"{home}/.claude/projects/-tmp-copilot-abc123/a1b2c3d4-e5f6-7890-abcd-ef1234567890/tool-results/12345.txt"
    # is_allowed_local_path requires the session's encoded cwd to be set
    token = _current_project_dir.set("-tmp-copilot-abc123")
    try:
        result = _validate_tool_access("Read", {"file_path": path}, sdk_cwd=SDK_CWD)
        assert result == {}
    finally:
        _current_project_dir.reset(token)


def test_read_claude_projects_settings_json_denied():
    """SDK-internal artifacts like settings.json are NOT accessible — only tool-results/ is."""
    home = os.path.expanduser("~")
    path = f"{home}/.claude/projects/-tmp-copilot-abc123/settings.json"
    token = _current_project_dir.set("-tmp-copilot-abc123")
    try:
        result = _validate_tool_access("Read", {"file_path": path}, sdk_cwd=SDK_CWD)
        assert _is_denied(result)
    finally:
        _current_project_dir.reset(token)


# -- Built-in Bash is blocked (use bash_exec MCP tool instead) ---------------


def test_bash_builtin_always_blocked():
    """SDK built-in Bash is blocked — bash_exec MCP tool with bubblewrap is used instead."""
    result = _validate_tool_access("Bash", {"command": "echo hello"}, sdk_cwd=SDK_CWD)
    assert _is_denied(result)


# -- Dangerous patterns ------------------------------------------------------


def test_dangerous_pattern_blocked():
    result = _validate_tool_access("SomeTool", {"cmd": "sudo rm -rf /"})
    assert _is_denied(result)


def test_subprocess_pattern_blocked():
    result = _validate_tool_access("SomeTool", {"code": "subprocess.run(...)"})
    assert _is_denied(result)


# -- User isolation ----------------------------------------------------------


def test_workspace_path_traversal_blocked():
    result = _validate_user_isolation(
        "workspace_read", {"path": "../../../etc/shadow"}, user_id="user-1"
    )
    assert _is_denied(result)


def test_workspace_absolute_path_allowed():
    """Workspace 'path' is a cloud storage key — leading '/' is normal."""
    result = _validate_user_isolation(
        "workspace_read", {"path": "/ASEAN/report.md"}, user_id="user-1"
    )
    assert result == {}


def test_workspace_normal_path_allowed():
    result = _validate_user_isolation(
        "workspace_read", {"path": "src/main.py"}, user_id="user-1"
    )
    assert result == {}


def test_non_workspace_tool_passes_isolation():
    result = _validate_user_isolation(
        "find_agent", {"query": "email"}, user_id="user-1"
    )
    assert result == {}


# -- Deny message quality ----------------------------------------------------


def test_blocked_tool_message_clarity():
    """Deny messages must include [SECURITY] and 'cannot be bypassed'."""
    reason = _reason(_validate_tool_access("bash", {}))
    assert "[SECURITY]" in reason
    assert "cannot be bypassed" in reason


def test_bash_builtin_blocked_message_clarity():
    reason = _reason(_validate_tool_access("Bash", {"command": "echo hello"}))
    assert "[SECURITY]" in reason
    assert "cannot be bypassed" in reason


# -- Task sub-agent hooks (require SDK) --------------------------------------


@pytest.fixture()
def _hooks():
    """Create security hooks and return (pre, post, post_failure) handlers."""
    hooks = create_security_hooks(user_id="u1", sdk_cwd=SDK_CWD, max_subtasks=2)
    pre = hooks["PreToolUse"][0].hooks[0]
    post = hooks["PostToolUse"][0].hooks[0]
    post_failure = hooks["PostToolUseFailure"][0].hooks[0]
    return pre, post, post_failure


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_task_always_blocked(_hooks):
    """Task is in BLOCKED_TOOLS — always denied regardless of input."""
    pre, _, _ = _hooks
    # Background Task
    result = await pre(
        {"tool_name": "Task", "tool_input": {"run_in_background": True, "prompt": "x"}},
        tool_use_id=None,
        context={},
    )
    assert _is_denied(result)

    # Foreground Task
    result = await pre(
        {"tool_name": "Task", "tool_input": {"prompt": "do stuff"}},
        tool_use_id="tu-1",
        context={},
    )
    assert _is_denied(result)
    assert "blocked" in _reason(result).lower()


# -- WebSearch cap -----------------------------------------------------------


@pytest.fixture()
def _hooks_search():
    """Create security hooks with low search cap for testing."""
    hooks = create_security_hooks(
        user_id="u1",
        sdk_cwd=SDK_CWD,
        max_subtasks=2,
        max_web_searches=3,
        max_tool_calls=100,
    )
    pre = hooks["PreToolUse"][0].hooks[0]
    return pre


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_web_search_allowed_under_cap(_hooks_search):
    """WebSearch calls under the cap should be allowed."""
    pre = _hooks_search
    for i in range(3):
        result = await pre(
            {"tool_name": "WebSearch", "tool_input": {"query": f"search {i}"}},
            tool_use_id=f"ws-{i}",
            context={},
        )
        assert not _is_denied(result), f"WebSearch {i} should be allowed"


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_web_search_denied_at_cap(_hooks_search):
    """WebSearch calls exceeding the cap should be denied with synthesis message."""
    pre = _hooks_search
    # Use up the cap
    for i in range(3):
        await pre(
            {"tool_name": "WebSearch", "tool_input": {"query": f"search {i}"}},
            tool_use_id=f"ws-cap-{i}",
            context={},
        )
    # Fourth should be denied
    result = await pre(
        {"tool_name": "WebSearch", "tool_input": {"query": "one more"}},
        tool_use_id="ws-cap-3",
        context={},
    )
    assert _is_denied(result)
    reason = _reason(result)
    assert "web search" in reason.lower()
    assert "per turn" in reason.lower()
    assert "synthesize" in reason.lower()


# -- Total tool call cap -----------------------------------------------------


@pytest.fixture()
def _hooks_tool_cap():
    """Create security hooks with low total tool call cap for testing."""
    hooks = create_security_hooks(
        user_id="u1",
        sdk_cwd=SDK_CWD,
        max_subtasks=2,
        max_web_searches=100,
        max_tool_calls=5,
    )
    pre = hooks["PreToolUse"][0].hooks[0]
    return pre


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_total_tool_calls_allowed_under_cap(_hooks_tool_cap):
    """Tool calls under the total cap should be allowed."""
    pre = _hooks_tool_cap
    for i in range(5):
        result = await pre(
            {"tool_name": "SomeTool", "tool_input": {"arg": i}},
            tool_use_id=f"tc-{i}",
            context={},
        )
        assert not _is_denied(result), f"Tool call {i} should be allowed"


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_total_tool_calls_denied_at_cap(_hooks_tool_cap):
    """Tool calls exceeding the total cap should be denied."""
    pre = _hooks_tool_cap
    # Use up the cap
    for i in range(5):
        await pre(
            {"tool_name": "SomeTool", "tool_input": {"arg": i}},
            tool_use_id=f"tc-cap-{i}",
            context={},
        )
    # Sixth should be denied
    result = await pre(
        {"tool_name": "SomeTool", "tool_input": {"arg": "over"}},
        tool_use_id="tc-cap-5",
        context={},
    )
    assert _is_denied(result)
    reason = _reason(result)
    assert "synthesize" in reason.lower()


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_task_blocked_regardless_of_slots(_hooks):
    """Task is always blocked (in BLOCKED_TOOLS), slots are irrelevant."""
    pre, _, _ = _hooks
    result = await pre(
        {"tool_name": "Task", "tool_input": {"prompt": "ok"}},
        tool_use_id="tu-fail-0",
        context={},
    )
    assert _is_denied(result)


# -- WebSearch denial doesn't consume total budget ---------------------------


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_websearch_denials_do_not_consume_total_cap():
    """Denied WebSearch calls should not count toward the total tool call budget."""
    hooks = create_security_hooks(
        user_id="u1",
        sdk_cwd=SDK_CWD,
        max_subtasks=2,
        max_web_searches=3,
        max_tool_calls=5,
    )
    pre = hooks["PreToolUse"][0].hooks[0]

    # Exhaust the WebSearch cap (3 allowed)
    for i in range(3):
        result = await pre(
            {"tool_name": "WebSearch", "tool_input": {"query": f"q{i}"}},
            tool_use_id=f"ws-budget-{i}",
            context={},
        )
        assert not _is_denied(result)

    # Next WebSearch should be denied
    result = await pre(
        {"tool_name": "WebSearch", "tool_input": {"query": "q3"}},
        tool_use_id="ws-budget-3",
        context={},
    )
    assert _is_denied(result)

    # The 3 allowed WebSearches consumed 3 of 5 total budget slots.
    # The denied one should NOT have consumed a slot.
    # So we should have 2 remaining non-WebSearch calls.
    for i in range(2):
        result = await pre(
            {"tool_name": "SomeTool", "tool_input": {}},
            tool_use_id=f"other-{i}",
            context={},
        )
        assert not _is_denied(result), f"SomeTool call {i} should be allowed"

    # 6th total call (3 WebSearch + 2 SomeTool + 1 denied WS that shouldn't count)
    # should be denied because we're at 5 total
    result = await pre(
        {"tool_name": "SomeTool", "tool_input": {}},
        tool_use_id="other-2",
        context={},
    )
    assert _is_denied(result), "Should hit total tool call cap"
