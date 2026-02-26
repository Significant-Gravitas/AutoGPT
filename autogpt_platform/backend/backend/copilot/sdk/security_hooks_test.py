"""Tests for SDK security hooks — workspace paths, tool access, and deny messages.

These are pure unit tests with no external dependencies (no SDK, no DB, no server).
They validate that the security hooks correctly block unauthorized paths,
tool access, and dangerous input patterns.
"""

import os

import pytest

from .security_hooks import _validate_tool_access, _validate_user_isolation
from .service import _is_tool_error_or_denial

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
    path = f"{home}/.claude/projects/-tmp-copilot-abc123/tool-results/12345.txt"
    result = _validate_tool_access("Read", {"file_path": path}, sdk_cwd=SDK_CWD)
    assert result == {}


def test_read_claude_projects_without_tool_results_denied():
    home = os.path.expanduser("~")
    path = f"{home}/.claude/projects/-tmp-copilot-abc123/settings.json"
    result = _validate_tool_access("Read", {"file_path": path}, sdk_cwd=SDK_CWD)
    assert _is_denied(result)


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
    from .security_hooks import create_security_hooks

    hooks = create_security_hooks(user_id="u1", sdk_cwd=SDK_CWD, max_subtasks=2)
    pre = hooks["PreToolUse"][0].hooks[0]
    post = hooks["PostToolUse"][0].hooks[0]
    post_failure = hooks["PostToolUseFailure"][0].hooks[0]
    return pre, post, post_failure


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_task_background_blocked(_hooks):
    """Task with run_in_background=true must be denied."""
    pre, _, _ = _hooks
    result = await pre(
        {"tool_name": "Task", "tool_input": {"run_in_background": True, "prompt": "x"}},
        tool_use_id=None,
        context={},
    )
    assert _is_denied(result)
    assert "foreground" in _reason(result).lower()


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_task_foreground_allowed(_hooks):
    """Task without run_in_background should be allowed."""
    pre, _, _ = _hooks
    result = await pre(
        {"tool_name": "Task", "tool_input": {"prompt": "do stuff"}},
        tool_use_id="tu-1",
        context={},
    )
    assert not _is_denied(result)


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_task_limit_enforced(_hooks):
    """Task spawns beyond max_subtasks should be denied."""
    pre, _, _ = _hooks
    # First two should pass
    for i in range(2):
        result = await pre(
            {"tool_name": "Task", "tool_input": {"prompt": "ok"}},
            tool_use_id=f"tu-limit-{i}",
            context={},
        )
        assert not _is_denied(result)

    # Third should be denied (limit=2)
    result = await pre(
        {"tool_name": "Task", "tool_input": {"prompt": "over limit"}},
        tool_use_id="tu-limit-2",
        context={},
    )
    assert _is_denied(result)
    assert "Maximum" in _reason(result)


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_task_slot_released_on_completion(_hooks):
    """Completing a Task should free a slot so new Tasks can be spawned."""
    pre, post, _ = _hooks
    # Fill both slots
    for i in range(2):
        result = await pre(
            {"tool_name": "Task", "tool_input": {"prompt": "ok"}},
            tool_use_id=f"tu-comp-{i}",
            context={},
        )
        assert not _is_denied(result)

    # Third should be denied — at capacity
    result = await pre(
        {"tool_name": "Task", "tool_input": {"prompt": "over"}},
        tool_use_id="tu-comp-2",
        context={},
    )
    assert _is_denied(result)

    # Complete first task — frees a slot
    await post(
        {"tool_name": "Task", "tool_input": {}},
        tool_use_id="tu-comp-0",
        context={},
    )

    # Now a new Task should be allowed
    result = await pre(
        {"tool_name": "Task", "tool_input": {"prompt": "after release"}},
        tool_use_id="tu-comp-3",
        context={},
    )
    assert not _is_denied(result)


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_task_slot_released_on_failure(_hooks):
    """A failed Task should also free its concurrency slot."""
    pre, _, post_failure = _hooks
    # Fill both slots
    for i in range(2):
        result = await pre(
            {"tool_name": "Task", "tool_input": {"prompt": "ok"}},
            tool_use_id=f"tu-fail-{i}",
            context={},
        )
        assert not _is_denied(result)

    # At capacity
    result = await pre(
        {"tool_name": "Task", "tool_input": {"prompt": "over"}},
        tool_use_id="tu-fail-2",
        context={},
    )
    assert _is_denied(result)

    # Fail first task — should free a slot
    await post_failure(
        {"tool_name": "Task", "tool_input": {}, "error": "something broke"},
        tool_use_id="tu-fail-0",
        context={},
    )

    # New Task should be allowed
    result = await pre(
        {"tool_name": "Task", "tool_input": {"prompt": "after failure"}},
        tool_use_id="tu-fail-3",
        context={},
    )
    assert not _is_denied(result)


# -- _is_tool_error_or_denial ------------------------------------------------


class TestIsToolErrorOrDenial:
    def test_none_content(self):
        assert _is_tool_error_or_denial(None) is False

    def test_empty_content(self):
        assert _is_tool_error_or_denial("") is False

    def test_benign_output(self):
        assert _is_tool_error_or_denial("All good, no issues.") is False

    def test_security_marker(self):
        assert _is_tool_error_or_denial("[SECURITY] Tool access blocked") is True

    def test_cannot_be_bypassed(self):
        assert _is_tool_error_or_denial("This restriction cannot be bypassed.") is True

    def test_not_allowed(self):
        assert _is_tool_error_or_denial("Operation not allowed in sandbox") is True

    def test_background_task_denial(self):
        assert (
            _is_tool_error_or_denial(
                "Background task execution is not supported. "
                "Run tasks in the foreground instead."
            )
            is True
        )

    def test_subtask_limit_denial(self):
        assert (
            _is_tool_error_or_denial(
                "Maximum 2 concurrent sub-tasks. "
                "Wait for running sub-tasks to finish, "
                "or continue in the main conversation."
            )
            is True
        )

    def test_denied_marker(self):
        assert (
            _is_tool_error_or_denial("Access denied: insufficient privileges") is True
        )

    def test_blocked_marker(self):
        assert _is_tool_error_or_denial("Request blocked by security policy") is True

    def test_failed_marker(self):
        assert _is_tool_error_or_denial("Failed to execute tool: timeout") is True

    def test_mcp_iserror(self):
        assert _is_tool_error_or_denial('{"isError": true, "content": []}') is True

    def test_benign_error_in_value(self):
        """Content like '0 errors found' should not trigger — 'error' was removed."""
        assert _is_tool_error_or_denial("0 errors found") is False

    def test_benign_permission_field(self):
        """Schema descriptions mentioning 'permission' should not trigger."""
        assert (
            _is_tool_error_or_denial(
                '{"fields": [{"name": "permission_level", "type": "int"}]}'
            )
            is False
        )

    def test_benign_not_found_in_listing(self):
        """File listing containing 'not found' in filenames should not trigger."""
        assert _is_tool_error_or_denial("readme.md\nfile-not-found-handler.py") is False
