"""Tests for SDK security hooks — workspace paths, tool access, and deny messages.

These are pure unit tests with no external dependencies (no SDK, no DB, no server).
They validate that the security hooks correctly block unauthorized paths,
tool access, and dangerous input patterns.
"""

import logging
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


def test_read_within_workspace_blocked():
    """Read of workspace files is denied — workspace reads must use the read_file MCP tool."""
    result = _validate_tool_access(
        "Read", {"file_path": f"{SDK_CWD}/file.txt"}, sdk_cwd=SDK_CWD
    )
    assert _is_denied(result)


def test_read_outside_workspace_blocked():
    """Read outside the workspace is denied."""
    result = _validate_tool_access(
        "Read", {"file_path": "/etc/passwd"}, sdk_cwd=SDK_CWD
    )
    assert _is_denied(result)


def test_write_builtin_blocked():
    """SDK built-in Write is blocked — all writes go through MCP Write tool."""
    result = _validate_tool_access(
        "Write", {"file_path": f"{SDK_CWD}/output.json"}, sdk_cwd=SDK_CWD
    )
    assert _is_denied(result)


def test_edit_builtin_blocked():
    """SDK built-in Edit is blocked — all edits go through MCP Edit tool."""
    result = _validate_tool_access(
        "Edit", {"file_path": f"{SDK_CWD}/src/main.py"}, sdk_cwd=SDK_CWD
    )
    assert _is_denied(result)


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


def test_read_tool_outputs_allowed():
    """tool-outputs/ paths should be allowed, same as tool-results/."""
    home = os.path.expanduser("~")
    path = f"{home}/.claude/projects/-tmp-copilot-abc123/a1b2c3d4-e5f6-7890-abcd-ef1234567890/tool-outputs/12345.txt"
    token = _current_project_dir.set("-tmp-copilot-abc123")
    try:
        result = _validate_tool_access("Read", {"file_path": path}, sdk_cwd=SDK_CWD)
        assert result == {}
    finally:
        _current_project_dir.reset(token)


def test_read_claude_projects_settings_json_denied():
    """SDK-internal artifacts like settings.json are NOT accessible — only tool-results/tool-outputs is."""
    home = os.path.expanduser("~")
    path = f"{home}/.claude/projects/-tmp-copilot-abc123/settings.json"
    token = _current_project_dir.set("-tmp-copilot-abc123")
    try:
        result = _validate_tool_access("Read", {"file_path": path}, sdk_cwd=SDK_CWD)
        assert _is_denied(result)
    finally:
        _current_project_dir.reset(token)


def test_read_cross_session_tool_results_denied():
    """Cross-session reads are blocked: session A cannot read session B's tool-results."""
    home = os.path.expanduser("~")
    # session A: encoded cwd is "-tmp-copilot-abc123"
    # session B: encoded cwd is "-tmp-copilot-other999"
    other_session_path = (
        f"{home}/.claude/projects/-tmp-copilot-other999/"
        "a1b2c3d4-e5f6-7890-abcd-ef1234567890/tool-results/secret.txt"
    )
    # Current session is abc123, not other999 — so the path should be denied.
    token = _current_project_dir.set("-tmp-copilot-abc123")
    try:
        result = _validate_tool_access(
            "Read", {"file_path": other_session_path}, sdk_cwd=SDK_CWD
        )
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
async def test_task_background_allowed(_hooks):
    """Task with run_in_background=true is allowed (SDK handles async lifecycle)."""
    pre, _, _ = _hooks
    result = await pre(
        {"tool_name": "Task", "tool_input": {"run_in_background": True, "prompt": "x"}},
        tool_use_id="tu-bg-1",
        context={},
    )
    assert not _is_denied(result)


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


# ---------------------------------------------------------------------------
# "Agent" tool name (SDK v2.x+ renamed "Task" → "Agent")
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_agent_background_allowed(_hooks):
    """Agent with run_in_background=true is allowed (SDK handles async lifecycle)."""
    pre, _, _ = _hooks
    result = await pre(
        {
            "tool_name": "Agent",
            "tool_input": {"run_in_background": True, "prompt": "x"},
        },
        tool_use_id="tu-agent-bg-1",
        context={},
    )
    assert not _is_denied(result)


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_agent_foreground_allowed(_hooks):
    """Agent without run_in_background should be allowed."""
    pre, _, _ = _hooks
    result = await pre(
        {"tool_name": "Agent", "tool_input": {"prompt": "do stuff"}},
        tool_use_id="tu-agent-1",
        context={},
    )
    assert not _is_denied(result)


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_background_agent_counts_against_limit(_hooks):
    """Background agents still consume concurrency slots."""
    pre, _, _ = _hooks
    # Two background agents fill the limit
    for i in range(2):
        result = await pre(
            {
                "tool_name": "Agent",
                "tool_input": {"run_in_background": True, "prompt": "bg"},
            },
            tool_use_id=f"tu-bglimit-{i}",
            context={},
        )
        assert not _is_denied(result)
    # Third (background or foreground) should be denied
    result = await pre(
        {
            "tool_name": "Agent",
            "tool_input": {"run_in_background": True, "prompt": "over"},
        },
        tool_use_id="tu-bglimit-2",
        context={},
    )
    assert _is_denied(result)
    assert "Maximum" in _reason(result)


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_agent_limit_enforced(_hooks):
    """Agent spawns beyond max_subtasks should be denied."""
    pre, _, _ = _hooks
    # First two should pass
    for i in range(2):
        result = await pre(
            {"tool_name": "Agent", "tool_input": {"prompt": "ok"}},
            tool_use_id=f"tu-agent-limit-{i}",
            context={},
        )
        assert not _is_denied(result)

    # Third should be denied (limit=2)
    result = await pre(
        {"tool_name": "Agent", "tool_input": {"prompt": "over limit"}},
        tool_use_id="tu-agent-limit-2",
        context={},
    )
    assert _is_denied(result)
    assert "Maximum" in _reason(result)


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_agent_slot_released_on_completion(_hooks):
    """Completing an Agent should free a slot so new Agents can be spawned."""
    pre, post, _ = _hooks
    # Fill both slots
    for i in range(2):
        result = await pre(
            {"tool_name": "Agent", "tool_input": {"prompt": "ok"}},
            tool_use_id=f"tu-agent-comp-{i}",
            context={},
        )
        assert not _is_denied(result)

    # Third should be denied — at capacity
    result = await pre(
        {"tool_name": "Agent", "tool_input": {"prompt": "over"}},
        tool_use_id="tu-agent-comp-2",
        context={},
    )
    assert _is_denied(result)

    # Complete first agent — frees a slot
    await post(
        {"tool_name": "Agent", "tool_input": {}},
        tool_use_id="tu-agent-comp-0",
        context={},
    )

    # Now a new Agent should be allowed
    result = await pre(
        {"tool_name": "Agent", "tool_input": {"prompt": "after release"}},
        tool_use_id="tu-agent-comp-3",
        context={},
    )
    assert not _is_denied(result)


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_agent_slot_released_on_failure(_hooks):
    """A failed Agent should also free its concurrency slot."""
    pre, _, post_failure = _hooks
    # Fill both slots
    for i in range(2):
        result = await pre(
            {"tool_name": "Agent", "tool_input": {"prompt": "ok"}},
            tool_use_id=f"tu-agent-fail-{i}",
            context={},
        )
        assert not _is_denied(result)

    # At capacity
    result = await pre(
        {"tool_name": "Agent", "tool_input": {"prompt": "over"}},
        tool_use_id="tu-agent-fail-2",
        context={},
    )
    assert _is_denied(result)

    # Fail first agent — should free a slot
    await post_failure(
        {"tool_name": "Agent", "tool_input": {}, "error": "something broke"},
        tool_use_id="tu-agent-fail-0",
        context={},
    )

    # New Agent should be allowed
    result = await pre(
        {"tool_name": "Agent", "tool_input": {"prompt": "after failure"}},
        tool_use_id="tu-agent-fail-3",
        context={},
    )
    assert not _is_denied(result)


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_mixed_task_agent_share_slots(_hooks):
    """Task and Agent share the same concurrency pool."""
    pre, post, _ = _hooks
    # Fill one slot with Task, one with Agent
    result = await pre(
        {"tool_name": "Task", "tool_input": {"prompt": "ok"}},
        tool_use_id="tu-mix-task",
        context={},
    )
    assert not _is_denied(result)

    result = await pre(
        {"tool_name": "Agent", "tool_input": {"prompt": "ok"}},
        tool_use_id="tu-mix-agent",
        context={},
    )
    assert not _is_denied(result)

    # Third (either name) should be denied
    result = await pre(
        {"tool_name": "Agent", "tool_input": {"prompt": "over"}},
        tool_use_id="tu-mix-over",
        context={},
    )
    assert _is_denied(result)

    # Release the Task slot
    await post(
        {"tool_name": "Task", "tool_input": {}},
        tool_use_id="tu-mix-task",
        context={},
    )

    # Now an Agent should be allowed
    result = await pre(
        {"tool_name": "Agent", "tool_input": {"prompt": "after task release"}},
        tool_use_id="tu-mix-new",
        context={},
    )
    assert not _is_denied(result)


# ---------------------------------------------------------------------------
# SubagentStart / SubagentStop hooks
# ---------------------------------------------------------------------------


@pytest.fixture()
def _subagent_hooks():
    """Create hooks and return (subagent_start, subagent_stop) handlers."""
    hooks = create_security_hooks(user_id="u1", sdk_cwd=SDK_CWD, max_subtasks=2)
    start = hooks["SubagentStart"][0].hooks[0]
    stop = hooks["SubagentStop"][0].hooks[0]
    return start, stop


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_subagent_start_hook_returns_empty(_subagent_hooks):
    """SubagentStart hook should return an empty dict (logging only)."""
    start, _ = _subagent_hooks
    result = await start(
        {"agent_id": "sa-123", "agent_type": "research"},
        tool_use_id=None,
        context={},
    )
    assert result == {}


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_subagent_stop_hook_returns_empty(_subagent_hooks):
    """SubagentStop hook should return an empty dict (logging only)."""
    _, stop = _subagent_hooks
    result = await stop(
        {
            "agent_id": "sa-123",
            "agent_type": "research",
            "agent_transcript_path": "/tmp/transcript.txt",
        },
        tool_use_id=None,
        context={},
    )
    assert result == {}


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_subagent_hooks_sanitize_inputs(_subagent_hooks, caplog):
    """SubagentStart/Stop should sanitize control chars from inputs."""
    start, stop = _subagent_hooks
    # Inject control characters (C0, DEL, C1, BiDi overrides, zero-width)
    # — hook should not raise AND logs must be clean
    with caplog.at_level(logging.DEBUG, logger="backend.copilot.sdk.security_hooks"):
        result = await start(
            {
                "agent_id": "sa\n-injected\r\x00\x7f",
                "agent_type": "safe\x80_type\x9f\ttab",
            },
            tool_use_id=None,
            context={},
        )
    assert result == {}
    # Control chars must be stripped from the logged values
    for record in caplog.records:
        assert "\x00" not in record.message
        assert "\r" not in record.message
        assert "\n" not in record.message
        assert "\x7f" not in record.message
        assert "\x80" not in record.message
        assert "\x9f" not in record.message
    assert "safe_type" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="backend.copilot.sdk.security_hooks"):
        result = await stop(
            {
                "agent_id": "sa\n-injected\x7f",
                "agent_type": "type\r\x80\x9f",
                "agent_transcript_path": "/tmp/\x00malicious\npath\u202a\u200b",
            },
            tool_use_id=None,
            context={},
        )
    assert result == {}
    for record in caplog.records:
        assert "\x00" not in record.message
        assert "\r" not in record.message
        assert "\n" not in record.message
        assert "\x7f" not in record.message
        assert "\u202a" not in record.message
        assert "\u200b" not in record.message
    assert "/tmp/maliciouspath" in caplog.text
