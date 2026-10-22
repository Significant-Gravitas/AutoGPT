"""Tests for SDK security hooks — workspace paths, tool access, and deny messages.

These are pure unit tests with no external dependencies (no SDK, no DB, no server).
They validate that the security hooks correctly block unauthorized paths
and tool access.
"""

import logging
import os
from unittest.mock import AsyncMock

import pytest

from .security_hooks import (
    _validate_tool_access,
    _validate_user_isolation,
    create_security_hooks,
)
from .tool_adapter import (
    get_copilot_tool_names,
    get_sdk_builtin_tools,
    get_sdk_disallowed_tools,
)

SDK_CWD = "/tmp/copilot-abc123"
ALLOWED = get_copilot_tool_names()


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


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.parametrize(
    "tool_name", ["ListAgents2", "mcp__copilot__not_registered", "mcp__other__tool"]
)
@pytest.mark.asyncio
async def test_tool_outside_allowed_set_denied(tool_name):
    pre = create_security_hooks("u1", sdk_cwd=SDK_CWD, allowed_tools=ALLOWED)[
        "PreToolUse"
    ][0].hooks[0]
    result = await pre(
        {"tool_name": tool_name, "tool_input": {}}, tool_use_id=None, context={}
    )
    assert "not available in this session" in _reason(result)


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.parametrize("use_e2b", [False, True])
@pytest.mark.asyncio
async def test_every_allowed_tool_passes_the_hook(use_e2b):
    allowed = get_copilot_tool_names(use_e2b=use_e2b)
    pre = create_security_hooks("u1", sdk_cwd=SDK_CWD, allowed_tools=allowed)[
        "PreToolUse"
    ][0].hooks[0]
    usable = set(allowed) - set(get_sdk_disallowed_tools(use_e2b=use_e2b))
    denied = [
        name
        for name in sorted(usable)
        if _is_denied(
            await pre(
                {"tool_name": name, "tool_input": {}}, tool_use_id=None, context={}
            )
        )
    ]
    assert denied == []


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.parametrize("use_e2b", [False, True])
@pytest.mark.asyncio
async def test_every_unusable_builtin_is_denied(use_e2b):
    allowed = get_copilot_tool_names(use_e2b=use_e2b)
    pre = create_security_hooks("u1", sdk_cwd=SDK_CWD, allowed_tools=allowed)[
        "PreToolUse"
    ][0].hooks[0]
    usable = set(allowed) - set(get_sdk_disallowed_tools(use_e2b=use_e2b))
    passed = [
        name
        for name in sorted(set(get_sdk_builtin_tools()) - usable)
        if not _is_denied(
            await pre(
                {"tool_name": name, "tool_input": {}}, tool_use_id=None, context={}
            )
        )
    ]
    assert passed == []


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
    result = _validate_tool_access(
        "Glob", {"path": f"{SDK_CWD}/src"}, sdk_cwd=SDK_CWD, allowed_tools=ALLOWED
    )
    assert result == {}


def test_grep_within_workspace_allowed():
    result = _validate_tool_access(
        "Grep", {"path": f"{SDK_CWD}/src"}, sdk_cwd=SDK_CWD, allowed_tools=ALLOWED
    )
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
    result = _validate_tool_access("Glob", {}, sdk_cwd=SDK_CWD, allowed_tools=ALLOWED)
    assert result == {}


def test_read_no_cwd_denies_absolute():
    """If no sdk_cwd is set, absolute paths are denied."""
    result = _validate_tool_access("Read", {"file_path": "/tmp/anything"})
    assert _is_denied(result)


# -- Tool-results directory --------------------------------------------------


@pytest.mark.parametrize(
    "relative",
    [
        "-tmp-copilot-abc123/a1b2c3d4-e5f6-7890-abcd-ef1234567890/tool-results/o",
        "-tmp-copilot-abc123/a1b2c3d4-e5f6-7890-abcd-ef1234567890/tool-outputs/o",
        "-tmp-copilot-abc123/settings.json",
        "-tmp-copilot-other999/a1b2c3d4-e5f6-7890-abcd-ef1234567890/tool-results/o",
    ],
)
def test_native_read_is_blocked_regardless_of_sdk_path(relative):
    path = os.path.join(os.path.expanduser("~"), ".claude", "projects", relative)
    result = _validate_tool_access(
        "Read", {"file_path": path}, sdk_cwd=SDK_CWD, allowed_tools={"Read"}
    )
    assert _is_denied(result)
    assert "blocked for security" in _reason(result)


# -- Built-in Bash is blocked (use bash_exec MCP tool instead) ---------------


def test_bash_builtin_always_blocked():
    """SDK built-in Bash is blocked — bash_exec MCP tool with bubblewrap is used instead."""
    result = _validate_tool_access("Bash", {"command": "echo hello"}, sdk_cwd=SDK_CWD)
    assert _is_denied(result)


# -- Prose that names a command ----------------------------------------------


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.parametrize("use_e2b", [False, True])
@pytest.mark.parametrize(
    "tool_name, tool_input",
    [
        (
            "Agent",
            {
                "description": "Audit the deploy script",
                "prompt": "Check whether deploy.sh calls sudo or subprocess.run",
            },
        ),
        (
            "Task",
            {
                "description": "Find risky calls",
                "prompt": "List every os.system and rm -rf in the repo",
            },
        ),
        (
            "TodoWrite",
            {
                "todos": [
                    {
                        "content": "Replace the sudo subprocess call",
                        "status": "pending",
                        "activeForm": "Replacing the sudo subprocess call",
                    }
                ]
            },
        ),
        (
            "mcp__copilot__bash_exec",
            {"command": "sudo pip install requests && python -c 'import subprocess'"},
        ),
    ],
)
@pytest.mark.asyncio
async def test_command_text_is_allowed_for_permitted_tools(
    tool_name, tool_input, use_e2b
):
    """Command words are not a substitute for tool authorization."""
    pre = create_security_hooks(
        "u1", sdk_cwd=SDK_CWD, allowed_tools=get_copilot_tool_names(use_e2b=use_e2b)
    )["PreToolUse"][0].hooks[0]
    result = await pre(
        {"tool_name": tool_name, "tool_input": tool_input},
        tool_use_id=None,
        context={},
    )
    assert not _is_denied(result)


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
    hooks = create_security_hooks(
        user_id="u1", sdk_cwd=SDK_CWD, max_subtasks=2, allowed_tools=ALLOWED
    )
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
    hooks = create_security_hooks(
        user_id="u1", sdk_cwd=SDK_CWD, max_subtasks=2, allowed_tools=ALLOWED
    )
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


# -- PostToolUse: mid-turn pending-message drain ------------------------------


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_post_tool_use_injects_followup_additional_context(
    monkeypatch,
):
    """Queued messages drain into ``additionalContext`` for any tool."""
    from unittest.mock import MagicMock

    from backend.copilot import context as ctx_mod
    from backend.copilot import pending_messages as pm_module

    session = MagicMock()
    session.session_id = "sess-post-inject"
    ctx_mod.set_execution_context(
        user_id="u1",
        session=session,
        sandbox=None,
        sdk_cwd=SDK_CWD,
    )

    drain = AsyncMock(
        return_value=[pm_module.PendingMessage(content="please also do X")]
    )

    async def fake_stash(_session_id, _messages):
        return None

    monkeypatch.setattr(
        "backend.copilot.pending_messages.drain_pending_messages", drain
    )
    monkeypatch.setattr(
        "backend.copilot.pending_messages.stash_pending_for_persist", fake_stash
    )

    hooks = create_security_hooks(
        user_id="u1", sdk_cwd=SDK_CWD, max_subtasks=2, allowed_tools=ALLOWED
    )
    post = hooks["PostToolUse"][0].hooks[0]

    result = await post(
        {
            "tool_name": "WebSearch",  # built-in — the path the old wrapper missed
            "tool_response": "search results here",
        },
        tool_use_id="tu-web-1",
        context={},
    )

    drain.assert_awaited_once_with("sess-post-inject", text_only=True)
    injected = result.get("hookSpecificOutput", {})
    assert injected.get("hookEventName") == "PostToolUse"
    assert "<user_follow_up>" in injected.get("additionalContext", "")
    assert "please also do X" in injected.get("additionalContext", "")


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_post_tool_use_no_pending_returns_empty(monkeypatch):
    from unittest.mock import MagicMock

    from backend.copilot import context as ctx_mod

    session = MagicMock()
    session.session_id = "sess-post-empty"
    ctx_mod.set_execution_context(
        user_id="u1", session=session, sandbox=None, sdk_cwd=SDK_CWD
    )

    drain = AsyncMock(return_value=[])

    monkeypatch.setattr(
        "backend.copilot.pending_messages.drain_pending_messages", drain
    )

    hooks = create_security_hooks(
        user_id="u1", sdk_cwd=SDK_CWD, max_subtasks=2, allowed_tools=ALLOWED
    )
    post = hooks["PostToolUse"][0].hooks[0]

    result = await post(
        {"tool_name": "mcp__copilot__run_block", "tool_response": "ok"},
        tool_use_id="tu-mcp-1",
        context={},
    )

    # No additionalContext means Claude gets the tool_result verbatim.
    drain.assert_awaited_once_with("sess-post-empty", text_only=True)
    assert "hookSpecificOutput" not in result


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_post_tool_use_drain_failure_returns_empty(monkeypatch):
    """A Redis blip must not corrupt the hook response."""
    from unittest.mock import MagicMock

    from backend.copilot import context as ctx_mod

    session = MagicMock()
    session.session_id = "sess-post-fail"
    ctx_mod.set_execution_context(
        user_id="u1", session=session, sandbox=None, sdk_cwd=SDK_CWD
    )

    drain = AsyncMock(side_effect=RuntimeError("redis down"))

    monkeypatch.setattr(
        "backend.copilot.pending_messages.drain_pending_messages", drain
    )

    hooks = create_security_hooks(
        user_id="u1", sdk_cwd=SDK_CWD, max_subtasks=2, allowed_tools=ALLOWED
    )
    post = hooks["PostToolUse"][0].hooks[0]

    result = await post(
        {"tool_name": "Read", "tool_response": "file body"},
        tool_use_id="tu-read-1",
        context={},
    )

    drain.assert_awaited_once_with("sess-post-fail", text_only=True)
    assert "hookSpecificOutput" not in result


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_post_tool_use_no_session_skips_drain(monkeypatch):
    from backend.copilot import context as ctx_mod

    ctx_mod.set_execution_context(
        user_id=None,
        session=None,
        sandbox=None,
        sdk_cwd=SDK_CWD,
    )

    drain = AsyncMock(return_value=[])

    monkeypatch.setattr(
        "backend.copilot.pending_messages.drain_pending_messages", drain
    )

    hooks = create_security_hooks(
        user_id=None, sdk_cwd=SDK_CWD, max_subtasks=2, allowed_tools=ALLOWED
    )
    post = hooks["PostToolUse"][0].hooks[0]

    result = await post(
        {"tool_name": "WebSearch", "tool_response": "x"},
        tool_use_id="tu-x",
        context={},
    )

    drain.assert_not_awaited()
    assert "hookSpecificOutput" not in result


# -- PostHog spans for the CLI's built-in tools -------------------------------


@pytest.fixture()
def _spans(monkeypatch):
    """A turn in progress, with PostHog's capture recorded."""
    from unittest.mock import MagicMock

    from backend.copilot import ai_spans
    from backend.copilot import context as ctx_mod
    from backend.copilot.llm_attribution import set_turn_trace_id

    monkeypatch.setattr(
        "backend.copilot.pending_messages.drain_pending_messages",
        AsyncMock(return_value=[]),
    )
    session = MagicMock()
    session.session_id = "sess-spans"
    ctx_mod.set_execution_context(
        user_id="u1", session=session, sandbox=None, sdk_cwd=SDK_CWD
    )
    set_turn_trace_id("turn-1")
    ai_spans.set_turn_span("turn-1")
    capture = MagicMock()
    monkeypatch.setattr(ai_spans.posthog_client, "capture", capture)
    yield capture
    set_turn_trace_id(None)
    ai_spans.set_turn_span(None)


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_a_builtin_tool_call_is_a_posthog_span(_hooks, _spans):
    pre, post, _ = _hooks
    task = {"prompt": "the user's private request"}

    await pre({"tool_name": "Task", "tool_input": task}, "tu-1", {})
    await post(
        {"tool_name": "Task", "tool_input": task, "tool_response": "results"},
        tool_use_id="tu-1",
        context={},
    )

    _spans.assert_called_once()
    distinct_id, event, properties = _spans.call_args.args
    assert (distinct_id, event) == ("u1", "$ai_span")
    assert properties["$ai_span_name"] == "Task"
    assert properties["$ai_trace_id"] == "turn-1"
    assert properties["$ai_session_id"] == "sess-spans"
    assert properties["$ai_is_error"] is False
    assert "private" not in repr(properties)
    assert "results" not in repr(properties)


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_a_failed_builtin_tool_is_an_error_span(_hooks, _spans):
    pre, _, post_failure = _hooks

    await pre({"tool_name": "Task", "tool_input": {"prompt": "x"}}, "tu-1", {})
    await post_failure(
        {"tool_name": "Task", "error": "the page said something private"},
        tool_use_id="tu-1",
        context={},
    )

    properties = _spans.call_args.args[2]
    assert properties["$ai_is_error"] is True
    assert "private" not in repr(properties)


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_a_refused_builtin_tool_sends_no_span(_hooks, _spans):
    pre, _, post_failure = _hooks

    result = await pre(
        {"tool_name": "Read", "tool_input": {"file_path": "/etc/passwd"}}, "tu-1", {}
    )
    await post_failure({"tool_name": "Read", "error": "denied"}, "tu-1", {})

    assert _is_denied(result)
    _spans.assert_not_called()


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_a_platform_tool_gets_its_span_from_basetool_not_the_hooks(
    _hooks, _spans
):
    pre, post, _ = _hooks

    await pre({"tool_name": "mcp__copilot__run_block", "tool_input": {}}, "tu-1", {})
    await post(
        {"tool_name": "mcp__copilot__run_block", "tool_response": "ok"}, "tu-1", {}
    )

    _spans.assert_not_called()


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_a_high_volume_builtin_tool_sends_no_span(_hooks, _spans):
    """The CLI's checklist and file lookups run tens of times a turn."""
    pre, post, _ = _hooks
    todos = {"todos": [{"content": "x", "status": "pending", "activeForm": "x"}]}

    result = await pre({"tool_name": "TodoWrite", "tool_input": todos}, "tu-1", {})
    await post(
        {"tool_name": "TodoWrite", "tool_input": todos, "tool_response": "ok"},
        tool_use_id="tu-1",
        context={},
    )

    assert not _is_denied(result)
    _spans.assert_not_called()


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
@pytest.mark.asyncio
async def test_builtin_tool_spans_turned_off_send_none(_spans):
    hooks = create_security_hooks(
        user_id="u1",
        sdk_cwd=SDK_CWD,
        max_subtasks=2,
        builtin_tool_spans=False,
        allowed_tools=ALLOWED,
    )
    pre = hooks["PreToolUse"][0].hooks[0]
    post = hooks["PostToolUse"][0].hooks[0]
    task = {"prompt": "x"}

    await pre({"tool_name": "Task", "tool_input": task}, "tu-1", {})
    await post(
        {"tool_name": "Task", "tool_input": task, "tool_response": "results"},
        tool_use_id="tu-1",
        context={},
    )

    _spans.assert_not_called()
