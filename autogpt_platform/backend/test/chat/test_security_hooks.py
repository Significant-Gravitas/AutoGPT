"""Tests for SDK security hooks — workspace paths, tool access, and deny messages.

These are pure unit tests with no external dependencies (no SDK, no DB, no server).
They validate that the security hooks correctly block unauthorized paths,
tool access, and dangerous input patterns.

Note: Bash command validation was removed — the SDK built-in Bash tool is not in
allowed_tools, and the bash_exec MCP tool has kernel-level network isolation
(unshare --net) making command-level parsing unnecessary.
"""

from backend.api.features.chat.sdk.security_hooks import (
    _validate_tool_access,
    _validate_workspace_path,
)

SDK_CWD = "/tmp/copilot-test-session"


def _is_denied(result: dict) -> bool:
    hook = result.get("hookSpecificOutput", {})
    return hook.get("permissionDecision") == "deny"


def _reason(result: dict) -> str:
    return result.get("hookSpecificOutput", {}).get("permissionDecisionReason", "")


# ============================================================
# Workspace path validation (Read, Write, Edit, etc.)
# ============================================================


class TestWorkspacePathValidation:
    def test_path_in_workspace(self):
        result = _validate_workspace_path(
            "Read", {"file_path": f"{SDK_CWD}/file.txt"}, SDK_CWD
        )
        assert not _is_denied(result)

    def test_path_outside_workspace(self):
        result = _validate_workspace_path("Read", {"file_path": "/etc/passwd"}, SDK_CWD)
        assert _is_denied(result)

    def test_tool_results_allowed(self):
        result = _validate_workspace_path(
            "Read",
            {"file_path": "~/.claude/projects/abc/tool-results/out.txt"},
            SDK_CWD,
        )
        assert not _is_denied(result)

    def test_claude_settings_blocked(self):
        result = _validate_workspace_path(
            "Read", {"file_path": "~/.claude/settings.json"}, SDK_CWD
        )
        assert _is_denied(result)

    def test_claude_projects_without_tool_results(self):
        result = _validate_workspace_path(
            "Read", {"file_path": "~/.claude/projects/abc/credentials.json"}, SDK_CWD
        )
        assert _is_denied(result)

    def test_no_path_allowed(self):
        """Glob/Grep without path defaults to cwd — should be allowed."""
        result = _validate_workspace_path("Grep", {"pattern": "foo"}, SDK_CWD)
        assert not _is_denied(result)

    def test_path_traversal_with_dotdot(self):
        result = _validate_workspace_path(
            "Read", {"file_path": f"{SDK_CWD}/../../../etc/passwd"}, SDK_CWD
        )
        assert _is_denied(result)


# ============================================================
# Tool access validation
# ============================================================


class TestToolAccessValidation:
    def test_blocked_tools(self):
        for tool in ("bash", "shell", "exec", "terminal", "command"):
            result = _validate_tool_access(tool, {})
            assert _is_denied(result), f"Tool '{tool}' should be blocked"

    def test_bash_builtin_blocked(self):
        """SDK built-in Bash (capital) is blocked as defence-in-depth."""
        result = _validate_tool_access("Bash", {"command": "echo hello"}, SDK_CWD)
        assert _is_denied(result)
        assert "Bash" in _reason(result)

    def test_workspace_tools_delegate(self):
        result = _validate_tool_access(
            "Read", {"file_path": f"{SDK_CWD}/file.txt"}, SDK_CWD
        )
        assert not _is_denied(result)

    def test_dangerous_pattern_blocked(self):
        result = _validate_tool_access("SomeUnknownTool", {"data": "sudo rm -rf /"})
        assert _is_denied(result)

    def test_safe_unknown_tool_allowed(self):
        result = _validate_tool_access("SomeSafeTool", {"data": "hello world"})
        assert not _is_denied(result)


# ============================================================
# Deny message quality (ntindle feedback)
# ============================================================


class TestDenyMessageClarity:
    """Deny messages must include [SECURITY] and 'cannot be bypassed'
    so the model knows the restriction is enforced, not a suggestion."""

    def test_blocked_tool_message(self):
        reason = _reason(_validate_tool_access("bash", {}))
        assert "[SECURITY]" in reason
        assert "cannot be bypassed" in reason

    def test_bash_builtin_blocked_message(self):
        reason = _reason(_validate_tool_access("Bash", {"command": "echo hello"}))
        assert "[SECURITY]" in reason
        assert "cannot be bypassed" in reason

    def test_workspace_path_message(self):
        reason = _reason(
            _validate_workspace_path("Read", {"file_path": "/etc/passwd"}, SDK_CWD)
        )
        assert "[SECURITY]" in reason
        assert "cannot be bypassed" in reason
