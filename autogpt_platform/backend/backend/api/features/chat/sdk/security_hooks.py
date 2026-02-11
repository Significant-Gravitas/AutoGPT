"""Security hooks for Claude Agent SDK integration.

This module provides security hooks that validate tool calls before execution,
ensuring multi-user isolation and preventing unauthorized operations.
"""

import logging
import os
import re
import shlex
from typing import Any, cast

from backend.api.features.chat.sdk.tool_adapter import MCP_TOOL_PREFIX

logger = logging.getLogger(__name__)

# Tools that are blocked entirely (CLI/system access)
BLOCKED_TOOLS = {
    "bash",
    "shell",
    "exec",
    "terminal",
    "command",
}

# Safe read-only commands allowed in the sandboxed Bash tool.
# These are data-processing / inspection utilities — no writes, no network.
ALLOWED_BASH_COMMANDS = {
    # JSON / structured data
    "jq",
    # Text processing
    "grep",
    "egrep",
    "fgrep",
    "rg",
    "head",
    "tail",
    "cat",
    "wc",
    "sort",
    "uniq",
    "cut",
    "tr",
    "sed",
    "awk",
    "column",
    "fold",
    "fmt",
    "nl",
    "paste",
    "rev",
    # File inspection (read-only)
    "find",
    "ls",
    "file",
    "stat",
    "du",
    "tree",
    "basename",
    "dirname",
    "realpath",
    # Utilities
    "echo",
    "printf",
    "date",
    "true",
    "false",
    "xargs",
    "tee",
    # Comparison / encoding
    "diff",
    "comm",
    "base64",
    "md5sum",
    "sha256sum",
}

# Tools allowed only when their path argument stays within the SDK workspace.
# The SDK uses these to handle oversized tool results (writes to tool-results/
# files, then reads them back) and for workspace file operations.
WORKSPACE_SCOPED_TOOLS = {"Read", "Write", "Edit", "Glob", "Grep"}

# Tools that get sandboxed Bash validation (command allowlist + workspace paths).
SANDBOXED_BASH_TOOLS = {"Bash"}

# Dangerous patterns in tool inputs
DANGEROUS_PATTERNS = [
    r"sudo",
    r"rm\s+-rf",
    r"dd\s+if=",
    r"/etc/passwd",
    r"/etc/shadow",
    r"chmod\s+777",
    r"curl\s+.*\|.*sh",
    r"wget\s+.*\|.*sh",
    r"eval\s*\(",
    r"exec\s*\(",
    r"__import__",
    r"os\.system",
    r"subprocess",
]


def _deny(reason: str) -> dict[str, Any]:
    """Return a hook denial response."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _validate_workspace_path(
    tool_name: str, tool_input: dict[str, Any], sdk_cwd: str | None
) -> dict[str, Any]:
    """Validate that a workspace-scoped tool only accesses allowed paths.

    Allowed directories:
    - The SDK working directory (``/tmp/copilot-<session>/``)
    - The SDK tool-results directory (``~/.claude/projects/…/tool-results/``)
    """
    path = tool_input.get("file_path") or tool_input.get("path") or ""
    if not path:
        # Glob/Grep without a path default to cwd which is already sandboxed
        return {}

    resolved = os.path.normpath(os.path.expanduser(path))

    # Allow access within the SDK working directory
    if sdk_cwd:
        norm_cwd = os.path.normpath(sdk_cwd)
        if resolved.startswith(norm_cwd + os.sep) or resolved == norm_cwd:
            return {}

    # Allow access to ~/.claude/projects/*/tool-results/ (big tool results)
    claude_dir = os.path.normpath(os.path.expanduser("~/.claude/projects"))
    if resolved.startswith(claude_dir + os.sep) and "tool-results" in resolved:
        return {}

    logger.warning(
        f"Blocked {tool_name} outside workspace: {path} (resolved={resolved})"
    )
    return _deny(
        f"Tool '{tool_name}' can only access files within the workspace directory."
    )


def _validate_bash_command(
    tool_input: dict[str, Any], sdk_cwd: str | None
) -> dict[str, Any]:
    """Validate a Bash command against the allowlist of safe commands.

    Only read-only data-processing commands are allowed (jq, grep, head, etc.).
    Blocks command substitution, output redirection, and disallowed executables.

    Uses ``shlex.split`` to properly handle quoted strings (e.g. jq filters
    containing ``|`` won't be mistaken for shell pipes).
    """
    command = tool_input.get("command", "")
    if not command or not isinstance(command, str):
        return _deny("Bash command is empty.")

    # Block command substitution — can smuggle arbitrary commands
    if "$(" in command or "`" in command:
        return _deny("Command substitution ($() or ``) is not allowed in Bash.")

    # Block output redirection — Bash should be read-only
    if re.search(r"(?<!\d)>{1,2}\s", command):
        return _deny("Output redirection (> or >>) is not allowed in Bash.")

    # Block /dev/ access (e.g., /dev/tcp for network)
    if "/dev/" in command:
        return _deny("Access to /dev/ is not allowed in Bash.")

    # Tokenize with shlex (respects quotes), then extract command names.
    # shlex preserves shell operators like | ; && || as separate tokens.
    try:
        tokens = shlex.split(command)
    except ValueError:
        return _deny("Malformed command (unmatched quotes).")

    # Walk tokens: the first non-assignment token after a pipe/separator is a command.
    expect_command = True
    for token in tokens:
        if token in ("|", "||", "&&", ";"):
            expect_command = True
            continue
        if expect_command:
            # Skip env var assignments (VAR=value)
            if "=" in token and not token.startswith("-"):
                continue
            cmd_name = os.path.basename(token)
            if cmd_name not in ALLOWED_BASH_COMMANDS:
                allowed = ", ".join(sorted(ALLOWED_BASH_COMMANDS))
                logger.warning(f"Blocked Bash command: {cmd_name}")
                return _deny(
                    f"Command '{cmd_name}' is not allowed. "
                    f"Allowed commands: {allowed}"
                )
            expect_command = False

    # Validate absolute file paths stay within workspace
    if sdk_cwd:
        norm_cwd = os.path.normpath(sdk_cwd)
        claude_dir = os.path.normpath(os.path.expanduser("~/.claude/projects"))
        for token in tokens:
            if not token.startswith("/"):
                continue
            resolved = os.path.normpath(token)
            if resolved.startswith(norm_cwd + os.sep) or resolved == norm_cwd:
                continue
            if resolved.startswith(claude_dir + os.sep) and "tool-results" in resolved:
                continue
            logger.warning(f"Blocked Bash path outside workspace: {token}")
            return _deny(
                f"Bash can only access files within the workspace directory. "
                f"Path '{token}' is outside the workspace."
            )

    return {}


def _validate_tool_access(
    tool_name: str, tool_input: dict[str, Any], sdk_cwd: str | None = None
) -> dict[str, Any]:
    """Validate that a tool call is allowed.

    Returns:
        Empty dict to allow, or dict with hookSpecificOutput to deny
    """
    # Block forbidden tools
    if tool_name in BLOCKED_TOOLS:
        logger.warning(f"Blocked tool access attempt: {tool_name}")
        return _deny(
            f"Tool '{tool_name}' is not available. "
            "Use the CoPilot-specific tools instead."
        )

    # Sandboxed Bash: only allowlisted commands, workspace-scoped paths
    if tool_name in SANDBOXED_BASH_TOOLS:
        return _validate_bash_command(tool_input, sdk_cwd)

    # Workspace-scoped tools: allowed only within the SDK workspace directory
    if tool_name in WORKSPACE_SCOPED_TOOLS:
        return _validate_workspace_path(tool_name, tool_input, sdk_cwd)

    # Check for dangerous patterns in tool input
    input_str = str(tool_input)

    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, input_str, re.IGNORECASE):
            logger.warning(
                f"Blocked dangerous pattern in tool input: {pattern} in {tool_name}"
            )
            return _deny("Input contains blocked pattern")

    return {}


def _validate_user_isolation(
    tool_name: str, tool_input: dict[str, Any], user_id: str | None
) -> dict[str, Any]:
    """Validate that tool calls respect user isolation."""
    # For workspace file tools, ensure path doesn't escape
    if "workspace" in tool_name.lower():
        path = tool_input.get("path", "") or tool_input.get("file_path", "")
        if path:
            # Check for path traversal
            if ".." in path or path.startswith("/"):
                logger.warning(
                    f"Blocked path traversal attempt: {path} by user {user_id}"
                )
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": "Path traversal not allowed",
                    }
                }

    return {}


def create_security_hooks(
    user_id: str | None, sdk_cwd: str | None = None
) -> dict[str, Any]:
    """Create the security hooks configuration for Claude Agent SDK.

    Includes security validation and observability hooks:
    - PreToolUse: Security validation before tool execution
    - PostToolUse: Log successful tool executions
    - PostToolUseFailure: Log and handle failed tool executions
    - PreCompact: Log context compaction events (SDK handles compaction automatically)

    Args:
        user_id: Current user ID for isolation validation
        sdk_cwd: SDK working directory for workspace-scoped tool validation

    Returns:
        Hooks configuration dict for ClaudeAgentOptions
    """
    try:
        from claude_agent_sdk import HookMatcher
        from claude_agent_sdk.types import HookContext, HookInput, SyncHookJSONOutput

        async def pre_tool_use_hook(
            input_data: HookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> SyncHookJSONOutput:
            """Combined pre-tool-use validation hook."""
            _ = context  # unused but required by signature
            tool_name = cast(str, input_data.get("tool_name", ""))
            tool_input = cast(dict[str, Any], input_data.get("tool_input", {}))

            # Strip MCP prefix for consistent validation
            is_copilot_tool = tool_name.startswith(MCP_TOOL_PREFIX)
            clean_name = tool_name.removeprefix(MCP_TOOL_PREFIX)

            # Only block non-CoPilot tools; our MCP-registered tools
            # (including Read for oversized results) are already sandboxed.
            if not is_copilot_tool:
                result = _validate_tool_access(clean_name, tool_input, sdk_cwd)
                if result:
                    return cast(SyncHookJSONOutput, result)

            # Validate user isolation
            result = _validate_user_isolation(clean_name, tool_input, user_id)
            if result:
                return cast(SyncHookJSONOutput, result)

            logger.debug(f"[SDK] Tool start: {tool_name}, user={user_id}")
            return cast(SyncHookJSONOutput, {})

        async def post_tool_use_hook(
            input_data: HookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> SyncHookJSONOutput:
            """Log successful tool executions for observability."""
            _ = context
            tool_name = cast(str, input_data.get("tool_name", ""))
            logger.debug(f"[SDK] Tool success: {tool_name}, tool_use_id={tool_use_id}")
            return cast(SyncHookJSONOutput, {})

        async def post_tool_failure_hook(
            input_data: HookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> SyncHookJSONOutput:
            """Log failed tool executions for debugging."""
            _ = context
            tool_name = cast(str, input_data.get("tool_name", ""))
            error = input_data.get("error", "Unknown error")
            logger.warning(
                f"[SDK] Tool failed: {tool_name}, error={error}, "
                f"user={user_id}, tool_use_id={tool_use_id}"
            )
            return cast(SyncHookJSONOutput, {})

        async def pre_compact_hook(
            input_data: HookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> SyncHookJSONOutput:
            """Log when SDK triggers context compaction.

            The SDK automatically compacts conversation history when it grows too large.
            This hook provides visibility into when compaction happens.
            """
            _ = context, tool_use_id
            trigger = input_data.get("trigger", "auto")
            logger.info(
                f"[SDK] Context compaction triggered: {trigger}, user={user_id}"
            )
            return cast(SyncHookJSONOutput, {})

        return {
            "PreToolUse": [HookMatcher(matcher="*", hooks=[pre_tool_use_hook])],
            "PostToolUse": [HookMatcher(matcher="*", hooks=[post_tool_use_hook])],
            "PostToolUseFailure": [
                HookMatcher(matcher="*", hooks=[post_tool_failure_hook])
            ],
            "PreCompact": [HookMatcher(matcher="*", hooks=[pre_compact_hook])],
        }
    except ImportError:
        # Fallback for when SDK isn't available - return empty hooks
        return {}
