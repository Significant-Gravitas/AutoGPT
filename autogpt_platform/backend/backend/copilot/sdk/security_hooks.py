"""Security hooks for Claude Agent SDK integration.

This module provides security hooks that validate tool calls before execution,
ensuring multi-user isolation and preventing unauthorized operations.
"""

import json
import logging
import os
import re
from collections.abc import Callable
from typing import Any, cast

from .tool_adapter import MCP_TOOL_PREFIX

logger = logging.getLogger(__name__)

# Tools that are blocked entirely (CLI/system access).
# "Bash" (capital) is the SDK built-in — it's NOT in allowed_tools but blocked
# here as defence-in-depth.  The agent uses mcp__copilot__bash_exec instead,
# which has kernel-level network isolation (unshare --net).
BLOCKED_TOOLS = {
    "Bash",
    "bash",
    "shell",
    "exec",
    "terminal",
    "command",
}

# Tools allowed only when their path argument stays within the SDK workspace.
# The SDK uses these to handle oversized tool results (writes to tool-results/
# files, then reads them back) and for workspace file operations.
WORKSPACE_SCOPED_TOOLS = {"Read", "Write", "Edit", "Glob", "Grep"}

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

    # Resolve relative paths against sdk_cwd (the SDK sets cwd so the LLM
    # naturally uses relative paths like "test.txt" instead of absolute ones).
    # Tilde paths (~/) are home-dir references, not relative — expand first.
    if path.startswith("~"):
        resolved = os.path.realpath(os.path.expanduser(path))
    elif not os.path.isabs(path) and sdk_cwd:
        resolved = os.path.realpath(os.path.join(sdk_cwd, path))
    else:
        resolved = os.path.realpath(path)

    # Allow access within the SDK working directory
    if sdk_cwd:
        norm_cwd = os.path.realpath(sdk_cwd)
        if resolved.startswith(norm_cwd + os.sep) or resolved == norm_cwd:
            return {}

    # Allow access to ~/.claude/projects/*/tool-results/ (big tool results)
    claude_dir = os.path.realpath(os.path.expanduser("~/.claude/projects"))
    tool_results_seg = os.sep + "tool-results" + os.sep
    if resolved.startswith(claude_dir + os.sep) and tool_results_seg in resolved:
        return {}

    logger.warning(
        f"Blocked {tool_name} outside workspace: {path} (resolved={resolved})"
    )
    workspace_hint = f" Allowed workspace: {sdk_cwd}" if sdk_cwd else ""
    return _deny(
        f"[SECURITY] Tool '{tool_name}' can only access files within the workspace "
        f"directory.{workspace_hint} "
        "This is enforced by the platform and cannot be bypassed."
    )


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
            f"[SECURITY] Tool '{tool_name}' is blocked for security. "
            "This is enforced by the platform and cannot be bypassed. "
            "Use the CoPilot-specific MCP tools instead."
        )

    # Workspace-scoped tools: allowed only within the SDK workspace directory
    if tool_name in WORKSPACE_SCOPED_TOOLS:
        return _validate_workspace_path(tool_name, tool_input, sdk_cwd)

    # Check for dangerous patterns in tool input
    # Use json.dumps for predictable format (str() produces Python repr)
    input_str = json.dumps(tool_input) if tool_input else ""

    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, input_str, re.IGNORECASE):
            logger.warning(
                f"Blocked dangerous pattern in tool input: {pattern} in {tool_name}"
            )
            return _deny(
                "[SECURITY] Input contains a blocked pattern. "
                "This is enforced by the platform and cannot be bypassed."
            )

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
    user_id: str | None,
    sdk_cwd: str | None = None,
    max_subtasks: int = 3,
    on_stop: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Create the security hooks configuration for Claude Agent SDK.

    Includes security validation and observability hooks:
    - PreToolUse: Security validation before tool execution
    - PostToolUse: Log successful tool executions
    - PostToolUseFailure: Log and handle failed tool executions
    - PreCompact: Log context compaction events (SDK handles compaction automatically)
    - Stop: Capture transcript path for stateless resume (when *on_stop* is provided)

    Args:
        user_id: Current user ID for isolation validation
        sdk_cwd: SDK working directory for workspace-scoped tool validation
        max_subtasks: Maximum Task (sub-agent) spawns allowed per session
        on_stop: Callback ``(transcript_path, sdk_session_id)`` invoked when
            the SDK finishes processing — used to read the JSONL transcript
            before the CLI process exits.

    Returns:
        Hooks configuration dict for ClaudeAgentOptions
    """
    try:
        from claude_agent_sdk import HookMatcher
        from claude_agent_sdk.types import HookContext, HookInput, SyncHookJSONOutput

        # Per-session counter for Task sub-agent spawns
        task_spawn_count = 0

        async def pre_tool_use_hook(
            input_data: HookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> SyncHookJSONOutput:
            """Combined pre-tool-use validation hook."""
            nonlocal task_spawn_count
            _ = context  # unused but required by signature
            tool_name = cast(str, input_data.get("tool_name", ""))
            tool_input = cast(dict[str, Any], input_data.get("tool_input", {}))

            # Rate-limit Task (sub-agent) spawns per session
            if tool_name == "Task":
                task_spawn_count += 1
                if task_spawn_count > max_subtasks:
                    logger.warning(
                        f"[SDK] Task limit reached ({max_subtasks}), user={user_id}"
                    )
                    return cast(
                        SyncHookJSONOutput,
                        _deny(
                            f"Maximum {max_subtasks} sub-tasks per session. "
                            "Please continue in the main conversation."
                        ),
                    )

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

        # --- Stop hook: capture transcript path for stateless resume ---
        async def stop_hook(
            input_data: HookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> SyncHookJSONOutput:
            """Capture transcript path when SDK finishes processing.

            The Stop hook fires while the CLI process is still alive, giving us
            a reliable window to read the JSONL transcript before SIGTERM.
            """
            _ = context, tool_use_id
            transcript_path = cast(str, input_data.get("transcript_path", ""))
            sdk_session_id = cast(str, input_data.get("session_id", ""))

            if transcript_path and on_stop:
                logger.info(
                    f"[SDK] Stop hook: transcript_path={transcript_path}, "
                    f"sdk_session_id={sdk_session_id[:12]}..."
                )
                on_stop(transcript_path, sdk_session_id)

            return cast(SyncHookJSONOutput, {})

        hooks: dict[str, Any] = {
            "PreToolUse": [HookMatcher(matcher="*", hooks=[pre_tool_use_hook])],
            "PostToolUse": [HookMatcher(matcher="*", hooks=[post_tool_use_hook])],
            "PostToolUseFailure": [
                HookMatcher(matcher="*", hooks=[post_tool_failure_hook])
            ],
            "PreCompact": [HookMatcher(matcher="*", hooks=[pre_compact_hook])],
        }

        if on_stop is not None:
            hooks["Stop"] = [HookMatcher(matcher=None, hooks=[stop_hook])]

        return hooks
    except ImportError:
        # Fallback for when SDK isn't available - return empty hooks
        logger.warning("claude-agent-sdk not available, security hooks disabled")
        return {}
