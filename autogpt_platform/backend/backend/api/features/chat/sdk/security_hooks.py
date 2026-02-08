"""Security hooks for Claude Agent SDK integration.

This module provides security hooks that validate tool calls before execution,
ensuring multi-user isolation and preventing unauthorized operations.
"""

import logging
import re
from typing import Any, cast

logger = logging.getLogger(__name__)

# Tools that are blocked entirely (CLI/system access)
BLOCKED_TOOLS = {
    "Bash",
    "bash",
    "shell",
    "exec",
    "terminal",
    "command",
    "Read",  # Block raw file read - use workspace tools instead
    "Write",  # Block raw file write - use workspace tools instead
    "Edit",  # Block raw file edit - use workspace tools instead
    "Glob",  # Block raw file glob - use workspace tools instead
    "Grep",  # Block raw file grep - use workspace tools instead
}

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


def _validate_tool_access(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Validate that a tool call is allowed.

    Returns:
        Empty dict to allow, or dict with hookSpecificOutput to deny
    """
    # Block forbidden tools
    if tool_name in BLOCKED_TOOLS:
        logger.warning(f"Blocked tool access attempt: {tool_name}")
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"Tool '{tool_name}' is not available. "
                    "Use the CoPilot-specific tools instead."
                ),
            }
        }

    # Check for dangerous patterns in tool input
    input_str = str(tool_input)

    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, input_str, re.IGNORECASE):
            logger.warning(
                f"Blocked dangerous pattern in tool input: {pattern} in {tool_name}"
            )
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Input contains blocked pattern",
                }
            }

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


def create_security_hooks(user_id: str | None) -> dict[str, Any]:
    """Create the security hooks configuration for Claude Agent SDK.

    Includes security validation and observability hooks:
    - PreToolUse: Security validation before tool execution
    - PostToolUse: Log successful tool executions
    - PostToolUseFailure: Log and handle failed tool executions
    - PreCompact: Log context compaction events (SDK handles compaction automatically)

    Args:
        user_id: Current user ID for isolation validation

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

            # Validate basic tool access
            result = _validate_tool_access(tool_name, tool_input)
            if result:
                return cast(SyncHookJSONOutput, result)

            # Validate user isolation
            result = _validate_user_isolation(tool_name, tool_input, user_id)
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


def create_strict_security_hooks(
    user_id: str | None,
    allowed_tools: list[str] | None = None,
) -> dict[str, Any]:
    """Create strict security hooks that only allow specific tools.

    Args:
        user_id: Current user ID
        allowed_tools: List of allowed tool names (defaults to CoPilot tools)

    Returns:
        Hooks configuration dict
    """
    try:
        from claude_agent_sdk import HookMatcher
        from claude_agent_sdk.types import HookContext, HookInput, SyncHookJSONOutput

        from .tool_adapter import RAW_TOOL_NAMES

        tools_list = allowed_tools if allowed_tools is not None else RAW_TOOL_NAMES
        allowed_set = set(tools_list)

        async def strict_pre_tool_use(
            input_data: HookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> SyncHookJSONOutput:
            """Strict validation that only allows whitelisted tools."""
            _ = context  # unused but required by signature
            tool_name = cast(str, input_data.get("tool_name", ""))
            tool_input = cast(dict[str, Any], input_data.get("tool_input", {}))

            # Remove MCP prefix if present
            clean_name = tool_name.removeprefix("mcp__copilot__")

            if clean_name not in allowed_set:
                logger.warning(f"Blocked non-whitelisted tool: {tool_name}")
                return cast(
                    SyncHookJSONOutput,
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": (
                                f"Tool '{tool_name}' is not in the allowed list"
                            ),
                        }
                    },
                )

            # Run standard validations using clean_name for consistent checks
            result = _validate_tool_access(clean_name, tool_input)
            if result:
                return cast(SyncHookJSONOutput, result)

            result = _validate_user_isolation(clean_name, tool_input, user_id)
            if result:
                return cast(SyncHookJSONOutput, result)

            logger.debug(
                f"[SDK Audit] Tool call: tool={tool_name}, "
                f"user={user_id}, tool_use_id={tool_use_id}"
            )
            return cast(SyncHookJSONOutput, {})

        return {
            "PreToolUse": [
                HookMatcher(matcher="*", hooks=[strict_pre_tool_use]),
            ],
        }
    except ImportError:
        return {}
