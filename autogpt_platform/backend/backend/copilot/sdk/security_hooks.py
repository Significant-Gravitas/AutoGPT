"""Security hooks for Claude Agent SDK integration.

This module provides security hooks that validate tool calls before execution,
ensuring multi-user isolation and preventing unauthorized operations.
"""

import json
import logging
import re
from collections.abc import Callable
from typing import Any, cast

from backend.copilot.context import (
    get_execution_context,
    is_allowed_local_path,
    is_sdk_tool_path,
)
from backend.copilot.pending_messages import drain_and_format_for_injection

from .tool_adapter import (
    BLOCKED_TOOLS,
    DANGEROUS_PATTERNS,
    MCP_TOOL_PREFIX,
    WORKSPACE_SCOPED_TOOLS,
    stash_pending_tool_output,
)

logger = logging.getLogger(__name__)

# The SDK CLI uses "Task" in older versions and "Agent" in v2.x+.
# Shared across all sessions — used by security hooks for sub-agent detection.
_SUBAGENT_TOOLS: frozenset[str] = frozenset({"Task", "Agent"})

# Unicode ranges stripped by _sanitize():
#   - BiDi overrides (U+202A-U+202E, U+2066-U+2069) can trick reviewers
#     into misreading code/logs.
#   - Zero-width characters (U+200B-U+200F, U+FEFF) can hide content.
_BIDI_AND_ZW_CHARS = set(
    chr(c)
    for r in (range(0x202A, 0x202F), range(0x2066, 0x206A), range(0x200B, 0x2010))
    for c in r
) | {"\ufeff"}


def _sanitize(value: str, max_len: int = 200) -> str:
    """Strip control characters and truncate for safe logging.

    Removes C0 (U+0000-U+001F), DEL (U+007F), C1 (U+0080-U+009F),
    Unicode BiDi overrides, and zero-width characters to prevent
    log injection and visual spoofing.
    """
    cleaned = "".join(
        c
        for c in value
        if c >= " "
        and c != "\x7f"
        and not ("\x80" <= c <= "\x9f")
        and c not in _BIDI_AND_ZW_CHARS
    )
    return cleaned[:max_len]


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

    For ``Read``: only SDK artifact paths (tool-results/, tool-outputs/) are
    permitted.  The workspace directory is served by the ``read_file`` MCP
    tool which enforces per-session isolation.

    For ``Glob`` / ``Grep``: the full workspace (sdk_cwd) is allowed in
    addition to SDK artifact paths.
    """
    path = tool_input.get("file_path") or tool_input.get("path") or ""
    if not path:
        # Glob/Grep without a path default to cwd which is already sandboxed
        return {}

    if tool_name == "Read":
        # Narrow carve-out: only allow SDK artifact paths for the native Read tool.
        # ``is_sdk_tool_path`` validates session membership via _current_project_dir,
        # preventing cross-session access to another session's tool-results directory.
        # All other file reads must go through the read_file MCP tool.
        if is_sdk_tool_path(path):
            return {}
        logger.warning(f"Blocked Read outside SDK artifact paths: {path}")
        return _deny(
            "[SECURITY] The SDK 'Read' tool can only access tool-results/ or "
            "tool-outputs/ paths. Use the 'read_file' MCP tool to read workspace files. "
            "This is enforced by the platform and cannot be bypassed."
        )

    if is_allowed_local_path(path, sdk_cwd):
        return {}

    logger.warning(f"Blocked {tool_name} outside workspace: {path}")
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
    # Workspace-scoped tools: allowed only within the SDK workspace directory.
    # Check this BEFORE the blocked-tools list because Read is blocked in
    # general but must remain accessible for tool-results/tool-outputs paths
    # that the SDK uses internally for oversized result handling.
    if tool_name in WORKSPACE_SCOPED_TOOLS:
        return _validate_workspace_path(tool_name, tool_input, sdk_cwd)

    # Block forbidden tools
    if tool_name in BLOCKED_TOOLS:
        logger.warning(f"Blocked tool access attempt: {tool_name}")
        return _deny(
            f"[SECURITY] Tool '{tool_name}' is blocked for security. "
            "This is enforced by the platform and cannot be bypassed. "
            "Use the CoPilot-specific MCP tools instead."
        )

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
        # The "path" param is a cloud storage key (e.g. "/ASEAN/report.md")
        # where a leading "/" is normal.  Only check for ".." traversal.
        # Filesystem paths (source_path, save_to_path) are validated inside
        # the tool itself via _validate_ephemeral_path.
        path = tool_input.get("path", "") or tool_input.get("file_path", "")
        if path and ".." in path:
            logger.warning(f"Blocked path traversal attempt: {path} by user {user_id}")
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
    on_compact: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Create the security hooks configuration for Claude Agent SDK.

    Includes security validation and observability hooks:
    - PreToolUse: Security validation before tool execution
    - PostToolUse: Log successful tool executions
    - PostToolUseFailure: Log and handle failed tool executions
    - PreCompact: Log context compaction events (SDK handles compaction automatically)
    - SubagentStart: Log sub-agent lifecycle start
    - SubagentStop: Log sub-agent lifecycle end

    Args:
        user_id: Current user ID for isolation validation
        sdk_cwd: SDK working directory for workspace-scoped tool validation
        max_subtasks: Maximum concurrent sub-agent spawns allowed per session
        on_compact: Callback invoked when SDK starts compacting context.
            Receives the transcript_path from the hook input.

    Returns:
        Hooks configuration dict for ClaudeAgentOptions
    """
    try:
        from claude_agent_sdk import HookMatcher
        from claude_agent_sdk.types import HookContext, HookInput, SyncHookJSONOutput

        # Per-session tracking for sub-agent concurrency.
        # Set of tool_use_ids that consumed a slot — len() is the active count.
        #
        # LIMITATION: For background (async) agents the SDK returns the
        # Agent/Task tool immediately with {isAsync: true}, which triggers
        # PostToolUse and releases the slot while the agent is still running.
        # SubagentStop fires later when the background process finishes but
        # does not currently hold a slot.  This means the concurrency limit
        # only gates *launches*, not true concurrent execution.  To fix this
        # we would need to track background agent_ids separately and release
        # in SubagentStop, but the SDK does not guarantee SubagentStop fires
        # for every background agent (e.g. on session abort).
        subagent_tool_use_ids: set[str] = set()

        async def pre_tool_use_hook(
            input_data: HookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> SyncHookJSONOutput:
            """Combined pre-tool-use validation hook."""
            _ = context  # unused but required by signature
            tool_name = cast(str, input_data.get("tool_name", ""))
            tool_input = cast(dict[str, Any], input_data.get("tool_input", {}))

            # Rate-limit sub-agent spawns per session.
            # The SDK CLI renamed "Task" → "Agent" in v2.x; handle both.
            if tool_name in _SUBAGENT_TOOLS:
                # Background agents are allowed — the SDK returns immediately
                # with {isAsync: true} and the model polls via TaskOutput.
                # Still count them against the concurrency limit.
                if len(subagent_tool_use_ids) >= max_subtasks:
                    logger.warning(
                        f"[SDK] Sub-agent limit reached ({max_subtasks}), "
                        f"user={user_id}"
                    )
                    return cast(
                        SyncHookJSONOutput,
                        _deny(
                            f"Maximum {max_subtasks} concurrent sub-agents. "
                            "Wait for running sub-agents to finish, "
                            "or continue in the main conversation."
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

            # Reserve the sub-agent slot only after all validations pass
            if tool_name in _SUBAGENT_TOOLS and tool_use_id is not None:
                subagent_tool_use_ids.add(tool_use_id)

            logger.debug(f"[SDK] Tool start: {tool_name}, user={user_id}")
            return cast(SyncHookJSONOutput, {})

        def _release_subagent_slot(tool_name: str, tool_use_id: str | None) -> None:
            """Release a sub-agent concurrency slot if one was reserved."""
            if tool_name in _SUBAGENT_TOOLS and tool_use_id in subagent_tool_use_ids:
                subagent_tool_use_ids.discard(tool_use_id)
                logger.info(
                    "[SDK] Sub-agent slot released, active=%d/%d, user=%s",
                    len(subagent_tool_use_ids),
                    max_subtasks,
                    user_id,
                )

        async def post_tool_use_hook(
            input_data: HookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> SyncHookJSONOutput:
            """Log successful tool executions and stash SDK built-in tool outputs.

            MCP tools stash their output in ``_execute_tool_sync`` before the
            SDK can truncate it.  SDK built-in tools (WebSearch, Read, etc.)
            are executed by the CLI internally — this hook captures their
            output so the response adapter can forward it to the frontend.
            """
            _ = context
            tool_name = cast(str, input_data.get("tool_name", ""))

            _release_subagent_slot(tool_name, tool_use_id)
            is_builtin = not tool_name.startswith(MCP_TOOL_PREFIX)
            safe_tool_use_id = _sanitize(str(tool_use_id or ""), max_len=12)
            logger.info(
                "[SDK] PostToolUse: %s (builtin=%s, tool_use_id=%s)",
                tool_name,
                is_builtin,
                safe_tool_use_id,
            )

            # Stash output for SDK built-in tools so the response adapter can
            # emit StreamToolOutputAvailable even when the CLI doesn't surface
            # a separate UserMessage with ToolResultBlock content.
            if is_builtin:
                tool_response = input_data.get("tool_response")
                if tool_response is not None:
                    resp_preview = _sanitize(str(tool_response), max_len=100)
                    logger.info(
                        "[SDK] Stashing builtin output for %s (%d chars): %s...",
                        tool_name,
                        len(str(tool_response)),
                        resp_preview,
                    )
                    stash_pending_tool_output(tool_name, tool_response)
                else:
                    logger.warning(
                        "[SDK] PostToolUse for builtin %s but tool_response is None",
                        tool_name,
                    )

            # Mid-turn drain: after ANY tool finishes (MCP or built-in), pull
            # any queued user follow-up messages and attach them to the
            # tool_result as ``additionalContext``.  This is the
            # protocol-legal mid-turn injection slot — Claude reads the
            # follow-up on the next LLM round without starting a new turn.
            # The drain helper also stashes a persist-queue copy so
            # ``sdk/service.py`` can append a matching user row to the UI.
            _, session = get_execution_context()
            followup = ""
            if session is not None and session.session_id:
                followup = await drain_and_format_for_injection(
                    session.session_id,
                    log_prefix="[SDK][PostToolUse]",
                )
            if followup:
                return cast(
                    SyncHookJSONOutput,
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PostToolUse",
                            "additionalContext": followup,
                        }
                    },
                )
            return cast(SyncHookJSONOutput, {})

        async def post_tool_failure_hook(
            input_data: HookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> SyncHookJSONOutput:
            """Log failed tool executions for debugging."""
            _ = context
            tool_name = cast(str, input_data.get("tool_name", ""))
            error = _sanitize(str(input_data.get("error", "Unknown error")))
            safe_tool_use_id = _sanitize(str(tool_use_id or ""))
            logger.warning(
                "[SDK] Tool failed: %s, error=%s, user=%s, tool_use_id=%s",
                tool_name,
                error,
                user_id,
                safe_tool_use_id,
            )

            _release_subagent_slot(tool_name, tool_use_id)

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
            trigger = _sanitize(str(input_data.get("trigger", "auto")), max_len=50)
            # Sanitize untrusted input: strip control chars for logging AND
            # for the value passed downstream.  read_compacted_entries()
            # validates against projects_base() as defence-in-depth, but
            # sanitizing here prevents log injection and rejects obviously
            # malformed paths early.
            transcript_path = _sanitize(
                str(input_data.get("transcript_path", "")), max_len=500
            )
            logger.info(
                "[SDK] Context compaction triggered: %s, user=%s, transcript_path=%s",
                trigger,
                user_id,
                transcript_path,
            )
            if on_compact is not None:
                on_compact(transcript_path)
            return cast(SyncHookJSONOutput, {})

        async def subagent_start_hook(
            input_data: HookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> SyncHookJSONOutput:
            """Log when a sub-agent starts execution."""
            _ = context, tool_use_id
            agent_id = _sanitize(str(input_data.get("agent_id", "?")))
            agent_type = _sanitize(str(input_data.get("agent_type", "?")))
            logger.info(
                "[SDK] SubagentStart: agent_id=%s, type=%s, user=%s",
                agent_id,
                agent_type,
                user_id,
            )
            return cast(SyncHookJSONOutput, {})

        async def subagent_stop_hook(
            input_data: HookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> SyncHookJSONOutput:
            """Log when a sub-agent stops."""
            _ = context, tool_use_id
            agent_id = _sanitize(str(input_data.get("agent_id", "?")))
            agent_type = _sanitize(str(input_data.get("agent_type", "?")))
            transcript = _sanitize(
                str(input_data.get("agent_transcript_path", "")), max_len=500
            )
            logger.info(
                "[SDK] SubagentStop: agent_id=%s, type=%s, user=%s, transcript=%s",
                agent_id,
                agent_type,
                user_id,
                transcript,
            )
            return cast(SyncHookJSONOutput, {})

        hooks: dict[str, Any] = {
            "PreToolUse": [HookMatcher(matcher="*", hooks=[pre_tool_use_hook])],
            "PostToolUse": [HookMatcher(matcher="*", hooks=[post_tool_use_hook])],
            "PostToolUseFailure": [
                HookMatcher(matcher="*", hooks=[post_tool_failure_hook])
            ],
            "PreCompact": [HookMatcher(matcher="*", hooks=[pre_compact_hook])],
            "SubagentStart": [HookMatcher(matcher="*", hooks=[subagent_start_hook])],
            "SubagentStop": [HookMatcher(matcher="*", hooks=[subagent_stop_hook])],
        }

        return hooks
    except ImportError:
        # Fallback for when SDK isn't available - return empty hooks
        logger.warning("claude-agent-sdk not available, security hooks disabled")
        return {}
