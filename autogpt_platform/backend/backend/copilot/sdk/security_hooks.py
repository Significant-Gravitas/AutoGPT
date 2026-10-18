"""Security hooks for Claude Agent SDK integration.

This module provides security hooks that validate tool calls before execution,
ensuring multi-user isolation and preventing unauthorized operations.
"""

import json
import logging
import re
from collections.abc import Awaitable, Callable, Collection, Iterable
from typing import Any, cast

from claude_agent_sdk.types import HookEvent, HookMatcher

from backend.copilot.context import get_execution_context, is_allowed_local_path
from backend.copilot.pending_messages import drain_and_format_for_injection

from .after_compaction import AFTER_COMPACTION_MAX_PARTS
from .after_compaction_handout import AfterCompactionHandout
from .governor import TurnGovernor
from .tool_adapter import (
    BLOCKED_TOOLS,
    DANGEROUS_PATTERNS,
    MCP_TOOL_PREFIX,
    SDK_SUBAGENT_TOOLS,
    WORKSPACE_SCOPED_TOOLS,
    stash_pending_tool_output,
)
from .tool_display import SDKToolDisplayBridge

logger = logging.getLogger(__name__)

# ``SessionStart`` is missing from the Python SDK's ``HookEvent`` literal, but
# the CLI runs SDK callback hooks for it: with matcher ``compact`` it fires
# right after the CLI compacts the conversation, and its
# ``additionalContext`` is saved into the session (an ``attachment`` line that
# ``strip_for_upload`` keeps), so later ``--resume`` turns still carry it.
# Measured with a stub API against claude-agent-sdk 0.2.161 / CLI 2.1.284 and
# 0.2.156 / 2.1.276: at a turn boundary and inside a tool loop, and across a
# ``--resume`` in a new CLI process.
_SESSION_START_EVENT = cast(HookEvent, "SessionStart")
# The CLI waits on this hook before it carries on after the summary, so it is
# bounded; a timeout only costs the re-sent context, never the turn.
_AFTER_COMPACTION_HOOK_TIMEOUT_S = 30.0

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


def _landing_refusal(governor: TurnGovernor | None) -> str | None:
    """The governor's refusal for the next tool call, if it is landing the turn.

    A governor failure lets the call run: the CLI's own budget and turn limits
    still stop the turn, and failing every tool call would strand it instead.
    """
    if governor is None:
        return None
    try:
        return governor.refusal()
    except Exception:
        logger.warning("[SDK] Turn governor could not decide a refusal", exc_info=True)
        return None


def _validate_workspace_path(
    tool_name: str, tool_input: dict[str, Any], sdk_cwd: str | None
) -> dict[str, Any]:
    """Validate that a workspace-scoped tool (``Glob`` / ``Grep``) only
    accesses the workspace (sdk_cwd) and the SDK's artifact paths.
    """
    path = tool_input.get("file_path") or tool_input.get("path") or ""
    if not path:
        # Glob/Grep without a path default to cwd which is already sandboxed
        return {}

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
    tool_name: str,
    tool_input: dict[str, Any],
    sdk_cwd: str | None = None,
    allowed_tools: Collection[str] = frozenset(),
) -> dict[str, Any]:
    """Validate that a tool call is allowed.

    A tool outside *allowed_tools* is denied: ``allowed_tools`` on
    ``ClaudeAgentOptions`` only pre-approves, so a CLI built-in that needs no
    permission runs unless this hook refuses it.

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

    if tool_name not in allowed_tools:
        return _deny_unlisted(tool_name)

    # Workspace-scoped tools: allowed only within the SDK workspace directory.
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


def _deny_unlisted(tool_name: str) -> dict[str, Any]:
    logger.warning(f"Blocked tool outside the session's allowed set: {tool_name}")
    return _deny(
        f"[SECURITY] Tool '{tool_name}' is not available in this session. "
        "This is enforced by the platform and cannot be bypassed."
    )


def _validate_user_isolation(
    tool_name: str, tool_input: dict[str, Any], user_id: str | None
) -> dict[str, Any]:
    """Validate that tool calls respect user isolation."""
    # For workspace file tools, ensure path doesn't escape
    if "workspace" in tool_name.lower():
        # The "path" param is a cloud storage key (e.g. "/ASEAN/report.md")
        # where a leading "/" is normal.  Only check for ".." traversal.
        # Filesystem paths (source_path, save_to_path) are validated inside
        # the tool itself via workdir.validate_ephemeral_path.
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


# Tools whose display name (block, agent, MCP tool) streams to the UI before
# the call finishes; the bridge tags their input with a call token.
_DISPLAY_BRIDGED_TOOLS: frozenset[str] = frozenset(
    {"run_agent", "run_capability", "resume_capability"}
)


def create_security_hooks(
    user_id: str | None,
    sdk_cwd: str | None = None,
    max_subtasks: int = 3,
    on_compact: Callable[[str], None] | None = None,
    tool_display_bridge: SDKToolDisplayBridge | None = None,
    context_after_compaction: Callable[[], Awaitable[list[str]]] | None = None,
    governor: TurnGovernor | None = None,
    *,
    allowed_tools: Iterable[str],
) -> dict[HookEvent, list[HookMatcher]]:
    """Create the security hooks configuration for Claude Agent SDK.

    Includes security validation and observability hooks:
    - PreToolUse: Security validation before tool execution
    - PostToolUse: Log successful tool executions
    - PostToolUseFailure: Log and handle failed tool executions
    - PreCompact: Log context compaction events (SDK handles compaction automatically)
    - SessionStart (``compact``): Re-send context the compaction summary dropped
    - SubagentStart: Log sub-agent lifecycle start
    - SubagentStop: Log sub-agent lifecycle end

    Args:
        user_id: Current user ID for isolation validation
        sdk_cwd: SDK working directory for workspace-scoped tool validation
        max_subtasks: Maximum concurrent sub-agent spawns allowed per session
        on_compact: Callback invoked when SDK starts compacting context.
            Receives the transcript_path from the hook input.
        context_after_compaction: Returns what to hand the model right after
            the CLI compacts the conversation — the first-turn blocks the
            summary does not carry — as at most
            ``AFTER_COMPACTION_MAX_PARTS`` parts, each within the CLI's
            per-callback cap.  ``[]`` sends nothing.
        governor: The turn's spend and step governor. Once it is landing the
            turn, every tool call -- the main agent's and its sub-agents' --
            is refused with a reason that asks for a checkpoint.
        allowed_tools: The ``allowed_tools`` passed to ``ClaudeAgentOptions``;
            every other tool is denied.

    Returns:
        Hooks configuration dict for ClaudeAgentOptions
    """
    allowed = frozenset(allowed_tools)
    try:
        from claude_agent_sdk import HookMatcher
        from claude_agent_sdk.types import HookContext, HookInput, SyncHookJSONOutput

        # Per-session tracking for sub-agent concurrency.
        # Set of tool_use_ids that consumed a slot — len() is the active count.
        # Sub-agents run in the foreground (route_launch_env sets
        # CLAUDE_CODE_DISABLE_BACKGROUND_TASKS), so the Agent/Task call lasts
        # as long as the sub-agent and PostToolUse releases the slot when it
        # finishes.
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

            refusal = _landing_refusal(governor)
            if governor is not None and refusal is not None:
                logger.info(
                    "[SDK] Refusing %s: turn is landing (%s), user=%s",
                    tool_name,
                    governor.landing,
                    user_id,
                )
                return cast(SyncHookJSONOutput, _deny(refusal))

            # Rate-limit sub-agent spawns per session.
            # The SDK CLI renamed "Task" → "Agent" in v2.x; handle both.
            if tool_name in SDK_SUBAGENT_TOOLS:
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

            # Our MCP tools sandbox themselves, so they skip the path and
            # pattern checks — but not the allowed set.
            if not is_copilot_tool:
                result = _validate_tool_access(
                    clean_name, tool_input, sdk_cwd, allowed_tools=allowed
                )
                if result:
                    return cast(SyncHookJSONOutput, result)
            elif tool_name not in allowed:
                return cast(SyncHookJSONOutput, _deny_unlisted(tool_name))

            # Validate user isolation
            result = _validate_user_isolation(clean_name, tool_input, user_id)
            if result:
                return cast(SyncHookJSONOutput, result)

            # Reserve the sub-agent slot only after all validations pass
            if tool_name in SDK_SUBAGENT_TOOLS and tool_use_id is not None:
                subagent_tool_use_ids.add(tool_use_id)

            logger.debug(f"[SDK] Tool start: {tool_name}, user={user_id}")
            if (
                is_copilot_tool
                and clean_name in _DISPLAY_BRIDGED_TOOLS
                and tool_use_id is not None
                and tool_display_bridge is not None
            ):
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "updatedInput": tool_display_bridge.prepare_call(
                            clean_name, tool_input, tool_use_id
                        ),
                    }
                }
            return cast(SyncHookJSONOutput, {})

        def _release_subagent_slot(tool_name: str, tool_use_id: str | None) -> None:
            """Release a sub-agent concurrency slot if one was reserved."""
            if tool_name in SDK_SUBAGENT_TOOLS and tool_use_id in subagent_tool_use_ids:
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
                    # Key by the call's input so the response adapter pops the
                    # output for the RIGHT tool_call_id when the model fired
                    # several same-name calls in parallel (OPEN-3158).
                    tool_input = input_data.get("tool_input")
                    stash_pending_tool_output(tool_name, tool_response, tool_input)
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

        async def _rebuild_after_compaction() -> list[str]:
            if context_after_compaction is None:
                return []
            try:
                return await context_after_compaction()
            except Exception:
                logger.exception(
                    "[SDK] Could not rebuild the context to re-send after"
                    " compaction (user=%s)",
                    user_id,
                )
                return []

        after_compaction = AfterCompactionHandout(_rebuild_after_compaction)

        def session_start_hook(part: int):
            async def hook(
                input_data: HookInput,
                tool_use_id: str | None,
                context: HookContext,
            ) -> SyncHookJSONOutput:
                """Hand the model back part of what the compaction dropped.

                The first-turn blocks (skills index, workflows, follow-ups,
                working dir, user context) live in the session's first user
                message, which compaction replaces with a summary.  The CLI
                starts these callbacks together and puts their
                ``additionalContext`` right after the summary in the order
                they answer; ``after_compaction`` rebuilds the context once
                per compaction and lets the parts answer in order.
                """
                _ = context, tool_use_id
                if input_data.get("source") != "compact":
                    return cast(SyncHookJSONOutput, {})
                text = await after_compaction.part(part)
                if not text:
                    return cast(SyncHookJSONOutput, {})
                return cast(
                    SyncHookJSONOutput,
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "SessionStart",
                            "additionalContext": text,
                        }
                    },
                )

            return hook

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

        hooks: dict[HookEvent, list[HookMatcher]] = {
            "PreToolUse": [HookMatcher(matcher="*", hooks=[pre_tool_use_hook])],
            "PostToolUse": [HookMatcher(matcher="*", hooks=[post_tool_use_hook])],
            "PostToolUseFailure": [
                HookMatcher(matcher="*", hooks=[post_tool_failure_hook])
            ],
            "PreCompact": [HookMatcher(matcher="*", hooks=[pre_compact_hook])],
            "SubagentStart": [HookMatcher(matcher="*", hooks=[subagent_start_hook])],
            "SubagentStop": [HookMatcher(matcher="*", hooks=[subagent_stop_hook])],
        }
        if context_after_compaction is not None:
            # One callback per part: the CLI caps each callback's context on
            # its own (``after_compaction.py``).
            hooks[_SESSION_START_EVENT] = [
                HookMatcher(
                    matcher="compact",
                    hooks=[
                        session_start_hook(part)
                        for part in range(AFTER_COMPACTION_MAX_PARTS)
                    ],
                    timeout=_AFTER_COMPACTION_HOOK_TIMEOUT_S,
                )
            ]
        return hooks
    except ImportError:
        # Fallback for when SDK isn't available - return empty hooks
        logger.warning("claude-agent-sdk not available, security hooks disabled")
        return {}
