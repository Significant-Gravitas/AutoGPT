"""Claude Agent SDK service layer for CoPilot chat completions."""

# isort: skip_file  — double-dot relative imports must stay relative to avoid Pyright type collisions

import asyncio
import contextlib
import contextvars
import base64
import functools
from copy import copy
import json
import logging
import os
import random
import re
import shutil
import sys
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict, cast

if TYPE_CHECKING:
    from ..permissions import CopilotPermissions
    from ..tree import TurnEnvelope

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultError,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from claude_agent_sdk.types import SystemPromptCustom
from langfuse import get_client, propagate_attributes
from langsmith.integrations.claude_agent_sdk import configure_claude_agent_sdk
from opentelemetry import trace as otel_trace
from pydantic import BaseModel

from backend.blocks.desktop._common import workspace_volume_mounts
from backend.copilot.model_router import RoutingSource
from backend.copilot.budget_gate import gate_auto_continue
from backend.copilot.budget_signal import build_turn_budget_block
from backend.copilot.continuation import as_model_prompt
from backend.copilot.feedback_db import RATEABLE_ROLES
from backend.copilot.graphiti.context import fetch_warm_context
from backend.copilot.markers import append_error_marker
from backend.copilot.provider_failure import ProviderFailure
from backend.copilot.segments import Segment, stamp_segment
from backend.copilot.session_kind import SessionKind
from backend.copilot.context_blocks import ContextBlock, render_context_blocks
from backend.data.understanding import BusinessUnderstanding
from backend.copilot.graphiti.ingest import enqueue_conversation_turn
from backend.copilot.learning.capture import capture_chat_turn, find_initiating_user_row
from backend.copilot.sdk.model_gateway.front import ModelGateway
from backend.copilot.sdk.model_gateway.usage import GatewayUsage
from backend.copilot.route_profile import RouteProfile
from backend.copilot.sdk.env import (
    GatewayEndpoint,
    common_launch_options,
    describe_sdk_context,
    route_launch_env,
)
from backend.copilot.sdk.launch_env import (
    launch_options,
)
from backend.copilot.sdk.turn_route import (
    resolve_turn_profile,
    start_route_gateway,
    turn_route_name,
)
from backend.copilot.sdk.cost_tracking import (
    TokenUsage as _TokenUsage,
    read_cli_session_usage,
)
from backend.copilot.tool_display import tool_calls_for_provider
from backend.copilot.credit_billing import credit_billed_turn, wait_for_cost
from backend.copilot.turn_budget import resolve_turn_budget
from backend.copilot.turn_pause import TurnPause, append_pause_marker
from backend.data.redis_client import get_redis_async
from backend.executor.cluster_lock import AsyncClusterLock
from backend.integrations.codex.transport import CodexCredentialLease
from backend.integrations.credential_lease import CredentialLease
from backend.util.exceptions import NotFoundError
from backend.copilot.gate import active_mode
from backend.copilot.gate.held import resolve_answered
from backend.util.llm.provider_billing import (
    PROVIDER_UNAVAILABLE_CODE,
    PROVIDER_UNAVAILABLE_MESSAGE,
    is_provider_out_of_credits,
    report_provider_out_of_credits,
)
from backend.util.feature_flag import Flag, is_feature_enabled
from backend.util.prompt import (
    DEFAULT_COMPRESSION_RESERVE,
    estimate_token_count,
    estimate_token_count_str,
    get_compression_target,
)
from backend.util.settings import Settings

from backend.copilot.sdk.context_window import BARE_MESSAGE_TOKEN_FLOOR

from ..config import ChatConfig, CopilotLLMModel
from ..constants import (
    COPILOT_ERROR_PREFIX,
    COPILOT_RETRYABLE_ERROR_PREFIX,
    FRIENDLY_TRANSIENT_MSG,
    HUNG_TOOL_CAP_SECONDS,
    STREAM_ERROR_MARKER,
    STREAM_INCOMPLETE_MARKER,
    STREAM_LOCK_PREFIX,
    is_transient_api_error,
    is_transient_assistant_error,
)
from ..session_cleanup import prune_orphan_tool_calls
from ..context import encode_cwd_for_cli, get_workspace_manager
from ..graphiti.config import is_enabled_for_user
from backend.data.llm_registry.llm_models import MODEL_DATE_SUFFIX_RE

from ..model import (
    ChatMessage,
    ChatSession,
    clear_pending_question,
    get_chat_session,
    maybe_append_user_message,
    upsert_chat_session,
)
from ..pending_message_helpers import (
    combine_pending_with_current,
    drain_pending_safe,
    drained_rows_entry,
    pending_texts_from,
    persist_pending_as_user_rows,
    persist_session_safe,
)
from ..pending_messages import (
    PendingMessage,
    drain_pending_for_persist,
    push_pending_message,
)
from ..permissions import (
    CopilotPermissions,
    apply_tool_permissions,
    denied_tool_names,
)
from ..prompting import (
    VOICE_TURN_INSTRUCTION,
    approval_mode_supplement,
    get_chat_platform_supplement,
    get_delegation_supplement,
    get_expert_oversight_supplement,
    get_team_building_supplement,
    get_graphiti_supplement,
)
from ..rate_limit import get_user_tier
from ..response_model import (
    StreamBaseResponse,
    StreamCheckpoint,
    StreamCompactionProgress,
    StreamError,
    StreamFinish,
    StreamProviderFailure,
    StreamFinishStep,
    StreamHeartbeat,
    StreamPendingDrained,
    StreamReasoningDelta,
    StreamReasoningEnd,
    StreamReasoningStart,
    StreamStart,
    StreamStartStep,
    StreamStatus,
    StreamTextDelta,
    StreamTextEnd,
    StreamTextStart,
    StreamToolDisplayAvailable,
    StreamToolInputAvailable,
    StreamToolOutputAvailable,
    StreamTurnPaused,
    StreamUsage,
)
from ..builder_context import (
    build_builder_context_turn_prefix,
    build_builder_system_prompt_suffix,
)
from ..expert_context import build_expert_identity_suffix
from ..service import (
    SKILLS_CONTEXT_TAG,
    _build_system_prompt,
    _is_langfuse_configured,
    _update_title_async,
    fetch_business_understanding,
    inject_context_blocks,
    injected_context_blocks,
    strip_injected_context_for_display,
    strip_user_context_tags,
)
from ..stream_checkpoint import turn_checkpoint
from ..thinking_stripper import ThinkingStripper
from ..token_tracking import persist_and_record_usage
from ..tools import (
    ToolGroup,
    expert_tool_disabled_groups,
    tool_names_in_groups,
)
from ..tools.e2b_sandbox import get_or_create_sandbox, pause_sandbox_direct
from ..tools.sandbox import WORKSPACE_PREFIX, make_session_path
from ..tools.seen_capabilities import build_seen_capabilities_notice
from ..tools.session_context import build_session_context
from ..tools.skills import (
    build_skills_update_notice,
    skills_context_block,
    start_session_skill_versions,
)
from ..tracking import track_user_message
from ..transcript import (
    _run_compression,
    TranscriptDownload,
    cleanup_stale_project_dirs,
    cli_session_path,
    compact_transcript,
    download_transcript,
    next_uncovered_sequence,
    prior_context_messages,
    projects_base,
    read_compacted_entries,
    read_compacted_entries_detailed,
    strip_for_upload,
    upload_transcript,
    validate_transcript,
)
from ..transcript_builder import TranscriptBuilder, TranscriptSnapshot
from .after_compaction import fit_after_compaction
from .compaction import (
    CompactionResult,
    CompactionStats,
    CompactionTracker,
    filter_compaction_messages,
    sdk_compaction_stats,
    transcript_stats,
)
from .governor import TurnGovernor
from .openrouter_cost import TurnGenerations, record_turn_cost_from_openrouter
from .query_blocks import scan_context_blocks
from .response_adapter import SDKResponseAdapter
from .resume_gap import (
    fetch_missing_rows,
    missing_range,
    sendable_gap,
    uncovered_rows,
)
from .security_hooks import create_security_hooks
from .system_prompt import SystemPromptParts, assemble_system_prompt
from .tool_adapter import (
    cap_late_tool_result,
    MCP_TOOL_PREFIX,
    create_copilot_mcp_server,
    get_copilot_tool_names,
    get_sdk_declared_tools,
    get_sdk_disallowed_tools,
    reset_pending_tool_outputs,
    reset_stash_event,
    reset_tool_failure_counters,
    sdk_declared_tool_name,
    set_execution_context,
    wait_for_stash,
)
from .tool_display import (
    SDKToolDisplayBridge,
    stamp_tool_display_name,
    strip_display_token,
)
from .turn_summary import TurnSummaryCollector
from .turn_summary_model import TurnUsage, upload_gate_reason

logger = logging.getLogger(__name__)
config = ChatConfig()


# On context-size errors the SDK query is retried with progressively
# less context: (1) original transcript → (2) compacted transcript →
# (3) no transcript (DB messages only).
# Non-context errors (network, auth, rate-limit) are NOT retried.
_MAX_STREAM_ATTEMPTS = 3

# Hard circuit breaker: abort the stream if the model sends this many
# consecutive tool calls with empty parameters (a sign of context
# saturation or serialization failure).  The MCP wrapper now returns
# guidance on the first empty call, giving the model a chance to
# self-correct.  The limit is generous to allow recovery attempts.
_EMPTY_TOOL_CALL_LIMIT = 5

# User-facing error shown when the empty-tool-call circuit breaker trips.
_CIRCUIT_BREAKER_ERROR_MSG = (
    "Unable to complete the tool call "
    "— this usually happens when the response is "
    "too large to fit in a single tool call. "
    "Try breaking your request into smaller parts."
)

# Two regimes: no tool pending → 30 min (SDK genuinely idle); tool pending →
# 2 h hard cap (lets long sub-AutoPilots run, still backstops a hung tool).
_IDLE_TIMEOUT_SECONDS = 30 * 60
_HUNG_TOOL_CAP_SECONDS = HUNG_TOOL_CAP_SECONDS


@dataclass
class _SDKLoopState:
    """Mutable per-attempt state for the SDK consume loop.

    Lives outside ``_StreamAccumulator`` so the consume helper can be a
    plain module-level async generator with a small fixed parameter list,
    instead of nesting the entire loop body deep inside
    ``_run_stream_attempt`` under a ``while True:`` wrapper.
    """

    last_real_msg_time: float
    last_flush_time: float
    msgs_since_flush: int = 0
    consecutive_empty_tool_calls: int = 0
    ended_with_stream_error: bool = False
    # Carried out of the helper so the caller can plumb idle-timeout /
    # transient / breaker error metadata into the ``_HandledStreamError``
    # raised after the loop finishes — without these fields the helper's
    # error context would be silently dropped.
    stream_error_msg: str | None = None
    stream_error_code: str | None = None
    # A billing refusal was seen; the loop reads on only to the
    # ``ResultMessage``, which carries the usage of the rounds already served.
    provider_refused: bool = False


async def _open_sdk_compaction_row(
    ctx: "_StreamContext", state: "_RetryState"
) -> list[StreamBaseResponse]:
    """Open the row for a CLI-side compaction the PreCompact hook announced.

    Sized off our own mirror of the CLI context: the transcript builder
    holds the entries the CLI is about to condense, so its token and turn
    counts are the ``tokensBefore`` that paces the bar and the
    ``messagesBefore`` the settled row reports.  Measured off the loop — a
    full context is hundreds of thousands of characters.
    """
    if not ctx.compaction.has_pending_start:
        return []
    stats = await asyncio.to_thread(
        transcript_stats,
        state.transcript_builder.entries_as_dicts(),
        model=_compression_model(),
    )
    return ctx.compaction.emit_start_if_ready(stats)


async def _measure_sdk_compaction(
    ctx: "_StreamContext", state: "_RetryState", *, final: bool = True
) -> tuple[bool, list[dict] | None, CompactionStats | None, str | None]:
    """Read what the CLI kept after compacting and size the row's payoff.

    Runs before the row closes so the settled output carries the numbers.
    Returns ``(measured, compacted, stats, after_source)``: ``measured``
    is False when no cycle was pending — or, unless *final*, when the
    session file does not hold this cycle's summary yet, so the cycle
    stays open for the next message.  The compacted entries are handed
    back so the caller can sync the transcript builder without a second
    read. ``after_source`` names how the post-compaction read resolved so
    a missing after-count stays diagnosable downstream.
    """
    # Let a PreCompact hook that raced this message land before we look —
    # ``emit_end_if_ready`` yields for the same reason.
    await asyncio.sleep(0)
    path = ctx.compaction.pending_transcript_path
    if path is None:
        return False, None, None, None
    compacted, after_source = await asyncio.to_thread(
        read_compacted_entries_detailed, path
    )
    after_source = ctx.compaction.resolve_after_source(compacted, after_source)
    if after_source in ("no_summary_line", "stale_summary"):
        if not final:
            # The boundary marker can land a beat before the summary line
            # is on disk.  Leave the cycle open; the next message reads
            # again and the turn's ResultMessage settles it either way.
            return False, None, None, None
        compacted = None
    else:
        ctx.compaction.note_summary(compacted)
    stats = await asyncio.to_thread(
        sdk_compaction_stats,
        state.transcript_builder.entries_as_dicts(),
        compacted,
        model=_compression_model(),
        start=ctx.compaction.start_stats,
    )
    return True, compacted, stats, after_source


async def _consume_sdk_until_done(
    client: ClaudeSDKClient,
    ctx: "_StreamContext",
    state: "_RetryState",
    acc: "_StreamAccumulator",
    loop_state: "_SDKLoopState",
) -> AsyncGenerator[StreamBaseResponse, None]:
    """One pass through the SDK's message stream — yields wire events.

    Returns when the SDK turn ends (``ResultMessage`` → StreamFinish, or
    iterator exhaustion).  The caller in ``_run_stream_attempt`` invokes
    this once per turn; if the adapter sets
    ``pending_thinking_only_reprompt`` after the first pass, the caller
    fires a synthetic re-prompt and invokes this again for the second
    pass — bounded to one re-prompt per turn.
    """
    async for sdk_msg in _iter_sdk_messages(
        client,
        wake=ctx.compaction.hook_fired,
        tool_display_wake=ctx.tool_display.ready if ctx.tool_display else None,
    ):
        for display in ctx.tool_display.drain() if ctx.tool_display else []:
            dispatched = _dispatch_response(
                display, acc, ctx, state, False, ctx.log_prefix
            )
            if dispatched is not None:
                yield dispatched
        # Heartbeat sentinel — refresh lock and keep SSE alive
        if sdk_msg is None:
            await ctx.lock.refresh()
            for ev in await _open_sdk_compaction_row(ctx, state):
                yield ev
            yield StreamHeartbeat()

            # Threshold flips to the long cap while a tool is pending; clock never resets.
            idle_seconds = time.monotonic() - loop_state.last_real_msg_time
            if loop_state.provider_refused and idle_seconds >= _HEARTBEAT_INTERVAL:
                break
            threshold = _idle_timeout_threshold(state.adapter)
            if idle_seconds >= threshold:
                unresolved_tool_names = sorted(
                    {
                        info.get("name", "unknown")
                        for tid, info in state.adapter.current_tool_calls.items()
                        if tid not in state.adapter.resolved_tool_calls
                    }
                )
                logger.error(
                    "%s Idle timeout after %.0fs (threshold=%ds, "
                    "unresolved tools: %s) — aborting stream",
                    ctx.log_prefix,
                    idle_seconds,
                    threshold,
                    ", ".join(unresolved_tool_names) or "none",
                )
                # The retryable marker written to the session omits
                # the `[code:<id>]` prefix — the AI SDK serializer
                # (`StreamError.to_sse`) attaches that automatically
                # on the wire so the frontend can still parse a
                # machine-readable code out of the otherwise opaque
                # `{type, errorText}` schema.
                loop_state.stream_error_code = "idle_timeout"
                tool_phrase = (
                    f" while running {_humanise_tool_list(unresolved_tool_names)}"
                    if unresolved_tool_names
                    else ""
                )
                loop_state.stream_error_msg = (
                    f"The response stopped{tool_phrase}. "
                    "This usually means a tool got stuck. Please try again."
                )
                _append_error_marker(
                    ctx.session, loop_state.stream_error_msg, retryable=True
                )
                yield StreamError(
                    errorText=loop_state.stream_error_msg,
                    code=loop_state.stream_error_code,
                )
                loop_state.ended_with_stream_error = True
                break
            continue

        loop_state.last_real_msg_time = time.monotonic()

        logger.info(
            "%s Received: %s %s (unresolved=%d, current=%d, resolved=%d)",
            ctx.log_prefix,
            type(sdk_msg).__name__,
            getattr(sdk_msg, "subtype", ""),
            len(state.adapter.current_tool_calls)
            - len(state.adapter.resolved_tool_calls),
            len(state.adapter.current_tool_calls),
            len(state.adapter.resolved_tool_calls),
        )

        # Capture OpenRouter generation IDs from each
        # ``AssistantMessage.message_id`` — when routed via OpenRouter
        # these are ``gen-...`` slugs we can use post-turn to query
        # ``/api/v1/generation?id=`` for the authoritative per-turn
        # cost and token counts (the CLI's ``total_cost_usd`` is
        # computed from a static Anthropic pricing table that
        # silently over-bills non-Anthropic routes).  Direct-Anthropic
        # turns produce ``msg_...`` IDs which the generation endpoint
        # doesn't know about — harmlessly ignored at reconcile time.
        if isinstance(sdk_msg, AssistantMessage):
            msg_id = sdk_msg.message_id
            if msg_id is not None and msg_id.startswith("gen-"):
                state.generations.add(msg_id)
            # Track the model the SDK actually used (the CLI picks one
            # when none was requested, as on a subscription).  Consumed
            # by the Moonshot cost-override decision and the turn's
            # model stamps.
            observed = getattr(sdk_msg, "model", None)
            if (
                isinstance(observed, str)
                and observed
                and observed != _CLI_SYNTHETIC_MODEL
            ):
                state.observed_model = observed
            if ctx.turn_summary is not None:
                ctx.turn_summary.observe_assistant(sdk_msg)

        # Live spend and step count: the governor lands the turn when either
        # nears its limit (see ``security_hooks`` PreToolUse). It reads the
        # assistant messages and the raw stream, where each response's real
        # output count arrives. Metering must never fail the turn it meters;
        # the CLI's own budget and turn limits still back it up.
        if state.governor is not None:
            try:
                state.governor.observe(sdk_msg)
            except Exception:
                logger.warning(
                    "%s Turn governor could not meter a message",
                    ctx.log_prefix,
                    exc_info=True,
                )

        if loop_state.provider_refused:
            if isinstance(sdk_msg, ResultMessage):
                _record_result_usage(sdk_msg, state, ctx.log_prefix)
                break
            continue

        # Checked before the message reaches the adapter, which would
        # otherwise stream the provider's "buy more credits" text as the reply.
        refusal = _platform_out_of_credits_refusal(sdk_msg, ctx)
        if refusal is not None:
            report_provider_out_of_credits(
                provider=_route_name(ctx),
                model=state.observed_model or getattr(state.options, "model", None),
                surface="copilot_sdk",
                error=refusal,
                session_id=ctx.session_id,
            )
            # Not yielded here: the consumer stops at the first StreamError and
            # closes this generator, so the outer loop yields it only after
            # the history marker carrying this message is written.
            loop_state.stream_error_msg = PROVIDER_UNAVAILABLE_MESSAGE
            loop_state.stream_error_code = PROVIDER_UNAVAILABLE_CODE
            loop_state.ended_with_stream_error = True
            if isinstance(sdk_msg, ResultMessage):
                _record_result_usage(sdk_msg, state, ctx.log_prefix)
                break
            loop_state.provider_refused = True
            continue

        # Log AssistantMessage API errors (e.g. invalid_request)
        # so we can debug Anthropic API 400s surfaced by the CLI.
        sdk_error = getattr(sdk_msg, "error", None)
        if isinstance(sdk_msg, AssistantMessage) and sdk_error:
            error_text = str(sdk_error)
            error_preview = str(sdk_msg.content)[:500]
            logger.error(
                "[SDK] [%s] AssistantMessage has error=%s, "
                "content_blocks=%d, content_preview=%s",
                ctx.session_id[:12],
                sdk_error,
                len(sdk_msg.content),
                error_preview,
            )

            # Intercept prompt-too-long errors surfaced as
            # AssistantMessage.error (not as a Python exception).
            # Re-raise so the outer retry loop can compact the
            # transcript and retry with reduced context.
            # Check both error_text and error_preview: sdk_error
            # being set confirms this is an error message (not user
            # content), so checking content is safe. The actual
            # error description (e.g. "Prompt is too long") may be
            # in the content, not the error type field
            # (e.g. error="invalid_request", content="Prompt is
            # too long").
            if _is_prompt_too_long(Exception(error_text)) or _is_prompt_too_long(
                Exception(error_preview)
            ):
                logger.warning(
                    "%s Prompt-too-long detected via AssistantMessage "
                    "error — raising for retry",
                    ctx.log_prefix,
                )
                raise RuntimeError("Prompt is too long")

            # Intercept transient API errors (5xx, overloaded, 429,
            # socket closed, ECONNRESET) — replace the raw message with
            # a user-friendly error text and use the retryable error
            # prefix so the frontend shows a retry button. The error
            # field carries the CLI's category; the content its text.
            if is_transient_assistant_error(category=error_text, content=error_preview):
                logger.warning(
                    "%s Transient Anthropic API error detected, "
                    "suppressing raw error text",
                    ctx.log_prefix,
                )
                loop_state.stream_error_msg = FRIENDLY_TRANSIENT_MSG
                loop_state.stream_error_code = "transient_api_error"
                # Do NOT yield StreamError or append error marker here.
                # The outer retry loop decides: if a retry is available it
                # yields StreamStatus("retrying…"); if retries are exhausted
                # it appends the marker and yields StreamError exactly once.
                # Yielding StreamError before the retry decision causes the
                # client to display an error that is immediately superseded.
                loop_state.ended_with_stream_error = True
                break

        # Determine if the message is a tool-only batch (all content
        # items are ToolUseBlocks) — such messages have no text output yet,
        # so we skip the wait_for_stash flush below.
        #
        # Note: parallel execution of tools is handled natively by the
        # SDK CLI via readOnlyHint annotations on tool definitions.
        is_tool_only = False
        if isinstance(sdk_msg, AssistantMessage) and sdk_msg.content:
            is_tool_only = all(
                isinstance(item, ToolUseBlock) for item in sdk_msg.content
            )

        # Race-condition fix: SDK hooks (PostToolUse) are
        # executed asynchronously via start_soon() — the next
        # message can arrive before the hook stashes output.
        # wait_for_stash() awaits an asyncio.Event signaled by
        # stash_pending_tool_output(), completing as soon as
        # the hook finishes (typically <1ms).  The sleep(0)
        # after lets any remaining concurrent hooks complete.
        #
        # Skip for parallel tool continuations: when the SDK
        # sends parallel tool calls as separate
        # AssistantMessages (each containing only
        # ToolUseBlocks), we must NOT wait/flush — the prior
        # tools are still executing concurrently.
        if (
            state.adapter.has_unresolved_tool_calls
            and isinstance(sdk_msg, (AssistantMessage, ResultMessage))
            and not is_tool_only
        ):
            if await wait_for_stash():
                await asyncio.sleep(0)
            else:
                logger.warning(
                    "%s Timed out waiting for PostToolUse "
                    "hook stash (%d unresolved tool calls)",
                    ctx.log_prefix,
                    len(state.adapter.current_tool_calls)
                    - len(state.adapter.resolved_tool_calls),
                )

        # Log ResultMessage details and capture token usage
        if isinstance(sdk_msg, ResultMessage):
            logger.info(
                "%s Received: ResultMessage %s "
                "(unresolved=%d, current=%d, resolved=%d, "
                "num_turns=%d, cost_usd=%s, result=%s)",
                ctx.log_prefix,
                sdk_msg.subtype,
                len(state.adapter.current_tool_calls)
                - len(state.adapter.resolved_tool_calls),
                len(state.adapter.current_tool_calls),
                len(state.adapter.resolved_tool_calls),
                sdk_msg.num_turns,
                sdk_msg.total_cost_usd,
                (sdk_msg.result or "")[:200],
            )
            if sdk_msg.subtype in (
                "error",
                "error_during_execution",
            ):
                logger.error(
                    "%s SDK execution failed with error: %s",
                    ctx.log_prefix,
                    sdk_msg.result or "(no error message provided)",
                )

            # Check for prompt-too-long regardless of subtype — the
            # SDK may return subtype="success" with result="Prompt is
            # too long" when the CLI rejects the prompt before calling
            # the API (cost_usd=0, no tokens consumed).  If we only
            # check the "error" subtype path, the stream appears to
            # complete normally, the synthetic error text is stored
            # in the transcript, and the session grows without bound.
            if _is_prompt_too_long(RuntimeError(sdk_msg.result or "")):
                # A later API call can fail after earlier calls incurred spend.
                # Preflight rejections report zero even on a resumed session.
                if sdk_msg.total_cost_usd:
                    _record_result_usage(sdk_msg, state, ctx.log_prefix)
                raise RuntimeError("Prompt is too long")

            _record_result_usage(sdk_msg, state, ctx.log_prefix)

        # Emit compaction end if SDK finished compacting.
        # Sync TranscriptBuilder with the CLI's active context.
        #
        # ``status`` (and ``init``) system messages keep arriving while the
        # CLI is still writing the summary — closing on those read the
        # session file too early: a false "no readable summary" alarm on
        # the first cycle, the previous cycle's summary counted as the
        # second's.  A cycle closes on the CLI's own ``compact_boundary``
        # or on the next non-system message; the ResultMessage closes it
        # unconditionally so nothing leaks across turns.
        closes_cycle = not isinstance(sdk_msg, SystemMessage) or (
            sdk_msg.subtype == "compact_boundary"
        )
        if closes_cycle:
            measured, compacted, end_stats, after_source = (
                await _measure_sdk_compaction(
                    ctx, state, final=isinstance(sdk_msg, ResultMessage)
                )
            )
        else:
            measured, compacted, end_stats, after_source = False, None, None, None
        # Close the row only with the measurement in hand: a pending cycle
        # that is still waiting for its summary must stay open.
        compact_result = (
            await ctx.compaction.emit_end_if_ready(
                ctx.session, end_stats, after_source=after_source
            )
            if measured
            else CompactionResult()
        )
        if compact_result.events:
            # Compaction events end with StreamFinishStep; open blocks must
            # close before it (see ``SDKResponseAdapter.end_open_blocks``).
            pre_close: list[StreamBaseResponse] = []
            state.adapter.end_open_blocks(pre_close)
            # Compaction events bypass the adapter, so sync step state
            # when a StreamFinishStep is present — otherwise the adapter
            # will skip StreamStartStep on the next AssistantMessage.
            if any(isinstance(ev, StreamFinishStep) for ev in compact_result.events):
                state.adapter.step_open = False
            for r in pre_close:
                yield r
        for ev in compact_result.events:
            yield ev
        entries_replaced = False
        if compact_result.just_ended:
            if not measured and compact_result.transcript_path:
                # The hook landed in the one yield between the measurement
                # and the close: the row went out without numbers, but the
                # builder must still mirror what the CLI kept.
                compacted = await asyncio.to_thread(
                    read_compacted_entries, compact_result.transcript_path
                )
            if compacted is not None:
                state.transcript_builder.replace_entries(
                    compacted, log_prefix=ctx.log_prefix
                )
                entries_replaced = True

        # --- Hard circuit breaker for empty tool calls ---
        breaker = _check_empty_tool_breaker(
            sdk_msg, loop_state.consecutive_empty_tool_calls, ctx, state
        )
        loop_state.consecutive_empty_tool_calls = breaker.count
        if breaker.tripped and breaker.error is not None:
            loop_state.stream_error_msg = breaker.error_msg
            loop_state.stream_error_code = breaker.error_code
            yield breaker.error
            loop_state.ended_with_stream_error = True
            break

        # --- Dispatch adapter responses ---
        adapter_responses = state.adapter.convert_message(sdk_msg)

        # Pre-create the new assistant message in the session BEFORE
        # yielding any events so it survives a GeneratorExit (client
        # disconnect) that interrupts the yield loop at StreamStartStep.
        #
        # Without this, the sequence is:
        #   tool result saved → intermediate flush → StreamStartStep
        #   yield → GeneratorExit → finally saves session with
        #   last_role=tool (the text response was generated but never
        #   appended because _dispatch_response(StreamTextDelta) was
        #   skipped).
        #
        # We only pre-create when:
        #   1. Tool results were received this turn (has_tool_results).
        #   2. The prior assistant message is already appended
        #      (has_appended_assistant) — so this is a post-tool turn.
        #   3. This batch contains StreamTextDelta — text IS coming, so
        #      we won't leave a spurious empty message for tool-only turns.
        #
        # Subsequent StreamTextDelta dispatches accumulate content into
        # acc.assistant_response in-place (ChatMessage is mutable), so
        # the DB record is updated without a second append.
        if (
            acc.has_tool_results
            and acc.has_appended_assistant
            and any(isinstance(r, StreamTextDelta) for r in adapter_responses)
        ):
            acc.assistant_response = ChatMessage(role="assistant", content="")
            acc.accumulated_tool_calls = []
            acc.has_tool_results = False
            ctx.session.messages.append(acc.assistant_response)
            # acc.has_appended_assistant stays True — placeholder is live

        # When StreamFinish is in this batch (ResultMessage), flush any
        # text buffered by the thinking stripper and inject it as a
        # StreamTextDelta BEFORE the StreamTextEnd so the Vercel AI SDK
        # receives the tail inside the still-open text block (correct
        # protocol order: TextDelta → TextEnd → FinishStep → Finish).
        tail_delta: StreamTextDelta | None = None
        if any(isinstance(r, StreamFinish) for r in adapter_responses):
            tail = acc.thinking_stripper.flush()
            if tail and not loop_state.ended_with_stream_error:
                # Do NOT manually append tail to acc.assistant_response.content
                # here — _dispatch_response handles that.  Doing it here would
                # double-append because _dispatch_response also updates the
                # accumulator.  Instead, mark the delta as pre-stripped so
                # _dispatch_response bypasses ThinkingStripper.process() for it
                # (re-processing could suppress a tail that looks like a partial
                # tag opener, e.g. "Hello <inter" → buffered again → lost).
                tail_delta = StreamTextDelta(id=state.adapter.text_block_id, delta=tail)
                insert_at = next(
                    (
                        i
                        for i, r in enumerate(adapter_responses)
                        if isinstance(r, (StreamTextEnd, StreamFinish))
                    ),
                    len(adapter_responses),
                )
                adapter_responses.insert(insert_at, tail_delta)
        for response in adapter_responses:
            dispatched = _dispatch_response(
                response,
                acc,
                ctx,
                state,
                entries_replaced,
                ctx.log_prefix,
                skip_strip=response is tail_delta,
            )
            if dispatched is not None:
                # The envelope goes out just ahead of the error it explains,
                # so a client acting on it has it in hand before the turn is
                # reported failed.
                if isinstance(dispatched, StreamError):
                    codex_failure = _provider_failure_for(ctx)
                    if codex_failure is not None:
                        yield StreamProviderFailure(failure=codex_failure.as_part())
                yield dispatched

            # Mid-turn follow-up persistence: the MCP tool wrapper drains
            # the primary pending buffer and stashes the drained
            # PendingMessages into the per-session persist queue.  Claude
            # has already seen them via the <user_follow_up> block
            # injected into the tool output.  Now — right after the
            # tool_result row has been appended to session.messages — we
            # pop the persist queue and append a real user ChatMessage
            # so the UI renders a proper user bubble in the correct
            # chronological position (after the tool_result, before the
            # assistant's continuing response).  Rollback re-queues into
            # the PRIMARY pending buffer so the next turn-start drain
            # picks them up if this persist silently fails.
            # Only run the follow-up persist if the tool_result row was
            # actually appended by _dispatch_response (currently always
            # true for this variant, but we guard so a future refactor
            # that conditionally skips the append can't silently land
            # a user row before a missing tool_result).
            if (
                isinstance(response, StreamToolOutputAvailable)
                and dispatched is not None
                and acc.has_tool_results
            ):
                followup_drained = await drain_pending_for_persist(
                    ctx.session.session_id
                )
                if followup_drained and await persist_pending_as_user_rows(
                    ctx.session,
                    state.transcript_builder,
                    followup_drained,
                    log_prefix=ctx.log_prefix,
                ):
                    yield drained_rows_entry(followup_drained)

        # Append assistant entry AFTER convert_message so that
        # any stashed tool results from the previous turn are
        # recorded first, preserving the required API order:
        # assistant(tool_use) → tool_result → assistant(text).
        # Skip if replace_entries just ran — the CLI session
        # file already contains this message.
        if isinstance(sdk_msg, AssistantMessage) and not entries_replaced:
            state.transcript_builder.append_assistant(
                content_blocks=_format_sdk_content_blocks(sdk_msg.content),
                model=sdk_msg.model,
            )

        # --- Intermediate persistence ---
        # Flush session messages to DB periodically so page reloads
        # show progress during long-running turns.
        #
        # IMPORTANT: Skip the flush while tool calls are pending
        # (tool_calls set on assistant but results not yet received).
        # The DB save is append-only (uses start_sequence), so if we
        # flush the assistant message before tool_calls are set on it
        # (text and tool_use arrive as separate SDK events), the
        # tool_calls update is lost — the next flush starts past it.
        #
        # With ``include_partial_messages=True`` the CLI delivers
        # hundreds of ``StreamEvent`` messages per turn — incrementing
        # ``loop_state.msgs_since_flush`` on each one trips the threshold long
        # before the assistant text is complete, saving a truncated
        # prefix that subsequent deltas can never extend (append-only).
        # Count only messages that produce a persisted row boundary
        # (AssistantMessage, UserMessage, ResultMessage) and skip
        # raw StreamEvents.  Also skip when text or reasoning is
        # still in-flight on the adapter: the row is live and a flush
        # would lock it at its current length.
        if not isinstance(sdk_msg, StreamEvent):
            loop_state.msgs_since_flush += 1
        now = time.monotonic()
        if not _intermediate_flush_blocked(acc, state.adapter) and (
            loop_state.msgs_since_flush >= _FLUSH_MESSAGE_THRESHOLD
            or (now - loop_state.last_flush_time) >= _FLUSH_INTERVAL_SECONDS
        ):
            checkpoint = None
            try:
                await asyncio.shield(upsert_chat_session(ctx.session))
                logger.debug(
                    "%s Intermediate flush: %d messages (msgs_since=%d, elapsed=%.1fs)",
                    ctx.log_prefix,
                    len(ctx.session.messages),
                    loop_state.msgs_since_flush,
                    now - loop_state.last_flush_time,
                )
                checkpoint = turn_checkpoint(ctx.session.messages, ctx.turn_start)
            except Exception as flush_err:
                logger.warning(
                    "%s Intermediate flush failed: %s",
                    ctx.log_prefix,
                    flush_err,
                )
            loop_state.last_flush_time = now
            loop_state.msgs_since_flush = 0
            if checkpoint is not None:
                yield checkpoint

        # --- Building-mode switch (enter_agent_building_mode) ---
        # Restart the attempt with the guide in the system prompt.
        if _ready_for_building_mode_restart(ctx.session, state.adapter):
            logger.info(
                f"{ctx.log_prefix} Building mode requested — interrupting "
                f"for prompt upgrade"
            )
            try:
                await client.interrupt()
            except Exception as exc:
                # Best-effort: client close in the caller's finally is the
                # fallback shutdown path (same as transient retries).
                logger.warning(
                    f"{ctx.log_prefix} Interrupt for building-mode restart "
                    f"failed: {exc}"
                )
            raise _BuildingModeRestart()

        if acc.stream_completed:
            break


# Continuation query sent when relaunching an attempt after the
# building-mode switch — the CLI session already holds the original user
# message and all partial work, so this only needs to orient the model.
_BUILDING_MODE_CONTINUATION = (
    "Building mode is now active — the complete agent-building guide is in "
    "your system prompt (<building_guide>) and survives context compaction. "
    "Continue working on the user's request from where you left off."
)

# Sent instead when the guide could not be loaded, so the model is never told
# a <building_guide> block is present that is not. The building-mode gates stay
# closed, which is correct — the guide really is absent.
_BUILDING_MODE_UNAVAILABLE_CONTINUATION = (
    "The agent-building guide could not be loaded into your system prompt. "
    "Continue working on the user's request from where you left off, and do "
    "not retry enter_agent_building_mode in this turn."
)

# Synthetic message injected when a turn ends with extended thinking but no
# visible TextBlock. Bounded to one re-prompt per turn — if the model still
# returns thinking-only the adapter promotes the last thinking block to
# visible text rather than calling another round.
_THINKING_ONLY_REPROMPT = (
    "Please write a brief user-facing summary of what you found, in plain "
    "prose. Do not use tools."
)

# Intermediate-flush thresholds for the SDK consume loop — periodic
# session-message flush so page reloads show progress on long turns.
_FLUSH_INTERVAL_SECONDS = 30.0
_FLUSH_MESSAGE_THRESHOLD = 10


def _intermediate_flush_blocked(
    acc: "_StreamAccumulator", adapter: SDKResponseAdapter
) -> bool:
    """Whether the periodic mid-turn session flush must be held off.

    The DB save is append-only (uses ``start_sequence``), so flushing mid-turn
    locks the trailing rows at their current sequence permanently. Hold the
    flush while any of these is true:

    - ``has_pending_tools`` — a tool call is dispatched but its result hasn't
      been recorded yet; the ``assistant(tool_calls)`` and ``tool`` result rows
      must persist together.
    - ``has_unsealed_assistant`` — a live assistant row hasn't acquired
      ``tool_calls`` yet. The model often emits a text preamble, pauses, then
      calls a tool — text and ``tool_use`` arrive as *separate* AssistantMessage
      events, and ``StreamToolInputAvailable`` attaches the ``tool_calls`` onto
      this same ``acc.assistant_response`` object. Flushing now would lock the
      row without ``tool_calls``; the later attach could never reach the DB,
      orphaning the ``tool`` result (no call row → the tool card vanishes on
      reload).
    - ``has_open_block`` — text/reasoning is still streaming, so the row would
      be locked at a truncated length.

    The guaranteed turn-end persist (the ``finally`` upsert) always saves the
    rows, so holding an intermediate flush never loses data.
    """
    has_pending_tools = (
        acc.has_appended_assistant
        and bool(acc.accumulated_tool_calls)
        and not acc.has_tool_results
    )
    has_unsealed_assistant = (
        acc.has_appended_assistant and not acc.accumulated_tool_calls
    )
    return has_pending_tools or has_unsealed_assistant or _has_open_block(adapter)


def _has_open_block(adapter: SDKResponseAdapter) -> bool:
    return (adapter.has_started_text and not adapter.has_ended_text) or (
        adapter.has_started_reasoning and not adapter.has_ended_reasoning
    )


def _hidden_short_names_for_permissions(
    permissions: CopilotPermissions | None,
) -> frozenset[str]:
    """Short tool names that should not be registered on the MCP server.

    ``allowed_tools``/``disallowed_tools`` only gate execution — denied
    calls still come back to the model with the CLI's canned "Permission
    to use ... has been denied" string, which the model then narrates as a
    Claude-Code-style approval prompt that doesn't exist in copilot.
    Hiding the tool from the MCP server removes it from the model's tool
    list entirely so it never reaches for the blocked name.
    """
    return denied_tool_names(permissions)


def _strip_synthetic_reprompt_from_cli_jsonl(content: bytes) -> bytes:
    """Drop the synthetic re-prompt user message AND its preceding empty
    thinking-only AssistantMessage from the CLI session JSONL.

    The CLI persists every ``client.query(...)`` call to its session file,
    including the synthetic re-prompt we send when a turn ends thinking-only.
    Leaving it in the uploaded JSONL pollutes ``--resume`` history on the next
    turn — the model would see a phantom user message asking it to summarise.

    We must also drop the empty thinking-only AssistantMessage that came
    immediately *before* the synthetic user message, otherwise the JSONL
    ends up with two AssistantMessage entries back-to-back (the empty one
    and the actual closing turn) without a user message between them —
    which violates Anthropic's role-alternation contract on resume.
    """
    if not content:
        return content
    parsed: list[tuple[bytes, dict | None]] = []
    for line in content.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped:
            parsed.append((line, None))
            continue
        try:
            entry = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            parsed.append((line, None))
            continue
        parsed.append((line, entry if isinstance(entry, dict) else None))

    drop: set[int] = set()
    for i, (_line, entry) in enumerate(parsed):
        if not _is_synthetic_reprompt_user_entry(entry):
            continue
        drop.add(i)
        # Walk backwards over blank / non-entry lines to the most recent
        # JSONL entry; if it's an empty-content AssistantMessage, drop it
        # too so the post-strip role alternation stays valid.
        j = i - 1
        while j >= 0 and parsed[j][1] is None:
            j -= 1
        if j >= 0 and _is_empty_assistant_entry(parsed[j][1]):
            drop.add(j)
    return b"".join(
        line for idx, (line, _entry) in enumerate(parsed) if idx not in drop
    )


def _is_synthetic_reprompt_user_entry(entry: dict | None) -> bool:
    if not entry or entry.get("type") != "user":
        return False
    msg = entry.get("message")
    if not isinstance(msg, dict) or msg.get("role") != "user":
        return False
    return _extract_user_message_text(msg.get("content")) == _THINKING_ONLY_REPROMPT


def _is_empty_assistant_entry(entry: dict | None) -> bool:
    """True for an AssistantMessage whose visible content is empty (no
    TextBlock / ToolUseBlock / non-empty text).  ThinkingBlocks alone count
    as empty for role-alternation purposes — the model emitted nothing the
    next turn would treat as an answer."""
    if not entry or entry.get("type") != "assistant":
        return False
    msg = entry.get("message")
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    for block in content:
        if not isinstance(block, dict):
            return False
        btype = block.get("type")
        # Both ``thinking`` and ``redacted_thinking`` (Anthropic's encrypted
        # thinking variant for safety-redacted content) count as empty for
        # role-alternation purposes — neither carries a user-visible answer.
        if btype in ("thinking", "redacted_thinking"):
            continue
        if btype == "text" and not (block.get("text") or "").strip():
            continue
        return False
    return True


def _extract_user_message_text(content: object) -> str | None:
    """Return the plain-text payload of a user message, or None if not text-only."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                return None
            if block.get("type") != "text":
                return None
            text = block.get("text")
            if not isinstance(text, str):
                return None
            texts.append(text)
        return "".join(texts) if texts else None
    return None


def _idle_timeout_threshold(adapter: SDKResponseAdapter) -> int:
    """Pick the idle-timeout threshold for the current heartbeat.

    Returns ``_HUNG_TOOL_CAP_SECONDS`` (longer) whenever any tool call is
    still pending, so a legitimately long operation isn't killed. Returns
    ``_IDLE_TIMEOUT_SECONDS`` (shorter) when nothing is pending — the SDK
    itself is idle with no work in flight.
    """
    if adapter.has_unresolved_tool_calls:
        return _HUNG_TOOL_CAP_SECONDS
    return _IDLE_TIMEOUT_SECONDS


# StreamError codes that should render as a retryable error in the UI (retry
# button) rather than a terminal ErrorCard. Codes appended via
# ``_append_error_marker`` directly already pass ``retryable=True``; this set
# covers the codes that flow through the adapter -> ``_dispatch_response``.
_RETRYABLE_STREAM_ERROR_CODES: frozenset[str] = frozenset(
    {"transient_api_error", "empty_completion"}
)


# Limits the platform sets on purpose: the turn spent its USD budget or made its
# maximum number of LLM calls, and the user is told how to carry on. They are
# outcomes, not failures, so they log at info: Sentry's LoggingIntegration files
# every ERROR record as an issue (AUTOGPT-SERVER-8C4).
_EXPECTED_STOP_CODES: frozenset[str] = frozenset(
    {"max_budget_exhausted", "max_turns_exhausted"}
)


# Handled-error codes whose StreamError the outer retry loop yields itself,
# after it has persisted the history marker, rather than the attempt.
_OUTER_LOOP_YIELDS_ERROR_CODES: frozenset[str] = frozenset(
    {"transient_api_error", PROVIDER_UNAVAILABLE_CODE}
)


# Event types that are ephemeral / cosmetic and must NOT be counted toward
# ``events_yielded`` in the transient-retry loop.  Counting them would prevent
# the backoff retry from firing because ``_next_transient_backoff`` returns
# ``None`` when ``events_yielded > 0``.
#
# Tool events are deliberately absent: a tool row, a call the model made or
# the CLI's own compaction, is output the retry cannot take back.  The retry
# rolls the attempt's rows out of the session, so a row the user watched would
# vanish on refresh, and the resent query can run a side-effecting tool (an
# email, an agent run) a second time.  Not retrying keeps every row that
# closed.  A compaction the failure cuts short is the exception: its row was
# opened but is never closed or persisted, and the frontend hides a row left
# unfinished.  Pre-query compaction rows are yielded by the retry loop itself,
# outside the attempt, so they never count.
_EPHEMERAL_EVENT_TYPES = (
    StreamHeartbeat,
    StreamCheckpoint,
    StreamPendingDrained,
    StreamStartStep,
    StreamFinishStep,
    # A compaction row's progress phases ride along with its tool events,
    # which count on their own.
    StreamCompactionProgress,
    # Transient StreamError and StreamStatus are ephemeral notifications,
    # not content.  Counting them would prevent the backoff retry from
    # firing because _next_transient_backoff() returns None when
    # events_yielded > 0.
    StreamError,
    StreamStatus,
)

# Patterns that indicate the prompt/request exceeds the model's context limit.
# Matched case-insensitively against the full exception chain.
_PROMPT_TOO_LONG_PATTERNS: tuple[str, ...] = (
    "prompt is too long",
    "request too large",
    "maximum context length",
    "context_length_exceeded",
    "input tokens exceed",
    "input is too long",
    "content length exceeds",
)


# Map raw SDK error patterns to user-friendly messages.
# Matched case-insensitively; first match wins.
_FRIENDLY_ERROR_MAP: tuple[tuple[str, str], ...] = (
    ("authentication", "Authentication failed. Please check your API key."),
    ("invalid api key", "Authentication failed. Please check your API key."),
    ("unauthorized", "Authentication failed. Please check your API key."),
    ("rate limit", "Rate limit exceeded. Please wait a moment and try again."),
    ("overloaded", "The AI service is currently overloaded. Please try again shortly."),
    ("server error", "The AI service encountered an internal error. Please retry."),
    ("timeout", "The request timed out. Please try again."),
    ("connection", "Connection error. Please check your network and retry."),
)


def _friendly_error_text(raw: str) -> str:
    """Map a raw SDK error string to a user-friendly message.

    Returns the mapped message if a known pattern is found, otherwise
    returns a generic sanitized version of the raw error.
    """
    lower = raw.lower()
    for pattern, friendly in _FRIENDLY_ERROR_MAP:
        if pattern in lower:
            return friendly
    # Fallback: sanitize but keep the original text for debugging
    return f"SDK stream error: {raw}"


async def _record_unreported_usage(
    state: "_RetryState", sdk_cwd: str, session_id: str | None, log_prefix: str
) -> None:
    """Charge what the launch's CLI spent but never reported in a result,
    read back from its session file once the CLI has exited."""
    snapshot = await asyncio.to_thread(
        read_cli_session_usage,
        sdk_cwd,
        state.options.resume or state.options.session_id or session_id,
        log_prefix,
    )
    state.usage.record_unreported(
        snapshot,
        state.observed_model or getattr(state.options, "model", None),
        log_prefix,
    )


def _record_result_usage(
    sdk_msg: ResultMessage, state: "_RetryState", log_prefix: str
) -> None:
    """Record this query using the model it actually ran on."""
    state.usage.record_result(
        sdk_msg.usage,
        sdk_msg.total_cost_usd,
        state.observed_model or getattr(state.options, "model", None),
        log_prefix,
    )
    if sdk_msg.usage:
        logger.info(
            "%s Token usage: uncached=%d, cache_read=%d, cache_create=%d, output=%d",
            log_prefix,
            state.usage.prompt_tokens,
            state.usage.cache_read_tokens,
            state.usage.cache_creation_tokens,
            state.usage.completion_tokens,
        )


def _platform_out_of_credits_refusal(
    sdk_msg: object, ctx: "_StreamContext"
) -> str | None:
    """The CLI's text for a billing refusal on the platform's own account.

    The CLI reports a provider error as an ``AssistantMessage`` carrying
    ``error`` (content is the provider's wording) and/or an error
    ``ResultMessage``. A turn on the user's own subscription (Codex, a
    Claude login) has a limit that is theirs to hear about, so it is left
    to the route's own envelope.
    """
    if ctx.profile is not None and not ctx.profile.platform_pays:
        return None
    if isinstance(sdk_msg, AssistantMessage) and sdk_msg.error:
        text = f"{sdk_msg.error} {sdk_msg.content}"
        if sdk_msg.error == "billing_error":
            return text
    elif isinstance(sdk_msg, ResultMessage) and (
        sdk_msg.is_error or sdk_msg.subtype in ("error", "error_during_execution")
    ):
        text = str(sdk_msg.result or "")
    else:
        return None
    return text if is_provider_out_of_credits(text) else None


def _context_limit_without_compaction(
    attempt: int, compaction: "CompactionTracker"
) -> str | None:
    """Why a first-attempt context-limit error is alarming, or None.

    The CLI is meant to compact well before the ceiling.  Reaching it on
    the first attempt with nothing landed means one of two things, and the
    message says which: the trigger fired but no summary was ever written
    (something answered its compaction request wrongly), or it never fired
    at all (the pin sits past the model's real window, or the threshold is
    above the ceiling).  Later attempts are our own retries and expected.
    """
    if attempt != 0 or compaction.landed_count > 0:
        return None
    if compaction.attempt_count > 0:
        return (
            f"context limit reached after {compaction.attempt_count} compaction "
            f"attempt(s) and none landed "
            f"({', '.join(compaction.attempt_sources)})"
        )
    return "context limit reached before any compaction was attempted"


def _is_prompt_too_long(err: BaseException) -> bool:
    """Return True if *err* indicates the prompt exceeds the model's limit.

    Walks the exception chain (`__cause__` / `__context__`) so that
    wrapped errors are detected too.
    """
    seen: set[int] = set()
    current: BaseException | None = err
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        msg = str(current).lower()
        if any(p in msg for p in _PROMPT_TOO_LONG_PATTERNS):
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_sdk_disconnect_error(exc: BaseException) -> bool:
    """Return True if *exc* is an expected SDK cleanup error from client disconnect.

    Two known patterns occur when ``GeneratorExit`` tears down the async
    generator and the SDK's ``__aexit__`` runs in a different context/task:

    * ``RuntimeError``: cancel scope exited in wrong task (anyio)
    * ``ValueError``: ContextVar token created in a different Context (OTEL)

    These are suppressed to avoid polluting Sentry with non-actionable noise.
    """
    if isinstance(exc, RuntimeError) and "cancel scope" in str(exc):
        return True
    if isinstance(exc, ValueError) and "was created in a different Context" in str(exc):
        return True
    return False


class ReducedContext(NamedTuple):
    builder: TranscriptBuilder
    transcript_lost: bool
    # Token budget for the retry's history compression, from the pinned
    # window: ``compaction_target_tokens`` when the transcript compaction
    # landed, else ``retry_target_tokens``, smaller each retry so
    # compress_context applies progressively more aggressive reduction
    # (LLM summarize → truncate → middle-out delete → trim).  None only when
    # built outside ``_reduce_context``.
    target_tokens: int | None = None


@dataclass
class _RetryState:
    """Mutable state passed to `_run_stream_attempt` instead of closures.

    Holds values that the retry loop modifies between attempts so the inner
    generator does not rely on reassigning closure variables.
    """

    options: ClaudeAgentOptions
    query_message: str
    compaction_stats: "CompactionStats | None"
    use_resume: bool
    transcript_msg_count: int
    adapter: SDKResponseAdapter
    transcript_builder: TranscriptBuilder
    usage: _TokenUsage
    # Token budget for history compression on retries (DB-message fallback path).
    # None = model-aware default.  Halved each retry for progressively more
    # aggressive compression (LLM summarize → truncate → middle-out → trim).
    target_tokens: int | None = None
    # Tracks whether the thinking-only-final-turn re-prompt has already
    # fired for this user turn.  Lives on ``_RetryState`` (not on
    # ``adapter``) so a transient retry that rebuilds the adapter does
    # not reset the per-turn cap to zero — otherwise multiple retries
    # could each fire their own re-prompt round.
    thinking_only_reprompted: bool = False
    # OpenRouter generations this turn is billed for, collected launch by
    # launch.  Populated from ``AssistantMessage.message_id`` when routed via
    # OpenRouter (``gen-...`` prefix).  Consumed by the finally block to
    # fire ``record_turn_cost_from_openrouter`` for non-Anthropic models —
    # the CLI's static-Anthropic-priced estimate is replaced with the
    # authoritative ``/generation`` total_cost.  Lives on ``_RetryState``
    # (not per-attempt ``_StreamAccumulator``) so it survives retries; a
    # transient retry absorbs the failed launch's (``_do_transient_backoff``).
    generations: TurnGenerations = dataclass_field(default_factory=TurnGenerations)
    # The *actually executed* model observed on ``AssistantMessage.model`` —
    # set even when ``state.options.model`` is unset (the CLI picked one).
    # The Moonshot cost override gates on this.
    observed_model: str | None = None
    # The turn's spend and step governor. On ``_RetryState`` so its meter
    # keeps what earlier attempts spent, and so every rebuilt adapter can be
    # handed the same one (see ``_fresh_adapter``).
    governor: TurnGovernor | None = None


def _fresh_adapter(state: _RetryState, message_id: str, session_id: str) -> None:
    """Replace the adapter for a new attempt, keeping the turn's governor."""
    state.adapter = SDKResponseAdapter(message_id=message_id, session_id=session_id)
    state.adapter.governor = state.governor


@dataclass
class _StreamContext:
    """Per-request variables shared across all retry attempts.

    Extracted so that `_run_stream_attempt` can be a module-level function
    rather than a closure, making it independently testable and reducing the
    cognitive load of the 970-line `stream_chat_completion_sdk` function.

    Scalar fields (IDs, paths, the message string) are set once before the
    retry loop and never reassigned.  `session`, `compaction`, and
    `lock` are **shared mutable references** whose internal state is
    modified by both the retry loop and `_run_stream_attempt` (e.g.
    `session.messages` is rolled back on retry, `compaction` tracks
    mid-stream compaction events).  Their *references* are constant even
    though the objects they point to are mutated.
    """

    session: ChatSession
    session_id: str
    log_prefix: str
    sdk_cwd: str
    current_message: str
    file_ids: list[str] | None
    message_id: str
    attachments: "PreparedAttachments"
    compaction: CompactionTracker
    lock: AsyncClusterLock
    # The shared model gateway this turn launches behind (Codex, the local
    # route). Carried here so the error path can ask it what actually
    # failed: by the time a provider failure reaches this layer it is CLI
    # text, and the gateway holds the last point at which it was typed.
    route_gateway: "ModelGateway | None" = None
    # The connected account the turn runs on, which a failure names.
    credential_id: str | None = None
    # The route this turn runs on (``copilot/route_profile.py``).
    profile: RouteProfile | None = None
    tool_display: SDKToolDisplayBridge | None = None
    # Index in ``session.messages`` of the stream's first row, for checkpoints.
    turn_start: int = 0
    turn_summary: TurnSummaryCollector | None = None
    # The turn's envelope is tainted. A pause marker records it, so the turn
    # that picks the task back up starts with the same taint.
    tainted: bool = False


# Below this token budget the model context is so tight that injecting any
# conversation history would likely exceed the limit regardless of content.
# _build_query_message returns the bare message when target_tokens falls to
# or below this floor, giving the user a response instead of a hard error.
# The retry budgets end on this value on purpose (``retry_target_tokens``).
_BARE_MESSAGE_TOKEN_FLOOR: int = BARE_MESSAGE_TOKEN_FLOOR


def _tail_budget_tokens(target_tokens: int | None, model: str) -> int:
    """Recent history kept verbatim when the copilot compresses: a quarter of
    the budget, never under 2K.  Sized in tokens, not messages, so eight 20K
    tool results can neither crowd out the summary nor be cut blind.  With no
    explicit target the budget is the same model default ``compress_context``
    itself falls back to, so the tail is never chosen by message count."""
    budget = (
        target_tokens if target_tokens is not None else get_compression_target(model)
    )
    return max(2_000, budget // 4)


async def _reduce_context(
    transcript_content: str,
    log_prefix: str,
    *,
    profile: RouteProfile,
    attempt: int = 1,
) -> ReducedContext:
    """Prepare reduced context for retry *attempt* (1 is the first retry).

    On the first retry, compacts the transcript via LLM summarization.
    On subsequent retries (or if compaction fails), drops the transcript
    entirely so the query is rebuilt from DB messages only.

    When no transcript is available (use_resume=False fallback path), returns
    a decreasing ``target_tokens`` budget so ``compress_context`` applies
    progressively more aggressive reduction (LLM summarize → content truncate
    → middle-out delete → first/last trim).  The budget applies in
    ``_build_query_message`` and is halved on each retry.

    ``transcript_lost`` is True when the transcript was dropped.
    """
    # The compactor LLM is fixed (``_compression_model``); every budget
    # below comes from the launch's own window and trigger, the ones whose
    # CLI autocompact threshold the retry has to land under.
    # Token budget for the DB fallback on this attempt (no-transcript path).
    budgets = profile.retry_target_tokens()
    retry_target = budgets[min(max(0, attempt - 1), len(budgets) - 1)]

    # First retry: try compacting our transcript builder state.
    # Note: the CLI native --resume file is not updated with the compacted
    # content (it would require emitting CLI-native JSONL format), so the
    # retry runs without --resume.
    if transcript_content and attempt <= 1:
        compaction_target = profile.compaction_target_tokens()
        compacted = await compact_transcript(
            transcript_content,
            model=_compression_model(),
            log_prefix=log_prefix,
            target_tokens=compaction_target,
        )
        if (
            compacted
            and compacted != transcript_content
            and validate_transcript(compacted)
        ):
            logger.info(
                "%s Using compacted transcript for retry (no --resume on this attempt)",
                log_prefix,
            )
            tb = TranscriptBuilder()
            tb.load_previous(compacted, log_prefix=log_prefix)
            # The retry still rebuilds its history from the DB rows; it is
            # held to the budget the transcript was compacted to, never to the
            # catalog's window, which can be far past the pin (Codex).
            return ReducedContext(tb, False, compaction_target)
        logger.warning("%s Compaction failed, dropping transcript", log_prefix)

    # Subsequent retry or compaction failed: drop transcript entirely.
    # Return retry_target so the caller compresses DB messages to that budget.
    logger.warning(
        "%s Dropping transcript, rebuilding from DB messages (target_tokens=%d)",
        log_prefix,
        retry_target,
    )
    return ReducedContext(TranscriptBuilder(), True, retry_target)


def _humanise_tool_list(names: list[str]) -> str:
    """Format a list of tool names for user-facing messages.

    ``["WebSearch"]``              → ``"'WebSearch'"``
    ``["WebSearch", "run_block"]`` → ``"'WebSearch' and 'run_block'"``
    Three or more items collapse to ``"'A', 'B', and 1 other"`` so the
    toast stays readable.
    """
    if not names:
        return ""
    quoted = [f"'{n}'" for n in names]
    if len(quoted) == 1:
        return quoted[0]
    if len(quoted) == 2:
        return f"{quoted[0]} and {quoted[1]}"
    extras = len(quoted) - 2
    suffix = "others" if extras > 1 else "other"
    return f"{quoted[0]}, {quoted[1]}, and {extras} {suffix}"


def _append_error_marker(
    session: ChatSession | None,
    display_msg: str,
    *,
    retryable: bool = False,
    failure: dict[str, Any] | None = None,
) -> None:
    """Append a copilot error marker to *session* so it persists across refresh.

    Delegates to the shared writer so every route produces the same row: the
    frontend's rendering contract lives in one place, and a failure recorded
    on a Codex turn carries the same envelope a platform turn would.
    """
    append_error_marker(session, display_msg, retryable=retryable, failure=failure)


def _provider_failure_for(ctx: "_StreamContext") -> ProviderFailure | None:
    """What the Codex gateway last named, if it named anything.

    The gateway is the last point where a provider failure is still a typed
    exception; downstream it is an HTTP status, then CLI text. Reading it
    here is what lets a Codex turn say "your ChatGPT login expired" rather
    than "the assistant ran into an error".

    ``None`` on the platform route, and on a Codex turn whose failure the
    gateway declined to name -- the caller keeps its existing behaviour.
    """
    return _attributed_failure(ctx.route_gateway, ctx.credential_id)


def _attributed_failure(
    gateway: ModelGateway | None, credential_id: str | None
) -> ProviderFailure | None:
    """The gateway's last failure, naming the connected account it happened
    on: the switch-connection dialog offers the user's other accounts by
    it."""
    failure = gateway.last_failure if gateway is not None else None
    if failure is None or failure.credential_id or not credential_id:
        return failure
    return failure.model_copy(update={"credential_id": credential_id})


def _is_error_marker(msg: ChatMessage) -> bool:
    """True if *msg* is an error marker emitted by ``_append_error_marker``."""
    if msg.role != "assistant" or not msg.content:
        return False
    return msg.content.startswith(COPILOT_ERROR_PREFIX) or msg.content.startswith(
        COPILOT_RETRYABLE_ERROR_PREFIX
    )


def _has_trailing_marker(session: ChatSession | None) -> bool:
    """True when the session's last message is a copilot system/error marker.

    Used to guard against double-persisting markers when multiple guards
    (StreamError dispatch, post-stream STREAM_ERROR_MARKER append, retry-loop
    ``_append_error_marker``) fire on the same turn.
    """
    if session is None or not session.messages:
        return False
    last = session.messages[-1]
    if last.role != "assistant" or not last.content:
        return False
    return (
        _is_error_marker(last)
        or last.content == STREAM_INCOMPLETE_MARKER
        or last.content == STREAM_ERROR_MARKER
    )


@dataclass
class _InterruptedAttempt:
    """Captured state of a failed SDK attempt, carried across the retry loop.

    The SDK always rolls back ``session.messages`` before deciding whether
    to retry (so attempt #2 starts clean). That rollback would otherwise
    discard everything the assistant produced — the user sees tokens stream
    live, then a refresh shows nothing. This dataclass holds the rolled-back
    messages plus the ``_HandledStreamError`` info needed to emit a final
    ``StreamError`` once the loop decides not to retry.

    The retry loop calls ``capture()`` on every failed attempt, ``clear()``
    on a successful retry (so prior rolled-back content is not replayed),
    and ``finalize()`` exactly once after the loop on final failure.
    """

    partial: list[ChatMessage] = dataclass_field(default_factory=list)
    # Populated by the ``except _HandledStreamError`` branch so the post-loop
    # block can restore the partial and (when the inner handler didn't) emit
    # the client-facing StreamError. Transient errors deliberately suppress
    # the early StreamError flash and rely on this post-loop emit.
    handled_error: "_HandledErrorInfo | None" = None

    def capture(
        self,
        session: ChatSession,
        transcript_builder: "TranscriptBuilder",
        transcript_snap: TranscriptSnapshot,
        pre_attempt_msg_count: int,
    ) -> None:
        """Roll back ``session.messages`` + transcript, keeping the partial.

        Trailing error markers appended inside ``_run_stream_attempt`` (idle
        timeout, circuit breaker) are stripped: re-attaching them would make
        the post-loop restore replay a stale marker before adding its own,
        leaving duplicate error bubbles.  ``STREAM_ERROR_MARKER`` is treated
        the same way — the post-loop ``finalize`` calls
        ``_append_error_marker`` which writes a fresh COPILOT_*_PREFIX row
        with the final display message.
        """
        tail = list(session.messages[pre_attempt_msg_count:])
        while tail and (
            _is_error_marker(tail[-1]) or tail[-1].content == STREAM_ERROR_MARKER
        ):
            tail.pop()
        self.partial = tail
        session.messages = session.messages[:pre_attempt_msg_count]
        transcript_builder.restore(transcript_snap)

    def clear(self) -> None:
        """Drop captured state — used on successful retry."""
        self.partial = []
        self.handled_error = None

    def finalize(
        self,
        session: ChatSession | None,
        state: "_RetryState | None",
        display_msg: str,
        *,
        retryable: bool,
        failure: dict[str, Any] | None = None,
    ) -> list[StreamBaseResponse]:
        """Re-attach partial + synthetic tool_result rows + error marker.

        Called exactly once after the retry loop on final-failure exit.
        Idempotent on empty state, so it's safe to call on paths where no
        rollback happened.

        Returns the ``StreamBaseResponse`` events produced by the safety
        flush so the caller can yield them to the client (the flush mutates
        adapter state, so a second flush elsewhere would return nothing and
        stale UI elements like spinners would stay open).
        """
        if session is None:
            return []
        if self.partial:
            session.messages.extend(self.partial)
            self.partial = []
        events = _flush_orphan_tool_uses_to_session(session, state)
        _append_error_marker(
            session,
            display_msg,
            retryable=retryable,
            failure=failure,
        )
        return events


async def _apply_building_mode_restart(
    *,
    session: "ChatSession",
    state: "_RetryState",
    sdk_options: "ClaudeAgentOptions",
    prompt_parts: SystemPromptParts,
    session_id: str,
    message_id: str,
    log_prefix: str,
) -> StreamStatus:
    """Reconfigure *state* to relaunch the attempt with the guide in the
    system prompt, returning the status event to yield.

    Not an error, not a rollback — everything produced so far stands, so the
    fresh adapter only carries over the transient-retry flags. Detection
    cannot re-fire: guide_in_system_prompt flips True on success,
    building_mode_requested flips False either way.
    """
    session.building_mode_requested = False
    # ``force``: the enter tool set the flag in this very turn, so re-deriving
    # "is this session building?" from persisted history asks a question the
    # caller already answered — and answers it wrong, because the tool call is
    # not in ``messages`` yet.
    building_suffix = await build_builder_system_prompt_suffix(session, force=True)
    session.guide_in_system_prompt = bool(building_suffix)
    if not building_suffix:
        # Only a guide-load failure reaches here now.
        logger.error(
            "%s Building-mode restart: guide suffix empty — relaunching "
            "without the guide (session_id=%s)",
            log_prefix,
            session.session_id,
        )
    # Re-validates an expert session, failing closed if the expert went away
    # mid-turn. The prompt keeps the identity the turn launched with.
    await build_expert_identity_suffix(
        session.user_id,
        session.expert_id,
        organization_id=session.organization_id,
        team_id=session.team_id,
    )
    # The turn's own prompt parts, so the delegation and chat-reading rules
    # stay with their tools, which remain registered across a restart.
    system_prompt = assemble_system_prompt(
        prompt_parts, building_suffix=building_suffix
    )
    sdk_options_restart = copy(sdk_options)
    sdk_options_restart.system_prompt = _configured_system_prompt_value(system_prompt)
    # Resume the CLI session the interrupted run was writing. The guide reaches
    # the resumed turn only with the live prompt: with the CLI's snapshot on,
    # the session resends the prompt it recorded on its first request
    # (system_prompt_snapshot_cli_test.py).
    sdk_options_restart.resume = session_id
    sdk_options_restart.session_id = None
    state.options = sdk_options_restart
    state.use_resume = True
    state.query_message = (
        _BUILDING_MODE_CONTINUATION
        if building_suffix
        else _BUILDING_MODE_UNAVAILABLE_CONTINUATION
    )
    # Fresh adapter, carrying over the same flags as a transient retry. Unlike
    # a transient retry, this relaunch keeps the CLI cost baseline: it is
    # deliberate, so the interrupted run's spend is the user's (its tokens
    # were charged from the session file before this relaunch).
    # NOTE: the transcript builder is NOT restored — its partial
    # entries are real; the relaunched run's `append_user` adds
    # the original message once more, which only surfaces in the
    # rare CLI-file-loss fallback path (cosmetic duplicate).
    prior_adapter = state.adapter
    _fresh_adapter(state, message_id, session_id)
    state.adapter.thinking_only_reprompted = state.thinking_only_reprompted
    if prior_adapter.emitted_real_content_to_wire:
        state.adapter.prior_attempt_emitted_visible_content = True
    if not building_suffix:
        return StreamStatus(message="Continuing without the agent guide…")
    return StreamStatus(message="Entering building mode — loading the agent guide…")


def _ready_for_building_mode_restart(
    session: "ChatSession",
    adapter: "SDKResponseAdapter",
) -> bool:
    """True when the attempt may restart for the building-mode prompt upgrade.

    Requires a clean message boundary — no unresolved tool calls — so the
    CLI session file is orphan-free and cleanly resumable; firing mid-tool-
    call would strand ``tool_use`` blocks without results. No-op once the
    guide is already in the system prompt (restart already happened).
    """
    return (
        session.building_mode_requested
        and not session.guide_in_system_prompt
        and not adapter.has_unresolved_tool_calls
    )


def _flush_orphan_tool_uses_to_session(
    session: "ChatSession | None",
    state: "_RetryState | None",
) -> list[StreamBaseResponse]:
    """Synthesize ``tool_result`` rows for ``tool_use`` blocks that never resolved.

    Re-attached partial work may carry orphan ``tool_use`` blocks; without
    matching ``tool_result`` rows the next turn's LLM call would error with
    ``tool_use_id without tool_result``. The adapter's safety-flush produces
    interrupted-marker results that satisfy the API contract.

    Returns the flushed events so callers can yield them to the client
    alongside persisting the synthetic rows in session history.
    """
    if session is None or state is None:
        return []
    if not state.adapter.has_unresolved_tool_calls:
        return []
    safety: list[StreamBaseResponse] = []
    state.adapter.flush_unresolved_tool_calls(safety)
    for resp in safety:
        if isinstance(resp, StreamToolOutputAvailable):
            content = (
                resp.output
                if isinstance(resp.output, str)
                else json.dumps(resp.output, ensure_ascii=False)
            )
            session.messages.append(
                ChatMessage(
                    role="tool",
                    content=content,
                    tool_call_id=resp.toolCallId,
                    name=_resolve_tool_result_name(
                        session, resp.toolCallId, resp.toolName
                    ),
                )
            )
    return safety


def _resolve_tool_result_name(
    session: "ChatSession",
    tool_call_id: str,
    tool_name: str | None,
) -> str | None:
    """Name for a persisted tool-result row, falling back to the matching
    assistant ``tool_call`` — a nameless row is unidentifiable in history,
    which is the exact defect the ``name`` column exists to fix."""
    if tool_name:
        return tool_name
    for msg in reversed(session.messages):
        for tc in msg.tool_calls or []:
            if tc.get("id") == tool_call_id:
                if name := tc.get("function", {}).get("name"):
                    return name
                logger.warning(
                    f"Persisting nameless tool-result row for session "
                    f"{session.session_id}: matching tool_call "
                    f"{tool_call_id} has no function name"
                )
                return None
    logger.warning(
        f"Persisting nameless tool-result row for session {session.session_id}: "
        f"no name provided and no matching tool_call {tool_call_id} in history"
    )
    return None


@dataclass(frozen=True)
class _FinalFailure:
    """Display message + stream code + retryable flag for a final-failure exit.

    Shared by the in-history error marker (via ``_InterruptedAttempt.finalize``)
    and the client-facing ``StreamError`` SSE yield so the two stay in sync.
    """

    display_msg: str
    code: str
    retryable: bool


def _is_raised_billing_refusal(err: BaseException) -> bool:
    """A billing refusal that reached us raised rather than streamed.

    Only a typed provider error or the CLI's own error result, which the SDK
    raises as ``ResultError`` once the CLI exits, is judged; ``str()`` of any
    other exception can quote the user's input or a page a tool fetched.
    """
    if is_provider_out_of_credits(err):
        return True
    seen: set[int] = set()
    current: BaseException | None = err
    while current is not None and id(current) not in seen:
        if isinstance(current, ResultError):
            return current.api_error_status == 402 or is_provider_out_of_credits(
                current.result
            )
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def _classify_final_failure(
    interrupted: _InterruptedAttempt,
    attempts_exhausted: bool,
    transient_exhausted: bool,
    stream_err: BaseException | None,
    platform_route: bool = True,
) -> _FinalFailure | None:
    """Pick the display message, stream code, and retryable flag for the exit.

    Returns ``None`` when no failure was recorded (success path) — the caller
    should skip both the history marker and the SSE yield in that case.
    ``platform_route`` is False on a Codex turn, whose billing refusal is the
    user's own limit and keeps the provider's wording.
    """
    if interrupted.handled_error is not None:
        return _FinalFailure(
            display_msg=interrupted.handled_error.error_msg,
            code=interrupted.handled_error.code,
            retryable=interrupted.handled_error.retryable,
        )
    # Judged before the exhausted-retry verdicts: those come from text
    # patterns that a billing refusal's wording can also match.
    if (
        stream_err is not None
        and platform_route
        and _is_raised_billing_refusal(stream_err)
    ):
        return _FinalFailure(
            display_msg=PROVIDER_UNAVAILABLE_MESSAGE,
            code=PROVIDER_UNAVAILABLE_CODE,
            retryable=True,
        )
    if attempts_exhausted:
        return _FinalFailure(
            display_msg=(
                "Your conversation is too long. "
                "Please start a new chat or clear some history."
            ),
            code="all_attempts_exhausted",
            retryable=False,
        )
    if transient_exhausted:
        return _FinalFailure(
            display_msg=FRIENDLY_TRANSIENT_MSG,
            code="transient_api_error",
            retryable=True,
        )
    if stream_err is not None:
        safe_err = str(stream_err).replace("\n", " ").replace("\r", "")[:500]
        return _FinalFailure(
            display_msg=_friendly_error_text(safe_err),
            code="sdk_stream_error",
            retryable=False,
        )
    return None


def _setup_langfuse_otel() -> None:
    """Configure OTEL tracing for the Claude Agent SDK → Langfuse.

    This uses LangSmith's built-in Claude Agent SDK integration to monkey-patch
    `ClaudeSDKClient`, capturing every tool call and model turn as OTEL spans.
    Spans are exported via OTLP to Langfuse (or any OTEL-compatible backend).

    To route traces elsewhere, override `OTEL_EXPORTER_OTLP_ENDPOINT` and
    `OTEL_EXPORTER_OTLP_HEADERS` environment variables — no code changes needed.
    """
    if not _is_langfuse_configured():
        return

    try:
        settings = Settings()
        pk = settings.secrets.langfuse_public_key
        sk = settings.secrets.langfuse_secret_key
        host = settings.secrets.langfuse_host

        # OTEL exporter config — these are only set if not already present,
        # so explicit env-var overrides always win.
        creds = base64.b64encode(f"{pk}:{sk}".encode()).decode()
        os.environ.setdefault("LANGSMITH_OTEL_ENABLED", "true")
        os.environ.setdefault("LANGSMITH_OTEL_ONLY", "true")
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", f"{host}/api/public/otel")
        os.environ.setdefault(
            "OTEL_EXPORTER_OTLP_HEADERS", f"Authorization=Basic {creds}"
        )
        os.environ.setdefault(
            "OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER",
            "autogpt-copilot-traces",
        )

        # Set the Langfuse environment via OTEL resource attributes so the
        # Langfuse server maps it to the first-class environment field.
        tracing_env = settings.secrets.langfuse_tracing_environment
        os.environ.setdefault(
            "OTEL_RESOURCE_ATTRIBUTES",
            f"langfuse.environment={tracing_env}",
        )

        configure_claude_agent_sdk(tags=["sdk"])
        logger.info(
            "OTEL tracing configured for Claude Agent SDK → %s [%s]", host, tracing_env
        )
    except Exception:
        logger.warning("OTEL setup skipped — failed to configure", exc_info=True)


_setup_langfuse_otel()


# Set to hold background tasks to prevent garbage collection
_background_tasks: set[asyncio.Task[Any]] = set()


_SDK_CWD_PREFIX = WORKSPACE_PREFIX

_last_sweep_time: float = 0.0
_SWEEP_INTERVAL_SECONDS = 300  # 5 minutes

# Heartbeat interval — keep SSE alive through proxies/LBs during tool execution.
# IMPORTANT: Must be less than frontend timeout (12s in useCopilotPage.ts)
_HEARTBEAT_INTERVAL = 10.0  # seconds


async def _safe_close_sdk_client(
    sdk_client: ClaudeSDKClient,
    log_prefix: str,
) -> None:
    """Close a ClaudeSDKClient, suppressing errors from client disconnect.

    When the SSE client disconnects mid-stream, ``GeneratorExit`` propagates
    through the async generator stack and causes ``ClaudeSDKClient.__aexit__``
    to run in a different async context or task than where the client was
    opened.  This triggers two known error classes:

    * ``ValueError``: ``<Token var=<ContextVar name='current_context'>>
      was created in a different Context`` — OpenTelemetry's
      ``context.detach()`` fails because the OTEL context token was
      created in the original generator coroutine but detach runs in
      the GC / cleanup coroutine (Sentry: AUTOGPT-SERVER-8BT).

    * ``RuntimeError``: ``Attempted to exit cancel scope in a different
      task than it was entered in`` — anyio's ``TaskGroup.__aexit__``
      detects that the cancel scope was entered in one task but is
      being exited in another (Sentry: AUTOGPT-SERVER-8BW).

    Both are harmless — the TCP connection is already dead and no
    resources leak.  Logging them at ``debug`` level keeps observability
    without polluting Sentry.
    """
    try:
        await sdk_client.__aexit__(None, None, None)
    except (ValueError, RuntimeError) as exc:
        if _is_sdk_disconnect_error(exc):
            # Expected during client disconnect — suppress to avoid Sentry noise.
            logger.debug(
                "%s SDK client cleanup error suppressed (client disconnect): %s: %s",
                log_prefix,
                type(exc).__name__,
                exc,
            )
        else:
            raise
    except GeneratorExit:
        # GeneratorExit can propagate through __aexit__ — suppress it here
        # since the generator is already being torn down.
        logger.debug(
            "%s SDK client cleanup GeneratorExit suppressed (client disconnect)",
            log_prefix,
        )
    except Exception:
        # Unexpected cleanup error — log at error level so Sentry captures it
        # (via its logging integration), but don't propagate since we're in
        # teardown and the caller cannot meaningfully handle this.
        logger.error(
            "%s Unexpected SDK client cleanup error",
            log_prefix,
            exc_info=True,
        )


async def _iter_sdk_messages(
    client: ClaudeSDKClient,
    wake: asyncio.Event | None = None,
    tool_display_wake: asyncio.Event | None = None,
) -> AsyncGenerator[Any, None]:
    """Yield SDK messages with heartbeat-based timeouts.

    Uses an explicit async iterator with non-cancelling heartbeats.

    ``wake`` cuts a heartbeat short: when it is set, the generator yields
    the heartbeat sentinel at once and clears the event instead of waiting
    out ``_HEARTBEAT_INTERVAL``.  The PreCompact hook sets it so the
    compaction row opens before the CLI's next message can close it.

    CRITICAL: we must NOT cancel `__anext__()` mid-flight — doing so
    (via `asyncio.timeout` or `wait_for`) corrupts the SDK's internal
    anyio memory stream, causing `StopAsyncIteration` on the next call
    and silently dropping all in-flight tool results.  Instead, wrap
    `__anext__()` in a `Task` and use `asyncio.wait()` with a
    timeout.  On timeout we yield a heartbeat sentinel but keep the Task
    alive so it can deliver the next message.

    Every fetch Task runs in one context, copied once up front.  The
    langsmith tracing wrapper around ``receive_response()`` stores the
    ``claude.conversation`` run in a ContextVar during the first fetch and
    parents each reply's ``claude.assistant.turn`` span on it; a Task given
    a fresh copy of the caller's context per message never sees that run,
    so every reply after the first message would drop out of the trace.
    Only one fetch is in flight at a time, so sharing the context is safe.

    Yields `None` on heartbeat timeout (caller should refresh locks and
    emit heartbeat events).  Yields the raw SDK message otherwise.
    On stream end (`StopAsyncIteration`), the generator returns normally.
    Any other exception from the SDK propagates to the caller.
    """
    msg_iter = client.receive_response().__aiter__()
    fetch_context = contextvars.copy_context()
    pending_task: asyncio.Task[Any] | None = None
    wake_tasks: dict[asyncio.Task[bool], asyncio.Event] = {}

    async def _next_msg() -> Any:
        """Await the next SDK message, wrapped for use with `asyncio.Task`."""
        return await msg_iter.__anext__()

    try:
        while True:
            if pending_task is None:
                pending_task = asyncio.create_task(_next_msg(), context=fetch_context)
            waiters: set[asyncio.Task[Any]] = {pending_task}
            for event in (wake, tool_display_wake):
                if event is not None and event not in wake_tasks.values():
                    wake_tasks[asyncio.create_task(event.wait())] = event
            waiters.update(wake_tasks)

            done, _ = await asyncio.wait(
                waiters,
                timeout=_HEARTBEAT_INTERVAL,
                return_when=asyncio.FIRST_COMPLETED,
            )

            completed_wakes = [task for task in wake_tasks if task in done]
            if completed_wakes:
                # Woken: hand the sentinel over BEFORE any message that
                # landed in the same tick, or the row the hook announced
                # would be closed by the very message meant to follow it.
                for task in completed_wakes:
                    wake_tasks.pop(task).clear()
                yield None
                continue

            if pending_task not in done:
                yield None  # heartbeat sentinel: the interval elapsed
                continue

            msg_task, pending_task = pending_task, None
            try:
                yield msg_task.result()
            except StopAsyncIteration:
                return
    finally:
        for task in (pending_task, *wake_tasks):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass


def _gateway_usage(gateway: ModelGateway) -> GatewayUsage | None:
    """What the upstream reported for the gateway's successful requests;
    None before any. A failed attempt's usage is metered apart
    (``failed_usage``) and logged per request by the gateway."""
    usage = gateway.usage
    return usage if usage.prompt_tokens or usage.output_tokens else None


def _cli_project_dir(sdk_cwd: str) -> str | None:
    """The CLI's project dir, where the OpenRouter reconcile sweeps for
    subagents' gen-IDs.

    ``sdk_cwd`` is the per-session CLI working directory; the CLI encodes it
    into the project-dir name the same way ``encode_cwd_for_cli`` does, and
    writes the main transcript + any ``subagents/`` alongside it under
    ``~/.claude/projects/<encoded>/``.  ``None`` when sdk_cwd isn't set
    (shouldn't happen in practice for SDK turns).
    """
    if not sdk_cwd:
        return None
    return os.path.join(
        os.path.expanduser("~/.claude/projects"), encode_cwd_for_cli(sdk_cwd)
    )


_MAX_TRANSIENT_BACKOFF_SECONDS = 30

# The model the CLI names on messages it writes itself (an API error, a
# refusal). No model served them, so they never price a turn's tokens or
# stand for the model it ran on.
_CLI_SYNTHETIC_MODEL = "<synthetic>"


def _compute_transient_backoff(attempt: int) -> int:
    """Return the exponential backoff delay (seconds) for a transient retry.

    ``attempt`` is 1-based: first retry → ~1 s, second → ~2 s, third → ~4 s,
    …, capped at :data:`_MAX_TRANSIENT_BACKOFF_SECONDS`.

    Full-jitter (``0.5 … 1.0 × base``) is applied to spread retries from
    multiple tenants hitting the same 429/529 simultaneously (thundering herd).
    The result is rounded to the nearest integer so the ``StreamStatus``
    message is human-readable.

    Extracted as a module-level pure function so it can be unit-tested
    independently of the closure that wraps it inside the retry loop.
    """
    base = min(_MAX_TRANSIENT_BACKOFF_SECONDS, 2 ** (attempt - 1))
    return max(1, round(base * (0.5 + random.random() * 0.5)))


def _next_transient_backoff(
    events_yielded: int,
    transient_retries: int,
    max_transient_retries: int,
) -> tuple[int | None, int]:
    """Decide whether to retry after a transient error.

    Returns ``(backoff_seconds, updated_retries)``.  ``backoff_seconds`` is
    ``None`` when no retry should be attempted — either content has already
    been streamed to the frontend (``events_yielded > 0``, retrying would
    produce duplicates) or the retry budget is exhausted.

    Extracted as a module-level pure function so it can be unit-tested
    independently of the retry loop.
    """
    if events_yielded > 0:
        return None, transient_retries
    new_retries = transient_retries + 1
    if new_retries > max_transient_retries:
        return None, new_retries
    return _compute_transient_backoff(new_retries), new_retries


async def _do_transient_backoff(
    backoff: int,
    state: "_RetryState",
    ctx: _StreamContext,
) -> AsyncIterator[StreamStatus]:
    """Emit a retry notification, sleep, and reset the SDK adapter.

    Yields a single :class:`StreamStatus` so the caller can forward it to
    the client, then sleeps for *backoff* seconds and resets ``state.adapter``.
    Usage already reported survives the retry, but the CLI cost baseline does
    not: the relaunch re-reads the total the resumed session restores, so
    whatever the failed attempt spent without reporting a result is not
    charged. Its OpenRouter generations, live or a subagent's, are absorbed
    for the same reason. The user pays for the attempt that succeeds, not for
    our failure.

    A launch that started a fresh CLI session (``--session-id``, no
    ``--resume``: a chat's first turn, or a turn whose session could not be
    restored) wrote that session's file before its call failed, and the CLI
    refuses the id again ("Session ID ... is already in use"). The file goes,
    so the relaunch can start the session again; the failed launch streamed
    nothing, so it holds nothing worth resuming.

    All of that is settled before the first yield. A turn torn down while it
    waits out the backoff, by a client that goes away or a stop, never comes
    back to this generator, yet the failed launch is no less ours.

    Extracted from both exception handlers in the retry loop to remove
    near-identical code duplication.
    """
    state.usage.cli_session_total_usd = None
    await state.generations.absorb_launch()
    fresh_session = state.options.session_id
    if ctx.sdk_cwd and fresh_session and state.options.resume is None:
        delete_stale_cli_session_file(ctx.sdk_cwd, fresh_session, ctx.log_prefix)
    yield StreamStatus(message=f"Connection interrupted, retrying in {backoff}s…")
    await asyncio.sleep(backoff)
    _fresh_adapter(state, ctx.message_id, ctx.session_id)


_CLI_STDERR_AUTH_RE = re.compile(
    r"(?i)(?P<prefix>\b(?:authorization|proxy-authorization|x-api-key|"
    r"anthropic[_-]auth[_-]token)\b[\"']?\s*[:=]\s*[\"']?)"
    r"(?:(?:bearer|basic)\s+)?[^,\s;\"'}\]]+"
)


def _redact_cli_stderr(line: str, *, secrets: tuple[str, ...] = ()) -> str:
    """Remove known capabilities and authorization values from CLI logs."""
    redacted = _CLI_STDERR_AUTH_RE.sub(r"\g<prefix>[REDACTED]", line)
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


async def _close_route_gateway_for_finally(
    gateway: ModelGateway,
    log_prefix: str,
) -> BaseException | None:
    """Finish gateway checkpointing and return any error for deferred raising."""
    close_task = asyncio.create_task(gateway.close())
    try:
        await asyncio.shield(close_task)
    except asyncio.CancelledError as cancellation:
        try:
            await close_task
        except BaseException as cleanup_error:
            logger.exception("%s Route gateway cleanup failed", log_prefix)
            return cleanup_error
        return cancellation
    except Exception as cleanup_error:
        logger.exception("%s Route gateway cleanup failed", log_prefix)
        return cleanup_error
    return None


def _raise_deferred_gateway_cleanup_error(
    cleanup_error: BaseException | None,
    turn_error: BaseException | None,
) -> None:
    if cleanup_error is None:
        return
    if turn_error is not None and turn_error is not cleanup_error:
        raise cleanup_error from turn_error
    raise cleanup_error


def _configured_system_prompt_value(system_prompt: str) -> str | SystemPromptCustom:
    """The ``system_prompt`` option for *system_prompt* under the current config.

    Every launch (first, context-size retry, building-mode restart) goes
    through here so they agree on the prompt's form.
    """
    return _build_system_prompt_value(
        system_prompt, live=config.claude_agent_live_system_prompt
    )


class _SdkToolOptions(TypedDict):
    tools: list[str]
    allowed_tools: list[str]
    disallowed_tools: list[str]


def _build_sdk_tool_options(
    permissions: CopilotPermissions | None,
    *,
    use_e2b: bool,
    disabled_groups: list[ToolGroup],
    denied_tools: frozenset[str] = frozenset(),
) -> _SdkToolOptions:
    """Build the three tool lists of a turn's :class:`ClaudeAgentOptions`.

    They travel together so they cannot drift: ``allowed_tools`` is what the
    PreToolUse hook enforces, and ``tools`` declares only the built-ins in it,
    so a CLI upgrade's new built-ins never reach the model.  *denied_tools*
    are short names of copilot tools taken out of ``allowed_tools``.
    """
    extra_builtins = config.claude_agent_extra_builtin_tools
    if permissions is not None:
        allowed, disallowed = apply_tool_permissions(
            permissions,
            use_e2b=use_e2b,
            disabled_groups=disabled_groups,
            extra_builtins=extra_builtins,
        )
    else:
        allowed = get_copilot_tool_names(
            use_e2b=use_e2b,
            disabled_groups=disabled_groups,
            extra_builtins=extra_builtins,
        )
        disallowed = get_sdk_disallowed_tools(use_e2b=use_e2b)

    denied_mcp = {f"{MCP_TOOL_PREFIX}{name}" for name in denied_tools}
    allowed = [name for name in allowed if name not in denied_mcp]
    return _SdkToolOptions(
        tools=get_sdk_declared_tools(allowed, disallowed),
        allowed_tools=allowed,
        disallowed_tools=disallowed,
    )


def _build_system_prompt_value(
    system_prompt: str, *, live: bool
) -> str | SystemPromptCustom:
    """Build the ``system_prompt`` argument for :class:`ClaudeAgentOptions`.

    *system_prompt* replaces the CLI's built-in prompt via ``--system-prompt``.

    When *live* is set it carries ``snapshot=False``, so the prompt this
    process was launched with is the one sent, on a new session and on
    ``--resume``.  With the snapshot on (the CLI's default when the key is
    absent), the CLI records the system prompt and every tool description on
    a session's first request and resends that record on every later request
    and resume until the session compacts, ignoring the prompt a later
    process was given.  Only a dict can carry ``snapshot`` (the SDK sends it
    on the initialize request; the prompt text still travels as the flag
    above), so the prompt is wrapped in :class:`SystemPromptCustom`.
    Without *live*, the value is the raw string passed before ``snapshot``
    existed.
    """
    if live:
        return SystemPromptCustom(type="custom", prompt=system_prompt, snapshot=False)
    return system_prompt


def _make_sdk_cwd(session_id: str) -> str:
    """Create a safe, session-specific working directory path.

    Delegates to :func:`~backend.copilot.tools.sandbox.make_session_path`
    (single source of truth for path sanitization) and adds a defence-in-depth
    assertion.
    """
    cwd = make_session_path(session_id)
    # Defence-in-depth: normpath + startswith is a CodeQL-recognised sanitizer
    cwd = os.path.normpath(cwd)
    if not cwd.startswith(_SDK_CWD_PREFIX):
        raise ValueError(f"SDK cwd escaped prefix: {cwd}")
    return cwd


def _write_cli_session_to_disk(
    content: bytes,
    sdk_cwd: str,
    session_id: str,
    log_prefix: str,
) -> bool:
    """Write downloaded CLI session bytes to disk so the CLI can --resume.

    Returns True on success, False if the path is invalid or the write fails.
    Path-traversal guard: rejects paths outside the CLI projects base.
    """
    session_file = cli_session_path(sdk_cwd, session_id)
    real_path = os.path.realpath(session_file)
    _pbase = projects_base()
    if not real_path.startswith(_pbase + os.sep):
        logger.warning(
            "%s CLI session restore path outside projects base: %s",
            log_prefix,
            os.path.basename(session_file),
        )
        return False
    try:
        os.makedirs(os.path.dirname(real_path), exist_ok=True)
        Path(real_path).write_bytes(content)
        logger.info(
            "%s Wrote CLI session to disk (%dB) for --resume",
            log_prefix,
            len(content),
        )
        return True
    except OSError as e:
        logger.warning(
            "%s Failed to write CLI session file %s: %s",
            log_prefix,
            os.path.basename(session_file),
            e.strerror or str(e),
        )
        return False


def delete_stale_cli_session_file(
    sdk_cwd: str,
    session_id: str,
    log_prefix: str,
) -> bool:
    """Delete the local CLI session file at the predictable path.

    Used so a subsequent CLI invocation with ``--session-id`` (no ``--resume``)
    doesn't trip ``"Session ID already in use"``.  Path-traversal guard:
    rejects paths outside the CLI projects base.

    Returns True if a file was deleted, False otherwise (missing, traversal,
    or unlink failure).
    """
    real_path = os.path.realpath(cli_session_path(sdk_cwd, session_id))
    if not real_path.startswith(projects_base() + os.sep):
        # Mirror ``_write_cli_session_to_disk``'s defense-in-depth: log
        # rather than fail silently when the resolved path escapes the
        # projects base.  In normal operation this is unreachable
        # (session_id is a server-generated UUID and ``cli_session_path``
        # is deterministic), so a hit indicates a config or tampering
        # issue that's worth surfacing.
        logger.warning(
            "%s CLI session delete path outside projects base: %s",
            log_prefix,
            os.path.basename(real_path),
        )
        return False
    # Direct unlink — no exists() check (avoids TOCTOU with the file being
    # deleted by another process between check and unlink).
    try:
        Path(real_path).unlink()
        logger.info(
            "%s Removed stale local CLI session file at %s",
            log_prefix,
            os.path.basename(real_path),
        )
        return True
    except FileNotFoundError:
        return False
    except OSError as unlink_err:
        # Sanitise log: basename + strerror only (no full path / no raw
        # exception which can echo absolute paths back in some libc errors).
        logger.warning(
            "%s Failed to remove stale local CLI session file %s: %s",
            log_prefix,
            os.path.basename(real_path),
            unlink_err.strerror or type(unlink_err).__name__,
        )
        return False


def read_cli_session_from_disk(
    sdk_cwd: str,
    session_id: str,
    log_prefix: str,
) -> bytes | None:
    """Read the CLI session JSONL file from disk after the SDK turn.

    Returns the file bytes, or None if the file is missing, outside the
    projects base, or unreadable.
    Path-traversal guard: rejects paths outside the CLI projects base.
    """
    session_file = cli_session_path(sdk_cwd, session_id)
    real_path = os.path.realpath(session_file)
    _pbase = projects_base()
    if not real_path.startswith(_pbase + os.sep):
        logger.warning(
            "%s CLI session file outside projects base, skipping upload: %s",
            log_prefix,
            os.path.basename(real_path),
        )
        return None
    try:
        raw_bytes = Path(real_path).read_bytes()
    except FileNotFoundError:
        logger.debug(
            "%s CLI session file not found, skipping upload: %s",
            log_prefix,
            os.path.basename(session_file),
        )
        return None
    except OSError as e:
        logger.warning(
            "%s Failed to read CLI session file %s: %s",
            log_prefix,
            os.path.basename(session_file),
            e.strerror or str(e),
        )
        return None

    # Strip stale thinking blocks and metadata entries before uploading.
    # Thinking blocks from non-last turns can be massive; keeping them causes
    # the CLI to auto-compact its session when the context window fills up,
    # silently losing conversation history.
    try:
        raw_text = raw_bytes.decode("utf-8")
        stripped_text = strip_for_upload(raw_text)
        stripped_bytes = stripped_text.encode("utf-8")
    except UnicodeDecodeError:
        logger.warning("%s CLI session is not valid UTF-8, uploading raw", log_prefix)
        return raw_bytes
    except (OSError, ValueError) as e:
        # OSError: encode/decode I/O failure; ValueError: malformed JSONL in strip.
        # Other unexpected exceptions are not silently swallowed here so they propagate
        # to the outer OSError handler and are logged with exc_info.
        logger.warning(
            "%s Failed to strip CLI session, uploading raw: %s", log_prefix, e
        )
        return raw_bytes

    if len(stripped_bytes) < len(raw_bytes):
        # Write back locally so same-pod turns also benefit.
        try:
            Path(real_path).write_bytes(stripped_bytes)
            logger.info(
                "%s Stripped CLI session: %dB → %dB",
                log_prefix,
                len(raw_bytes),
                len(stripped_bytes),
            )
        except OSError as e:
            # write_bytes failed — stripped content is still valid for GCS upload even
            # though the local write-back failed (same-pod optimization silently skipped).
            logger.warning(
                "%s Failed to write back stripped CLI session: %s",
                log_prefix,
                e.strerror or str(e),
            )
    return stripped_bytes


def process_cli_restore(
    cli_restore: TranscriptDownload,
    sdk_cwd: str,
    session_id: str,
    log_prefix: str,
) -> tuple[str, bool]:
    """Validate and write a restored CLI session to disk.

    Decodes bytes → UTF-8, strips progress entries and stale thinking blocks,
    validates the result, then writes the stripped content to disk so the CLI
    can ``--resume`` from it.

    Returns ``(stripped_content, success)`` where ``success=False`` means the
    content was invalid or the disk write failed (caller should skip --resume).
    """
    try:
        raw_bytes = cli_restore.content
        raw_str = (
            raw_bytes.decode("utf-8") if isinstance(raw_bytes, bytes) else raw_bytes
        )
    except UnicodeDecodeError:
        logger.warning(
            "%s CLI session content is not valid UTF-8, skipping", log_prefix
        )
        return "", False

    stripped = strip_for_upload(raw_str)
    is_valid = validate_transcript(stripped)
    # Use len(raw_str) rather than len(cli_restore.content) so the unit is always
    # characters (raw_str is always str at this point regardless of input type).
    # lines_stripped = original lines minus remaining lines after stripping.
    _original_lines = len(raw_str.strip().split("\n")) if raw_str.strip() else 0
    _remaining_lines = len(stripped.strip().split("\n")) if stripped.strip() else 0
    logger.info(
        "%s Restored CLI session: %dB raw, %d lines stripped, msg_count=%d, valid=%s",
        log_prefix,
        len(raw_str),
        _original_lines - _remaining_lines,
        cli_restore.message_count,
        is_valid,
    )
    if not is_valid:
        logger.warning(
            "%s CLI session content invalid after strip — running without --resume",
            log_prefix,
        )
        return "", False

    stripped_bytes = stripped.encode("utf-8")
    if not _write_cli_session_to_disk(stripped_bytes, sdk_cwd, session_id, log_prefix):
        return "", False

    return stripped, True


async def _cleanup_sdk_tool_results(cwd: str) -> None:
    """Remove SDK session artifacts for a specific working directory.

    Cleans up the ephemeral working directory `/tmp/copilot-<session>/`.

    Also sweeps stale CLI project directories (older than 12 h) to prevent
    unbounded disk growth.  The sweep is best-effort, rate-limited to once
    every 5 minutes, and capped at 50 directories per sweep.

    Security: *cwd* MUST be created by `_make_sdk_cwd()` which sanitizes
    the session_id.
    """
    normalized = os.path.normpath(cwd)
    if not normalized.startswith(_SDK_CWD_PREFIX):
        logger.warning("[SDK] Rejecting cleanup for path outside workspace: %s", cwd)
        return

    await asyncio.to_thread(shutil.rmtree, normalized, True)

    # Best-effort sweep of old project dirs to prevent disk leak.
    # Pass the encoded cwd so only this session's project directory is swept,
    # which is safe in multi-tenant environments.
    global _last_sweep_time
    now = time.time()
    if now - _last_sweep_time >= _SWEEP_INTERVAL_SECONDS:
        _last_sweep_time = now
        encoded = encode_cwd_for_cli(normalized)
        await asyncio.to_thread(cleanup_stale_project_dirs, encoded)


def _format_sdk_content_blocks(blocks: list) -> list[dict[str, Any]]:
    """Convert SDK content blocks to transcript format.

    Handles TextBlock, ToolUseBlock, ToolResultBlock, and ThinkingBlock.
    Raw dicts (e.g. ``redacted_thinking`` blocks that the SDK may not have
    a typed class for) are passed through verbatim to preserve them in the
    transcript.  Unknown typed block objects are logged and skipped.
    """
    result: list[dict[str, Any]] = []
    for block in blocks or []:
        if isinstance(block, TextBlock):
            result.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolUseBlock):
            result.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": strip_display_token(block.input),
                }
            )
        elif isinstance(block, ToolResultBlock):
            tool_result_entry: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": block.tool_use_id,
                "content": block.content,
            }
            if block.is_error:
                tool_result_entry["is_error"] = True
            result.append(tool_result_entry)
        elif isinstance(block, ThinkingBlock):
            result.append(
                {
                    "type": "thinking",
                    "thinking": block.thinking,
                    "signature": block.signature,
                }
            )
        elif isinstance(block, dict) and "type" in block:
            # Preserve raw dict blocks (e.g. redacted_thinking) verbatim.
            result.append(block)
        else:
            logger.warning(
                "[SDK] Unknown content block type: %s."
                " This may indicate a new SDK version with additional block types.",
                type(block).__name__,
            )
    return result


# The summarizer's model on a turn with no platform route (Codex alone).
_COMPRESSION_FALLBACK_MODEL = "anthropic/claude-sonnet-5-5"


def _compression_model() -> str:
    """The model whose context window governs pre-query compression.

    ``_compress_messages`` hands this to ``_run_compression``, and the
    pre-query predictor sizes its estimate against the same name.  Both read
    it here rather than naming a model each: the runtime model (Codex 400K,
    Opus 120K) and the compressor's model have different windows, so any
    call site that picks its own opens compaction rows for work the
    compressor won't do — or does the work with no row open at all.
    The platform route's standard tier, as configured.
    """
    return config.tier_model("standard") or _COMPRESSION_FALLBACK_MODEL


def _route_name(ctx: "_StreamContext") -> str:
    """The route a stream runs on, for provider-failure reports."""
    return ctx.profile.name if ctx.profile is not None else config.platform_route


def _to_compress_dict(msg: ChatMessage) -> dict[str, Any]:
    """One ``ChatMessage`` in the dict shape the compressor reads.

    Shared by ``_compress_messages`` (what actually gets compressed) and
    ``_will_compact`` (what gets estimated).  ``_msg_tokens`` counts the
    serialized ``function.arguments`` of every tool call, and copilot
    assistant rows carry whole graph JSON in ``create_agent``/``edit_agent``
    payloads — an estimate built from ``role``/``content`` alone undercounts
    the tool-heaviest sessions by the widest margin.
    """
    payload: dict[str, Any] = {"role": msg.role}
    if msg.content:
        payload["content"] = msg.content
    if msg.tool_calls:
        payload["tool_calls"] = tool_calls_for_provider(msg.tool_calls)
    if msg.tool_call_id:
        payload["tool_call_id"] = msg.tool_call_id
    return payload


# Characters per token, measured with ``o200k_base`` over the payload shapes
# this history actually carries: base64-ish blobs 1.5, CJK 1.8, numeric IDs
# 2.6, graph JSON 2.7, source code 4.6, English prose 4.9-6.5.
# ``estimate_token_count`` then scales raw counts by up to 1.5x for the
# Claude-5 tokenizer generation, which drags the dense end down to ~1.0
# characters per *estimated* token.
#
# The two bounds bracket the estimate rather than replacing it: at or below
# ``limit * _MIN_CHARS_PER_TOKEN`` no history can reach the threshold, above
# ``limit * _MAX_CHARS_PER_TOKEN`` none can miss it, so both ends are decided
# by arithmetic alone.  Only the band between them pays for a tokenizer pass,
# and that band is bounded by construction — at the production limit (~138K
# tokens) the widest input tiktoken can see is ~830K characters, ~80ms.  The
# megabyte histories that made tokenizing expensive in the first place are
# exactly the ones the upper cut answers for free.
_MIN_CHARS_PER_TOKEN = 1.0
_MAX_CHARS_PER_TOKEN = 6.0

# Pacing hint only, never the decision: when the bounds settle the question
# without tokenizing there is no estimate to report, and ``summarizing`` with
# no ``tokens_before`` animates a 500K compaction exactly like a 20K one.  The
# midpoint of the measured range is wrong by ~2x either way — good enough to
# pick a progress curve, which is all the client does with it.
_NOMINAL_CHARS_PER_TOKEN = 3.0


@dataclass(frozen=True)
class _CompactionForecast:
    """Whether the upcoming turn compresses, and how big the input is.

    ``tokens_before`` is the size the prediction was made from; it rides
    the ``summarizing`` phase so the client can pace its progress curve.
    ``None`` when no size was established at all.
    """

    expected: bool
    tokens_before: int | None = None


_NO_COMPACTION = _CompactionForecast(expected=False)


def _payload_chars(rows: list[ChatMessage]) -> int:
    """Characters the estimator would tokenize for *rows*.

    Content plus tool-call names and serialized arguments — the fields
    ``_to_compress_dict`` hands the estimator.  The per-message wrapper
    tokens ``_msg_tokens`` adds are left out; they are absorbed by the
    width of the bounds above.
    """
    total = 0
    for msg in rows:
        total += len(msg.content or "")
        for call in msg.tool_calls or []:
            function = call.get("function", {})
            total += len(str(function.get("name") or ""))
            total += len(str(function.get("arguments") or ""))
    return total


def _will_compact(
    messages: list[ChatMessage], model: str, *, target_tokens: int
) -> _CompactionForecast:
    """Cheap pre-check: would ``_compress_messages`` do real work?

    ``target_tokens`` is the same pin-derived budget handed to
    ``_build_query_message``; ``model`` only picks the tokenizer.

    Mirrors ``compress_context``'s early-return condition (``prompt.py``:
    ``original_count + reserve <= target_tokens``) so the UI can open a
    compaction row *before* the expensive LLM summarization starts instead of
    announcing it afterwards.

    A character count decides the clear-cut cases outright; only histories
    that land in the ambiguous band near the threshold are tokenized, and
    that band is bounded (see ``_MIN_CHARS_PER_TOKEN``), so the tokenizer
    can never run on the huge histories that would make it expensive.

    Being wrong is survivable in both directions — a false negative falls
    back to a self-contained row, a false positive is retired by
    ``abort_pre_query`` — but a false positive flashes a bar on screen, so
    the tokenizer is spent exactly where the bounds cannot separate the two.
    Any failure means no prediction: this is cosmetic, and a tiktoken
    download failure or an unknown model must not take the stream with it.
    """
    rows = [m for m in filter_compaction_messages(messages) if m.role != "reasoning"]
    if len(rows) < 2:
        return _NO_COMPACTION
    chars = _payload_chars(rows)
    try:
        # ``estimated > limit`` is ``estimated + reserve > target``, the
        # negation of the compressor's early return.
        limit = target_tokens - DEFAULT_COMPRESSION_RESERVE
        if chars <= limit * _MIN_CHARS_PER_TOKEN:
            return _NO_COMPACTION
        if chars > limit * _MAX_CHARS_PER_TOKEN:
            return _CompactionForecast(
                expected=True,
                tokens_before=int(chars / _NOMINAL_CHARS_PER_TOKEN),
            )
        estimated = estimate_token_count(
            [_to_compress_dict(m) for m in rows], model=model
        )
    except Exception:
        logger.warning(
            "[SDK] Compaction pre-check failed on %d rows (%d chars) —"
            " skipping the prediction (the row still opens after the fact)",
            len(rows),
            chars,
            exc_info=True,
        )
        return _NO_COMPACTION
    return _CompactionForecast(expected=estimated > limit, tokens_before=estimated)


def _expect_pre_query_compaction(
    messages: list[ChatMessage],
    model: str,
    *,
    use_resume: bool,
    transcript_msg_count: int,
    session_msg_ceiling: int,
    prior_messages: "list[ChatMessage] | None" = None,
    target_tokens: int,
) -> _CompactionForecast:
    """Predict whether ``_build_query_message`` will compress on this turn.

    Mirrors ``_build_query_message``'s branch selection so the compaction
    row only opens when the branch actually taken feeds ``_compress_messages``
    a slice that would compress.  On the steady-state resume path the
    transcript covers the full history and nothing is compressed — running
    ``_will_compact`` over the whole session there returns a false positive
    on EVERY turn once cumulative history exceeds the compression target,
    opening a row that is immediately aborted.

    ``prior_messages`` must be the same value handed to
    ``_build_query_message``: on the no-resume path it is the compacted
    transcript + hole + gap, which is normally far smaller than the raw
    session and is non-``None`` on essentially every turn with a baseline
    transcript.  Predicting over the full session there is the mirror-image
    error — the branch compresses a slice this function never looked at.

    ``target_tokens`` must be the same value handed to
    ``_build_query_message``: the turn's ``_compaction_target_tokens``, never
    below 10,000 tokens, so the builder's bare-message floor, which only a
    retry's budget reaches, never applies on the path this predicts.

    False negatives are safe: ``emit_pre_query_end`` emits a self-contained
    row when compression happens without a prediction.  That covers slices
    this estimator cannot see, like cap-engaged hole rows fetched from the DB.
    """
    prior = [
        m for m in messages[: max(0, session_msg_ceiling - 1)] if m.role != "reasoning"
    ]
    if not use_resume:
        if session_msg_ceiling <= 1:
            return _NO_COMPACTION
        source = prior_messages if prior_messages is not None else prior
        return _will_compact(source, model, target_tokens=target_tokens)
    if transcript_msg_count <= 0:
        # ``use_resume`` with no covered rows compresses nothing.
        return _NO_COMPACTION
    prior_rows = messages[: max(0, session_msg_ceiling - 1)]
    gap = sendable_gap(uncovered_rows(prior_rows, transcript_msg_count))
    if not gap:
        # Scenario A: --resume covers the full context; nothing is compressed.
        return _NO_COMPACTION
    return _will_compact(gap, model, target_tokens=target_tokens)


def _retry_reduced_context(
    *,
    reduced: ReducedContext,
    compaction_stats: "CompactionStats | None",
) -> bool:
    """Did a context-reduction retry actually summarize anything?

    The retry opens its compaction row optimistically, before
    ``_reduce_context`` runs, so the close is a claim about work that may
    not have happened.  The row says the conversation was condensed, and
    only two things earn that: the transcript came back summarized
    (``transcript_lost`` false — the drop branch is the only other exit),
    or ``_build_query_message`` compressed DB history, which surfaces as
    ``compaction_stats``.

    Dropping the transcript is deliberately NOT enough.  It reduces
    context, but by discarding history rather than summarizing it, and
    ``compact_transcript`` failing lands there just as surely as a second
    retry does — so a guess made before ``_reduce_context`` ran would
    persist a durable "Condensed" row for a conversation that was
    truncated.  The non-retry path already refuses to call a drop a
    summarize; this keeps the retry path honest about the same thing.
    """
    return not reduced.transcript_lost or compaction_stats is not None


async def _compress_messages(
    messages: list[ChatMessage],
    target_tokens: int | None = None,
) -> tuple[list[ChatMessage], bool, "CompactionStats | None"]:
    """Compress a list of messages if they exceed the token threshold.

    Delegates to `_run_compression` (`transcript.py`) which centralizes
    the "try LLM, fallback to truncation" pattern with timeouts.  Both
    `_compress_messages` and `compact_transcript` share this helper so
    client acquisition and error handling are consistent.

    ``target_tokens`` sets a hard ceiling for the compressed output.  The
    turn always passes the pin-derived budget (``_compaction_target_tokens``
    or a retry budget); ``None`` is only for callers outside the turn, and
    lets ``compress_context`` fall back to its own model-aware default.

    See also:
        `_run_compression` — shared compression with timeout guards.
        `compact_transcript` — compresses JSONL transcript entries.
        `CompactionTracker` — emits UI events for mid-stream compaction.
    """
    # ``role="reasoning"`` rows are persisted for frontend replay only — they
    # aren't valid OpenAI roles and ``compress_context`` would either drop or
    # malform them.  Strip here so every caller is covered (``_build_query_message``
    # already filters upstream, but ``_seed_transcript`` and any future caller
    # don't, and centralising the filter avoids per-call-site drift).
    messages = [
        m for m in filter_compaction_messages(messages) if m.role != "reasoning"
    ]

    if len(messages) < 2:
        return messages, False, None

    # Convert ChatMessages to dicts for compress_context
    messages_dict = [_to_compress_dict(msg) for msg in messages]

    try:
        model = _compression_model()
        result = await _run_compression(
            messages_dict,
            model,
            "[SDK]",
            target_tokens=target_tokens,
            keep_recent_tokens=_tail_budget_tokens(target_tokens, model),
            # History is rendered as text (see _format_conversation_context),
            # so a middle-out cut of a tool call's arguments is safe here.
            truncate_tool_arguments=True,
        )
    except Exception as exc:
        # Both the LLM summarize path AND the truncation fallback inside
        # ``_run_compression`` failed (timeouts, tokenization error, etc.).
        # Returning the originals here would silently feed the same
        # too-long payload back into the retry loop, guaranteeing another
        # ``Prompt is too long`` and burning the retry budget for no
        # progress.  Drop history entirely instead — the caller will fall
        # back to the bare current message, which is the tightest
        # compression we can offer without an LLM, and is the only thing
        # that can definitively recover a session whose stored history
        # exceeds the model's context window.
        logger.warning(
            "[SDK] _compress_messages failed — dropping %d messages to bare"
            " message to guarantee retry progress: %s",
            len(messages),
            exc,
        )
        # A drop is not a summarize: the row must not claim "condensed" and
        # must not count as a completed compaction.  It still has to close
        # honestly — the user just lost their history, and the settled row
        # is where they learn it — so the stats say only what happened.
        return [], True, CompactionStats(dropped=True, messages_before=len(messages))

    if result.was_compacted:
        logger.info(
            "[SDK] Context compacted: %d -> %d tokens (%d summarized, %d dropped)",
            result.original_token_count,
            result.token_count,
            result.messages_summarized,
            result.messages_dropped,
        )
        # Convert compressed dicts back to ChatMessages
        compacted = [
            ChatMessage(
                role=m["role"],
                content=m.get("content"),
                tool_calls=m.get("tool_calls"),
                tool_call_id=m.get("tool_call_id"),
            )
            for m in result.messages
        ]
        stats = CompactionStats(
            tokens_before=result.original_token_count,
            tokens_after=result.token_count,
            messages_before=len(messages),
            messages_after=len(compacted),
        )
        return compacted, True, stats

    return messages, False, None


def _session_messages_to_transcript(messages: list[ChatMessage]) -> str:
    """Convert session ChatMessages to JSONL transcript for ``--resume``.

    Reconstructs proper ``tool_use`` and ``tool_result`` content blocks from
    :attr:`ChatMessage.tool_calls` and :attr:`ChatMessage.tool_call_id` so the
    Claude CLI receives full structural context when no previous transcript file
    is available (e.g. first turn after a storage failure or compaction drop).

    This gives the model the same fidelity as an on-disk session JSONL file —
    preserving tool call names, IDs, inputs, and *complete* (un-truncated)
    tool results — rather than the plain-text injection produced by
    :func:`_format_conversation_context` (which renders tool calls and results
    as text and discards structural linkage).

    Args:
        messages: Prior session messages, typically ``session.messages[:-1]``
            (all turns except the current user query).

    Returns:
        A JSONL string suitable for writing to a temp file and passing as
        ``ClaudeAgentOptions.resume``.  Returns an empty string if the input
        list is empty after filtering compaction entries.
    """
    filtered = filter_compaction_messages(messages)
    if not filtered:
        return ""
    builder = TranscriptBuilder()
    for msg in filtered:
        if msg.role == "user" and msg.content:
            builder.append_user(msg.content)
        elif msg.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls or []:
                try:
                    tc_input: dict[str, Any] = json.loads(
                        tc.get("function", {}).get("arguments", "{}")
                    )
                except (json.JSONDecodeError, ValueError):
                    tc_input = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "input": tc_input,
                    }
                )
            if blocks:
                builder.append_assistant(blocks)
        elif msg.role == "tool":
            if msg.tool_call_id:
                builder.append_tool_result(
                    tool_use_id=msg.tool_call_id,
                    content=msg.content or "",
                )
            else:
                # Malformed tool message — no tool_call_id to link to an
                # assistant tool_use block.  Skip to avoid an unmatched
                # tool_result entry in the builder (which would confuse --resume).
                logger.warning("[SDK] Skipping tool gap message with no tool_call_id")
    return builder.to_jsonl()


def _format_conversation_context(messages: list[ChatMessage]) -> str | None:
    """Format conversation messages into a context prefix for the user message.

    Includes user messages, assistant text, tool call summaries, and
    tool result summaries so the agent retains full context about what
    tools were invoked and their outcomes.

    The server blocks a first-turn row carries (``<available_skills>``,
    ``<memory_context>``, ``<user_context>`` …) are peeled off: inside the
    history the system prompt tells the model to ignore them, and the
    no-``--resume`` fallback re-sends fresh copies of all but
    ``<memory_context>`` ahead of the history.

    Returns None if there are no messages to format.
    """
    if not messages:
        return None

    # Filter out compaction messages first, then format
    messages = filter_compaction_messages(messages)

    lines: list[str] = []
    for msg in messages:
        if msg.role == "user":
            if content := strip_injected_context_for_display(msg.content or ""):
                lines.append(f"User: {content}")
        elif msg.role == "assistant":
            if msg.content:
                lines.append(f"You responded: {msg.content}")
            # A tool is written under the name the SDK declares it by: the
            # model copies a bare ``run_capability`` from text into a call
            # the CLI refuses as undeclared.
            for call in msg.tool_calls or []:
                function = call.get("function", {}) if isinstance(call, dict) else {}
                name = sdk_declared_tool_name(function.get("name") or "tool")
                lines.append(f"You called {name}: {function.get('arguments') or ''}")
        elif msg.role == "tool":
            # Rendered as the compressor left it.  Cutting every tool result
            # to 500 characters here made the compression budget a fiction,
            # and dropping tool calls entirely hid what the agent had done.
            lines.append(f"Tool output: {msg.content or ''}")

    if not lines:
        return None

    return "<conversation_history>\n" + "\n".join(lines) + "\n</conversation_history>"


async def _build_query_message(
    current_message: str,
    session: ChatSession,
    use_resume: bool,
    transcript_msg_count: int,
    session_id: str,
    *,
    session_msg_ceiling: int | None = None,
    target_tokens: int | None = None,
    prior_messages: "list[ChatMessage] | None" = None,
    expect_compaction: bool = False,
    context_prefix: str = "",
    turn_summary: TurnSummaryCollector | None = None,
) -> tuple[str, "CompactionStats | None"]:
    """Build the query message with appropriate context.

    When ``use_resume=True``, the CLI has the full session via ``--resume``;
    only a gap-fill prefix is injected when the transcript is stale.

    When ``use_resume=False``, the CLI starts a fresh session with no prior
    context, so the full prior session is always compressed and injected via
    ``_format_conversation_context``.  ``compress_context`` handles size
    reduction internally (LLM summarize → content truncate → middle-out delete
    → first/last trim).  ``target_tokens`` decreases on each retry to force
    progressively more aggressive compression when the first attempt exceeds
    context limits.

    Args:
        session_msg_ceiling: If provided, treat ``session.messages`` as if it
            only has this many entries when computing the gap slice.  Pass
            ``len(session.messages)`` captured *before* appending any pending
            messages so that mid-turn drains do not skew the gap calculation
            and cause pending messages to be duplicated in both the gap context
            and ``current_message``.
        expect_compaction: What the caller's pre-check predicted.  Logged only,
            beside the branch actually taken, so a row that opens without
            compression (or compression with no row) is one grep away.
        context_prefix: The first-turn server blocks to lead the query with
            when ``use_resume=False`` on a session with history — the fresh
            CLI session has nothing else to carry them.  Ignored on the
            ``--resume`` paths and below the bare-message token floor.  Its
            size is taken out of ``target_tokens`` before the history is
            compressed.

    Returns:
        Tuple of (query_message, compaction_stats or None).
    """
    msg_count = len(session.messages)
    # Use the ceiling if supplied (prevents pending-message duplication when
    # messages were appended to session.messages after the drain but before
    # this function is called).
    effective_count = (
        session_msg_ceiling if session_msg_ceiling is not None else msg_count
    )
    # Exclude the current user message and any pending messages appended after
    # the ceiling snapshot — only history up to effective_count-1 is in scope.
    # max(0, ...) guards against a theoretical 0-message ceiling (brand-new
    # session) where -1 would select all-but-last instead of an empty slice.
    prior_rows = session.messages[: max(0, effective_count - 1)]
    # ``role="reasoning"`` rows are persisted for frontend replay only and are
    # never present in the CLI JSONL (extended_thinking is embedded inside
    # assistant entries), so no history built here includes them.
    prior = [m for m in prior_rows if m.role != "reasoning"]

    # Every branch below makes at most one ``_compress_messages`` call and then
    # returns, so this is written exactly once per invocation.
    compaction_stats: CompactionStats | None = None

    logger.info(
        "[SDK] [%s] Context path: use_resume=%s, transcript_msg_count=%d,"
        " db_msg_count=%d, target_tokens=%s, expect_compaction=%s,"
        " prior_messages=%s",
        session_id[:8],
        use_resume,
        transcript_msg_count,
        msg_count,
        target_tokens,
        expect_compaction,
        len(prior_messages) if prior_messages is not None else None,
    )

    if use_resume and transcript_msg_count > 0:
        return await _resume_query_with_gap(
            current_message,
            prior_rows,
            transcript_msg_count=transcript_msg_count,
            msg_count=msg_count,
            session_id=session_id,
            target_tokens=target_tokens,
            turn_summary=turn_summary,
        )

    elif not use_resume and effective_count > 1:
        # No --resume: the CLI starts a fresh session with no prior context.
        # Injecting only the post-transcript gap would omit the transcript-covered
        # prefix entirely, so always compress the full prior session here.
        # compress_context handles size reduction internally (LLM summarize →
        # content truncate → middle-out delete → first/last trim).

        # Final escape hatch: if the token budget is at or below the floor,
        # the model context is so tight that even fully compressed history
        # would risk a "prompt too long" error.  Return the bare message so
        # the user always gets a response rather than a hard failure.
        if target_tokens is not None and target_tokens <= _BARE_MESSAGE_TOKEN_FLOOR:
            logger.warning(
                "[SDK] [%s] target_tokens=%d at or below floor (%d) —"
                " skipping history injection to guarantee response delivery"
                " (session has %d messages)",
                session_id[:8],
                target_tokens,
                _BARE_MESSAGE_TOKEN_FLOOR,
                msg_count,
            )
            return current_message, compaction_stats

        source = prior_messages if prior_messages is not None else prior
        history_target = target_tokens
        if target_tokens is not None and context_prefix:
            history_target = max(
                _BARE_MESSAGE_TOKEN_FLOOR,
                target_tokens - estimate_token_count_str(context_prefix),
            )
        logger.warning(
            "[SDK] [%s] No --resume for %d-message session — compressing context "
            "(source=%s, target_tokens=%s, context_prefix_chars=%d)",
            session_id[:8],
            msg_count,
            "transcript+gap" if prior_messages is not None else "full-db",
            history_target,
            len(context_prefix),
        )
        compressed, _was_compressed, stats = await _compress_messages(
            source, history_target
        )
        compaction_stats = stats
        history_context = _format_conversation_context(compressed)
        if history_context:
            logger.info(
                "[SDK] [%s] Fallback context built: compressed=%s, context_bytes=%d",
                session_id[:8],
                stats is not None,
                len(history_context),
            )
            if turn_summary is not None:
                turn_summary.note_history_rows(
                    len(source), compressed=stats is not None
                )
            return (
                f"{context_prefix}{history_context}"
                f"\n\nNow, the user says:\n{current_message}",
                compaction_stats,
            )
        logger.warning(
            "[SDK] [%s] Fallback context empty after compression"
            " (%d messages) — sending message without history",
            session_id[:8],
            len(source),
        )
        return context_prefix + current_message, compaction_stats

    return current_message, compaction_stats


async def _resume_query_with_gap(
    current_message: str,
    prior_rows: list[ChatMessage],
    *,
    transcript_msg_count: int,
    msg_count: int,
    session_id: str,
    target_tokens: int | None,
    turn_summary: TurnSummaryCollector | None = None,
) -> tuple[str, "CompactionStats | None"]:
    """The query for a ``--resume`` turn: the rows the transcript lacks, then
    the message.

    ``transcript_msg_count`` is the next uncovered DB sequence, so the gap is
    every loaded row at or above it, plus the rows between it and the start of
    a capped window, which are fetched from the DB.  Error markers and
    compaction rows are never sent.
    """
    hole_range = missing_range(prior_rows, transcript_msg_count)
    hole = (
        await fetch_missing_rows(session_id, hole_range, "[SDK]") if hole_range else []
    )
    gap = sendable_gap((hole or []) + uncovered_rows(prior_rows, transcript_msg_count))
    rows_left_out = hole_range is not None and hole is None
    if rows_left_out:
        logger.warning(
            "[SDK] [%s] Cap-engaged + empty sequence-based gap: window may"
            " not contain post-watermark rows (transcript=%d, db=%d)",
            session_id[:8],
            transcript_msg_count,
            msg_count,
        )
    if not gap:
        if not rows_left_out:
            logger.info(
                "[SDK] [%s] --resume covers full context (%d messages)",
                session_id[:8],
                transcript_msg_count,
            )
        return current_message, None
    compressed, _was_compressed, stats = await _compress_messages(gap, target_tokens)
    gap_context = _format_conversation_context(compressed)
    if not gap_context:
        logger.warning(
            "[SDK] [%s] Transcript stale: gap produced empty context"
            " (%d msgs, transcript=%d/%d) — sending message without gap prefix",
            session_id[:8],
            len(gap),
            transcript_msg_count,
            msg_count,
        )
        return current_message, stats
    _log_gap_sent(
        session_id, hole_range, transcript_msg_count, msg_count, gap, stats, gap_context
    )
    if turn_summary is not None:
        turn_summary.note_history_rows(len(gap), compressed=stats is not None)
    return f"{gap_context}\n\nNow, the user says:\n{current_message}", stats


def _log_gap_sent(
    session_id: str,
    hole_range: tuple[int, int] | None,
    transcript_msg_count: int,
    msg_count: int,
    gap: list[ChatMessage],
    stats: "CompactionStats | None",
    gap_context: str,
) -> None:
    if hole_range is not None:
        logger.info(
            "[SDK] [%s] Cap-engaged stale watermark (%d) — sequence-"
            "based gap=%d msgs from windowed view (compressed=%s, "
            "context_bytes=%d)",
            session_id[:8],
            transcript_msg_count,
            len(gap),
            stats is not None,
            len(gap_context),
        )
        return
    logger.info(
        "[SDK] Transcript stale: covers %d of %d messages, "
        "gap=%d (compressed=%s), gap_context_bytes=%d",
        transcript_msg_count,
        msg_count,
        len(gap),
        stats is not None,
        len(gap_context),
    )


def _transcript_watermark_after_turn(messages: list[ChatMessage]) -> int:
    """The watermark to store with the session file a finished turn wrote.

    It is the next uncovered DB sequence over every row of the turn.  A
    mid-turn follow-up row is covered like any other: its text reached the CLI
    as the PostToolUse hook's additionalContext, which the CLI keeps in the
    session file and replays on resume.  A lower watermark re-sends the rows
    that follow the follow-up, not the follow-up.
    """
    return next_uncovered_sequence(messages)


# Claude API vision-supported image types.
_VISION_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})

# Max size for embedding images directly in the user message (20 MiB raw).
_MAX_INLINE_IMAGE_BYTES = 20 * 1024 * 1024

# Matches characters unsafe for filenames.
_UNSAFE_FILENAME = re.compile(r"[^\w.\-]")


def _save_to_sdk_cwd(sdk_cwd: str, filename: str, content: bytes) -> str:
    """Write file content to the SDK ephemeral directory.

    Returns the absolute path.  Adds a numeric suffix on name collisions.
    """
    safe = _UNSAFE_FILENAME.sub("_", filename) or "file"
    candidate = os.path.join(sdk_cwd, safe)
    if os.path.exists(candidate):
        stem, ext = os.path.splitext(safe)
        idx = 1
        while os.path.exists(candidate):
            candidate = os.path.join(sdk_cwd, f"{stem}_{idx}{ext}")
            idx += 1
    with open(candidate, "wb") as f:
        f.write(content)
    return candidate


class PreparedAttachments(BaseModel):
    """Result of preparing file attachments for a query."""

    hint: str = ""
    """Text hint describing the files (appended to the user message)."""

    image_blocks: list[dict[str, Any]] = []
    """Claude API image content blocks to embed in the user message."""


async def _prepare_file_attachments(
    file_ids: list[str],
    user_id: str,
    session_id: str,
    sdk_cwd: str,
) -> PreparedAttachments:
    """Download workspace files and prepare them for Claude.

    Images (PNG/JPEG/GIF/WebP) are embedded directly as vision content blocks
    in the user message so Claude can see them without tool calls.

    Non-image files (PDFs, text, etc.) are referenced by ``file_id`` so the
    model retrieves them with ``read_workspace_file``, which works in every
    execution mode. A copy is also written to *sdk_cwd* for non-E2B tooling,
    but the model is never pointed at that host-side path.

    Returns a :class:`PreparedAttachments` with a text hint and any image
    content blocks.
    """
    empty = PreparedAttachments(hint="", image_blocks=[])
    if not file_ids or not user_id:
        return empty

    try:
        manager = await get_workspace_manager(user_id, session_id)
    except Exception:
        logger.warning(
            "Failed to create workspace manager for file attachments",
            exc_info=True,
        )
        return empty

    image_blocks: list[dict[str, Any]] = []
    file_descriptions: list[str] = []

    for fid in file_ids:
        try:
            file_info = await manager.get_file_info(fid)
            if file_info is None:
                continue
            content = await manager.read_file_by_id(fid)
            mime = (file_info.mime_type or "").split(";")[0].strip().lower()

            # Images: embed directly in the user message as vision blocks
            if mime in _VISION_MIME_TYPES and len(content) <= _MAX_INLINE_IMAGE_BYTES:
                b64 = base64.b64encode(content).decode("ascii")
                image_blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": b64,
                        },
                    }
                )
                file_descriptions.append(
                    f"- {file_info.name} ({mime}, "
                    f"{file_info.size_bytes:,} bytes) [embedded as image]"
                )
            else:
                # Non-image files: surface via read_workspace_file by file_id.
                # In E2B mode (prod default) the model's file tools operate on
                # the remote sandbox filesystem, so a host-side sdk_cwd path
                # is unreadable there — the workspace tool works in both modes.
                # A local copy is still written for non-E2B tooling, but the
                # model is never pointed at that path.
                _save_to_sdk_cwd(sdk_cwd, file_info.name, content)
                # ``file_id=<uuid>`` (not ``file_id:``) — chat-share
                # allowlisting extracts references via _FILE_ID_RE, which
                # matches exactly this shape (data/sharing/workspace_refs.py).
                file_descriptions.append(
                    f"- {file_info.name} ({mime}, "
                    f"{file_info.size_bytes:,} bytes) file_id={fid}"
                )
        except Exception:
            logger.warning("Failed to prepare file %s", fid[:12], exc_info=True)

    if not file_descriptions:
        return empty

    noun = "file" if len(file_descriptions) == 1 else "files"
    has_non_images = len(file_descriptions) > len(image_blocks)
    read_hint = (
        " Use the read_workspace_file tool with the file_id to view non-image"
        " files (pass save_to_path to copy one into the working directory)."
        if has_non_images
        else ""
    )
    hint = (
        f"[The user attached {len(file_descriptions)} {noun}.{read_hint}\n"
        + "\n".join(file_descriptions)
        + "]"
    )
    return PreparedAttachments(hint=hint, image_blocks=image_blocks)


@dataclass
class _StreamAccumulator:
    """Mutable state accumulated during a single streaming attempt.

    Tracks the assistant message being built, tool calls, and flags that
    control session-message bookkeeping in the dispatch loop.
    """

    assistant_response: ChatMessage
    accumulated_tool_calls: list[dict[str, Any]]
    has_appended_assistant: bool = False
    has_tool_results: bool = False
    stream_completed: bool = False
    thinking_stripper: ThinkingStripper = dataclass_field(
        default_factory=ThinkingStripper,
    )
    # Currently-open reasoning block for this turn.  Each StreamReasoningStart
    # creates a new ChatMessage(role="reasoning"), each delta appends to its
    # content, and StreamReasoningEnd clears the reference.  Rows are persisted
    # inline with text/tool rows so they survive session reload; the reader
    # filters role="reasoning" out of LLM context.
    reasoning_response: ChatMessage | None = None
    tool_display_names: dict[str, str] = dataclass_field(default_factory=dict)


def _dispatch_response(
    response: StreamBaseResponse,
    acc: "_StreamAccumulator",
    ctx: "_StreamContext",
    state: "_RetryState",
    entries_replaced: bool,
    log_prefix: str,
    skip_strip: bool = False,
) -> StreamBaseResponse | None:
    """Process a single adapter response and update session/accumulator state.

    Returns the response to yield to the client, or `None` if the response
    should be suppressed (e.g. `StreamStart` duplicates).

    Handles:
    - Logging tool events and errors
    - Persisting error markers
    - Accumulating text deltas into `assistant_response`
    - Appending tool input/output to session messages and transcript
    - Detecting `StreamFinish`

    Args:
        skip_strip: When True, bypass ThinkingStripper.process() for this delta.
            Used for the flushed tail delta which is already stripped content.
    """
    if isinstance(response, StreamStart):
        return None

    if isinstance(
        response,
        (StreamToolInputAvailable, StreamToolOutputAvailable),
    ):
        extra = ""
        if isinstance(response, StreamToolOutputAvailable):
            out_len = len(str(response.output))
            extra = f", output_len={out_len}"
        logger.info(
            "%s Tool event: %s, tool=%s%s",
            log_prefix,
            type(response).__name__,
            getattr(response, "toolName", "N/A"),
            extra,
        )

    # Persist error markers so they survive page refresh
    if isinstance(response, StreamError):
        logger.log(
            logging.INFO if response.code in _EXPECTED_STOP_CODES else logging.ERROR,
            "%s Sending error to frontend: %s (code=%s)",
            log_prefix,
            response.errorText,
            response.code,
        )
        failure = _provider_failure_for(ctx)
        _append_error_marker(
            ctx.session,
            failure.message if failure else response.errorText,
            retryable=(
                failure.retryable
                if failure
                else response.code in _RETRYABLE_STREAM_ERROR_CODES
            ),
            failure=failure.as_part() if failure else None,
        )

    # Persist the pause so the Paused card survives the end of the stream.
    if isinstance(response, StreamTurnPaused):
        pause = TurnPause.model_validate(response.pause)
        logger.info(
            "%s Turn paused (%s, checkpoint_saved=%s)",
            log_prefix,
            pause.reason,
            pause.checkpoint_saved,
        )
        append_pause_marker(ctx.session, pause, tainted=ctx.tainted)

    if isinstance(response, StreamReasoningStart):
        acc.reasoning_response = ChatMessage(role="reasoning", content="")
        messages = ctx.session.messages
        # The post-tool placeholder pre-created for this message's text goes
        # after its thinking, in the order the wire carries them.
        placeholder = acc.assistant_response
        if (
            messages
            and messages[-1] is placeholder
            and placeholder.sequence is None
            and not placeholder.content
            and not placeholder.tool_calls
        ):
            messages.insert(len(messages) - 1, acc.reasoning_response)
        else:
            messages.append(acc.reasoning_response)

    elif isinstance(response, StreamReasoningDelta):
        if acc.reasoning_response is not None:
            acc.reasoning_response.content = (acc.reasoning_response.content or "") + (
                response.delta or ""
            )

    elif isinstance(response, StreamReasoningEnd):
        acc.reasoning_response = None

    elif isinstance(response, StreamTextDelta):
        raw_delta = response.delta or ""
        if skip_strip:
            # Pre-stripped tail from ThinkingStripper.flush() — bypass process()
            # to avoid re-suppressing content that looks like a partial tag opener.
            delta = raw_delta
        else:
            # Strip <internal_reasoning> / <thinking> tags that non-extended-
            # thinking models (e.g. Sonnet) may emit as visible text.
            delta = acc.thinking_stripper.process(raw_delta)
            if not delta:
                # Stripper is buffering a potential tag — suppress this event.
                return None
        # Replace the delta with the stripped version for the SSE client.
        response = StreamTextDelta(id=response.id, delta=delta)
        if acc.has_tool_results and acc.has_appended_assistant:
            acc.assistant_response = ChatMessage(role="assistant", content=delta)
            acc.accumulated_tool_calls = []
            acc.has_appended_assistant = False
            acc.has_tool_results = False
            ctx.session.messages.append(acc.assistant_response)
            acc.has_appended_assistant = True
        else:
            acc.assistant_response.content = (
                acc.assistant_response.content or ""
            ) + delta
            if not acc.has_appended_assistant:
                ctx.session.messages.append(acc.assistant_response)
                acc.has_appended_assistant = True

    elif isinstance(response, StreamToolDisplayAvailable):
        display = response.data
        acc.tool_display_names[display.toolCallId] = display.displayName
        stamp_tool_display_name(
            [acc.assistant_response, *ctx.session.messages],
            display.toolCallId,
            display.displayName,
        )

    elif isinstance(response, StreamToolInputAvailable):
        # A new assistant row opens after tool results, for a call as for text.
        if acc.has_tool_results and acc.has_appended_assistant:
            acc.assistant_response = ChatMessage(role="assistant", content="")
            acc.accumulated_tool_calls = []
            acc.has_appended_assistant = False
            acc.has_tool_results = False
        acc.accumulated_tool_calls.append(
            {
                "id": response.toolCallId,
                "type": "function",
                "function": {
                    "name": response.toolName,
                    "arguments": json.dumps(response.input or {}),
                },
            }
        )
        if name := acc.tool_display_names.get(response.toolCallId):
            acc.accumulated_tool_calls[-1]["display_name"] = name
        acc.assistant_response.tool_calls = acc.accumulated_tool_calls
        acc.assistant_response.mark_tool_calls_pending_save()
        if not acc.has_appended_assistant:
            ctx.session.messages.append(acc.assistant_response)
            acc.has_appended_assistant = True

    elif isinstance(response, StreamToolOutputAvailable):
        # Dedupe: the response adapter can emit the same tool_use_id more than
        # once when the CLI re-delivers a ToolResultBlock (e.g. after a retry
        # or when a parallel-tool UserMessage is processed alongside a flush).
        # Guard at persistence time — the first emission already wrote the row
        # (via the pop_pending_tool_output stash, so it has clean text), and a
        # duplicate would land a second row with the raw MCP list fallback
        # content (breaking frontend widgets and inflating conversation tokens).
        already_persisted = any(
            m.role == "tool" and m.tool_call_id == response.toolCallId
            for m in ctx.session.messages
        )
        if already_persisted:
            logger.info(
                "%s Skipping duplicate tool_result for toolCallId=%s",
                log_prefix,
                response.toolCallId,
            )
            # Return None so the caller's ``if dispatched is not None: yield``
            # short-circuits — the duplicate event stays off the SSE stream
            # (so the frontend doesn't render a second widget) and the
            # mid-turn follow-up persist doesn't double-fire (its guard is
            # ``dispatched is not None``).
            return None
        content = (
            response.output
            if isinstance(response.output, str)
            else json.dumps(response.output, ensure_ascii=False)
        )
        ctx.session.messages.append(
            ChatMessage(
                role="tool",
                content=content,
                tool_call_id=response.toolCallId,
                name=_resolve_tool_result_name(
                    ctx.session, response.toolCallId, response.toolName
                ),
            )
        )
        if not entries_replaced:
            state.transcript_builder.append_tool_result(
                tool_use_id=response.toolCallId,
                content=content,
            )
        acc.has_tool_results = True

    elif isinstance(response, StreamFinish):
        acc.stream_completed = True

    return response


class _BuildingModeRestart(Exception):
    """Raised by ``_consume_sdk_until_done`` when ``enter_agent_building_mode``
    was called: the outer loop relaunches this attempt with the agent-building
    guide in the system prompt (``--resume`` on the same CLI session).

    NOT an error and NOT a rollback — everything the attempt produced so far
    (tool calls, results, text) stands, in the client stream, the session
    history, and the CLI session file.
    """


class _HandledStreamError(Exception):
    """Raised by `_run_stream_attempt` when an attempt fails and the outer
    retry loop must roll back session state.

    Two sub-cases:

    * ``already_yielded=True`` (default) — a ``StreamError`` was already sent
      to the client inside ``_run_stream_attempt`` (circuit-breaker, idle
      timeout, etc.).  The outer loop must **not** yield another one.
    * ``already_yielded=False`` — the error is transient and the outer loop
      will decide whether to retry or surface the error.  If retrying it
      yields a ``StreamStatus("retrying…")``; if exhausted it yields the
      ``StreamError`` itself so the client sees it only once.

    Attributes:
        error_msg: The user-facing error message to persist.
        code: Machine-readable error code (e.g. ``circuit_breaker_empty_tool_calls``).
        retryable: Whether the frontend should offer a retry button.
        already_yielded: ``True`` when ``StreamError`` was already sent to the
            client before this exception was raised.
    """

    def __init__(
        self,
        message: str,
        error_msg: str | None = None,
        code: str | None = None,
        retryable: bool = True,
        already_yielded: bool = True,
    ):
        super().__init__(message)
        self.error_msg = error_msg
        self.code = code
        self.retryable = retryable
        self.already_yielded = already_yielded


@dataclass(frozen=True)
class _HandledErrorInfo:
    """Carries a `_HandledStreamError`'s decisions out of the retry loop.

    Set inside the `except _HandledStreamError` branch and consumed by the
    post-loop block, which restores the partial and (if the inner handler
    didn't already do it) yields the client-facing StreamError.
    """

    error_msg: str
    code: str
    retryable: bool
    already_yielded: bool


@dataclass
class _EmptyToolBreakResult:
    """Result of checking for empty tool calls in a single AssistantMessage."""

    count: int  # Updated consecutive counter
    tripped: bool  # Whether the circuit breaker fired
    error: StreamError | None  # StreamError to yield (if tripped)
    error_msg: str | None  # Error message (if tripped)
    error_code: str | None  # Error code (if tripped)


@functools.cache
def _no_arg_tool_names() -> frozenset[str]:
    """Tool names whose schema declares zero arguments (required + properties).

    A ``ToolUseBlock`` with ``input == {}`` for one of these is a *legitimate*
    invocation of a no-arg tool, NOT the model-saturation failure mode the
    empty-tool-call breaker targets (sessions where the model emits ``{}`` for
    EVERY arg-needing tool because context-saturation broke argument
    serialization). Cached because the registry is module-level and immutable
    after import. Includes both bare and MCP-prefixed names because
    ``ToolUseBlock.name`` carries the MCP prefix when the tool is registered
    through the copilot MCP server.
    """
    from backend.copilot.tools import TOOL_REGISTRY

    bare = {
        name
        for name, tool in TOOL_REGISTRY.items()
        if not (tool.parameters.get("required") or tool.parameters.get("properties"))
    }
    return frozenset(bare | {f"{MCP_TOOL_PREFIX}{name}" for name in bare})


def _check_empty_tool_breaker(
    sdk_msg: object,
    consecutive: int,
    ctx: _StreamContext,
    state: "_RetryState",
) -> _EmptyToolBreakResult:
    """Detect consecutive empty tool calls and trip the circuit breaker.

    Returns an ``_EmptyToolBreakResult`` with the updated counter and, if the
    breaker tripped, the ``StreamError`` to yield plus the error metadata.
    """
    if not isinstance(sdk_msg, AssistantMessage):
        return _EmptyToolBreakResult(consecutive, False, None, None, None)

    no_arg = _no_arg_tool_names()
    empty_tools = [
        b.name
        for b in sdk_msg.content
        if isinstance(b, ToolUseBlock) and not b.input and b.name not in no_arg
    ]
    if not empty_tools:
        # Reset on any non-empty-tool AssistantMessage (including text-only
        # messages — any() over empty content is False).  Legitimate no-arg
        # tools (e.g. ``get_agent_building_guide``) also reset the counter
        # since they're a normal model action, not the saturation failure.
        return _EmptyToolBreakResult(0, False, None, None, None)

    consecutive += 1

    # Log full diagnostics on first occurrence only; subsequent hits just
    # log the counter to reduce noise.
    if consecutive == 1:
        logger.warning(
            "%s Empty tool call detected (%d/%d): "
            "tools=%s, model=%s, error=%s, "
            "block_types=%s, cumulative_usage=%s",
            ctx.log_prefix,
            consecutive,
            _EMPTY_TOOL_CALL_LIMIT,
            empty_tools,
            sdk_msg.model,
            sdk_msg.error,
            [type(b).__name__ for b in sdk_msg.content],
            {
                "prompt": state.usage.prompt_tokens,
                "completion": state.usage.completion_tokens,
                "cache_read": state.usage.cache_read_tokens,
            },
        )
    else:
        logger.warning(
            "%s Empty tool call detected (%d/%d): tools=%s",
            ctx.log_prefix,
            consecutive,
            _EMPTY_TOOL_CALL_LIMIT,
            empty_tools,
        )

    if consecutive < _EMPTY_TOOL_CALL_LIMIT:
        return _EmptyToolBreakResult(consecutive, False, None, None, None)

    logger.error(
        "%s Circuit breaker: aborting stream after %d "
        "consecutive empty tool calls. "
        "This is likely caused by the model attempting "
        "to write content too large for a single tool "
        "call's output token limit. The model should "
        "write large files in chunks using bash_exec "
        "with cat >> (append).",
        ctx.log_prefix,
        consecutive,
    )
    error_msg = _CIRCUIT_BREAKER_ERROR_MSG
    error_code = "circuit_breaker_empty_tool_calls"
    _append_error_marker(ctx.session, error_msg, retryable=True)
    return _EmptyToolBreakResult(
        count=consecutive,
        tripped=True,
        error=StreamError(errorText=error_msg, code=error_code),
        error_msg=error_msg,
        error_code=error_code,
    )


async def _run_stream_attempt(
    ctx: _StreamContext,
    state: "_RetryState",
) -> AsyncIterator[StreamBaseResponse]:
    """Run one SDK streaming attempt.

    Opens a `ClaudeSDKClient`, sends the query, iterates SDK messages with
    heartbeat timeouts, dispatches adapter responses, and performs post-stream
    cleanup (safety-net flush, CLI-side end-of-stream notice when the
    iterator drains without a ``ResultMessage``).

    Yields stream events.  On stream error the exception propagates to the
    caller so the retry loop can rollback and retry.

    Args:
        ctx: Per-request context shared across retry attempts.  Scalar
            fields (IDs, paths, message string) are set once and never
            reassigned.  `session`, `compaction`, and `lock` are
            shared mutable references: `session.messages` is rolled back
            on retry, `compaction` tracks mid-stream compaction events,
            and `lock` is refreshed during heartbeats.  Their references
            are constant even though the objects they point to are mutated.
        state: Mutable retry state — holds values that the retry loop
            modifies between attempts (options, query, adapter, etc.).

    See also:
        `stream_chat_completion_sdk` — owns the retry loop that calls this
        function up to `_MAX_STREAM_ATTEMPTS` times with reduced context.
    """
    # First, so whatever fails from here on is this launch's to absorb.
    await state.generations.start_launch()
    acc = _StreamAccumulator(
        assistant_response=ChatMessage(role="assistant", content=""),
        accumulated_tool_calls=[],
    )
    # --- Intermediate persistence tracking ---
    # Flush session messages to DB periodically so page reloads show progress
    # during long-running turns (see incident d2f7cba3: 82-min turn lost on refresh).

    # Use manual __aenter__/__aexit__ instead of ``async with`` so we can
    # suppress SDK cleanup errors that occur when the SSE client disconnects
    # mid-stream.  GeneratorExit causes the SDK's ``__aexit__`` to run in a
    # different async context/task than where the client was opened, which
    # triggers:
    #   - ValueError: ContextVar token mismatch (AUTOGPT-SERVER-8BT)
    #   - RuntimeError: cancel scope in wrong task  (AUTOGPT-SERVER-8BW)
    # Both are harmless — the TCP connection is already dead.
    # CLI subprocess spawn + MCP init can take seconds on cold starts —
    # narrate it so the status doesn't sit on the context-prep message.
    yield StreamStatus(message="Starting the assistant…")
    # An interruption can precede ResultMessage. A building-mode relaunch
    # charges the interrupted run's unreported tokens first and keeps the
    # baseline; a transient retry clears it (see ``_do_transient_backoff``).
    # The baseline is the total the CLI restores, and only ``--resume``
    # restores one: a launch with ``session_id`` starts at zero, so don't
    # read its file.
    if state.usage.cli_session_total_usd is None:
        baseline = await asyncio.to_thread(
            read_cli_session_usage, ctx.sdk_cwd, state.options.resume, ctx.log_prefix
        )
        state.usage.start_cli_session(baseline)
    sdk_client = ClaudeSDKClient(options=launch_options(state.options))
    client = await sdk_client.__aenter__()
    try:
        logger.info(
            "%s Sending query — resume=%s, total_msgs=%d, "
            "query_len=%d, attached_files=%d, image_blocks=%d",
            ctx.log_prefix,
            state.use_resume,
            len(ctx.session.messages),
            len(state.query_message),
            len(ctx.file_ids) if ctx.file_ids else 0,
            len(ctx.attachments.image_blocks),
        )

        ctx.compaction.reset_for_query()

        # Narrate the silent gap between dispatching the query and the
        # SDK's first real chunk — usually <1s but can stretch to several
        # seconds on cold-starts or large contexts. The frontend prefers
        # this over the generic "Thinking…" copy; fast turns replace it
        # with content immediately.
        yield StreamStatus(message="Reading your message\u2026")

        if ctx.attachments.image_blocks:
            content_blocks: list[dict[str, Any]] = [
                *ctx.attachments.image_blocks,
                {"type": "text", "text": state.query_message},
            ]
            user_msg = {
                "type": "user",
                "message": {"role": "user", "content": content_blocks},
                "parent_tool_use_id": None,
                "session_id": ctx.session_id,
            }
            if client._transport is None:  # noqa: SLF001
                raise RuntimeError("ClaudeSDKClient transport is not initialized")
            await client._transport.write(json.dumps(user_msg) + "\n")  # noqa: SLF001
            state.transcript_builder.append_user(
                content=[
                    *ctx.attachments.image_blocks,
                    {"type": "text", "text": ctx.current_message},
                ]
            )
        else:
            await client.query(state.query_message, session_id=ctx.session_id)
            state.transcript_builder.append_user(content=ctx.current_message)

        loop_state = _SDKLoopState(
            last_real_msg_time=time.monotonic(),
            last_flush_time=time.monotonic(),
        )

        async for ev in _consume_sdk_until_done(client, ctx, state, acc, loop_state):
            yield ev

        if (
            state.adapter.pending_thinking_only_reprompt
            and not state.thinking_only_reprompted
            and not loop_state.ended_with_stream_error
        ):
            state.adapter.pending_thinking_only_reprompt = False
            state.thinking_only_reprompted = True
            state.adapter.thinking_only_reprompted = True
            # Re-prompt round must still trip the placeholder guard if model returns thinking-only again.
            state.adapter._text_since_last_tool_result = False
            # Round 1's thinking content must not be surfaced as round 2's
            # promote-thinking fallback if round 2 itself produces no
            # thinking — that would show stale reasoning to the user as
            # if it were the answer to the re-prompt.
            state.adapter._last_thinking_content = ""
            acc.stream_completed = False
            # The previous round's tool_result is no longer "fresh" for the
            # post-tool placeholder pre-create branch — clearing prevents the
            # re-prompt round from spuriously appending an empty assistant
            # ChatMessage before its first text delta lands.
            acc.has_tool_results = False
            # Force the re-prompt's first text delta to allocate a NEW
            # ``acc.assistant_response`` ChatMessage instead of accumulating
            # into the previous (empty thinking-only) one.  Without this the
            # two logical assistant turns get fused into a single DB row.
            acc.has_appended_assistant = False
            # Also swap in a fresh ``assistant_response`` so the dispatch
            # code doesn't smuggle round 1's stale ``tool_calls`` list into
            # round 2's reply when it eventually appends to session.messages
            # — that would re-persist the previous turn's tool calls beside
            # the re-prompt's text and double the assistant row.
            acc.assistant_response = ChatMessage(role="assistant", content="")
            acc.accumulated_tool_calls = []
            # Reset the empty-tool-call breaker counter so a borderline
            # round-1 streak doesn't trip prematurely on the very first
            # re-prompt AssistantMessage.
            loop_state.consecutive_empty_tool_calls = 0
            # Restart the idle-timeout clock for the re-prompt round —
            # otherwise a long round 1 (e.g. 29 min) plus a tiny delay
            # before the first re-prompt message would push the clock
            # past the 30-min threshold and trip a phantom idle timeout.
            loop_state.last_real_msg_time = time.monotonic()
            logger.info(
                "%s Re-prompting model for closing summary "
                "after thinking-only final turn",
                ctx.log_prefix,
            )
            await client.query(
                _THINKING_ONLY_REPROMPT,
                session_id=ctx.session_id,
            )
            async for ev in _consume_sdk_until_done(
                client, ctx, state, acc, loop_state
            ):
                yield ev
    finally:
        await _safe_close_sdk_client(sdk_client, ctx.log_prefix)

    # --- Post-stream processing (only on success) ---
    if state.adapter.has_unresolved_tool_calls:
        logger.warning(
            "%s %d unresolved tool(s) after stream — flushing",
            ctx.log_prefix,
            len(state.adapter.current_tool_calls)
            - len(state.adapter.resolved_tool_calls),
        )
        safety_responses: list[StreamBaseResponse] = []
        state.adapter.flush_unresolved_tool_calls(safety_responses)
        for response in safety_responses:
            if isinstance(
                response,
                (StreamToolInputAvailable, StreamToolOutputAvailable),
            ):
                logger.info(
                    "%s Safety flush: %s, tool=%s",
                    ctx.log_prefix,
                    type(response).__name__,
                    getattr(response, "toolName", "N/A"),
                )
            if isinstance(response, StreamToolOutputAvailable):
                state.transcript_builder.append_tool_result(
                    tool_use_id=response.toolCallId,
                    content=(
                        response.output
                        if isinstance(response.output, str)
                        else json.dumps(response.output, ensure_ascii=False)
                    ),
                )
            yield response

    hangup_pause: TurnPause | None = None
    if not acc.stream_completed and not loop_state.ended_with_stream_error:
        await _record_unreported_usage(
            state, ctx.sdk_cwd, ctx.session_id, ctx.log_prefix
        )
        # User cancels raise ``asyncio.CancelledError`` upstream; reaching this
        # branch means the CLI hung up — per-query budget exhausted, max_turns,
        # OOM, or crash — without ever emitting a ResultMessage.
        logger.info(
            "%s Stream ended without ResultMessage — likely CLI-side kill "
            "(budget/turns/crash)",
            ctx.log_prefix,
        )
        closing_responses: list[StreamBaseResponse] = []
        state.adapter.end_open_blocks(closing_responses)
        for r in closing_responses:
            yield r
        # While landing, the hangup ended the landing: still a pause.
        hangup_pause = (
            state.governor.hangup_pause() if state.governor is not None else None
        )
        if hangup_pause is not None:
            yield StreamTurnPaused(pause=hangup_pause.as_part())
        else:
            notice_block_id = str(uuid.uuid4())
            yield StreamTextStart(id=notice_block_id)
            yield StreamTextDelta(id=notice_block_id, delta=STREAM_INCOMPLETE_MARKER)
            yield StreamTextEnd(id=notice_block_id)
            ctx.session.messages.append(
                ChatMessage(role="assistant", content=STREAM_INCOMPLETE_MARKER)
            )

    if (
        acc.assistant_response.content or acc.assistant_response.tool_calls
    ) and not acc.has_appended_assistant:
        ctx.session.messages.append(acc.assistant_response)
    if hangup_pause is not None:
        append_pause_marker(ctx.session, hangup_pause, tainted=ctx.tainted)

    # SECRT-2333: persist a visible-marker ChatMessage when the turn ended
    # with ``ended_with_stream_error=True`` and no prior marker is on the
    # tail.  The on-wire StreamError dies with the SSE connection (client
    # disconnect, network) — this row keeps the chat history honest on
    # reload, even if the outer retry loop's ``interrupted.finalize`` later
    # appends its own COPILOT_ERROR_PREFIX entry on the post-rollback
    # restore.  The capture/restore cycle pops trailing error markers
    # before re-attaching, so this row is never double-rendered.
    if loop_state.ended_with_stream_error and not _has_trailing_marker(ctx.session):
        ctx.session.messages.append(
            ChatMessage(role="assistant", content=STREAM_ERROR_MARKER)
        )

    # Raise so the outer retry loop can rollback session messages.
    # already_yielded=False for transient_api_error: StreamError was NOT
    # sent to the client yet (the outer loop does it when retries are
    # exhausted, avoiding a premature error flash before the retry).
    if loop_state.ended_with_stream_error:
        raise _HandledStreamError(
            "Stream error handled",
            error_msg=loop_state.stream_error_msg,
            code=loop_state.stream_error_code,
            already_yielded=(
                loop_state.stream_error_code not in _OUTER_LOOP_YIELDS_ERROR_CODES
            ),
        )


async def _seed_transcript(
    session: ChatSession,
    transcript_builder: TranscriptBuilder,
    transcript_msg_count: int,
    log_prefix: str,
    msg_ceiling: int,
    seed_target: int,
) -> tuple[str, int]:
    """Seed the transcript builder from compressed DB messages.

    Called when ``use_resume=False`` and no prior transcript exists in
    storage, so the builder holds the conversation so far and a first
    retry has a transcript to compact.

    ``msg_ceiling`` is ``len(session.messages)`` as of *before* the compaction
    row was persisted.  Slicing off just the last entry is not enough: closing
    a pre-query compaction row appends a synthetic assistant+tool pair, so the
    naive ``[:-1]`` leaves the current user message inside the seed — and the
    turn appends it again via ``append_user``, duplicating the live message in
    the transcript and inflating the watermark by two.

    Returns ``(transcript_content, transcript_msg_count)`` updated values —
    ``""`` and the count passed in if seeding is not possible.
    """
    if msg_ceiling <= 1:
        return "", transcript_msg_count

    _prior = session.messages[: msg_ceiling - 1]
    _comp, _, _ = await _compress_messages(_prior, seed_target)
    if not _comp:
        return "", transcript_msg_count

    _seeded = _session_messages_to_transcript(_comp)
    if not _seeded or not validate_transcript(_seeded):
        return "", transcript_msg_count

    transcript_builder.load_previous(_seeded, log_prefix=log_prefix)
    logger.info(
        "%s Seeded transcript from %d compressed DB messages"
        " for next-turn upload (seed_target_tokens=%d)",
        log_prefix,
        len(_comp),
        seed_target,
    )
    return _seeded, len(_prior)


@dataclass
class _RestoreResult:
    """Return value from ``_restore_cli_session_for_turn``."""

    transcript_content: str = ""
    use_resume: bool = False
    resume_file: str | None = None
    transcript_msg_count: int = 0
    context_messages: "list[ChatMessage] | None" = None
    download_bytes: int | None = None


async def _restore_cli_session_for_turn(
    user_id: str | None,
    session_id: str,
    session: "ChatSession",
    sdk_cwd: str,
    transcript_builder: "TranscriptBuilder",
    log_prefix: str,
    history_ceiling: int | None = None,
) -> _RestoreResult:
    """Download, validate and restore a CLI session for ``--resume`` on this turn.

    Performs a single GCS round-trip to fetch the session bytes + message_count
    watermark.  Falls back to DB-message reconstruction when GCS has no session
    (first turn or upload missed), from the rows below *history_ceiling*
    (all of them when ``None``).

    Returns a ``_RestoreResult`` with all transcript-related state ready for the
    caller to merge into its local variables.
    """
    result = _RestoreResult()

    if not (user_id and len(session.messages) > 1):
        return result

    try:
        cli_restore = await download_transcript(
            user_id, session_id, log_prefix=log_prefix
        )
    except Exception as restore_err:
        logger.warning(
            "%s CLI session restore failed, continuing without --resume: %s",
            log_prefix,
            restore_err,
        )
        cli_restore = None
    if cli_restore is not None:
        result.download_bytes = len(cli_restore.content)

    # Only SDK-written transcripts resume. One the old fast path wrote
    # (mode="baseline": a local chat from before the one-engine upgrade) is
    # treated as no transcript, so the turn rebuilds its context from the
    # chat's rows.
    if cli_restore is not None and cli_restore.mode != "sdk":
        logger.info(
            "%s Transcript written by mode=%r is not resumed; rebuilding "
            "context from the chat's rows",
            log_prefix,
            cli_restore.mode,
        )
        cli_restore = None

    # Validate, strip, and write to disk — delegate to helper to reduce
    # function complexity.  Writing an invalid/corrupt file to disk then
    # falling back to "no --resume" would cause the CLI to fail with
    # "Session ID already in use" because the file exists at the expected
    # session path, so we validate BEFORE any disk write.
    stripped = ""
    if cli_restore is not None:
        stripped, ok = process_cli_restore(cli_restore, sdk_cwd, session_id, log_prefix)
        if not ok:
            cli_restore = None

    if cli_restore is None:
        # Validation failed or GCS returned no session.  Delete any
        # existing local session file so the CLI doesn't reject the
        # session_id with "Session ID already in use".  T1 may have
        # left a valid file at this path; we clear it so the fallback
        # path (session_id= without --resume) can create a new session.
        delete_stale_cli_session_file(sdk_cwd, session_id, log_prefix)

    if cli_restore is not None:
        result.transcript_content = stripped
        transcript_builder.load_previous(stripped, log_prefix=log_prefix)
        result.use_resume = True
        result.resume_file = session_id
        result.transcript_msg_count = cli_restore.message_count
        return result

    # No valid --resume source: build the context from the chat's rows. The
    # last row is the current message, so the rows past *history_ceiling*
    # (an auto-continue turn's other queued messages, all part of its current
    # message) are left out first.
    history = session.messages[:history_ceiling]
    context_msgs = prior_context_messages(history)
    result.context_messages = context_msgs
    result.transcript_msg_count = len(history) - 1
    logger.info(
        "%s Context built from the chat's rows: %d messages (will inject "
        "as <conversation_history>)",
        log_prefix,
        len(context_msgs),
    )

    return result


async def _maybe_prepend_builder_context(
    session: ChatSession,
    user_id: str | None,
    is_user_message: bool,
    query_message: str,
) -> str:
    """Prepend the per-turn ``<builder_context>`` block to the user message.

    No-op for non-user messages and for sessions without a bound graph.
    Extracted from the SDK stream body so Pyright's complexity analyser
    stays within budget on the already-large ``stream_chat_completion_sdk``.
    """
    if not is_user_message or not session.metadata.builder_graph_id:
        return query_message
    block = await build_builder_context_turn_prefix(session, user_id)
    return block + query_message if block else query_message


async def _maybe_prepend_skills_update(
    session: ChatSession,
    user_id: str | None,
    is_user_message: bool,
    query_message: str,
) -> str:
    """Prepend the per-turn ``<skills_update>`` drift notice, if any.

    Compares the live skill registry against the ``<available_skills>``
    index baked into the session history at session start. No-op for
    non-user turns, anonymous turns, and steady-state sessions — and for
    a query that already leads with a fresh index: the first turn, and a
    turn that could not resume and re-sends the first turn's context.
    Query-only: the notice is never persisted, so a later turn re-diffs
    from the same baseline and the reminder clears itself once the
    session restarts.
    """
    if not is_user_message or not user_id:
        return query_message
    if any(
        tag == SKILLS_CONTEXT_TAG for tag, _ in scan_context_blocks(query_message)[0]
    ):
        # A session's first index starts the baseline its notices diff from.
        await start_session_skill_versions(
            user_id, session.expert_id, session.session_id
        )
        return query_message
    try:
        notice = await build_skills_update_notice(
            user_id,
            session_id=session.session_id,
            expert_id=session.expert_id,
            prior_contents=[
                m.content or "" for m in session.messages if m.role == "user"
            ],
        )
    except Exception:
        logger.exception("[skills] failed to build skills update notice")
        return query_message
    return notice + query_message if notice else query_message


def _maybe_prepend_seen_capabilities(
    session: ChatSession,
    is_user_message: bool,
    query_message: str,
) -> str:
    """Prepend the per-turn ``<seen_capabilities>`` notice, if any.

    Derived from the tool calls persisted in ``session.messages`` so the
    model is told which ids it already described / ran and which skills it
    already loaded instead of re-discovering them after a resume or a
    compaction (SECRT-2791). Same query-only contract as the skills-update
    notice: never persisted, re-derived every turn, no-op for non-user
    turns and for a first turn with no history.
    """
    if not is_user_message:
        return query_message
    notice = build_seen_capabilities_notice(session)
    return notice + query_message if notice else query_message


async def stream_chat_completion_sdk(  # pyright: ignore[reportGeneralTypeIssues]
    session_id: str,
    message: str | None = None,
    is_user_message: bool = True,
    user_id: str | None = None,
    session: ChatSession | None = None,
    file_ids: list[str] | None = None,
    permissions: "CopilotPermissions | None" = None,
    envelope: "TurnEnvelope | None" = None,
    model: CopilotLLMModel | None = None,
    request_arrival_at: float = 0.0,
    organization_id: str | None = None,
    team_id: str | None = None,
    credential_lease: CredentialLease | CodexCredentialLease | None = None,
    message_metadata: dict[str, Any] | None = None,
    turn_start: int | None = None,
    continued_pending: list[PendingMessage] | None = None,
    session_kind: SessionKind | None = None,
    **_kwargs: Any,
) -> AsyncGenerator[StreamBaseResponse, None]:
    # Pyright's complexity heuristic bails on this ~1500 LoC function (retry
    # loop with context-overflow fallback + transient backoff + partial-work
    # preservation). Splitting the retry loop further hurts readability —
    # branches share mutable state (session, adapter, transcript builder,
    # usage accumulators) that doesn't pass cleanly through helpers. The
    # suppression only silences the complexity bailout; real type errors in
    # the function body still surface.
    """Stream chat completion using Claude Agent SDK.

    Args:
        file_ids: Optional workspace file IDs attached to the user's message.
            Images are embedded as vision content blocks; other files are
            saved to the SDK working directory for the Read tool.
        mode: Accepted for signature compatibility with the baseline path.
            The SDK path does not currently branch on this value.
        model: Per-request model preference from the frontend toggle.
            'advanced' → Claude Opus; 'standard' → global config default.
            Takes priority over per-user LaunchDarkly targeting.
    """
    if session is None:
        session = await get_chat_session(session_id, user_id)

    if not session:
        raise NotFoundError(
            f"Session {session_id} not found. Please create a new session first."
        )

    # Type narrowing: session is guaranteed ChatSession after the check above
    session = cast(ChatSession, session)

    expert_session_suffix = await build_expert_identity_suffix(
        session.user_id,
        session.expert_id,
        organization_id=session.organization_id,
        team_id=session.team_id,
    )
    expert_identity_validated = True

    # Stamping state for THIS turn: messages beyond this index were created
    # by the current turn and get model/routing_source at persist time.
    # Pre-feature history rows (NULL model) must never be back-stamped.
    routing_source: "RoutingSource" = "env"

    # The session row is the tenancy anchor; the turn entry's org/team only
    # backfills sessions created before org tagging (pre-migration rows).
    if session.organization_id is None and organization_id:
        session.organization_id = organization_id
        session.team_id = team_id

    # Clean up ALL trailing error markers from previous turn before starting
    # a new turn.  Multiple markers can accumulate when a mid-stream error is
    # followed by a cleanup error in __aexit__ (both append a marker).
    while (
        len(session.messages) > 0
        and session.messages[-1].role == "assistant"
        and session.messages[-1].content
        and (
            COPILOT_ERROR_PREFIX in session.messages[-1].content
            or COPILOT_RETRYABLE_ERROR_PREFIX in session.messages[-1].content
        )
    ):
        logger.info(
            "[SDK] [%s] Removing stale error marker from previous turn",
            session_id[:12],
        )
        session.messages.pop()

    # Drop orphan tool_use + trailing stop-marker rows left by a previous
    # Stop mid-tool-call so the next turn's --resume transcript is well-formed.
    prune_orphan_tool_calls(session.messages, log_prefix=f"[SDK] [{session_id[:12]}]")

    # Capture the turn-start boundary AFTER the error-marker and orphan-tool
    # cleanup above: those pops shrink session.messages, so a count taken
    # before them would overshoot and leave this turn's first assistant
    # row(s) below _stamp_turn_messages' start_index — unstamped — exactly
    # on the error-recovery turns whose routing we most want recorded.
    pre_turn_message_count = len(session.messages)

    # The turn's kind, read once for every reader below: the executor's,
    # or this session's own for a caller that dispatches no turn.
    kind = session_kind or SessionKind.for_turn(session)
    turn_summary = TurnSummaryCollector(
        session_id, voice=kind.voice, is_user_message=is_user_message
    )

    # Strip any user-injected <user_context> tags on every turn.
    # Only the server-injected prefix on the first message is trusted.
    if message:
        message = strip_user_context_tags(message)

    # A reply is the only thing that clears a Home "Needs You" question.
    # Unconditional on the append result: the HTTP path pre-saves the user
    # message, so the append is a no-op dedup there.
    if is_user_message and message and message.strip():
        await clear_pending_question(session)

    continued_entry: StreamPendingDrained | None = None
    # Where the history this turn's query may carry ends: ``None`` means just
    # before the last row, the turn's own message.
    history_ceiling: int | None = None
    if continued_pending:
        # An auto-continue call streams into the turn it continues: its user
        # rows are the drained messages, one each, as a turn-start drain's are.
        # Together they are the turn's message, so the history ends before the
        # first of them.
        history_ceiling = len(session.messages) + 1
        if not await persist_pending_as_user_rows(
            session,
            None,
            continued_pending,
            log_prefix=f"[SDK][{session_id[:12]}]",
        ):
            return  # rolled back and re-queued for the next turn
        continued_entry = drained_rows_entry(continued_pending)
        _user_message_appended = False
    else:
        _user_message_appended = maybe_append_user_message(
            session, message, is_user_message, message_metadata
        )
    # Whether the turn answers a hire's kickoff is read once its own rows
    # are in: the replies an auto-continue picks up, or a message nothing
    # saved before dispatch, become the latest user rows only here.
    kind = kind.with_turn_rows(session)
    if (_user_message_appended or continued_entry) and is_user_message:
        track_user_message(
            user_id=user_id,
            session_id=session_id,
            message_length=len(message or ""),
            expert_id=session.expert_id,
            origin=session.metadata.origin,
            source_platform=session.metadata.source_platform,
        )

    # Structured log prefix: [SDK][<session>][T<turn>]
    # Turn = number of user messages (1-based), computed AFTER appending the new message.
    turn = sum(1 for m in session.messages if m.role == "user")
    log_prefix = f"[SDK][{session_id[:12]}][T{turn}]"

    # Persist the appended user message to DB immediately so page refreshes
    # during a long-running turn (e.g. auto-continue whose sleep/bash call
    # blocks for minutes) show the user bubble. routes.py pre-saves the
    # user message before direct POSTs so maybe_append_user_message returns
    # False there (duplicate) — this branch only fires for internal callers
    # that did NOT pre-save, most notably the auto-continue recursive call
    # below.
    #
    # If the persist fails, roll back the in-memory append: otherwise
    # session.messages[-1] carries a ``sequence=None`` ghost row, and a
    # later turn-start drain (from a pending message queued during this
    # turn) would trip the "no sequence" RuntimeError and crash the turn.
    if _user_message_appended and is_user_message:
        session = await persist_session_safe(session, log_prefix)
        if session.messages and session.messages[-1].sequence is None:
            # Eager persist swallowed a transient DB failure and left the
            # in-memory append without a sequence. Roll back so the session
            # stays consistent with the DB and raise so the caller can
            # re-queue any drained content. Without this, a later
            # turn-start drain would trip the "no sequence" RuntimeError
            # and lose the fresh pending messages it just LPOPed.
            session.messages.pop()
            raise RuntimeError(
                f"{log_prefix} Eager persist of user message failed; "
                f"in-memory append rolled back"
            )

    # The stream's rows start after the message that triggered it.
    if turn_start is None:
        turn_start = len(session.messages)

    # Generate title for new sessions (first user message)
    if is_user_message and not session.title:
        user_messages = [m for m in session.messages if m.role == "user"]
        if len(user_messages) == 1:
            first_message = user_messages[0].content or message or ""
            if first_message:
                task = asyncio.create_task(
                    _update_title_async(session_id, first_message, user_id)
                )
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)

    message_id = str(uuid.uuid4())
    stream_id = str(uuid.uuid4())
    turn_summary.record_turn(
        turn=turn,
        message_id=message_id,
        session=session,
        permissions=permissions,
        envelope=envelope,
        kind=kind,
    )
    e2b_sandbox = None
    use_resume = False
    resume_file: str | None = None
    transcript_builder = TranscriptBuilder()
    sdk_cwd = ""

    # Acquire stream lock to prevent concurrent streams to the same session
    lock = AsyncClusterLock(
        redis=await get_redis_async(),
        key=f"{STREAM_LOCK_PREFIX}{session_id}",
        owner_id=stream_id,
        timeout=config.stream_lock_ttl,
    )

    lock_owner = await lock.try_acquire()
    if lock_owner != stream_id:
        # Another stream is active
        logger.warning(
            "%s Session already has an active stream: %s", log_prefix, lock_owner
        )
        yield StreamError(
            errorText="Another stream is already active for this session. "
            "Please wait or stop it.",
            code="stream_already_active",
        )
        return

    # OTEL context manager — initialized inside the try and cleaned up in finally.
    _otel_ctx: Any = None
    # Parent Langfuse span for the turn — created so that the
    # ``openrouter-cost-reconcile`` backfill event has a stable trace_id to
    # attach to even though it fires after the SDK-emitted spans end.
    # ``propagate_attributes`` alone doesn't create a span, so without this
    # wrapper ``get_current_trace_id()`` returns None at the finally block.
    _lf_span: Any = None
    has_history = len(session.messages) > 1
    transcript_content: str = ""
    state: _RetryState | None = None

    # Token usage accumulators — populated from ResultMessage at end of turn
    turn_prompt_tokens = 0  # uncached input tokens only
    turn_completion_tokens = 0
    turn_cache_read_tokens = 0
    turn_cache_creation_tokens = 0
    turn_cost_usd: float | None = None
    graphiti_enabled = False
    pre_attempt_msg_count = 0
    # Checkpoint turn_start can span auto-continues; capture only this call.
    learning_turn_start = len(session.messages)
    learning_initiating_user = find_initiating_user_row(session, message)
    learning_initiating_text = message
    # State of the latest failed attempt: rolled-back messages + any
    # _HandledStreamError info to emit on final-failure exit. The retry loop
    # mutates this via capture()/clear(); the post-loop block calls
    # finalize() once.
    interrupted = _InterruptedAttempt()
    # Defaults ensure the finally block can always reference these safely even when
    # an early return (e.g. sdk_cwd error) skips their normal assignment below.
    sdk_model: str | None = None
    # ``compaction`` is constructed after several fallible setup steps below
    # (sandbox/system-prompt/CLI-restore). The except branch needs to close
    # a pre-query row on ANY failure in this function, including ones raised
    # before that point, so it must tolerate ``compaction`` still being None.
    compaction: CompactionTracker | None = None
    profile: RouteProfile | None = None
    route_gateway: ModelGateway | None = None
    tool_display_bridge = SDKToolDisplayBridge()
    deferred_gateway_cleanup_error: BaseException | None = None
    turn_segment = _sdk_serving_segment(credential_lease)
    # Wall-clock timestamp captured before the CLI runs so the
    # OpenRouter reconcile can filter subagent JSONLs by mtime — only
    # files created during THIS turn contribute gen-IDs.  Without this
    # the sweep would pick up prior turns' compaction files that persist
    # under ``<session_id>/subagents/``, double-billing the user.
    turn_start_ts = time.time()

    # Initialised before the retry loop so every code path that reads it after
    # the loop (post-stream upload guards, finally-block bookkeeping) sees a
    # bound name even when the loop never enters its happy path.
    ended_with_stream_error = False
    final_checkpoint: StreamCheckpoint | None = None
    # Built with the CLI options below; read after the loop to keep a paused
    # turn from auto-continuing.
    governor: TurnGovernor | None = None

    # Make sure there is no more code between the lock acquisition and try-block.
    try:
        # Build system prompt (reuses non-SDK path with Langfuse support).
        # Pre-compute the cwd here so the exact working directory path can be
        # injected into the supplement instead of the generic placeholder.
        # Catch ValueError early so the failure yields a clean StreamError rather
        # than propagating outside the stream error-handling path.
        try:
            sdk_cwd = _make_sdk_cwd(session_id)
            os.makedirs(sdk_cwd, exist_ok=True)
        except (ValueError, OSError) as e:
            logger.error("%s Invalid SDK cwd: %s", log_prefix, e)
            turn_summary.note_error("sdk_cwd_error")
            yield StreamError(
                errorText="Unable to initialize working directory.",
                code="sdk_cwd_error",
            )
            return

        # Narrate the parallel setup below (sandbox spin-up, system prompt
        # build, CLI session restore) — it can take several seconds and the
        # generator cannot yield mid-gather.
        yield StreamStatus(
            message=(
                "Restoring your session…" if has_history else "Preparing workspace…"
            )
        )

        # --- Run independent async I/O operations in parallel ---
        # E2B sandbox setup, system prompt build (Langfuse + DB), Graphiti
        # warm-context, and CLI session restore are all independent network
        # calls. Running them concurrently saves ~500-1000ms vs sequential.

        # Captured outside the closure: `session` is narrowed to ChatSession
        # here, but that narrowing does not carry into nested functions.
        owner_expert_id = session.expert_id

        async def _setup_e2b():
            """Set up E2B sandbox if configured, return sandbox or None."""
            nonlocal e2b_sandbox
            if not (e2b_api_key := config.active_e2b_api_key):
                if config.use_e2b_sandbox:
                    logger.warning(
                        "[E2B] [%s] E2B sandbox enabled but no API key configured "
                        "(CHAT_E2B_API_KEY / E2B_API_KEY) — falling back to bubblewrap",
                        session_id[:12],
                    )
                return None
            try:
                # An expert session runs on the expert's own persistent box;
                # everything else gets a per-session sandbox.
                sandbox = await get_or_create_sandbox(
                    session_id,
                    api_key=e2b_api_key,
                    template=config.e2b_sandbox_template,
                    timeout=config.e2b_sandbox_timeout,
                    on_timeout=config.e2b_sandbox_on_timeout,
                    volume_mounts=workspace_volume_mounts(user_id, owner_expert_id),
                    expert_id=owner_expert_id,
                    user_id=user_id,
                )
                # Publish the live box before the gather returns: if a sibling
                # setup leg fails, the finally below still pauses it and
                # releases the expert's turn slot instead of leaking both.
                e2b_sandbox = sandbox
            except Exception as e2b_err:
                logger.error(
                    "[E2B] [%s] Setup failed: %s",
                    session_id[:12],
                    e2b_err,
                    exc_info=True,
                )
                return None

            return sandbox

        # The E2B leg runs as its own task: if a sibling leg fails first, the
        # box it may already have opened (and the expert turn it counted)
        # must still be published so the finally below pauses and releases it.
        e2b_task = asyncio.create_task(_setup_e2b())
        try:
            (
                base_system_prompt,
                understanding,
                (graphiti_enabled, warm_ctx),
                _restore,
            ) = await asyncio.gather(
                _build_system_prompt(),
                fetch_business_understanding(user_id if not has_history else None),
                _fetch_graphiti_context(user_id, session, message),
                # Restore CLI session — single GCS round-trip covers both
                # --resume and builder state.  message_count watermark lives
                # in the companion .meta.json alongside the session file.
                _restore_cli_session_for_turn(
                    user_id,
                    session_id,
                    session,
                    sdk_cwd,
                    transcript_builder,
                    log_prefix,
                    history_ceiling=history_ceiling,
                ),
            )
        except BaseException:
            with contextlib.suppress(BaseException):
                e2b_sandbox = await e2b_task
            raise
        e2b_sandbox = await e2b_task

        use_e2b = e2b_sandbox is not None
        auto_mode_supplement = approval_mode_supplement(
            await active_mode(user_id, session)
        )
        # The whole expert-team surface rides the hire-experts flag, failing
        # closed for anonymous turns.  Resolved here rather than at the
        # tool-hiding site below so the delegation rules can be gated on the
        # same boolean — a flag-off turn must not be told to call tools that
        # were never registered with the MCP server.
        experts_enabled = bool(user_id) and await is_feature_enabled(
            Flag.HIRE_EXPERTS, user_id, default=False
        )
        # Append appropriate supplement (Claude gets tool schemas automatically)
        prompt_parts = SystemPromptParts(
            base=base_system_prompt,
            use_e2b=use_e2b,
            expert_session=bool(session.expert_id),
            delegation=get_delegation_supplement() if experts_enabled else "",
            oversight=get_expert_oversight_supplement(
                experts_enabled=experts_enabled, expert_id=session.expert_id
            ),
            team_building=get_team_building_supplement(
                experts_enabled=experts_enabled, expert_id=session.expert_id
            ),
            chat_platform=get_chat_platform_supplement(
                session.metadata.source_platform
            ),
            graphiti=get_graphiti_supplement() if graphiti_enabled else "",
            auto_mode=auto_mode_supplement,
            expert_identity=expert_session_suffix,
        )
        # Append the builder-session block (graph id+name + full building
        # guide) AFTER the shared supplements so the system prompt is
        # byte-identical across turns of the same builder session — Claude's
        # prompt cache keeps the ~20KB guide warm for the whole session.
        # Empty string for non-builder sessions preserves cross-user caching.
        builder_session_suffix = await build_builder_system_prompt_suffix(session)
        # Per-turn runtime flag (never persisted): lets the guide gate and
        # get_agent_building_guide skip redundant guide round-trips when the
        # guide is already in this turn's cached system prompt.
        session.sdk_turn_active = True
        # Turn-scoped, as the baseline's turn-end clear makes it: without
        # this the buffer the adapter fills would carry a previous turn's
        # calls into a gate that asks about *this* turn.
        session.clear_inflight_tool_calls()
        session.guide_in_system_prompt = bool(builder_session_suffix)
        system_prompt = assemble_system_prompt(
            prompt_parts, building_suffix=builder_session_suffix
        )

        transcript_content = _restore.transcript_content
        use_resume = _restore.use_resume
        resume_file = _restore.resume_file
        transcript_msg_count = _restore.transcript_msg_count
        restore_context_messages = _restore.context_messages
        turn_summary.note_download(_restore.download_bytes)

        yield StreamStart(messageId=message_id, sessionId=session_id)
        if continued_entry is not None:
            yield continued_entry

        set_execution_context(
            user_id,
            session,
            sandbox=e2b_sandbox,
            sdk_cwd=sdk_cwd,
            permissions=permissions,
            envelope=envelope,
        )

        disabled_tool_groups: list[ToolGroup] = []
        if not graphiti_enabled:
            disabled_tool_groups.append("graphiti")
        # ``experts_enabled`` was resolved with the system-prompt supplements
        # above; the role split lives in the shared helper.
        disabled_tool_groups.extend(
            expert_tool_disabled_groups(
                experts_enabled=experts_enabled, expert_id=session.expert_id
            )
        )

        # Hide both permission-denied tools AND group-disabled tools at
        # registration. ``allowed_tools`` filtering alone routes group-
        # disabled calls through the same auto-deny path the per-permission
        # hiding is meant to eliminate (CLI returns "Permission to use ...
        # has been denied", which the model narrates as a fake Allow/Deny
        # prompt).
        # get_agent_building_guide is redundant on the SDK path —
        # enter_agent_building_mode puts the same guide compaction-proof
        # into the system prompt. Hiding it removes the tempting-but-worse
        # fallback; read_skill("agent_building_guide") remains as escape
        # hatch. The baseline path (no prompt machinery) keeps the tool.

        # What the turn's kind does not offer: a hire's kickoff is narrowed
        # to the onboarding card, and a session no person drives is not
        # offered the staffing tools its own guard would refuse. Hiding is
        # the enforcement here: an unregistered MCP tool does not exist for
        # the CLI, so there is no call left to deny.
        hidden_tools = (
            _hidden_short_names_for_permissions(permissions)
            | tool_names_in_groups(disabled_tool_groups)
            | {"get_agent_building_guide"}
            | kind.hidden_tools()
        )
        # run_capability reaches deferred tools by id; the same hidden set
        # must bound it, so it travels with the turn's execution context.
        set_execution_context(
            user_id,
            session,
            sandbox=e2b_sandbox,
            sdk_cwd=sdk_cwd,
            permissions=permissions,
            envelope=envelope,
            hidden_tools=hidden_tools,
        )
        mcp_server = create_copilot_mcp_server(
            use_e2b=use_e2b,
            hidden_tool_names=hidden_tools,
            tool_display_bridge=tool_display_bridge,
        )

        # The route this turn runs on, built once: the model as the route
        # writes it, the window and trigger, the credential. Retries and the
        # building-mode restart reuse it.
        profile = await resolve_turn_profile(
            config,
            model=model,
            session_id=session_id,
            user_id=user_id,
            credential_lease=credential_lease,
        )
        sdk_model, routing_source = profile.model, profile.routing_source
        route_gateway = await start_route_gateway(profile, credential_lease)

        # sdk_cwd routes the CLI's temp dir into the per-session workspace
        # so sub-agent output files land inside sdk_cwd (see route_launch_env).
        sdk_env = route_launch_env(
            profile,
            sdk_cwd=sdk_cwd,
            gateway=(
                GatewayEndpoint(
                    base_url=route_gateway.base_url,
                    auth_token=route_gateway.auth_token,
                )
                if route_gateway is not None
                else None
            ),
            session_id=session_id,
            user_id=user_id,
        )
        # What this turn's subprocess was pinned to (window/trigger/flags —
        # never secrets). The turn summary reads the pin off the launch's env
        # and records this source; the context-limit alarm quotes the line.
        turn_summary.note_window_source(profile.window_source)
        context_summary = describe_sdk_context(
            route=profile.name,
            model=sdk_model,
            sdk_env=sdk_env,
            window_source=profile.window_source,
        )

        # Track SDK-internal compaction (PreCompact hook → start, next msg → end)
        compaction = CompactionTracker()

        tool_options = _build_sdk_tool_options(
            permissions,
            use_e2b=use_e2b,
            disabled_groups=disabled_tool_groups,
            denied_tools=kind.hidden_tools(),
        )
        # Lands the turn with a checkpoint near its budget or step check-in.
        # A route without a USD budget (a ChatGPT subscription pays for its
        # own turns) gets the check-in only. A turn billed to credits gets
        # what its balance covers.
        credit_turn = credit_billed_turn(user_id)
        governor = TurnGovernor(
            budget=(
                None
                if not profile.row.usd_budget
                else (
                    credit_turn.budget
                    if credit_turn is not None
                    else await resolve_turn_budget(user_id)
                )
            ),
            step_checkin=config.sdk_step_checkin,
        )

        security_hooks = create_security_hooks(
            user_id,
            sdk_cwd=sdk_cwd,
            max_subtasks=config.claude_agent_max_subtasks,
            on_compact=compaction.on_compact,
            tool_display_bridge=tool_display_bridge,
            context_after_compaction=functools.partial(
                _context_after_compaction,
                session,
                user_id,
                kind,
                sdk_cwd=sdk_cwd,
                use_e2b=use_e2b,
                log_prefix=log_prefix,
                hidden_tools=hidden_tools,
                turn_summary=turn_summary,
            ),
            governor=governor,
            allowed_tools=tool_options["allowed_tools"],
        )

        def _on_stderr(line: str) -> None:
            """Log a stderr line emitted by the Claude CLI subprocess."""
            sid = session_id[:12] if session_id else "?"
            capability = route_gateway.auth_token if route_gateway is not None else ""
            safe_line = _redact_cli_stderr(line, secrets=(capability,))
            logger.info("[SDK] [%s] CLI stderr: %s", sid, safe_line.rstrip())

        # With the live prompt, this turn's prompt is the one sent even when
        # resuming a session that started with another.
        system_prompt_value = _configured_system_prompt_value(system_prompt)

        sdk_options = ClaudeAgentOptions(
            system_prompt=system_prompt_value,
            mcp_servers={"copilot": mcp_server},
            **common_launch_options(),
            **tool_options,
            hooks=security_hooks,
            cwd=sdk_cwd,
            stderr=_on_stderr,
        )
        # A backstop a little past the governor's step check-in, which
        # normally lands the turn first.
        sdk_options.max_turns = governor.cli_max_turns
        if profile.row.usd_budget:
            # A Codex subscription consumes the connected account's
            # allowance; a USD ceiling there would be a fabricated policy.
            # A backstop above the turn's budget, in the CLI's price units:
            # the governor lands the turn before it fires.
            sdk_options.max_budget_usd = governor.cli_budget_usd(sdk_model)
        if profile.row.anthropic_thinking:
            # Adaptive thinking at the route's effort, with the legacy budget
            # for models that still read one. Kimi K2.6 takes it through
            # OpenRouter's ``reasoning`` extension (#12871).
            sdk_options.thinking = {"type": "adaptive"}
            sdk_options.effort = profile.thinking_effort
            sdk_options.max_thinking_tokens = profile.thinking_budget
        if sdk_model:
            sdk_options.model = sdk_model
        if config.sdk_include_partial_messages:
            # Opt into per-token streaming — the CLI emits raw Anthropic
            # ``content_block_delta`` events as ``StreamEvent`` messages
            # ahead of each summary ``AssistantMessage`` so reasoning and
            # text land on the wire token-by-token (matching the baseline
            # path's UX shipped in #12873).  ``SDKResponseAdapter`` consumes
            # the partial stream via ``_handle_stream_event`` and emits
            # only the tail diff from the subsequent summary, so content
            # never double-emits and a summary-only short block still
            # reaches the UI.
            sdk_options.include_partial_messages = True

        if sdk_env:
            sdk_options.env = sdk_env
        if use_resume and resume_file:
            # --resume {uuid} implies the session UUID — do NOT also pass
            # --session-id here.  CLI >=2.1.97 rejects the combination of
            # --session-id + --resume unless --fork-session is also given.
            sdk_options.resume = resume_file
        else:
            # Set session_id whenever NOT resuming so the CLI writes the
            # native session file to a predictable path for
            # upload_transcript() after the turn.  This covers:
            #   • T1 fresh: no prior history, first SDK turn.
            #   • Mode-switch T1: has_history=True (prior baseline turns in
            #     DB) but no CLI session file was ever uploaded — the CLI has
            #     never been invoked with this session_id before.
            #   • T2+ without --resume (restore failed): no session file was
            #     restored to local storage (download_transcript returned
            #     None), so no conflict with an existing file.
            # When --resume is active the session_id is already implied by
            # the resume file; passing it again would be rejected by the CLI.
            sdk_options.session_id = session_id

        adapter = SDKResponseAdapter(
            message_id=message_id,
            session_id=session_id,
        )
        adapter.governor = governor

        # Propagate user_id/session_id as OTEL context attributes so the
        # langsmith tracing integration attaches them to every span.  This
        # is what Langfuse (or any OTEL backend) maps to its native
        # user/session fields.
        _user_tier = await get_user_tier(user_id) if user_id else None
        _otel_metadata: dict[str, str] = {
            "resume": str(use_resume),
            "conversation_turn": str(turn),
            "tool_surface": "registry",
        }
        if _user_tier:
            _otel_metadata["subscription_tier"] = _user_tier.value

        # Open a Langfuse parent span so the trace_id is observable from
        # the finally block — ``propagate_attributes`` only annotates an
        # existing span, it does not create one.
        try:
            _lf_span = get_client().start_as_current_span(name="copilot-sdk-turn")
            _lf_span.__enter__()
            # Pin compaction events to this trace explicitly; the cycle
            # that emits them closes deep inside the awaited message loop.
            compaction.trace_id = get_client().get_current_trace_id()
        except Exception:
            logger.debug("Failed to open Langfuse parent span", exc_info=True)
            _lf_span = None

        _otel_ctx = propagate_attributes(
            user_id=user_id,
            session_id=session_id,
            trace_name="copilot-sdk",
            tags=["sdk"],
            metadata=_otel_metadata,
        )
        _otel_ctx.__enter__()

        current_message = message or ""
        if not current_message and session.messages:
            last_user = [m for m in session.messages if m.role == "user"]
            if last_user:
                current_message = last_user[-1].content or ""

        # Capture the message count *before* draining so _build_query_message
        # can compute the gap slice without including the newly-drained pending
        # messages.  Pending messages are both appended to session.messages AND
        # concatenated into current_message; without the ceiling the gap slice
        # would extend into the pending messages and duplicate them in the
        # model's input context (gap_context + current_message both containing
        # them).  An auto-continue turn's queued rows were persisted at its
        # start, so its ceiling was taken before them.
        _pre_drain_msg_count = (
            history_ceiling if history_ceiling is not None else len(session.messages)
        )

        # Drain any messages the user queued via POST /messages/pending
        # while the previous turn was running (or since the session was
        # idle).  Messages are drained ATOMICALLY — one LPOP with count
        # removes them all at once, so a concurrent push lands *after*
        # the drain and stays queued for the next turn instead of being
        # lost between LPOP and clear.  File IDs and context are
        # preserved via format_pending_as_user_message.
        #
        # The drained content is combined in chronological (typing) order:
        # pending messages were queued DURING the previous turn, so they
        # were typed BEFORE the current /stream message.  Putting pending
        # first — ``pending → current`` — matches the order the user
        # actually sent them and avoids the "I typed A then B but it shows
        # up as B then A" confusion.  The already-saved user message in
        # the DB is updated via update_message_content_by_sequence to
        # include the pending texts, avoiding the duplicate INSERT that
        # inserting them as rows and persisting would cause (routes.py has
        # already saved the user message at sequence N before the executor
        # runs, so an incremental upsert would write a second copy at N+1).
        # Narrate the context-assembly phase (pending drain, user-context
        # injection, attachment prep, query/transcript build) before the
        # SDK client spawns.
        yield StreamStatus(message="Preparing conversation context…")

        # Answered cards first: their results ride the same fold as pending.
        pending_messages = await resolve_answered(
            user_id, session, cap=cap_late_tool_result
        ) + await drain_pending_safe(session_id, log_prefix)
        if pending_messages:
            logger.info(
                "%s Draining %d pending message(s) at turn start",
                log_prefix,
                len(pending_messages),
            )
            # NOTE: combining and per-row persistence both happen *after*
            # ``inject_context_blocks`` below — see the comment near that
            # call.  At this point ``current_message`` is still the
            # original turn-starting send (no pending text yet).

        if not current_message.strip():
            turn_summary.note_error("empty_prompt")
            yield StreamError(
                errorText="Message cannot be empty.",
                code="empty_prompt",
            )
            return

        # Strip any user-injected <user_context> tags from current_message.
        # On --resume, current_message may come from session history which was
        # already sanitized on the original turn; strip again as defence-in-depth.
        current_message = strip_user_context_tags(current_message)
        if not is_user_message:
            # A continuation note is stored as a system row; the model gets
            # its instruction without the rendering marker.
            current_message = as_model_prompt(current_message)

        # On the first turn inject user context into the message before building
        # the query so that _build_query_message sees the full prefixed content.
        # The system prompt is now static (same for all users) so the LLM can
        # cache it across sessions.
        #
        # On resume (has_history=True) we intentionally skip re-injection: the
        # transcript already contains the <user_context> and <memory_context>
        # prefixes from the original turn (persisted to the DB via
        # inject_context_blocks), so the SDK replay carries context continuity
        # without us prepending them again.
        #
        # A later turn that could NOT resume (no SDK transcript in storage,
        # a baseline-written one, or one that failed validation) starts a
        # fresh CLI session that has never seen those prefixes, so the same
        # blocks, memory aside, are rebuilt and lead the query instead — see
        # ``_rebuild_first_turn_context``.  They ride the query only; the
        # stored row keeps the user's own words.  (The CLI's own compaction
        # drops them too; the ``SessionStart`` hook wired up with the
        # security hooks above re-sends them after every compaction.)
        resume_fallback_prefix: str | None = None
        if not has_history:
            first_turn_blocks = await _first_turn_context_blocks(
                session,
                user_id,
                kind,
                sdk_cwd=sdk_cwd,
                use_e2b=use_e2b,
                understanding=understanding,
                warm_ctx=warm_ctx,
            )
            # inject_context_blocks prepends the blocks AFTER
            # sanitize_user_supplied_context runs — so the trusted
            # server-injected blocks are never stripped by the sanitizer —
            # and persists the fully prefixed message to DB.
            prefixed_message = await inject_context_blocks(
                first_turn_blocks, current_message, session_id, session.messages
            )
            if prefixed_message is not None:
                current_message = prefixed_message
        elif not use_resume:
            resume_fallback_prefix = render_context_blocks(
                await _rebuild_first_turn_context(
                    session,
                    user_id,
                    kind,
                    sdk_cwd=sdk_cwd,
                    use_e2b=use_e2b,
                    log_prefix=log_prefix,
                    reason="no-resume fallback",
                )
            )

        # Now that ``inject_context_blocks`` has wrapped + persisted the
        # ORIGINAL turn-starting send into its row, fold any pending
        # chips into the prompt the model sees AND persist each pending
        # message as its own raw-text user row in the DB.  Order matters
        # because:
        #
        #   - inject targets the last user row (= routes.py-saved row at
        #     this point) and writes envelopes + the value passed in.
        #     If we'd combined first, the wrapped row would carry the
        #     chip text too, and persisting the chip as a separate row
        #     below would make the chip appear twice in the UI.
        #   - inject targets the last user row.  If we'd persisted
        #     pending rows first, inject would land on a pending row
        #     instead of the original send.
        #
        # ``transcript_builder=None``: the model still sees the combined
        # text below as a single user turn that gets written to the
        # transcript at turn-end via ``append_user(current_message)``.
        # Adding each pending to the transcript here would triple-count
        # them in the next turn's ``--resume`` context.
        if pending_messages:
            # Persist FIRST.  Only fold pending into the model's prompt
            # when persistence succeeded — if the helper rolled back and
            # re-queued the pending into Redis, leaving ``current_message``
            # untouched ensures the NEXT turn's drain doesn't double-
            # combine (re-queued pending + combined-from-this-turn) into
            # the model's context.
            persisted_ok = await persist_pending_as_user_rows(
                session,
                None,
                pending_messages,
                log_prefix=log_prefix,
            )
            if persisted_ok:
                yield drained_rows_entry(pending_messages)
                current_message = combine_pending_with_current(
                    pending_messages,
                    current_message,
                    request_arrival_at=request_arrival_at,
                )

        # One budget for the forecast and the build alike, sized from the
        # window this turn's subprocess is pinned to — never the catalog.
        pre_query_target = profile.compaction_target_tokens()
        forecast = _expect_pre_query_compaction(
            session.messages,
            _compression_model(),
            use_resume=use_resume,
            transcript_msg_count=transcript_msg_count,
            session_msg_ceiling=_pre_drain_msg_count,
            prior_messages=restore_context_messages,
            target_tokens=pre_query_target,
        )
        if forecast.expected:
            for ev in compaction.emit_pre_query_start(forecast.tokens_before):
                yield ev

        # Live budget, every turn — the CLI's own ``max_budget_usd`` reminder is
        # per-query and knows nothing of the tree. Prepended to the query only,
        # never to ``current_message``: that is what the transcript records and
        # the next turn replays, and it must not accumulate one stale figure
        # per turn.
        budget_status = await build_turn_budget_block(envelope, user_id)
        # A voice turn's instruction follows the user's words, on the query
        # only: added after the inbound strip, so it is the platform's own,
        # and never written to ``current_message`` or the stored row.
        turn_text = budget_status + current_message
        if kind.voice:
            turn_text = f"{turn_text}\n\n{VOICE_TURN_INSTRUCTION}"

        query_message, compaction_stats = await _build_query_message(
            turn_text,
            session,
            use_resume,
            transcript_msg_count,
            session_id,
            session_msg_ceiling=_pre_drain_msg_count,
            prior_messages=restore_context_messages,
            target_tokens=pre_query_target,
            expect_compaction=forecast.expected,
            context_prefix=resume_fallback_prefix or "",
            turn_summary=turn_summary,
        )

        # Snapshot before the close persists its synthetic assistant+tool pair:
        # ``_seed_transcript`` slices ``session.messages`` and would otherwise
        # read the compaction row as history and the current user message as
        # prior context, seeding a transcript that repeats the live turn.
        pre_compaction_msg_count = len(session.messages)
        if compaction_stats is not None:
            for ev in compaction.emit_pre_query_end(session, compaction_stats):
                yield ev
        elif forecast.expected:
            for ev in compaction.abort_pre_query():
                yield ev
        # If files are attached, prepare them: images become vision
        # content blocks in the user message, other files go to sdk_cwd.
        attachments = await _prepare_file_attachments(
            file_ids or [], user_id or "", session_id, sdk_cwd
        )
        if attachments.hint:
            query_message = f"{query_message}\n\n{attachments.hint}"

        # warm_ctx is among the first-turn blocks above (warm_ctx= kwarg).
        # No separate injection needed here.

        # Inject per-turn builder context when the session is bound to a
        # graph via ``metadata.builder_graph_id``.  Runs on EVERY user turn
        # (including resumes) so the LLM always sees the live graph snapshot
        # — if the user edits the graph between turns, the next turn carries
        # the updated nodes/links; the agent-building guide rides in the
        # system prompt.  Not persisted to the transcript: the snapshot is
        # stale-by-definition after the turn ends.
        query_message = await _maybe_prepend_builder_context(
            session, user_id, is_user_message, query_message
        )
        # Skill-drift notice — same query-only contract as builder
        # context: never persisted, re-diffed every turn.
        query_message = await _maybe_prepend_skills_update(
            session, user_id, is_user_message, query_message
        )
        # Already-seen capability record — same query-only contract.
        query_message = _maybe_prepend_seen_capabilities(
            session, is_user_message, query_message
        )
        turn_summary.record_query(
            query_message,
            use_resume=use_resume,
            covered_rows=transcript_msg_count,
            session_rows=len(session.messages),
        )

        # When running without --resume and no prior transcript in storage,
        # seed the transcript builder from compressed DB messages, so it holds
        # the conversation so far and a first retry has a transcript to compact.
        if not use_resume and not transcript_content:
            transcript_content, transcript_msg_count = await _seed_transcript(
                session,
                transcript_builder,
                transcript_msg_count,
                log_prefix,
                pre_compaction_msg_count,
                profile.seed_target_tokens(),
            )

        # Build the per-request context carrier (shared across attempts).
        # Scalar fields are immutable; session/compaction/lock are shared
        # mutable references (see `_StreamContext` docstring for details).
        stream_ctx = _StreamContext(
            session=session,
            session_id=session_id,
            log_prefix=log_prefix,
            sdk_cwd=sdk_cwd,
            current_message=current_message,
            file_ids=file_ids,
            message_id=message_id,
            attachments=attachments,
            compaction=compaction,
            lock=lock,
            route_gateway=route_gateway,
            credential_id=(
                credential_lease.credentials.id
                if credential_lease is not None
                else None
            ),
            profile=profile,
            tool_display=tool_display_bridge,
            turn_start=turn_start,
            turn_summary=turn_summary,
            tainted=envelope is not None and envelope.tainted,
        )

        # ---------------------------------------------------------------
        # Retry loop: original → compacted → no transcript
        # ---------------------------------------------------------------
        attempts_exhausted = False
        transient_exhausted = False
        stream_err: Exception | None = None

        transient_retries = 0
        max_transient_retries = config.claude_agent_max_transient_retries

        state = _RetryState(
            options=sdk_options,
            query_message=query_message,
            compaction_stats=compaction_stats,
            use_resume=use_resume,
            transcript_msg_count=transcript_msg_count,
            adapter=adapter,
            transcript_builder=transcript_builder,
            usage=_TokenUsage(priced=profile.prices_tokens),
            governor=governor,
            # Subagent files are read only on a turn the reconcile bills.
            generations=TurnGenerations(
                (
                    _cli_project_dir(sdk_cwd)
                    if profile.row.cost_source == "openrouter_reconcile"
                    else None
                ),
                session_id,
                turn_start_ts,
            ),
        )

        attempt = 0
        _last_reset_attempt = -1
        # Attempt 0 runs on the context built above; every later attempt
        # reduces context exactly once.  Transient retries and building-mode
        # restarts `continue` back to the loop head without advancing
        # `attempt`, and re-running the reduction there would summarize the
        # history a second time and persist a second compaction row for the
        # same logical compaction (mirrors the `_last_reset_attempt` guard).
        _last_reduced_attempt = 0
        while attempt < _MAX_STREAM_ATTEMPTS:
            # Reset transient retry counter per context-level attempt so
            # each attempt (original, compacted, no-transcript) gets the
            # full retry budget for transient errors.
            # Only reset when the attempt number actually changes —
            # transient retries `continue` back to the loop top without
            # incrementing `attempt`, so resetting unconditionally would
            # create an infinite retry loop.
            if attempt != _last_reset_attempt:
                transient_retries = 0
                _last_reset_attempt = attempt
            # Clear any stale stash signal from the previous attempt so
            # wait_for_stash() doesn't fire prematurely on a leftover event.
            reset_stash_event()
            # Drop orphaned tool outputs stashed by a rolled-back attempt —
            # the name-keyed FIFO would otherwise serve them (off-by-one) to
            # this attempt's tool calls, corrupting frontend tool payloads.
            reset_pending_tool_outputs()
            tool_display_bridge.reset()
            # Reset tool-level circuit breaker so failures from a previous
            # (rolled-back) attempt don't carry over to the fresh attempt.
            reset_tool_failure_counters()
            if attempt != _last_reduced_attempt:
                _last_reduced_attempt = attempt
                logger.info(
                    "%s Retrying with reduced context (%d/%d)",
                    log_prefix,
                    attempt + 1,
                    _MAX_STREAM_ATTEMPTS,
                )
                yield StreamStatus(message="Optimizing conversation context\u2026")
                for ev in compaction.emit_pre_query_start():
                    yield ev

                ctx = await _reduce_context(
                    transcript_content,
                    log_prefix,
                    profile=profile,
                    attempt=attempt,
                )
                state.transcript_builder = ctx.builder
                state.use_resume = False
                state.transcript_msg_count = 0
                state.target_tokens = ctx.target_tokens

                # Rebuild SDK options and query for the reduced context. The
                # options start from the last launch's, so the retry sends the
                # prompt that launch sent: after a building-mode restart, the
                # one with the guide (``guide_in_system_prompt`` stays True).
                sdk_options_retry = copy(state.options)
                # No --resume on a context retry. Whether we entered with
                # ``session_id`` (T1, mode-switch) or with ``--resume`` (T2+),
                # we want the recovery turn's CLI write to land on the
                # predictable ``cli_session_path(.., session_id)`` so the
                # post-turn ``upload_transcript`` actually picks up the
                # rescued (compacted) content.  Without this, a T2+ retry
                # would drop session_id to dodge "Session ID already in use",
                # write to a random path, and the upload would silently grab
                # the stale pre-failure file — leaving GCS bloated and
                # guaranteeing the next turn re-trips prompt-too-long.
                if sdk_cwd:
                    delete_stale_cli_session_file(sdk_cwd, session_id, log_prefix)
                sdk_options_retry.resume = None
                sdk_options_retry.session_id = session_id
                # The new CLI total starts at zero; retain the turn's spend.
                state.usage.cli_session_total_usd = None
                state.options = sdk_options_retry
                # Retry intentionally omits prior_messages (transcript+gap context) and
                # falls back to full session.messages[:-1] from DB — the authoritative
                # source.  transcript+gap is an optimisation for the first attempt only;
                # on retry the extra overhead of full-DB context is acceptable.
                # Keep the ``budget_status +`` prefix through any reflow of this
                # call: dropping it silently un-ships the retry path's budget line.
                # A resumed turn's retry drops --resume, so it needs the
                # first-turn context just as a turn that never resumed does.
                if has_history and resume_fallback_prefix is None:
                    resume_fallback_prefix = render_context_blocks(
                        await _rebuild_first_turn_context(
                            session,
                            user_id,
                            kind,
                            sdk_cwd=sdk_cwd,
                            use_e2b=use_e2b,
                            log_prefix=log_prefix,
                            reason="no-resume fallback",
                        )
                    )
                (
                    state.query_message,
                    state.compaction_stats,
                ) = await _build_query_message(
                    turn_text,
                    session,
                    state.use_resume,
                    state.transcript_msg_count,
                    session_id,
                    session_msg_ceiling=_pre_drain_msg_count,
                    target_tokens=state.target_tokens,
                    expect_compaction=True,
                    context_prefix=resume_fallback_prefix or "",
                    turn_summary=turn_summary,
                )
                if _retry_reduced_context(
                    reduced=ctx,
                    compaction_stats=state.compaction_stats,
                ):
                    for ev in compaction.emit_pre_query_end(
                        session, state.compaction_stats
                    ):
                        yield ev
                else:
                    for ev in compaction.abort_pre_query():
                        yield ev
                if attachments.hint:
                    state.query_message = f"{state.query_message}\n\n{attachments.hint}"
                # warm_ctx is already baked into current_message via
                # inject_context_blocks — no separate injection needed.
                # Re-inject per-turn builder context so retries carry the
                # same live graph snapshot as the initial attempt.
                state.query_message = await _maybe_prepend_builder_context(
                    session, user_id, is_user_message, state.query_message
                )
                state.query_message = await _maybe_prepend_skills_update(
                    session, user_id, is_user_message, state.query_message
                )
                state.query_message = _maybe_prepend_seen_capabilities(
                    session, is_user_message, state.query_message
                )
                turn_summary.record_query(
                    state.query_message,
                    use_resume=state.use_resume,
                    covered_rows=state.transcript_msg_count,
                    session_rows=len(session.messages),
                    rebuilt=True,
                )
                prior_adapter = state.adapter
                _fresh_adapter(state, message_id, session_id)
                # Carry the per-turn re-prompt cap forward so a transient
                # retry mid-turn does not unlock another re-prompt round.
                state.adapter.thinking_only_reprompted = state.thinking_only_reprompted
                # Forward only REAL prior content (text or non-empty-fallback
                # tool result) so the empty-completion guard on this retry
                # adapter is suppressed exactly when the user has actually
                # received content — and not when the prior attempt only
                # emitted reasoning, which would otherwise hide a genuinely
                # silent failure on the retry.
                if prior_adapter.emitted_real_content_to_wire:
                    state.adapter.prior_attempt_emitted_visible_content = True

            turn_summary.note_cli_launch()
            pre_attempt_msg_count = len(session.messages)
            # Snapshot transcript builder state — it maintains an
            # independent _entries list from session.messages, so rolling
            # back session.messages alone would leave duplicate entries
            # from the failed attempt in the uploaded transcript.
            transcript_snap = state.transcript_builder.snapshot()
            events_yielded = 0

            try:
                async for event in _run_stream_attempt(stream_ctx, state):
                    if not isinstance(event, _EPHEMERAL_EVENT_TYPES):
                        events_yielded += 1
                    yield event
                # Discard any state captured from prior failed attempts so
                # outer cleanup paths don't replay pre-retry content.
                interrupted.clear()
                break  # Stream completed — exit retry loop
            except asyncio.CancelledError:
                logger.warning(
                    "%s Streaming cancelled (attempt %d/%d)",
                    log_prefix,
                    attempt + 1,
                    _MAX_STREAM_ATTEMPTS,
                )
                raise
            except _BuildingModeRestart:
                # enter_agent_building_mode: relaunch this attempt with the
                # agent-building guide in the system prompt. Not an error,
                # not a rollback — everything produced so far stands (client
                # stream, session history, CLI session file), so no
                # interrupted.capture / snapshot restore here. Detection
                # cannot re-fire: guide_in_system_prompt flips True on
                # success, building_mode_requested flips False either way.
                # The interrupted run never reported a result, and the
                # relaunch's reports only its own query: charge what the
                # run wrote to its session file now that its CLI has exited.
                await _record_unreported_usage(state, sdk_cwd, session_id, log_prefix)
                expert_identity_validated = False
                restart_status = await _apply_building_mode_restart(
                    session=session,
                    state=state,
                    sdk_options=sdk_options,
                    prompt_parts=prompt_parts,
                    session_id=session_id,
                    message_id=message_id,
                    log_prefix=log_prefix,
                )
                expert_identity_validated = True
                # The restart sends a new query: the summary describes it.
                turn_summary.record_query(
                    state.query_message,
                    use_resume=state.use_resume,
                    covered_rows=state.transcript_msg_count,
                    session_rows=len(session.messages),
                )
                yield restart_status
                continue
            except _HandledStreamError as exc:
                # _run_stream_attempt already yielded a StreamError and
                # appended an error marker.  We only need to rollback
                # session messages and set the error flag — do NOT set
                # stream_err so the post-loop code won't emit a
                # duplicate StreamError.
                interrupted.capture(
                    session,
                    state.transcript_builder,
                    transcript_snap,
                    pre_attempt_msg_count,
                )
                # Check if this is a transient error we can retry with backoff.
                # exc.code is the only reliable signal — str(exc) is always the
                # static "Stream error handled — StreamError already yielded" message.
                if exc.code == "transient_api_error":
                    backoff, transient_retries = _next_transient_backoff(
                        events_yielded, transient_retries, max_transient_retries
                    )
                    if backoff is not None:
                        logger.warning(
                            "%s Transient error — retrying in %ds (%d/%d)",
                            log_prefix,
                            backoff,
                            transient_retries,
                            max_transient_retries,
                        )
                        async for evt in _do_transient_backoff(
                            backoff, state, stream_ctx
                        ):
                            yield evt
                        continue  # retry the same context-level attempt
                logger.warning(
                    "%s Stream error handled in attempt "
                    "(attempt %d/%d, code=%s, events_yielded=%d)",
                    log_prefix,
                    attempt + 1,
                    _MAX_STREAM_ATTEMPTS,
                    exc.code or "transient",
                    events_yielded,
                )
                interrupted.handled_error = _HandledErrorInfo(
                    error_msg=exc.error_msg or FRIENDLY_TRANSIENT_MSG,
                    code=exc.code or "transient_api_error",
                    retryable=exc.retryable,
                    already_yielded=exc.already_yielded,
                )
                ended_with_stream_error = True
                break
            except Exception as e:
                stream_err = e
                is_context_error = _is_prompt_too_long(e)
                is_transient = is_transient_api_error(str(e))
                if is_context_error:
                    alarm = _context_limit_without_compaction(attempt, compaction)
                    if alarm is not None:
                        # ERROR so it reaches Sentry with the pin that
                        # shaped the turn; the warning below is the
                        # ordinary retry bookkeeping.
                        logger.error("%s %s (%s)", log_prefix, alarm, context_summary)
                logger.warning(
                    "%s Stream error (attempt %d/%d, context_error=%s, "
                    "transient=%s, events_yielded=%d): %s",
                    log_prefix,
                    attempt + 1,
                    _MAX_STREAM_ATTEMPTS,
                    is_context_error,
                    is_transient,
                    events_yielded,
                    stream_err,
                    exc_info=True,
                )
                interrupted.capture(
                    session,
                    state.transcript_builder,
                    transcript_snap,
                    pre_attempt_msg_count,
                )
                if events_yielded > 0:
                    # Events were already sent to the frontend and cannot be
                    # unsent.  Retrying would produce duplicate/inconsistent
                    # output, so treat this as a final error.
                    logger.warning(
                        "%s Not retrying — %d events already yielded",
                        log_prefix,
                        events_yielded,
                    )
                    ended_with_stream_error = True
                    break
                # Transient API errors (ECONNRESET, 429, 5xx) — retry
                # with exponential backoff via the shared helper.
                if is_transient:
                    backoff, transient_retries = _next_transient_backoff(
                        events_yielded, transient_retries, max_transient_retries
                    )
                    if backoff is not None:
                        logger.warning(
                            "%s Transient exception — retrying in %ds (%d/%d)",
                            log_prefix,
                            backoff,
                            transient_retries,
                            max_transient_retries,
                        )
                        async for evt in _do_transient_backoff(
                            backoff, state, stream_ctx
                        ):
                            yield evt
                        continue  # retry same context-level attempt
                    # Retries exhausted — persist retryable marker so the
                    # frontend shows "Try again" after refresh.
                    # Mirrors the _HandledStreamError exhausted-retry path
                    # at line ~2310.
                    transient_exhausted = True
                    ended_with_stream_error = True
                    break

                if not is_context_error:
                    # Non-context, non-transient errors (auth, fatal)
                    # should not trigger compaction — surface immediately.
                    ended_with_stream_error = True
                    break
                attempt += 1  # advance to next context-level attempt
                continue
        else:
            # while condition became False — all attempts exhausted without
            # break.
            ended_with_stream_error = True
            attempts_exhausted = True
            logger.error(
                "%s All %d query attempts exhausted: %s",
                log_prefix,
                _MAX_STREAM_ATTEMPTS,
                stream_err,
            )
        # Consolidated final-failure handling. _classify_final_failure picks
        # the display message + stream code + retryable flag, finalize() adds
        # the history marker and produces the safety-flush events that close
        # stale UI widgets on the client, and the StreamError yield below
        # surfaces the same message over SSE. The _HandledStreamError path
        # sets ``already_yielded=True`` for non-transient errors (circuit
        # breaker, idle timeout) whose inner handler already yielded — skip
        # the re-yield in that case.
        if ended_with_stream_error:
            failure = _classify_final_failure(
                interrupted,
                attempts_exhausted,
                transient_exhausted,
                stream_err,
                platform_route=profile.platform_pays,
            )
            if (
                failure is not None
                and failure.code == PROVIDER_UNAVAILABLE_CODE
                and interrupted.handled_error is None
            ):
                report_provider_out_of_credits(
                    provider=profile.name,
                    model=state.observed_model if state is not None else None,
                    surface="copilot_sdk",
                    error=stream_err or "",
                    session_id=session_id,
                )
            if failure is not None:
                turn_summary.note_error(failure.code)
                provider_failure = _provider_failure_for(stream_ctx)
                cleanup_events: list[StreamBaseResponse] = []
                if state is not None:
                    state.adapter.end_open_blocks(cleanup_events)
                cleanup_events.extend(
                    interrupted.finalize(
                        session,
                        state,
                        failure.display_msg,
                        retryable=failure.retryable,
                        failure=(
                            provider_failure.as_part() if provider_failure else None
                        ),
                    )
                )
                for response in cleanup_events:
                    yield response
                already_yielded = (
                    interrupted.handled_error is not None
                    and interrupted.handled_error.already_yielded
                )
                if not already_yielded:
                    yield StreamError(errorText=failure.display_msg, code=failure.code)

        # Copy token usage from retry state to outer-scope accumulators
        # so the finally block can persist them.
        if state is not None:
            turn_prompt_tokens = state.usage.prompt_tokens
            turn_completion_tokens = state.usage.completion_tokens
            turn_cache_read_tokens = state.usage.cache_read_tokens
            turn_cache_creation_tokens = state.usage.cache_creation_tokens
            turn_cost_usd = state.usage.cost_usd

        # Emit token usage to the client (must be in try to reach SSE stream).
        # Session persistence of usage is in finally to stay consistent with
        # rate-limit recording even if an exception interrupts between here
        # and the finally block.
        if turn_prompt_tokens > 0 or turn_completion_tokens > 0:
            # total_tokens = prompt (uncached input) + completion (output).
            # Cache tokens are tracked separately and excluded from total
            # so that the semantics match the baseline path (OpenRouter)
            # which folds cache into prompt_tokens. Keeping total_tokens
            # = prompt + completion everywhere makes cross-path comparisons
            # and session-level aggregation consistent.
            total_tokens = turn_prompt_tokens + turn_completion_tokens
            yield StreamUsage(
                prompt_tokens=turn_prompt_tokens,
                completion_tokens=turn_completion_tokens,
                total_tokens=total_tokens,
                cache_read_tokens=turn_cache_read_tokens,
                cache_creation_tokens=turn_cache_creation_tokens,
            )

        if ended_with_stream_error:
            logger.warning(
                "%s Stream ended with SDK error after %d messages (compaction=%s)",
                log_prefix,
                len(session.messages),
                compaction.get_log_summary(),
            )
        else:
            logger.info(
                "%s Stream completed successfully with %d messages (compaction=%s)",
                log_prefix,
                len(session.messages),
                compaction.get_log_summary(),
            )
    except GeneratorExit:
        # GeneratorExit is raised when the async generator is closed by the
        # caller (e.g. client disconnect, page refresh).  We MUST release the
        # stream lock here because the ``finally`` block at the end of this
        # function may not execute when GeneratorExit propagates through nested
        # async generators.  Without this, the lock stays held for its full TTL
        # and the user sees "Another stream is already active" on every retry.
        logger.warning("%s GeneratorExit — releasing stream lock", log_prefix)
        await lock.release()
        raise
    except BaseException as e:
        # Catch BaseException to handle both Exception and CancelledError
        # (CancelledError inherits from BaseException in Python 3.8+)
        if isinstance(e, asyncio.CancelledError):
            logger.warning("%s Session cancelled", log_prefix)
            error_msg = "Operation cancelled"
        else:
            error_msg = str(e) or type(e).__name__
            # SDK cleanup errors are expected during client disconnect —
            # log as warning rather than error to reduce Sentry noise.
            # These are normally caught by _safe_close_sdk_client but
            # can escape in edge cases (e.g. GeneratorExit timing).
            if _is_sdk_disconnect_error(e):
                logger.warning(
                    "%s SDK cleanup error (client disconnect): %s",
                    log_prefix,
                    error_msg,
                )
            else:
                logger.error("%s Error: %s", log_prefix, error_msg, exc_info=True)

        is_transient = is_transient_api_error(error_msg)
        if is_transient:
            display_msg, code = FRIENDLY_TRANSIENT_MSG, "transient_api_error"
        else:
            display_msg, code = error_msg, "sdk_error"

        # Append error marker + restore any rolled-back partial when the retry
        # loop didn't already finalize. ``interrupted`` is empty on success and
        # on paths where the retry loop's own post-loop finalize() already ran,
        # so this is a no-op for those and only kicks in for unhandled errors
        # that bypass the retry-loop handlers entirely.
        if not ended_with_stream_error:
            interrupted.finalize(
                session,
                state,
                display_msg,
                retryable=is_transient,
                failure=(
                    gateway_failure.as_part()
                    if (
                        gateway_failure := _attributed_failure(
                            route_gateway,
                            (
                                credential_lease.credentials.id
                                if credential_lease is not None
                                else None
                            ),
                        )
                    )
                    else None
                ),
            )
            logger.debug(
                "%s Appended error marker, will be persisted in finally",
                log_prefix,
            )

        # Yield StreamError for immediate feedback (only for non-cancellation errors)
        # Skip for CancelledError and SDK disconnect cleanup errors — these
        # are not actionable by the user and the SSE connection is already dead.
        is_cancellation = isinstance(
            e, asyncio.CancelledError
        ) or _is_sdk_disconnect_error(e)
        if not is_cancellation:
            # An open pre-query row (emit_pre_query_start already yielded,
            # but the matching emit_pre_query_end/abort_pre_query never ran
            # because the error hit between the two) must be closed here —
            # otherwise the client is left with a permanently-open
            # compaction row. abort_pre_query is a no-op when nothing is
            # open, so this is safe to call unconditionally.
            if compaction is not None:
                for ev in compaction.abort_pre_query():
                    yield ev
            yield StreamError(errorText=display_msg, code=code)

        raise
    finally:
        turn_error = sys.exception()
        # The summary in the ``finally`` below reads these, so they are bound
        # before any cleanup step can raise.
        langfuse_trace_id: str | None = None
        _cost_reconciled = False
        # Why the CLI session upload below is skipped, or ``None``: the
        # upload and the summary read this one answer.
        upload_skip = upload_gate_reason(
            user_id=user_id,
            started=bool(sdk_cwd) and session is not None and state is not None,
            ended_with_stream_error=ended_with_stream_error,
        )
        try:
            tool_display_bridge.reset()
            # Pending messages are drained atomically at the start of each
            # turn (see drain_pending_messages call above), so there's
            # nothing to clean up here — any message pushed after that
            # point belongs to the next turn.

            if route_gateway is not None:
                deferred_gateway_cleanup_error = await _close_route_gateway_for_finally(
                    route_gateway,
                    log_prefix,
                )

                # The gateway read the upstream's own usage on every request:
                # on a gateway route it is the turn's count, and nothing
                # prices it (Codex is the account's, local the operator's).
                turn_cost_usd = None
                gateway_usage = _gateway_usage(route_gateway)
                if gateway_usage is not None:
                    turn_prompt_tokens = gateway_usage.input_tokens
                    turn_completion_tokens = gateway_usage.output_tokens
                    turn_cache_read_tokens = gateway_usage.cache_read_tokens
                    turn_cache_creation_tokens = gateway_usage.cache_write_tokens

            # --- Close OTEL context (with cost attributes) ---
            # Captured before __exit__ so the reconcile task (launched below,
            # after the span closes) can attach a backfill event to this turn's
            # Langfuse trace.  Without it, Langfuse shows the rate-card estimate
            # only — for non-Anthropic OpenRouter routes that's wildly wrong.
            if _otel_ctx is not None:
                try:
                    span = otel_trace.get_current_span()
                    if span and span.is_recording():
                        span.set_attribute(
                            "gen_ai.usage.prompt_tokens", turn_prompt_tokens
                        )
                        span.set_attribute(
                            "gen_ai.usage.completion_tokens", turn_completion_tokens
                        )
                        span.set_attribute(
                            "gen_ai.usage.cache_read_tokens", turn_cache_read_tokens
                        )
                        span.set_attribute(
                            "gen_ai.usage.cache_creation_tokens",
                            turn_cache_creation_tokens,
                        )
                        if turn_cost_usd is not None:
                            span.set_attribute("gen_ai.usage.cost_usd", turn_cost_usd)
                except Exception:
                    logger.debug("Failed to set OTEL cost attributes", exc_info=True)
                try:
                    _otel_ctx.__exit__(*sys.exc_info())
                except Exception:
                    logger.warning("OTEL context teardown failed", exc_info=True)
            if _lf_span is not None:
                # Capture from our Langfuse parent span before tearing it down;
                # tracks the lifetime of ``_lf_span`` so the trace id is still
                # available if ``_otel_ctx`` was never entered.
                try:
                    langfuse_trace_id = get_client().get_current_trace_id()
                except Exception:
                    logger.debug("Failed to capture Langfuse trace_id", exc_info=True)
                try:
                    _lf_span.__exit__(*sys.exc_info())
                except Exception:
                    logger.warning(
                        "Langfuse parent span teardown failed", exc_info=True
                    )

            # --- Persist token usage to session + rate-limit counters ---
            # Both must live in finally so they stay consistent even when an
            # exception interrupts the try block after StreamUsage was yielded.
            effective_model = sdk_model or _compression_model()
            # ``state`` is populated lazily inside the retry loop; when the
            # turn exits before the first attempt runs (e.g. very early
            # validation error) it's still None, so it has no generations.
            collected_gen_ids: list[str] = (
                list(state.generations.ids) if state is not None else []
            )
            absorbed_gen_ids: set[str] = (
                state.generations.absorbed if state is not None else set()
            )
            _cost_reconciled = bool(
                profile is not None
                and profile.row.cost_source == "openrouter_reconcile"
                and collected_gen_ids
            )

            if profile is not None and profile.row.cost_source == "codex_gateway":
                await persist_and_record_usage(
                    session=session,
                    user_id=user_id,
                    prompt_tokens=turn_prompt_tokens,
                    completion_tokens=turn_completion_tokens,
                    cache_read_tokens=turn_cache_read_tokens,
                    cache_creation_tokens=0,
                    log_prefix=log_prefix,
                    cost_usd=None,
                    model=effective_model,
                    provider=profile.row.cost_log_provider,
                    credential_id_override=(
                        credential_lease.credentials.id if credential_lease else None
                    ),
                    extra_metadata={"billing_mode": profile.row.payer},
                    execution_path="codex_claude_sdk",
                )
                # The gateway's own counts ride the turn summary beside the
                # CLI's. Re-derived (not reused from above): on early exits the
                # gateway was never built and that local is unbound.
                if route_gateway is not None:
                    gateway_usage = _gateway_usage(route_gateway)
                    turn_summary.note_codex_usage(
                        input_tokens=(
                            gateway_usage.prompt_tokens if gateway_usage else None
                        ),
                        cached_input_tokens=(
                            gateway_usage.cache_read_tokens if gateway_usage else None
                        ),
                        boundary_peak_tokens=route_gateway.peak_prompt_tokens,
                    )
            elif _cost_reconciled and profile is not None:
                # One background write, from OpenRouter's own bill for every
                # generation of the turn (``/generation?id=``), covering every
                # vendor. The rate-card price is only its fallback, for a turn
                # whose lookups fail. The sync path below is skipped, so
                # nothing is counted twice. Brief window (~0.5-2s) where the
                # rate-limit counter is unaware of this turn.
                cost_reconcile_task = asyncio.create_task(
                    record_turn_cost_from_openrouter(
                        session=session,
                        user_id=user_id,
                        model=effective_model,
                        prompt_tokens=turn_prompt_tokens,
                        completion_tokens=turn_completion_tokens,
                        cache_read_tokens=turn_cache_read_tokens,
                        cache_creation_tokens=turn_cache_creation_tokens,
                        generation_ids=collected_gen_ids,
                        cli_project_dir=_cli_project_dir(sdk_cwd),
                        cli_session_id=session_id,
                        turn_start_ts=turn_start_ts,
                        fallback_cost_usd=turn_cost_usd,
                        api_key=(
                            profile.api_key.get_secret_value()
                            if profile.api_key
                            else None
                        ),
                        log_prefix=log_prefix,
                        langfuse_trace_id=langfuse_trace_id,
                        absorbed_generation_ids=absorbed_gen_ids,
                    )
                )
                _background_tasks.add(cost_reconcile_task)
                cost_reconcile_task.add_done_callback(_background_tasks.discard)
                # A turn billed to credits is charged once this cost is in.
                wait_for_cost(cost_reconcile_task)
            elif profile is not None:
                # Priced on the platform's rate card from the tokens the
                # provider reported; the CLI's own figure is only logged.
                # Where the user's subscription pays, nothing is charged.
                if state is not None and turn_cost_usd is not None:
                    logger.info(
                        "%s Rate card $%.6f, CLI estimate $%.6f",
                        log_prefix,
                        turn_cost_usd,
                        state.usage.cli_cost_usd,
                    )
                await persist_and_record_usage(
                    session=session,
                    user_id=user_id,
                    prompt_tokens=turn_prompt_tokens,
                    completion_tokens=turn_completion_tokens,
                    cache_read_tokens=turn_cache_read_tokens,
                    cache_creation_tokens=turn_cache_creation_tokens,
                    log_prefix=log_prefix,
                    cost_usd=turn_cost_usd if profile.platform_pays else None,
                    model=effective_model,
                    provider=profile.row.cost_log_provider,
                    extra_metadata=(
                        None
                        if profile.platform_pays
                        else {
                            "billing_mode": profile.row.payer,
                            "estimated_cost_usd": turn_cost_usd,
                        }
                    ),
                )

            # --- Persist session messages ---
            # This MUST run in finally to persist messages even when the generator
            # is stopped early (e.g., user clicks stop, processor breaks stream loop).
            # Without this, messages disappear after refresh because they were never
            # saved to the database.
            if session is not None:
                # sdk_model (not effective_model): in subscription mode no
                # model was requested — the CLI's default must not read as a
                # fallback divergence against a fabricated "request".
                _stamp_turn_messages(
                    session.messages,
                    start_index=pre_turn_message_count,
                    requested_model=sdk_model,
                    actual_model=state.observed_model if state is not None else None,
                    routing_source=routing_source,
                )
                _stamp_turn_trace_id(
                    session.messages,
                    start_index=pre_turn_message_count,
                    trace_id=langfuse_trace_id,
                )
                # What this turn ran on, recorded on the turn rather than read
                # back off the session later, so a route change cannot rewrite it.
                stamp_segment(
                    session.messages,
                    pre_turn_message_count,
                    turn_segment,
                )
                try:
                    await asyncio.shield(upsert_chat_session(session))
                    logger.info(
                        "%s Session persisted in finally with %d messages",
                        log_prefix,
                        len(session.messages),
                    )
                    if turn_start is not None and not (
                        state is not None and _has_open_block(state.adapter)
                    ):
                        final_checkpoint = turn_checkpoint(session.messages, turn_start)
                except Exception as persist_err:
                    logger.error(
                        "%s Failed to persist session in finally: %s",
                        log_prefix,
                        persist_err,
                        exc_info=True,
                    )

            # --- Pause E2B sandbox to stop billing between turns ---
            # Fire-and-forget: pausing is best-effort and must not block the
            # response or the transcript upload.  The task is anchored to
            # _background_tasks to prevent garbage collection.
            # Use pause_sandbox_direct to skip the Redis lookup and reconnect
            # round-trip — e2b_sandbox is the live object from this turn.
            if e2b_sandbox is not None:
                task = asyncio.create_task(
                    pause_sandbox_direct(
                        e2b_sandbox,
                        session_id,
                        expert_id=session.expert_id if session else None,
                    )
                )
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)

            # --- Graphiti: ingest conversation turn for temporal memory ---
            if _graphiti_ingest_allowed(
                expert_identity_validated=expert_identity_validated,
                graphiti_enabled=graphiti_enabled,
                user_id=user_id,
                message=message,
                is_user_message=is_user_message,
            ):
                # Extract last assistant message from THIS TURN only (not all
                # session history) to avoid distilling stale content from prior
                # turns when the current turn errors before producing output.
                _this_turn_msgs = (
                    session.messages[pre_attempt_msg_count:] if session else []
                )
                _assistant_msgs = [
                    m.content or "" for m in _this_turn_msgs if m.role == "assistant"
                ]
                _last_assistant = _assistant_msgs[-1] if _assistant_msgs else ""

                _ingest_task = asyncio.create_task(
                    _enqueue_graphiti_turn(
                        user_id,
                        session,
                        session_id,
                        message,
                        _last_assistant,
                    )
                )
                _background_tasks.add(_ingest_task)
                _ingest_task.add_done_callback(_background_tasks.discard)

            # --- Skill learning: record this turn as a learning source ---
            # Independent of Graphiti: a memory-service failure must not change
            # what the nightly learner can consider. Fire-and-forget; the
            # helper is flag-gated and never raises.
            if expert_identity_validated and user_id and message and is_user_message:
                _capture_task = asyncio.create_task(
                    capture_chat_turn(
                        user_id,
                        session,
                        list(session.messages[learning_turn_start:]) if session else [],
                        message,
                        initiating_user=learning_initiating_user,
                        initiating_text=learning_initiating_text,
                    )
                )
                _background_tasks.add(_capture_task)
                _capture_task.add_done_callback(_background_tasks.discard)

            # --- Upload CLI native session file for cross-pod --resume ---
            # The CLI writes its native session JSONL after each turn completes.
            # The companion .meta.json carries the message_count watermark and mode
            # so the next turn can restore both --resume context and gap-fill state
            # in a single GCS round-trip via download_transcript().
            # asyncio.shield: if the outer finally-block coroutine is cancelled
            # while awaiting shield, the CancelledError propagates (BaseException,
            # not caught by `except Exception`) letting the caller handle
            # cancellation, while the shielded inner coroutine continues running
            # to completion so the upload is not lost.
            #
            # NOTE: upload is attempted regardless of state.use_resume — even when
            # this turn ran without --resume (restore failed or first T2+ on a new
            # pod), the T1 session file at the expected path may still be present
            # and should be re-uploaded so the next turn can resume from it.
            # read_cli_session_from_disk returns None when the file is absent, so
            # this is always safe.
            #
            # ``upload_skip is None`` implies both; spelled out for the type
            # checker.
            if upload_skip is None and session is not None and state is not None:
                logger.info(
                    "%s Attempting CLI session upload (use_resume=%s, has_history=%s)",
                    log_prefix,
                    state.use_resume,
                    has_history,
                )
                try:
                    # Read the CLI's native session file from disk (written by the CLI
                    # after the turn), then upload the bytes to GCS.
                    _cli_content = read_cli_session_from_disk(
                        sdk_cwd, session_id, log_prefix
                    )
                    if _cli_content:
                        _cli_content = _strip_synthetic_reprompt_from_cli_jsonl(
                            _cli_content
                        )
                        # Watermark = next uncovered DB sequence: the CLI wrote
                        # the session file after the turn, so it covers every
                        # row through this turn.
                        _jsonl_covered = _transcript_watermark_after_turn(
                            session.messages
                        )
                        await asyncio.shield(
                            upload_transcript(
                                user_id=user_id,
                                session_id=session_id,
                                content=_cli_content,
                                message_count=_jsonl_covered,
                                mode="sdk",
                                log_prefix=log_prefix,
                            )
                        )
                    turn_summary.note_upload(_cli_content)
                except Exception as cli_upload_err:
                    turn_summary.note_upload_failed()
                    logger.warning(
                        "%s CLI session upload failed in finally: %s",
                        log_prefix,
                        cli_upload_err,
                    )

            try:
                if sdk_cwd:
                    await _cleanup_sdk_tool_results(sdk_cwd)
            except Exception:
                logger.warning("%s SDK cleanup failed", log_prefix, exc_info=True)
            finally:
                # Release stream lock to allow new streams for this session
                await lock.release()

            _raise_deferred_gateway_cleanup_error(
                deferred_gateway_cleanup_error,
                turn_error,
            )
        finally:
            # Logged whatever the cleanup above did, also when a step of it
            # raised. Nothing evaluated for this call may raise: it would
            # replace the error in flight. Hence ``model_construct``.
            turn_summary.finish(
                error=turn_error,
                session=session,
                options=state.options if state is not None else None,
                compaction=compaction,
                transport=turn_route_name(profile, credential_lease, config),
                model_requested=sdk_model,
                model_answered=state.observed_model if state is not None else None,
                e2b=e2b_sandbox is not None,
                usage=TurnUsage.model_construct(
                    input_tokens=turn_prompt_tokens,
                    cache_read_tokens=turn_cache_read_tokens,
                    cache_write_tokens=turn_cache_creation_tokens,
                    output_tokens=turn_completion_tokens,
                    cost_usd=turn_cost_usd,
                ),
                codex=turn_route_name(profile, credential_lease, config) == "codex",
                cost_reconcile=_cost_reconciled,
                upload_gate=upload_skip,
                ended_with_stream_error=ended_with_stream_error,
            )
            turn_summary.emit(langfuse_trace_id)

    # Yielded here, not in ``finally``: an async generator cannot yield while
    # it is being closed, and a closed turn has nobody left to read it.
    if final_checkpoint is not None:
        yield final_checkpoint

    # -------------------------------------------------------------------------
    # Auto-continue: drain any messages the user queued AFTER the turn-start
    # drain window and process them as a new turn automatically.
    #
    # This code only executes on NORMAL turn completion.  GeneratorExit and
    # BaseException both re-raise inside their except blocks, so the generator
    # closes before reaching here — messages queued during a cancelled turn are
    # preserved in Redis for the next manual turn.
    #
    # A paused turn doesn't continue either: it stopped because it ran out of
    # budget or reached its check-in, and the user picks it back up from the
    # Paused card. Anything queued meanwhile stays queued for that turn.
    #
    # Nor does a platform turn the allowance can no longer cover: nobody
    # pressed send for the continuation, so nothing else would check it. The
    # queued messages stay queued and a not-started pause takes their place.
    # -------------------------------------------------------------------------
    if not ended_with_stream_error and not (governor is not None and governor.landed):
        # A turn on a Codex lease is paid by the user's ChatGPT subscription.
        # The continuation would have kept this turn's envelope, so a refusal
        # records its taint.
        auto_continue = await gate_auto_continue(
            session,
            user_id,
            platform_billed=credential_lease is None,
            tainted=envelope is not None and envelope.tainted,
        )
        if auto_continue.pause is not None:
            yield StreamTurnPaused(pause=auto_continue.pause.as_part())
        _auto_pending_messages = (
            await drain_pending_safe(session_id, log_prefix)
            if auto_continue.proceed
            else []
        )
        if _auto_pending_messages:
            logger.info(
                "%s Auto-continuing with %d pending message(s) queued after turn start",
                log_prefix,
                len(_auto_pending_messages),
            )
            # Combine all pending messages into one turn so they are processed
            # together rather than sequentially. The recursive call may itself
            # drain further messages queued while this turn runs.
            _auto_combined = "\n\n".join(pending_texts_from(_auto_pending_messages))
            # Race guard: drain_pending_safe has already LPOPed the messages
            # from Redis. If another request acquires the session lock in the
            # window between our lock.release() above and the recursive call's
            # try_acquire() below, that recursive call exits with
            # "stream_already_active" and the drained messages would be
            # permanently lost. Detect that sentinel on the first yielded
            # event and push the drained messages back to Redis so the
            # competing stream's turn-start drain picks them up — preserving
            # the original ``file_ids`` / ``context`` metadata (sentry
            # r3105523410 — text-only requeue silently stripped it).
            _auto_requeued = False
            _first_auto_event = True

            async def _requeue_drained(reason: str) -> None:
                logger.warning(
                    "%s Auto-continue %s; re-queueing %d drained message(s)",
                    log_prefix,
                    reason,
                    len(_auto_pending_messages),
                )
                for _pm in _auto_pending_messages:
                    try:
                        await push_pending_message(session_id, _pm)
                    except Exception:
                        logger.exception(
                            "%s Failed to re-queue auto-continue message",
                            log_prefix,
                        )

            try:
                async for event in stream_chat_completion_sdk(
                    session_id=session_id,
                    message=_auto_combined,
                    is_user_message=True,
                    user_id=user_id,
                    session=session,
                    file_ids=None,
                    permissions=permissions,
                    # Same turn continuing, so it keeps its envelope. Omitting
                    # it would default to None and clear the contextvar for the
                    # remainder of the turn: tool enforcement off, spend
                    # uncharged, and the next spawn minted as an unbounded root.
                    envelope=envelope,
                    model=model,
                    organization_id=organization_id,
                    team_id=team_id,
                    credential_lease=credential_lease,
                    # A new reading for the queued messages it answers: not
                    # spoken, still unattended and from the same source if
                    # this was. Whether it answers a kickoff is read once
                    # their rows are in.
                    session_kind=SessionKind.for_turn(
                        session,
                        scheduled=kind.unattended,
                        from_chat_platform=kind.from_chat_platform,
                    ),
                    turn_start=turn_start,
                    continued_pending=_auto_pending_messages,
                ):
                    if _first_auto_event:
                        _first_auto_event = False
                        if (
                            isinstance(event, StreamError)
                            and getattr(event, "code", None) == "stream_already_active"
                        ):
                            await _requeue_drained("lost lock race")
                            _auto_requeued = True
                            # Suppress the stale "already active" error —
                            # the competing stream will emit its own events.
                            continue
                    yield event
            except Exception:
                # Eager-persist rollback or any other failure inside the
                # recursive call before messages were consumed. Push the
                # drained texts back so the next turn picks them up.
                if not _auto_requeued:
                    await _requeue_drained("raised during recursive call")
                raise
            if _auto_requeued:
                return


async def _fetch_graphiti_context(
    user_id: str | None,
    session: ChatSession,
    message: str | None,
) -> tuple[bool, str]:
    """Check Graphiti flag and fetch warm context in one shot.

    Returns ``(graphiti_enabled, warm_ctx)`` where ``warm_ctx`` is a
    pre-loaded fact bundle injected into the first user message (not the
    system prompt) so the system prompt stays identical across users and
    sessions, enabling cross-session Anthropic prompt-cache hits.  Skips
    the fetch on follow-up turns (history > 1 message) and when the user
    is anonymous.
    """
    enabled = await is_enabled_for_user(user_id)
    if not enabled:
        return False, ""
    if not (user_id and len(session.messages) <= 1):
        return True, ""

    ctx = (
        await fetch_warm_context(user_id, message or "", expert_id=session.expert_id)
        or ""
    )
    return True, ctx


async def _first_turn_context_blocks(
    session: ChatSession,
    user_id: str | None,
    kind: SessionKind,
    *,
    sdk_cwd: str,
    use_e2b: bool,
    understanding: BusinessUnderstanding | None,
    warm_ctx: str = "",
) -> list[ContextBlock]:
    """The blocks that lead the session's first user message, for *kind*.

    The first turn writes them on its row; a context that has lost that message
    gets them again, rebuilt (``_rebuild_first_turn_context``).  Besides the
    shared ones (``injected_context_blocks``): the working directory; the
    session's follow-ups, so the model can answer "cancel that" / "what did I
    schedule" without a ``list_schedules`` round-trip (it degrades to the bare
    session_id line on a scheduler RPC hiccup); and the per-user skill index,
    which names the declared ``mcp__copilot__run_capability`` the model calls.
    All of them land in the per-turn user message, after the cache breakpoint,
    so they do not bust the cross-session prefix cache.  A skill-index failure
    must not block the turn: it is logged and the index left out.  ``warm_ctx``
    is the first turn's Graphiti memory; a rebuild has none.
    """
    env_ctx = f"working_dir: {sdk_cwd}" if sdk_cwd and not use_e2b else ""
    session_ctx = ""
    if user_id:
        session_ctx = await build_session_context(
            session_id=session.session_id, user_id=user_id
        )
    skills = None
    try:
        skills = await skills_context_block(
            user_id,
            expert_id=session.expert_id,
            run_capability_tool=f"{MCP_TOOL_PREFIX}run_capability",
        )
    except Exception:
        logger.exception("[skills] failed to build skills_ctx — proceeding without it")
    return await injected_context_blocks(
        understanding,
        warm_ctx=warm_ctx,
        env_ctx=env_ctx,
        session_ctx=session_ctx,
        skills=skills,
        user_id=user_id,
        expert_id=session.expert_id,
        kickoff_turn=kind.expert_kickoff,
    )


# Leads the blocks the ``SessionStart`` hook hands back after a compaction, so
# the model reads them as the first message's blocks, not as something new.
_AFTER_COMPACTION_LEAD = (
    "The conversation was just compacted, and the summary does not carry the "
    "server-injected blocks from the start of its first user message. Here "
    "they are again, rebuilt now; treat them exactly as that message's blocks:"
    "\n\n"
)


async def _context_after_compaction(
    session: ChatSession,
    user_id: str | None,
    kind: SessionKind,
    *,
    sdk_cwd: str,
    use_e2b: bool,
    log_prefix: str,
    hidden_tools: frozenset[str] = frozenset(),
    turn_summary: TurnSummaryCollector | None = None,
) -> list[str]:
    """What the ``SessionStart`` hook hands the model after the CLI compacts.

    Compaction replaces the session's first user message with a summary, and
    the first-turn blocks went with it — for the rest of the turn and for
    every turn that resumes the compacted session.  Returned as the parts
    one hook callback each delivers, every one within the CLI's hook-output
    cap (``fit_after_compaction``); ``[]`` when there is nothing to re-send.
    ``hidden_tools`` are the tools this turn does not offer: a block is not
    shortened with a pointer to one of them.  The hook runs this once per
    compaction the CLI completes, and each run counts on *turn_summary*.
    """
    if turn_summary is not None:
        turn_summary.note_after_compaction_resend()
    blocks = await _rebuild_first_turn_context(
        session,
        user_id,
        kind,
        sdk_cwd=sdk_cwd,
        use_e2b=use_e2b,
        log_prefix=log_prefix,
        reason="compaction",
    )
    fitted = fit_after_compaction(
        _AFTER_COMPACTION_LEAD, blocks, hidden_tools=hidden_tools
    )
    if fitted.dropped or fitted.shortened:
        logger.warning(
            "%s First-turn context re-sent after compaction was cut to fit the"
            " CLI hook-output cap: %d of %d chars in %d parts; dropped=%s,"
            " shortened=%s",
            log_prefix,
            fitted.delivered_chars,
            fitted.rebuilt_chars,
            len(fitted.parts),
            fitted.dropped,
            fitted.shortened,
        )
    return fitted.parts


async def _rebuild_first_turn_context(
    session: ChatSession,
    user_id: str | None,
    kind: SessionKind,
    *,
    sdk_cwd: str,
    use_e2b: bool,
    log_prefix: str,
    reason: str,
) -> list[ContextBlock]:
    """The first-turn blocks again, for a context that has lost them.

    The first turn's blocks reach later model calls only through the
    session's first user message.  Two things take that message away: a
    later turn that cannot ``--resume`` (``_build_query_message`` puts the
    rendered blocks ahead of ``<conversation_history>``), and the CLI's own
    compaction, which replaces it with a summary
    (``_context_after_compaction``).  These are the first turn's own blocks,
    from the same builders — skills index, expert or team workflows,
    follow-ups, working directory and user context.  Graphiti memory
    (``<memory_context>``) is not among them: it is fetched for the first
    turn only (``_fetch_graphiti_context``) and is not re-sent here.  Best
    effort: any failure is logged and ``[]`` returned, and the turn goes on
    without them.
    """
    try:
        understanding = await fetch_business_understanding(user_id)
        blocks = await _first_turn_context_blocks(
            session,
            user_id,
            kind,
            sdk_cwd=sdk_cwd,
            use_e2b=use_e2b,
            understanding=understanding,
        )
    except Exception:
        logger.exception(
            "%s Could not rebuild the first-turn context (%s) — going on without it",
            log_prefix,
            reason,
        )
        return []
    logger.info(
        "%s Rebuilt first-turn context (%s): %d chars; skills=%s, user_context=%s",
        log_prefix,
        reason,
        len(render_context_blocks(blocks)),
        any(block.tag == SKILLS_CONTEXT_TAG for block in blocks),
        understanding is not None,
    )
    return blocks


def _graphiti_ingest_allowed(
    *,
    expert_identity_validated: bool,
    graphiti_enabled: bool,
    user_id: str | None,
    message: str | None,
    is_user_message: bool,
) -> bool:
    return bool(
        expert_identity_validated
        and graphiti_enabled
        and user_id
        and message
        and is_user_message
    )


async def _enqueue_graphiti_turn(
    user_id: str,
    session: ChatSession,
    session_id: str,
    message: str,
    assistant_msg: str,
) -> None:
    await enqueue_conversation_turn(
        user_id,
        session_id,
        message,
        assistant_msg=assistant_msg,
        expert_id=session.expert_id,
    )


def _canonical_model(model: str) -> str:
    """Spelling-insensitive model identity for fallback detection:
    vendor prefix stripped, dots→dashes, trailing -YYYYMMDD dropped."""
    tail = model.lower().split("/")[-1]
    if tail.startswith("anthropic."):
        tail = tail[len("anthropic.") :]
    return MODEL_DATE_SUFFIX_RE.sub("", tail.replace(".", "-"))


def _same_model(a: str, b: str | None) -> bool:
    return b is not None and _canonical_model(a) == _canonical_model(b)


def _sdk_serving_segment(
    credential_lease: CredentialLease | CodexCredentialLease | None,
) -> Segment:
    """The immutable route that actually serves this SDK turn."""
    if credential_lease is None:
        return Segment("platform", None, is_segment_zero=False)
    return Segment(
        "codex",
        credential_lease.credentials.id,
        is_segment_zero=False,
    )


def _stamp_turn_messages(
    messages: list[ChatMessage],
    *,
    start_index: int,
    requested_model: str | None,
    actual_model: str | None,
    routing_source: "RoutingSource",
) -> None:
    """Stamp THIS turn's assistant messages with the serving model and
    routing layer.

    The SDK adapters build messages far from the resolution context, so
    the stamp lands at persist time — bounded to ``start_index`` (the
    session length when the turn began) so pre-feature history rows with
    NULL model are never back-stamped with today's values, and never
    overwriting an already-stamped row.

    ``actual_model`` is the model observed on ``AssistantMessage.model``
    (survives retries). When it names a DIFFERENT MODEL than the requested
    resolution the CLI's overload fallback served the turn — the stamp
    records the actual model with ``routing_source="fallback"`` so
    analytics never attribute a fallback-served turn to the layer that
    routed a different model. The comparison is canonicalized (vendor
    prefix, dot/dash, date-suffix spellings) so a spelling difference
    between the request and the CLI's report never fakes a fallback.

    ``requested_model=None`` means no specific model was requested
    (subscription mode lets the CLI pick its default) — there is nothing
    to diverge from, so the observed model stamps with the original
    routing source, never "fallback".
    """
    model = actual_model or requested_model
    if (
        actual_model is not None
        and requested_model is not None
        and not _same_model(actual_model, requested_model)
    ):
        routing_source = "fallback"
    for msg in messages[start_index:]:
        if msg.role == "assistant" and msg.model is None:
            msg.model = model
            msg.routing_source = routing_source
            if msg.sequence is not None:
                # Row already flushed to the DB mid-turn — flag it so the
                # save path back-fills the columns (insert only covers
                # unsequenced rows).
                msg.stamps_pending_save = True


def _stamp_turn_trace_id(
    messages: list[ChatMessage],
    *,
    start_index: int,
    trace_id: str | None,
) -> None:
    """Record the turn's Langfuse trace on the reply rows it wrote.

    A thumbs up/down on the reply is scored against this trace (see
    ``backend.copilot.feedback``). Every rateable role is stamped: the UI
    names a reply bubble after its last assistant *or* reasoning row, and a
    rating of either must find the trace. Bounded to the turn and never
    overwriting, exactly like ``_stamp_turn_messages``; rows flushed mid-turn
    ride the same stamps back-fill.
    """
    if not trace_id:
        return
    for msg in messages[start_index:]:
        if msg.role in RATEABLE_ROLES and msg.langfuse_trace_id is None:
            msg.langfuse_trace_id = trace_id
            if msg.sequence is not None:
                msg.stamps_pending_save = True
