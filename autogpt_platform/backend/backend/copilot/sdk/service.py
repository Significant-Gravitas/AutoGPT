"""Claude Agent SDK service layer for CoPilot chat completions."""

# isort: skip_file  — double-dot relative imports must stay relative to avoid Pyright type collisions

import asyncio
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
from typing import TYPE_CHECKING, Any, NamedTuple, NotRequired, cast

if TYPE_CHECKING:
    from ..permissions import CopilotPermissions

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    StreamEvent,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from claude_agent_sdk.types import SystemPromptPreset
from langfuse import get_client, propagate_attributes
from langsmith.integrations.claude_agent_sdk import configure_claude_agent_sdk
from opentelemetry import trace as otel_trace
from pydantic import BaseModel

from backend.data.redis_client import get_redis_async
from backend.executor.cluster_lock import AsyncClusterLock
from backend.util.exceptions import NotFoundError
from backend.util.settings import Settings

from ..config import ChatConfig, CopilotLlmModel, CopilotMode
from ..constants import (
    COPILOT_ERROR_PREFIX,
    COPILOT_RETRYABLE_ERROR_PREFIX,
    FRIENDLY_TRANSIENT_MSG,
    STREAM_INCOMPLETE_MARKER,
    STREAM_LOCK_PREFIX,
    is_transient_api_error,
)
from ..session_cleanup import prune_orphan_tool_calls
from ..context import encode_cwd_for_cli, get_workspace_manager
from ..graphiti.config import is_enabled_for_user
from ..model_router import resolve_model
from ..moonshot import (
    is_moonshot_model as _is_moonshot_model,
    override_cost_usd as _override_cost_for_moonshot,
)
from ..model import (
    ChatMessage,
    ChatSession,
    get_chat_session,
    maybe_append_user_message,
    upsert_chat_session,
)
from ..pending_message_helpers import (
    combine_pending_with_current,
    drain_pending_safe,
    pending_texts_from,
    persist_pending_as_user_rows,
    persist_session_safe,
)
from ..pending_messages import (
    drain_pending_for_persist,
    push_pending_message,
)
from ..permissions import apply_tool_permissions
from ..prompting import get_graphiti_supplement, get_sdk_supplement
from ..rate_limit import (
    get_global_rate_limits,
    get_remaining_usd_budget,
    get_user_tier,
)
from ..response_model import (
    StreamBaseResponse,
    StreamError,
    StreamFinish,
    StreamFinishStep,
    StreamHeartbeat,
    StreamReasoningDelta,
    StreamReasoningEnd,
    StreamReasoningStart,
    StreamStart,
    StreamStartStep,
    StreamStatus,
    StreamTextDelta,
    StreamTextEnd,
    StreamTextStart,
    StreamToolInputAvailable,
    StreamToolInputStart,
    StreamToolOutputAvailable,
    StreamUsage,
)
from ..builder_context import (
    build_builder_context_turn_prefix,
    build_builder_system_prompt_suffix,
)
from ..service import (
    _build_system_prompt,
    _is_langfuse_configured,
    _update_title_async,
    inject_user_context,
    strip_user_context_tags,
)
from ..thinking_stripper import ThinkingStripper
from ..token_tracking import persist_and_record_usage
from ..tools import ToolGroup
from ..tools.e2b_sandbox import get_or_create_sandbox, pause_sandbox_direct
from ..tools.sandbox import WORKSPACE_PREFIX, make_session_path
from ..tracking import track_user_message
from ..transcript import (
    _run_compression,
    TranscriptDownload,
    cleanup_stale_project_dirs,
    cli_session_path,
    compact_transcript,
    download_transcript,
    extract_context_messages,
    projects_base,
    read_compacted_entries,
    strip_for_upload,
    upload_transcript,
    validate_transcript,
)
from ..transcript_builder import TranscriptBuilder, TranscriptSnapshot
from .compaction import CompactionTracker, filter_compaction_messages
from .env import build_sdk_env  # noqa: F401 — re-export for backward compat
from .openrouter_cost import record_turn_cost_from_openrouter
from .response_adapter import SDKResponseAdapter
from .security_hooks import create_security_hooks
from .tool_adapter import (
    MCP_TOOL_PREFIX,
    create_copilot_mcp_server,
    get_copilot_tool_names,
    get_sdk_disallowed_tools,
    reset_stash_event,
    reset_tool_failure_counters,
    set_execution_context,
    wait_for_stash,
)

logger = logging.getLogger(__name__)
config = ChatConfig()


class _SystemPromptPreset(SystemPromptPreset, total=False):
    """Extends :class:`SystemPromptPreset` with ``exclude_dynamic_sections``.

    The field was added to the upstream TypedDict in claude-agent-sdk 0.1.59.
    Until the package is pinned to that version we declare it locally so Pyright
    accepts the kwarg without a ``# type: ignore`` comment.
    """

    exclude_dynamic_sections: NotRequired[bool]


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
    "AutoPilot was unable to complete the tool call "
    "— this usually happens when the response is "
    "too large to fit in a single tool call. "
    "Try breaking your request into smaller parts."
)

# Two regimes: no tool pending → 30 min (SDK genuinely idle); tool pending →
# 2 h hard cap (lets long sub-AutoPilots run, still backstops a hung tool).
_IDLE_TIMEOUT_SECONDS = 30 * 60
_HUNG_TOOL_CAP_SECONDS = 2 * 60 * 60

# Floor on the per-query SDK budget — too small and the CLI refuses to
# start a turn at all.  Caller (``_resolve_dynamic_max_budget_usd``) clamps
# to this so a user near (but not yet at) their cap still gets a chance to
# wrap up gracefully instead of hitting an immediate hard stop.
_MAX_BUDGET_USD_FLOOR = 0.5


async def _resolve_dynamic_max_budget_usd(user_id: str | None) -> float:
    """Pick the per-query ``max_budget_usd`` for this user *now*.

    Returns the smaller of:
      * ``config.claude_agent_max_budget_usd`` (the static per-query cap)
      * the user's remaining daily/weekly USD cap (from Redis)

    Falls back to the static cap when no ``user_id`` is available (e.g. an
    internal turn run without auth).  Floored at ``_MAX_BUDGET_USD_FLOOR``
    so a near-capped user still gets enough headroom to surface the
    "wrap up" reminder rather than failing to dispatch.
    """
    static_cap = config.claude_agent_max_budget_usd
    if not user_id:
        return static_cap
    daily_limit, weekly_limit, _ = await get_global_rate_limits(
        user_id,
        config.daily_cost_limit_microdollars,
        config.weekly_cost_limit_microdollars,
    )
    # Sentinel ``-1.0`` from ``get_remaining_usd_budget`` means Redis was
    # unavailable.  In that case fall back to the static per-query cap —
    # we don't actually know the user is near their limit, and clamping
    # to the floor would shrink every turn to $0.50 while Redis is in a
    # brown-out.  The pre-turn gate already failed closed at 503 in this
    # branch, so this defensive fallback only matters in edge paths that
    # bypass the gate.
    remaining = await get_remaining_usd_budget(
        user_id=user_id,
        daily_cost_limit=daily_limit,
        weekly_cost_limit=weekly_limit,
        floor_usd=-1.0,
    )
    if remaining < 0 or remaining == float("inf"):
        return static_cap
    return max(_MAX_BUDGET_USD_FLOOR, min(static_cap, remaining))


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
    async for sdk_msg in _iter_sdk_messages(client):
        # Heartbeat sentinel — refresh lock and keep SSE alive
        if sdk_msg is None:
            await ctx.lock.refresh()
            for ev in ctx.compaction.emit_start_if_ready():
                yield ev
            yield StreamHeartbeat()

            # Threshold flips to the long cap while a tool is pending; clock never resets.
            idle_seconds = time.monotonic() - loop_state.last_real_msg_time
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
                    f"AutoPilot stopped responding{tool_phrase}. "
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
            if (
                msg_id is not None
                and msg_id.startswith("gen-")
                and msg_id not in state.generation_ids
            ):
                state.generation_ids.append(msg_id)
            # Track the model the SDK actually used — when a fallback
            # activates, this differs from ``state.options.model``.
            # Consumed by the Moonshot cost-override decision so we
            # don't mis-bill a fallback-Anthropic response at
            # Moonshot rates (or a fallback-Moonshot at Anthropic
            # rates).
            observed = getattr(sdk_msg, "model", None)
            if isinstance(observed, str) and observed:
                state.observed_model = observed

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

            # Intercept transient API errors (socket closed,
            # ECONNRESET) — replace the raw message with a
            # user-friendly error text and use the retryable
            # error prefix so the frontend shows a retry button.
            # Check both the error field and content for patterns.
            if is_transient_api_error(error_text) or is_transient_api_error(
                error_preview
            ):
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
                raise RuntimeError("Prompt is too long")

            # Capture token usage from ResultMessage.
            # Anthropic reports cached tokens separately:
            #   input_tokens = uncached only
            #   cache_read_input_tokens = served from cache
            #   cache_creation_input_tokens = written to cache
            if sdk_msg.usage:
                # Use `or 0` instead of a default in .get() because
                # OpenRouter may include the key with a null value (e.g.
                # {"cache_read_input_tokens": null}) for models that don't
                # yet report cache tokens, making .get("key", 0) return
                # None rather than the fallback 0.
                state.usage.prompt_tokens += sdk_msg.usage.get("input_tokens") or 0
                state.usage.cache_read_tokens += (
                    sdk_msg.usage.get("cache_read_input_tokens") or 0
                )
                state.usage.cache_creation_tokens += (
                    sdk_msg.usage.get("cache_creation_input_tokens") or 0
                )
                state.usage.completion_tokens += sdk_msg.usage.get("output_tokens") or 0
                logger.info(
                    "%s Token usage: uncached=%d, cache_read=%d, "
                    "cache_create=%d, output=%d",
                    ctx.log_prefix,
                    state.usage.prompt_tokens,
                    state.usage.cache_read_tokens,
                    state.usage.cache_creation_tokens,
                    state.usage.completion_tokens,
                )
            if sdk_msg.total_cost_usd is not None:
                # Default: trust the CLI-reported value.  Accurate for
                # Anthropic models (the CLI's bundled pricing table is
                # Anthropic-authored), and becomes the sync-path cost
                # when the reconcile is disabled or fails.
                # Prefer the ACTUALLY executed model
                # (``state.observed_model`` from ``AssistantMessage.model``)
                # over the requested primary (``state.options.model``)
                # so a fallback activation doesn't mis-route pricing.
                active_model = state.observed_model or getattr(
                    state.options, "model", None
                )
                if _is_moonshot_model(active_model):
                    # Moonshot slug — the CLI doesn't know Moonshot's
                    # rate card and silently bills at Sonnet rates
                    # (~5x over-charge).  Replace with the rate-card
                    # estimate so the in-stream ``cost_usd`` and the
                    # reconcile's lookup-fail fallback reflect
                    # reality.  Reconcile
                    # (``record_turn_cost_from_openrouter``) still
                    # overrides this value when every gen-ID lookup
                    # succeeds.
                    state.usage.cost_usd = _override_cost_for_moonshot(
                        model=active_model,
                        sdk_reported_usd=sdk_msg.total_cost_usd,
                        prompt_tokens=state.usage.prompt_tokens,
                        completion_tokens=state.usage.completion_tokens,
                        cache_read_tokens=state.usage.cache_read_tokens,
                        cache_creation_tokens=state.usage.cache_creation_tokens,
                    )
                else:
                    state.usage.cost_usd = sdk_msg.total_cost_usd

        # Emit compaction end if SDK finished compacting.
        # Sync TranscriptBuilder with the CLI's active context.
        compact_result = await ctx.compaction.emit_end_if_ready(ctx.session)
        if compact_result.events:
            # Compaction events end with StreamFinishStep, which maps to
            # Vercel AI SDK's "finish-step" — that clears activeTextParts.
            # Close any open text block BEFORE the compaction events so
            # the text-end arrives before finish-step, preventing
            # "text-end for missing text part" errors on the frontend.
            pre_close: list[StreamBaseResponse] = []
            state.adapter._end_text_if_open(pre_close)
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
            compacted = await asyncio.to_thread(
                read_compacted_entries,
                compact_result.transcript_path,
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
                # Persistence (via _dispatch_response) always runs so the
                # session transcript keeps role='reasoning' rows; the
                # wire is gated so UI can suppress rendering.
                if not state.adapter.render_reasoning_in_ui and isinstance(
                    dispatched,
                    (
                        StreamReasoningStart,
                        StreamReasoningDelta,
                        StreamReasoningEnd,
                    ),
                ):
                    continue
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
                    # Track CLI-JSONL-invisible rows so the upload
                    # watermark excludes them and the next turn's
                    # detect_gap picks them up as gap-fill.
                    state.midturn_user_rows += len(followup_drained)

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
        has_pending_tools = (
            acc.has_appended_assistant
            and acc.accumulated_tool_calls
            and not acc.has_tool_results
        )
        adapter = state.adapter
        has_open_block = (adapter.has_started_text and not adapter.has_ended_text) or (
            adapter.has_started_reasoning and not adapter.has_ended_reasoning
        )
        if (
            not has_pending_tools
            and not has_open_block
            and (
                loop_state.msgs_since_flush >= _FLUSH_MESSAGE_THRESHOLD
                or (now - loop_state.last_flush_time) >= _FLUSH_INTERVAL_SECONDS
            )
        ):
            try:
                await asyncio.shield(upsert_chat_session(ctx.session))
                logger.debug(
                    "%s Intermediate flush: %d messages "
                    "(msgs_since=%d, elapsed=%.1fs)",
                    ctx.log_prefix,
                    len(ctx.session.messages),
                    loop_state.msgs_since_flush,
                    now - loop_state.last_flush_time,
                )
            except Exception as flush_err:
                logger.warning(
                    "%s Intermediate flush failed: %s",
                    ctx.log_prefix,
                    flush_err,
                )
            loop_state.last_flush_time = now
            loop_state.msgs_since_flush = 0

        if acc.stream_completed:
            break


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


# Event types that are ephemeral / cosmetic and must NOT be counted toward
# ``events_yielded`` in the transient-retry loop.  Counting them would prevent
# the backoff retry from firing because ``_next_transient_backoff`` returns
# ``None`` when ``events_yielded > 0``.
_EPHEMERAL_EVENT_TYPES = (
    StreamHeartbeat,
    # Compaction UI events are cosmetic and must not block retry — they're
    # emitted before the SDK query on compacted attempts.
    StreamStartStep,
    StreamFinishStep,
    StreamToolInputStart,
    StreamToolInputAvailable,
    StreamToolOutputAvailable,
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


def _is_tool_only_message(sdk_msg: object) -> bool:
    """Return True if *sdk_msg* is an AssistantMessage containing only ToolUseBlocks.

    Such a message represents a parallel tool-call batch (no text output yet).
    The ``bool(…content)`` guard prevents vacuous-truth evaluation on an empty list.
    """
    return (
        isinstance(sdk_msg, AssistantMessage)
        and bool(sdk_msg.content)
        and all(isinstance(b, ToolUseBlock) for b in sdk_msg.content)
    )


class ReducedContext(NamedTuple):
    builder: TranscriptBuilder
    use_resume: bool
    resume_file: str | None
    transcript_lost: bool
    tried_compaction: bool
    # Token budget for history compression on the DB-message fallback path.
    # None means "use model-aware default".  Halved on each retry so
    # compress_context applies progressively more aggressive reduction
    # (LLM summarize → content truncate → middle-out delete → first/last trim).
    target_tokens: int | None = None


@dataclass
class _TokenUsage:
    """Token usage accumulators for a single turn.

    Separated from `_RetryState` because usage is reset between retry
    attempts independently of the retry-control fields, and is read by
    the outer `stream_chat_completion_sdk` scope after the retry loop
    completes.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float | None = None

    def reset(self) -> None:
        """Reset all accumulators for a new attempt."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cache_read_tokens = 0
        self.cache_creation_tokens = 0
        self.cost_usd = None


@dataclass
class _RetryState:
    """Mutable state passed to `_run_stream_attempt` instead of closures.

    Holds values that the retry loop modifies between attempts so the inner
    generator does not rely on reassigning closure variables.
    """

    options: ClaudeAgentOptions
    query_message: str
    was_compacted: bool
    use_resume: bool
    resume_file: str | None
    transcript_msg_count: int
    adapter: SDKResponseAdapter
    transcript_builder: TranscriptBuilder
    usage: _TokenUsage
    # Token budget for history compression on retries (DB-message fallback path).
    # None = model-aware default.  Halved each retry for progressively more
    # aggressive compression (LLM summarize → truncate → middle-out → trim).
    target_tokens: int | None = None
    # Count of user rows inserted MID-TURN by the follow-up persist path
    # (``StreamToolOutputAvailable`` handler).  The CLI JSONL does NOT contain
    # these as separate user entries — they are embedded inside tool_result
    # text via the MCP wrapper injection, which the CLI may even strip when
    # the tool output exceeds its internal size cap.  Tracking them separately
    # lets the upload path subtract from ``message_count`` so the next turn's
    # ``detect_gap`` picks them up as gap-fill entries instead of assuming the
    # JSONL already covers them.
    midturn_user_rows: int = 0
    # Tracks whether the thinking-only-final-turn re-prompt has already
    # fired for this user turn.  Lives on ``_RetryState`` (not on
    # ``adapter``) so a transient retry that rebuilds the adapter does
    # not reset the per-turn cap to zero — otherwise multiple retries
    # could each fire their own re-prompt round.
    thinking_only_reprompted: bool = False
    # OpenRouter generation IDs collected across all attempts of this turn.
    # Populated from ``AssistantMessage.message_id`` when routed via
    # OpenRouter (``gen-...`` prefix).  Consumed by the finally block to
    # fire ``record_turn_cost_from_openrouter`` for non-Anthropic models —
    # the CLI's static-Anthropic-priced estimate is replaced with the
    # authoritative ``/generation`` total_cost.  Lives on ``_RetryState``
    # (not per-attempt ``_StreamAccumulator``) so it survives retries.
    generation_ids: list[str] = dataclass_field(default_factory=list)
    # The *actually executed* model observed on ``AssistantMessage.model`` —
    # differs from ``state.options.model`` (the requested primary) when
    # ``_resolve_fallback_model`` swaps to a fallback mid-attempt.  The
    # Moonshot cost override gates on this so a Moonshot-→-Anthropic
    # fallback doesn't get mis-billed at Moonshot rates, and vice versa.
    observed_model: str | None = None


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


# Per-retry token budgets for the no-transcript (use_resume=False) path.
# When there is no CLI native session to --resume, context is built from DB
# messages via _format_conversation_context.  For large sessions this text
# can exceed the model context window; each retry halves the token budget so
# compress_context applies progressively more aggressive reduction:
#   LLM summarize → content truncate → middle-out delete → first/last trim.
# Index 0 = first retry, 1 = second retry; last value applies beyond that.
_RETRY_TARGET_TOKENS: tuple[int, ...] = (50_000, 5_000)

# Below this token budget the model context is so tight that injecting any
# conversation history would likely exceed the limit regardless of content.
# _build_query_message returns the bare message when target_tokens falls to
# or below this floor, giving the user a response instead of a hard error.
_BARE_MESSAGE_TOKEN_FLOOR: int = 5_000

# Tight token budget for seeding the transcript builder on turns where no
# CLI native session exists.  Kept below _RETRY_TARGET_TOKENS[0] so the
# seeded JSONL upload stays compact and future gap injections are small.
_SEED_TARGET_TOKENS: int = 30_000

# Headroom subtracted from the CLI's autocompact threshold when sizing our
# own retry-path compaction target.  Without this gap the post-compact
# context would land just under the CLI's threshold and the next assistant
# message would tip it back over → CLI immediately re-compacts → cascade.
_COMPACTION_HEADROOM_TOKENS: int = 20_000


def _compaction_target_tokens(model: str) -> int:
    """Compaction target consistent with the CLI's autocompact threshold.

    Mirrors the bundled CLI's ``i6_()`` formula for autocompact:
    ``min(window * pct/100, window - 13K)``, then subtracts a 20K headroom
    so post-compaction context sits comfortably below the CLI's trigger and
    a follow-up assistant message doesn't immediately re-trigger.
    Floors at 10K to preserve at least some history budget.
    """
    from backend.util.prompt import DEFAULT_TOKEN_THRESHOLD, get_context_window

    window = get_context_window(model)
    if window is None:
        return DEFAULT_TOKEN_THRESHOLD
    pct = config.claude_agent_autocompact_pct_override
    cli_buffer = 13_000  # E88 in the bundled CLI
    if pct > 0 and not _is_moonshot_model(model):
        cli_threshold = min(window * pct // 100, window - cli_buffer)
    else:
        cli_threshold = window - cli_buffer
    return max(10_000, cli_threshold - _COMPACTION_HEADROOM_TOKENS)


async def _reduce_context(
    transcript_content: str,
    tried_compaction: bool,
    session_id: str,
    sdk_cwd: str,
    log_prefix: str,
    attempt: int = 1,
    runtime_model: str | None = None,
) -> ReducedContext:
    """Prepare reduced context for a retry attempt.

    On the first retry, compacts the transcript via LLM summarization.
    On subsequent retries (or if compaction fails), drops the transcript
    entirely so the query is rebuilt from DB messages only.

    When no transcript is available (use_resume=False fallback path), returns
    a decreasing ``target_tokens`` budget so ``compress_context`` applies
    progressively more aggressive reduction (LLM summarize → content truncate
    → middle-out delete → first/last trim).  The budget applies in
    ``_build_query_message`` and is halved on each retry.

    ``transcript_lost`` is True when the transcript was dropped (caller
    should set ``skip_transcript_upload``).
    """
    # Token budget for the DB fallback on this attempt (no-transcript path).
    idx = max(0, attempt - 1)
    retry_target = _RETRY_TARGET_TOKENS[min(idx, len(_RETRY_TARGET_TOKENS) - 1)]

    # First retry: try compacting our transcript builder state.
    # Note: the CLI native --resume file is not updated with the compacted
    # content (it would require emitting CLI-native JSONL format), so the
    # retry runs without --resume.  The compacted builder state is still
    # useful for the eventual upload_transcript call that seeds future turns.
    if transcript_content and not tried_compaction:
        # The compactor LLM is fixed (config.thinking_standard_model); the
        # token target is sized against the RUNTIME model since that's the
        # one whose CLI autocompact threshold we're trying to land below.
        target_model = runtime_model or config.thinking_standard_model
        compacted = await compact_transcript(
            transcript_content,
            model=config.thinking_standard_model,
            log_prefix=log_prefix,
            target_tokens=_compaction_target_tokens(target_model),
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
            return ReducedContext(tb, False, None, False, True)
        logger.warning("%s Compaction failed, dropping transcript", log_prefix)

    # Subsequent retry or compaction failed: drop transcript entirely.
    # Return retry_target so the caller compresses DB messages to that budget.
    logger.warning(
        "%s Dropping transcript, rebuilding from DB messages (target_tokens=%d)",
        log_prefix,
        retry_target,
    )
    return ReducedContext(TranscriptBuilder(), False, None, True, True, retry_target)


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
) -> None:
    """Append a copilot error marker to *session* so it persists across refresh."""
    if session is None:
        return
    prefix = COPILOT_RETRYABLE_ERROR_PREFIX if retryable else COPILOT_ERROR_PREFIX
    session.messages.append(
        ChatMessage(role="assistant", content=f"{prefix} {display_msg}")
    )


def _is_error_marker(msg: ChatMessage) -> bool:
    """True if *msg* is an error marker emitted by ``_append_error_marker``."""
    if msg.role != "assistant" or not msg.content:
        return False
    return msg.content.startswith(COPILOT_ERROR_PREFIX) or msg.content.startswith(
        COPILOT_RETRYABLE_ERROR_PREFIX
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
        leaving duplicate error bubbles.
        """
        tail = list(session.messages[pre_attempt_msg_count:])
        while tail and _is_error_marker(tail[-1]):
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
        _append_error_marker(session, display_msg, retryable=retryable)
        return events


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
                ChatMessage(role="tool", content=content, tool_call_id=resp.toolCallId)
            )
    return safety


@dataclass(frozen=True)
class _FinalFailure:
    """Display message + stream code + retryable flag for a final-failure exit.

    Shared by the in-history error marker (via ``_InterruptedAttempt.finalize``)
    and the client-facing ``StreamError`` SSE yield so the two stay in sync.
    """

    display_msg: str
    code: str
    retryable: bool


def _classify_final_failure(
    interrupted: _InterruptedAttempt,
    attempts_exhausted: bool,
    transient_exhausted: bool,
    stream_err: BaseException | None,
) -> _FinalFailure | None:
    """Pick the display message, stream code, and retryable flag for the exit.

    Returns ``None`` when no failure was recorded (success path) — the caller
    should skip both the history marker and the SSE yield in that case.
    """
    if interrupted.handled_error is not None:
        return _FinalFailure(
            display_msg=interrupted.handled_error.error_msg,
            code=interrupted.handled_error.code,
            retryable=interrupted.handled_error.retryable,
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
) -> AsyncGenerator[Any, None]:
    """Yield SDK messages with heartbeat-based timeouts.

    Uses an explicit async iterator with non-cancelling heartbeats.

    CRITICAL: we must NOT cancel `__anext__()` mid-flight — doing so
    (via `asyncio.timeout` or `wait_for`) corrupts the SDK's internal
    anyio memory stream, causing `StopAsyncIteration` on the next call
    and silently dropping all in-flight tool results.  Instead, wrap
    `__anext__()` in a `Task` and use `asyncio.wait()` with a
    timeout.  On timeout we yield a heartbeat sentinel but keep the Task
    alive so it can deliver the next message.

    Yields `None` on heartbeat timeout (caller should refresh locks and
    emit heartbeat events).  Yields the raw SDK message otherwise.
    On stream end (`StopAsyncIteration`), the generator returns normally.
    Any other exception from the SDK propagates to the caller.
    """
    msg_iter = client.receive_response().__aiter__()
    pending_task: asyncio.Task[Any] | None = None

    async def _next_msg() -> Any:
        """Await the next SDK message, wrapped for use with `asyncio.Task`."""
        return await msg_iter.__anext__()

    try:
        while True:
            if pending_task is None:
                pending_task = asyncio.create_task(_next_msg())

            done, _ = await asyncio.wait({pending_task}, timeout=_HEARTBEAT_INTERVAL)

            if not done:
                yield None  # heartbeat sentinel
                continue

            pending_task = None
            try:
                yield done.pop().result()
            except StopAsyncIteration:
                return
    finally:
        if pending_task is not None and not pending_task.done():
            pending_task.cancel()
            try:
                await pending_task
            except (asyncio.CancelledError, StopAsyncIteration):
                pass


def _normalize_model_name(raw_model: str) -> str:
    """Normalize a model name for the **actual** SDK CLI transport.

    Three transports (see ``ChatConfig.effective_transport``):

    1. **OpenRouter** — the canonical OpenRouter slug is
       ``"<vendor>/<model>"`` (e.g. ``"anthropic/claude-opus-4-6"``,
       ``"moonshotai/kimi-k2-6"``).  Pass the prefixed name through
       unchanged so OpenRouter can route to the correct provider.  Anthropic
       names happen to also resolve when stripped, but non-Anthropic vendors
       (Moonshot, Google, etc.) do not — keeping the prefix is the only form
       that works for every model in the catalog.
    2. **Subscription / Direct Anthropic** — strip the OpenRouter
       ``anthropic/`` prefix and convert dots to hyphens
       (``"claude-opus-4.6"`` → ``"claude-opus-4-6"``).  The CLI subprocess
       (subscription mode) and the Anthropic Messages API both reject the
       prefix and dot-separated versions.  Raises ``ValueError`` when a
       non-Anthropic vendor slug is paired with these transports — silently
       stripping ``moonshotai/`` would send ``kimi-k2-6`` to the Anthropic
       API / CLI and produce an opaque ``model_not_found`` error far from
       the misconfiguration source.

    Gating on the **actual transport** (not just config shape) matters
    because subscription mode and OpenRouter config can coexist —
    ``CHAT_USE_CLAUDE_CODE_SUBSCRIPTION=true`` paired with a populated
    ``CHAT_BASE_URL`` / ``CHAT_API_KEY`` (left over from an earlier
    OpenRouter setup) used to incorrectly pass ``anthropic/claude-opus-4-7``
    to the CLI subprocess, which the CLI rejects.
    """
    if config.effective_transport == "openrouter":
        return raw_model
    model = raw_model
    if "/" in model:
        vendor, model = model.split("/", 1)
        if vendor != "anthropic":
            raise ValueError(
                f"{config.effective_transport!r} transport requires an "
                f"Anthropic model, got vendor={vendor!r} from "
                f"model={raw_model!r}. Set CHAT_THINKING_STANDARD_MODEL/"
                f"CHAT_THINKING_ADVANCED_MODEL to an anthropic/* slug, or "
                f"enable OpenRouter."
            )
    return model.replace(".", "-")


def _resolve_sdk_model() -> str | None:
    """Resolve the SDK-CLI model name from static config (no LD lookup).

    ``config.claude_agent_model`` is an explicit override that wins
    unconditionally.  When the Claude Code subscription is enabled and no
    override is set, returns ``None`` so the CLI picks the model for the
    user's subscription plan.  Otherwise derives from
    ``config.thinking_standard_model``.

    For per-user routing (LaunchDarkly overrides), see
    :func:`_resolve_sdk_model_for_request`.
    """
    if config.claude_agent_model:
        return config.claude_agent_model
    if config.use_claude_code_subscription:
        return None
    return _normalize_model_name(config.thinking_standard_model)


async def _resolve_thinking_model_for_user(
    tier: "CopilotLlmModel",
    user_id: str | None,
) -> str:
    """LD-aware thinking-tier model pick for a specific user.

    Consults ``copilot-model-routing[thinking][{tier}]`` and falls back
    to the ``ChatConfig`` default on missing user / missing flag.
    """
    return await resolve_model("thinking", tier, user_id, config=config)


def _resolve_fallback_model() -> str | None:
    """Resolve the fallback model name via :func:`_normalize_model_name`.

    Returns ``None`` when no fallback is configured (empty string).
    """
    raw = config.claude_agent_fallback_model
    if not raw:
        return None
    return _normalize_model_name(raw)


def _resolve_env_model(sdk_model: str | None, fallback_model: str | None) -> str | None:
    """Pick the model that drives ``build_sdk_env``'s model-aware gates.

    Use the fallback when it's Moonshot so a 529-triggered swap to Kimi
    still suppresses ``CLAUDE_AUTOCOMPACT_PCT_OVERRIDE``.
    """
    if fallback_model and _is_moonshot_model(fallback_model):
        return fallback_model
    return sdk_model


async def _resolve_sdk_model_for_request(
    model: "CopilotLlmModel | None",
    session_id: str,
    user_id: str | None = None,
) -> str | None:
    """Resolve the SDK model string for a turn.

    Priority (highest first):
    1. ``config.claude_agent_model`` — unconditional override, bypasses LD.
    2. LaunchDarkly ``copilot-model-routing[thinking][{tier}]`` if it
       serves a value different from the config default for *user_id*.
       An LD-served override wins over subscription mode so admins can
       route specific users to a specific model without flipping
       subscription on/off.
    3. ``config.use_claude_code_subscription`` on the standard tier —
       returns ``None`` so the CLI picks the subscription default (this
       branch fires when LD has no opinion, i.e. the value equals the
       config default).
    4. ``ChatConfig`` static default for the tier.
    """
    if config.claude_agent_model:
        return config.claude_agent_model

    tier_name: "CopilotLlmModel" = "advanced" if model == "advanced" else "standard"
    # Strip at read time so a stray trailing space in ``CHAT_*_MODEL`` (a
    # common ``.env`` pitfall) doesn't make the ``resolved == tier_default``
    # comparison below spuriously diverge — ``resolve_model`` already strips
    # the LD side, so both halves must end up whitespace-normalised to stay
    # equal when they're semantically equal.  Downstream ``_normalize_model_name``
    # also benefits from the strip.
    tier_default = (
        config.thinking_advanced_model
        if tier_name == "advanced"
        else config.thinking_standard_model
    ).strip()

    resolved = await _resolve_thinking_model_for_user(tier_name, user_id)

    # Subscription mode on standard tier only wins when LD has no opinion
    # (value == config default ⇒ admin hasn't explicitly pointed this
    # user somewhere).  Any LD override — even to the same value with
    # stripped whitespace normalised — is an explicit admin choice that
    # must be honoured.  Without this, a subscription-mode deployment
    # silently ignores the ``copilot-model-routing[thinking][standard]``
    # flag entirely, which defeats the point of cohort-based routing.
    ld_overrides_default = resolved != tier_default
    if (
        not ld_overrides_default
        and tier_name == "standard"
        and config.use_claude_code_subscription
    ):
        logger.info(
            "[SDK] [%s] Subscription default (tier=standard, LD unset)",
            session_id[:12] if session_id else "?",
        )
        return None
    try:
        sdk_model = _normalize_model_name(resolved)
    except ValueError as exc:
        # The per-user LD value didn't pass ``_normalize_model_name``'s
        # vendor check (most commonly: a ``moonshotai/kimi-*`` slug on a
        # direct-Anthropic deployment that has no OpenRouter route).  Fail
        # soft to the TIER-SPECIFIC config default — using the generic
        # ``_resolve_sdk_model()`` here would pin advanced-tier requests to
        # ``thinking_standard_model`` (Sonnet) whenever LD misconfigures
        # the advanced cell, silently downgrading the user's chosen tier.
        try:
            sdk_model = _normalize_model_name(tier_default)
        except ValueError:
            # Config default is *also* invalid for the active routing
            # mode — this is a deployment-level misconfig that the
            # ``model_validator`` should catch at startup.  Re-raise the
            # original LD error so the issue surfaces loudly rather than
            # returning something misleading.
            raise exc
        logger.warning(
            "[SDK] [%s] LD model %r rejected for tier=%s (%s); falling "
            "back to tier default %s",
            session_id[:12] if session_id else "?",
            resolved,
            tier_name,
            exc,
            sdk_model,
        )
        return sdk_model
    logger.info(
        "[SDK] [%s] Resolved model for tier=%s: %s",
        session_id[:12] if session_id else "?",
        tier_name,
        sdk_model,
    )
    return sdk_model


_MAX_TRANSIENT_BACKOFF_SECONDS = 30


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
    message_id: str,
    session_id: str,
) -> AsyncIterator[StreamStatus]:
    """Emit a retry notification, sleep, and reset the SDK adapter.

    Yields a single :class:`StreamStatus` so the caller can forward it to
    the client, then sleeps for *backoff* seconds and resets ``state.adapter``
    and ``state.usage`` so the next attempt starts clean.

    Extracted from both exception handlers in the retry loop to remove
    near-identical code duplication.
    """
    yield StreamStatus(message=f"Connection interrupted, retrying in {backoff}s…")
    await asyncio.sleep(backoff)
    state.adapter = SDKResponseAdapter(
        message_id=message_id,
        session_id=session_id,
        render_reasoning_in_ui=config.render_reasoning_in_ui,
    )
    state.usage.reset()


def _is_fallback_stderr(line: str) -> bool:
    """Return True if a CLI stderr line signals fallback-model activation.

    Matches ``"fallback model"`` case-insensitively.  Uses the specific
    two-word phrase rather than just ``"fallback"`` to avoid false positives
    from unrelated CLI stderr lines (tool retries, cached-result fallbacks).

    Extracted as a module-level pure function so it can be unit-tested
    without wiring up the full ``_on_stderr`` closure.
    """
    return "fallback model" in line.lower()


def _build_system_prompt_value(
    system_prompt: str,
    *,
    cross_user_cache: bool,
) -> str | SystemPromptPreset:
    """Build the ``system_prompt`` argument for :class:`ClaudeAgentOptions`.

    When *cross_user_cache* is enabled, returns a :class:`SystemPromptPreset`
    with ``exclude_dynamic_sections=True`` so every turn — Turn 1 *and*
    resumed turns — shares the same static prefix and hits the cross-user
    prompt cache.  Our custom *system_prompt* is appended after the preset.

    Requires CLI ≥ 2.1.98 (older CLIs crash when ``excludeDynamicSections``
    is combined with ``--resume``).  The SDK bundles CLI 2.1.116 at
    ``claude-agent-sdk >= 0.1.64``, so the pin in ``pyproject.toml`` is
    the single source of truth — no external install needed.

    When *cross_user_cache* is disabled, the raw *system_prompt* string is
    returned.  Note this causes the CLI to REPLACE its built-in prompt via
    ``--system-prompt`` (vs ``--append-system-prompt`` for the preset),
    which loses Claude Code's default prompt and its cache markers entirely.

    An empty *system_prompt* is accepted: the preset dict will have
    ``append: ""`` which the SDK treats as no custom suffix.
    """
    if cross_user_cache:
        logger.debug("Using SystemPromptPreset for cross-user prompt cache")
        return _SystemPromptPreset(
            type="preset",
            preset="claude_code",
            append=system_prompt,
            exclude_dynamic_sections=True,
        )
    logger.debug("Cross-user prompt cache disabled, using raw string")
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
                    "input": block.input,
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


async def _compress_messages(
    messages: list[ChatMessage],
    target_tokens: int | None = None,
) -> tuple[list[ChatMessage], bool]:
    """Compress a list of messages if they exceed the token threshold.

    Delegates to `_run_compression` (`transcript.py`) which centralizes
    the "try LLM, fallback to truncation" pattern with timeouts.  Both
    `_compress_messages` and `compact_transcript` share this helper so
    client acquisition and error handling are consistent.

    ``target_tokens`` sets a hard ceiling for the compressed output so
    callers can enforce a tighter budget on retries.  When ``None``,
    ``compress_context`` uses the model-aware default.

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
        return messages, False

    # Convert ChatMessages to dicts for compress_context
    messages_dict = []
    for msg in messages:
        msg_dict: dict[str, Any] = {"role": msg.role}
        if msg.content:
            msg_dict["content"] = msg.content
        if msg.tool_calls:
            msg_dict["tool_calls"] = msg.tool_calls
        if msg.tool_call_id:
            msg_dict["tool_call_id"] = msg.tool_call_id
        messages_dict.append(msg_dict)

    try:
        result = await _run_compression(
            messages_dict,
            config.thinking_standard_model,
            "[SDK]",
            target_tokens=target_tokens,
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
            "[SDK] _compress_messages failed — dropping history to bare"
            " message to guarantee retry progress: %s",
            exc,
        )
        return [], True

    if result.was_compacted:
        logger.info(
            "[SDK] Context compacted: %d -> %d tokens (%d summarized, %d dropped)",
            result.original_token_count,
            result.token_count,
            result.messages_summarized,
            result.messages_dropped,
        )
        # Convert compressed dicts back to ChatMessages
        return [
            ChatMessage(
                role=m["role"],
                content=m.get("content"),
                tool_calls=m.get("tool_calls"),
                tool_call_id=m.get("tool_call_id"),
            )
            for m in result.messages
        ], True

    return messages, False


def _session_messages_to_transcript(messages: list[ChatMessage]) -> str:
    """Convert session ChatMessages to JSONL transcript for ``--resume``.

    Reconstructs proper ``tool_use`` and ``tool_result`` content blocks from
    :attr:`ChatMessage.tool_calls` and :attr:`ChatMessage.tool_call_id` so the
    Claude CLI receives full structural context when no previous transcript file
    is available (e.g. first turn after a storage failure or compaction drop).

    This gives the model the same fidelity as an on-disk session JSONL file —
    preserving tool call names, IDs, inputs, and *complete* (un-truncated)
    tool results — rather than the lossy plain-text injection produced by
    :func:`_format_conversation_context` (which caps tool results at 500 chars
    and discards structural linkage).

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

    Returns None if there are no messages to format.
    """
    if not messages:
        return None

    # Filter out compaction messages first, then format
    messages = filter_compaction_messages(messages)

    lines: list[str] = []
    for msg in messages:
        if msg.role == "user":
            if msg.content:
                lines.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            if msg.content:
                lines.append(f"You responded: {msg.content}")
            # Omit tool_calls — any text representation gets mimicked
            # by the model. Tool results below provide the context.
        elif msg.role == "tool":
            content = msg.content or ""
            lines.append(f"Tool output: {content[:500]}")

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
) -> tuple[str, bool]:
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

    Returns:
        Tuple of (query_message, was_compacted).
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
    prior = session.messages[: max(0, effective_count - 1)]
    # ``role="reasoning"`` rows are persisted for frontend replay only and are
    # never present in the CLI JSONL (extended_thinking is embedded inside
    # assistant entries).  The watermark — ``transcript_msg_count`` — counts
    # non-reasoning rows (see _jsonl_covered upload), so we must filter reasoning
    # out of ``prior`` too; otherwise the ``prior[transcript_msg_count - 1]``
    # watermark-alignment check trips on a reasoning row (instead of the
    # expected assistant) and the gap injection is skipped, dropping real
    # mid-turn user rows from the next LLM query.
    prior = [m for m in prior if m.role != "reasoning"]

    logger.info(
        "[SDK] [%s] Context path: use_resume=%s, transcript_msg_count=%d,"
        " db_msg_count=%d, target_tokens=%s",
        session_id[:8],
        use_resume,
        transcript_msg_count,
        msg_count,
        target_tokens,
    )

    if use_resume and transcript_msg_count > 0:
        if transcript_msg_count < effective_count - 1:
            # Sanity-check the watermark: the last covered position should be
            # an assistant turn.  A user-role message here means the count is
            # misaligned (e.g. a message was deleted and DB positions shifted).
            # Skip the gap rather than injecting wrong context — the CLI session
            # loaded via --resume still has good history.
            if prior[transcript_msg_count - 1].role != "assistant":
                logger.warning(
                    "[SDK] [%s] Watermark misaligned: prior[%d].role=%r"
                    " (expected 'assistant') — skipping gap to avoid"
                    " injecting wrong context (transcript=%d, db=%d)",
                    session_id[:8],
                    transcript_msg_count - 1,
                    prior[transcript_msg_count - 1].role,
                    transcript_msg_count,
                    msg_count,
                )
                return current_message, False
            gap = prior[transcript_msg_count:]
            compressed, was_compressed = await _compress_messages(gap, target_tokens)
            gap_context = _format_conversation_context(compressed)
            if gap_context:
                logger.info(
                    "[SDK] Transcript stale: covers %d of %d messages, "
                    "gap=%d (compressed=%s), gap_context_bytes=%d",
                    transcript_msg_count,
                    msg_count,
                    len(gap),
                    was_compressed,
                    len(gap_context),
                )
                return (
                    f"{gap_context}\n\nNow, the user says:\n{current_message}",
                    was_compressed,
                )
            logger.warning(
                "[SDK] [%s] Transcript stale: gap produced empty context"
                " (%d msgs, transcript=%d/%d) — sending message without gap prefix",
                session_id[:8],
                len(gap),
                transcript_msg_count,
                msg_count,
            )
        else:
            logger.info(
                "[SDK] [%s] --resume covers full context (%d messages)",
                session_id[:8],
                transcript_msg_count,
            )
        return current_message, False

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
            return current_message, False

        source = prior_messages if prior_messages is not None else prior
        logger.warning(
            "[SDK] [%s] No --resume for %d-message session — compressing context "
            "(source=%s, target_tokens=%s)",
            session_id[:8],
            msg_count,
            "transcript+gap" if prior_messages is not None else "full-db",
            target_tokens,
        )
        compressed, was_compressed = await _compress_messages(source, target_tokens)
        history_context = _format_conversation_context(compressed)
        if history_context:
            logger.info(
                "[SDK] [%s] Fallback context built: compressed=%s, context_bytes=%d",
                session_id[:8],
                was_compressed,
                len(history_context),
            )
            return (
                f"{history_context}\n\nNow, the user says:\n{current_message}",
                was_compressed,
            )
        logger.warning(
            "[SDK] [%s] Fallback context empty after compression"
            " (%d messages) — sending message without history",
            session_id[:8],
            len(source),
        )

    return current_message, False


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

    Non-image files (PDFs, text, etc.) are saved to *sdk_cwd* so the CLI's
    built-in Read tool can access them.

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
                # Non-image files: save to sdk_cwd for Read tool access
                local_path = _save_to_sdk_cwd(sdk_cwd, file_info.name, content)
                file_descriptions.append(
                    f"- {file_info.name} ({mime}, "
                    f"{file_info.size_bytes:,} bytes) saved to {local_path}"
                )
        except Exception:
            logger.warning("Failed to prepare file %s", fid[:12], exc_info=True)

    if not file_descriptions:
        return empty

    noun = "file" if len(file_descriptions) == 1 else "files"
    has_non_images = len(file_descriptions) > len(image_blocks)
    read_hint = " Use the Read tool to view non-image files." if has_non_images else ""
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
        logger.error(
            "%s Sending error to frontend: %s (code=%s)",
            log_prefix,
            response.errorText,
            response.code,
        )
        _append_error_marker(
            ctx.session,
            response.errorText,
            retryable=response.code in _RETRYABLE_STREAM_ERROR_CODES,
        )

    if isinstance(response, StreamReasoningStart):
        acc.reasoning_response = ChatMessage(role="reasoning", content="")
        ctx.session.messages.append(acc.reasoning_response)

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

    elif isinstance(response, StreamToolInputAvailable):
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
        acc.assistant_response.tool_calls = acc.accumulated_tool_calls
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
    sdk_client = ClaudeSDKClient(options=state.options)
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
        if state.was_compacted:
            for ev in ctx.compaction.emit_pre_query(ctx.session):
                yield ev

        # Narrate the silent gap between dispatching the query and the
        # SDK's first real chunk — usually <1s but can stretch to several
        # seconds on cold-starts or large contexts. The frontend prefers
        # this over the generic "Thinking…" copy; fast turns replace it
        # with content immediately.
        yield StreamStatus(message="Contacting the model\u2026")

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

    if not acc.stream_completed and not loop_state.ended_with_stream_error:
        # User cancels raise ``asyncio.CancelledError`` upstream; reaching this
        # branch means the CLI hung up — per-query budget exhausted, max_turns,
        # OOM, or crash — without ever emitting a ResultMessage.
        logger.info(
            "%s Stream ended without ResultMessage — likely CLI-side kill "
            "(budget/turns/crash)",
            ctx.log_prefix,
        )
        closing_responses: list[StreamBaseResponse] = []
        state.adapter._end_text_if_open(closing_responses)
        for r in closing_responses:
            yield r
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

    # Raise so the outer retry loop can rollback session messages.
    # already_yielded=False for transient_api_error: StreamError was NOT
    # sent to the client yet (the outer loop does it when retries are
    # exhausted, avoiding a premature error flash before the retry).
    if loop_state.ended_with_stream_error:
        raise _HandledStreamError(
            "Stream error handled",
            error_msg=loop_state.stream_error_msg,
            code=loop_state.stream_error_code,
            already_yielded=(loop_state.stream_error_code != "transient_api_error"),
        )


async def _seed_transcript(
    session: ChatSession,
    transcript_builder: TranscriptBuilder,
    transcript_covers_prefix: bool,
    transcript_msg_count: int,
    log_prefix: str,
) -> tuple[str, bool, int]:
    """Seed the transcript builder from compressed DB messages.

    Called when ``use_resume=False`` and no prior transcript exists in storage
    so that ``upload_transcript`` saves a compact version for future turns.
    This ensures the next turn can use the full-session compression path with
    the benefit of an already-compressed baseline, and a restored CLI session
    on the next pod gets a usable compact base even for sessions that started
    on old pods.

    Returns ``(transcript_content, transcript_covers_prefix, transcript_msg_count)``
    updated values — unchanged if seeding is not possible.
    """
    if len(session.messages) <= 1:
        return "", transcript_covers_prefix, transcript_msg_count

    _prior = session.messages[:-1]
    _comp, _ = await _compress_messages(_prior, _SEED_TARGET_TOKENS)
    if not _comp:
        return "", transcript_covers_prefix, transcript_msg_count

    _seeded = _session_messages_to_transcript(_comp)
    if not _seeded or not validate_transcript(_seeded):
        return "", transcript_covers_prefix, transcript_msg_count

    transcript_builder.load_previous(_seeded, log_prefix=log_prefix)
    logger.info(
        "%s Seeded transcript from %d compressed DB messages"
        " for next-turn upload (seed_target_tokens=%d)",
        log_prefix,
        len(_comp),
        _SEED_TARGET_TOKENS,
    )
    return _seeded, True, len(_prior)


@dataclass
class _RestoreResult:
    """Return value from ``_restore_cli_session_for_turn``."""

    transcript_content: str = ""
    transcript_covers_prefix: bool = True
    use_resume: bool = False
    resume_file: str | None = None
    transcript_msg_count: int = 0
    baseline_download: "TranscriptDownload | None" = None
    context_messages: "list[ChatMessage] | None" = None


async def _restore_cli_session_for_turn(
    user_id: str | None,
    session_id: str,
    session: "ChatSession",
    sdk_cwd: str,
    transcript_builder: "TranscriptBuilder",
    log_prefix: str,
) -> _RestoreResult:
    """Download, validate and restore a CLI session for ``--resume`` on this turn.

    Performs a single GCS round-trip to fetch the session bytes + message_count
    watermark.  Falls back to DB-message reconstruction when GCS has no session
    (first turn or upload missed).

    Returns a ``_RestoreResult`` with all transcript-related state ready for the
    caller to merge into its local variables.
    """
    result = _RestoreResult()

    if not (config.claude_agent_use_resume and user_id and len(session.messages) > 1):
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

    # Only attempt --resume for SDK-written transcripts.
    # Baseline-written transcripts use TranscriptBuilder format (synthetic IDs,
    # stripped fields) that may not be valid for --resume.
    if cli_restore is not None and cli_restore.mode != "sdk":
        logger.info(
            "%s Transcript written by mode=%r — skipping --resume, "
            "will use transcript content + gap for context",
            log_prefix,
            cli_restore.mode,
        )
        result.baseline_download = cli_restore  # keep for extract_context_messages
        cli_restore = None

    # Validate, strip, and write to disk — delegate to helper to reduce
    # function complexity.  Writing an invalid/corrupt file to disk then
    # falling back to "no --resume" would cause the CLI to fail with
    # "Session ID already in use" because the file exists at the expected
    # session path, so we validate BEFORE any disk write.
    stripped = ""
    if cli_restore is not None and sdk_cwd:
        stripped, ok = process_cli_restore(cli_restore, sdk_cwd, session_id, log_prefix)
        if not ok:
            result.transcript_covers_prefix = False
            cli_restore = None

    if cli_restore is None and sdk_cwd:
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

    # No valid --resume source (mode="baseline" or no GCS file).
    # Build context from transcript content + gap, falling back to full DB.
    # extract_context_messages handles both: non-None baseline_download uses
    # the compacted transcript + gap; None falls back to all prior DB messages.
    context_msgs = extract_context_messages(result.baseline_download, session.messages)
    result.context_messages = context_msgs
    result.transcript_msg_count = (
        result.baseline_download.message_count
        if result.baseline_download is not None
        and result.baseline_download.message_count > 0
        else len(session.messages) - 1
    )
    result.transcript_covers_prefix = True
    logger.info(
        "%s Context built from %s: %d messages (transcript watermark=%d, "
        "will inject as <conversation_history>)",
        log_prefix,
        (
            "baseline transcript + gap"
            if result.baseline_download is not None
            else "DB fallback"
        ),
        len(context_msgs),
        result.transcript_msg_count,
    )

    # Load baseline transcript content into builder so the upload path has accurate state.
    # Also sets result.transcript_content so the _seed_transcript guard in the caller
    # (``not transcript_content``) does not overwrite this builder state with a DB
    # reconstruction — which would duplicate entries since load_previous appends.
    if result.baseline_download is not None:
        try:
            raw_for_builder = result.baseline_download.content
            if isinstance(raw_for_builder, bytes):
                raw_for_builder = raw_for_builder.decode("utf-8")
            stripped = strip_for_upload(raw_for_builder)
            if validate_transcript(stripped):
                transcript_builder.load_previous(stripped, log_prefix=log_prefix)
                result.transcript_content = stripped
        except (UnicodeDecodeError, ValueError, OSError) as _load_err:
            # UnicodeDecodeError: non-UTF-8 content; ValueError: malformed JSONL in
            # strip_for_upload; OSError: encode/decode I/O failure.  Unexpected
            # exceptions propagate so programming errors are not silently masked.
            logger.debug(
                "%s Could not load baseline transcript into builder: %s",
                log_prefix,
                _load_err,
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


async def stream_chat_completion_sdk(  # pyright: ignore[reportGeneralTypeIssues]
    session_id: str,
    message: str | None = None,
    is_user_message: bool = True,
    user_id: str | None = None,
    session: ChatSession | None = None,
    file_ids: list[str] | None = None,
    permissions: "CopilotPermissions | None" = None,
    mode: CopilotMode | None = None,
    model: CopilotLlmModel | None = None,
    request_arrival_at: float = 0.0,
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

    # Strip any user-injected <user_context> tags on every turn.
    # Only the server-injected prefix on the first message is trusted.
    if message:
        message = strip_user_context_tags(message)

    _user_message_appended = maybe_append_user_message(
        session, message, is_user_message
    )
    if _user_message_appended and is_user_message:
        track_user_message(
            user_id=user_id,
            session_id=session_id,
            message_length=len(message or ""),
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
    e2b_sandbox = None
    use_resume = False
    resume_file: str | None = None
    transcript_builder = TranscriptBuilder()
    sdk_cwd = ""
    # True when transcript_builder represents a full prefix of session history.
    # First turn (<=1 prior message) is fully covered even without a download.
    # Set to True when load_previous succeeds; stays False when download fails
    # on a session with prior messages, preventing a partial upload that would
    # mislead _build_query_message into skipping gap reconstruction next turn.
    transcript_covers_prefix = True

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
    skip_transcript_upload = False
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
    # State of the latest failed attempt: rolled-back messages + any
    # _HandledStreamError info to emit on final-failure exit. The retry loop
    # mutates this via capture()/clear(); the post-loop block calls
    # finalize() once.
    interrupted = _InterruptedAttempt()
    # Defaults ensure the finally block can always reference these safely even when
    # an early return (e.g. sdk_cwd error) skips their normal assignment below.
    sdk_model: str | None = None
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
            yield StreamError(
                errorText="Unable to initialize working directory.",
                code="sdk_cwd_error",
            )
            return
        # --- Run independent async I/O operations in parallel ---
        # E2B sandbox setup, system prompt build (Langfuse + DB), Graphiti
        # warm-context, and CLI session restore are all independent network
        # calls. Running them concurrently saves ~500-1000ms vs sequential.

        async def _setup_e2b():
            """Set up E2B sandbox if configured, return sandbox or None."""
            if not (e2b_api_key := config.active_e2b_api_key):
                if config.use_e2b_sandbox:
                    logger.warning(
                        "[E2B] [%s] E2B sandbox enabled but no API key configured "
                        "(CHAT_E2B_API_KEY / E2B_API_KEY) — falling back to bubblewrap",
                        session_id[:12],
                    )
                return None
            try:
                sandbox = await get_or_create_sandbox(
                    session_id,
                    api_key=e2b_api_key,
                    template=config.e2b_sandbox_template,
                    timeout=config.e2b_sandbox_timeout,
                    on_timeout=config.e2b_sandbox_on_timeout,
                )
            except Exception as e2b_err:
                logger.error(
                    "[E2B] [%s] Setup failed: %s",
                    session_id[:12],
                    e2b_err,
                    exc_info=True,
                )
                return None

            return sandbox

        (
            e2b_sandbox,
            (base_system_prompt, understanding),
            (graphiti_enabled, warm_ctx),
            _restore,
        ) = await asyncio.gather(
            _setup_e2b(),
            _build_system_prompt(user_id if not has_history else None),
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
            ),
        )

        use_e2b = e2b_sandbox is not None
        # Append appropriate supplement (Claude gets tool schemas automatically)

        graphiti_supplement = get_graphiti_supplement() if graphiti_enabled else ""
        # Append the builder-session block (graph id+name + full building
        # guide) AFTER the shared supplements so the system prompt is
        # byte-identical across turns of the same builder session — Claude's
        # prompt cache keeps the ~20KB guide warm for the whole session.
        # Empty string for non-builder sessions preserves cross-user caching.
        builder_session_suffix = await build_builder_system_prompt_suffix(session)
        system_prompt = (
            base_system_prompt
            + get_sdk_supplement(use_e2b=use_e2b)
            + graphiti_supplement
            + builder_session_suffix
        )

        transcript_content = _restore.transcript_content
        transcript_covers_prefix = _restore.transcript_covers_prefix
        use_resume = _restore.use_resume
        resume_file = _restore.resume_file
        transcript_msg_count = _restore.transcript_msg_count
        restore_context_messages = _restore.context_messages

        yield StreamStart(messageId=message_id, sessionId=session_id)

        set_execution_context(
            user_id,
            session,
            sandbox=e2b_sandbox,
            sdk_cwd=sdk_cwd,
            permissions=permissions,
        )

        if not config.api_key and not config.use_claude_code_subscription:
            raise RuntimeError(
                "No API key configured. Set OPEN_ROUTER_API_KEY, "
                "CHAT_API_KEY, or ANTHROPIC_API_KEY for API access, "
                "or CHAT_USE_CLAUDE_CODE_SUBSCRIPTION=true to use "
                "Claude Code CLI subscription (requires `claude login`)."
            )

        mcp_server = create_copilot_mcp_server(use_e2b=use_e2b)

        # Resolve model (request tier → LD per-user override → config default).
        # Done BEFORE build_sdk_env so model-aware env vars (e.g. the
        # Moonshot autocompact gate) can branch on the resolved slug.
        sdk_model = await _resolve_sdk_model_for_request(model, session_id, user_id)
        fallback_model = _resolve_fallback_model()

        # sdk_cwd routes the CLI's temp dir into the per-session workspace
        # so sub-agent output files land inside sdk_cwd (see build_sdk_env).
        sdk_env = build_sdk_env(
            session_id=session_id,
            user_id=user_id,
            sdk_cwd=sdk_cwd,
            model=_resolve_env_model(sdk_model, fallback_model),
        )

        # Track SDK-internal compaction (PreCompact hook → start, next msg → end)
        compaction = CompactionTracker()

        security_hooks = create_security_hooks(
            user_id,
            sdk_cwd=sdk_cwd,
            max_subtasks=config.claude_agent_max_subtasks,
            on_compact=compaction.on_compact,
        )

        disabled_tool_groups: list[ToolGroup] = []
        if not graphiti_enabled:
            disabled_tool_groups.append("graphiti")

        if permissions is not None:
            allowed, disallowed = apply_tool_permissions(
                permissions, use_e2b=use_e2b, disabled_groups=disabled_tool_groups
            )
        else:
            allowed = get_copilot_tool_names(
                use_e2b=use_e2b, disabled_groups=disabled_tool_groups
            )
            disallowed = get_sdk_disallowed_tools(use_e2b=use_e2b)

        def _on_stderr(line: str) -> None:
            """Log a stderr line emitted by the Claude CLI subprocess."""
            nonlocal fallback_model_activated_per_attempt
            sid = session_id[:12] if session_id else "?"
            logger.info("[SDK] [%s] CLI stderr: %s", sid, line.rstrip())
            # Detect SDK fallback-model activation via the module-level pure
            # helper so the detection logic can be unit-tested independently.
            # Sets the per-attempt flag which is preserved across transient
            # retries so the user notification is never lost.
            if not fallback_model_activated_per_attempt and _is_fallback_stderr(line):
                fallback_model_activated_per_attempt = True
                logger.warning(
                    "[SDK] [%s] Fallback model activated — primary model "
                    "overloaded, switching to fallback",
                    sid,
                )

        # Use SystemPromptPreset with exclude_dynamic_sections=True on
        # every turn — including resumed ones — so all turns share the
        # same static prefix and hit the cross-user prompt cache.
        #
        # Requires CLI ≥ 2.1.98 (older CLIs crash when excludeDynamicSections
        # is combined with --resume).  claude-agent-sdk >= 0.1.64 bundles
        # CLI 2.1.116, so the pin in pyproject.toml is sufficient — no
        # external install or env-var override needed.
        system_prompt_value = _build_system_prompt_value(
            system_prompt,
            cross_user_cache=config.claude_agent_cross_user_prompt_cache,
        )

        sdk_options = ClaudeAgentOptions(
            system_prompt=system_prompt_value,
            mcp_servers={"copilot": mcp_server},
            allowed_tools=allowed,
            disallowed_tools=disallowed,
            hooks=security_hooks,
            cwd=sdk_cwd,
            max_buffer_size=config.claude_agent_max_buffer_size,
            stderr=_on_stderr,
            # --- P0 guardrails ---
            # fallback_model: SDK auto-retries with this cheaper model on
            # 529 (overloaded) errors, avoiding user-visible failures.
            fallback_model=fallback_model,
            # max_turns: hard cap on agentic tool-use loops per query to
            # prevent runaway execution from burning budget.
            max_turns=config.agent_max_turns,
            # max_budget_usd: per-query spend ceiling enforced by the CLI.
            # Sized to the smaller of the configured per-query default and
            # the user's *actual* remaining daily/weekly USD cap so the
            # CLI's "wrap up gracefully" reminder fires when they're close
            # to the real limit, not the static $10 default.
            max_budget_usd=await _resolve_dynamic_max_budget_usd(user_id),
            # thinking: specify extended thinking mode. Thinking tokens are
            # billed at output rate ($75/M for Opus) and account for ~54%
            # of total cost.  The CLI silently ignores this field for
            # models without native extended thinking, so it is safe to
            # pass unconditionally.
            # NOTE: Claude 4.7+ does not support capped thinking token
            # budget: use `effort` instead to steer thinking effort.
            thinking={"type": "adaptive"},
            # effort: applies to models with extended thinking (Sonnet,
            # Opus, Mythos) and Kimi K2.6 via OpenRouter's ``reasoning``
            # extension (#12871).
            effort=(
                "medium"
                if mode == "fast"
                else (config.claude_agent_thinking_effort or "high")
            ),
        )
        # max_thinking_tokens: legacy cost cap retained for non-4.7 models
        # and Kimi K2.6.  Setting to 0 acts as the kill switch (same as
        # baseline): omit the kwarg so the CLI falls back to its default.
        if config.claude_agent_max_thinking_tokens > 0:
            sdk_options.max_thinking_tokens = config.claude_agent_max_thinking_tokens
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
        # Optional explicit Claude Code CLI binary path (decouples the
        # bundled SDK version from the CLI version we run — needed because
        # the CLI bundled in 0.1.46+ is broken against OpenRouter).  Falls
        # back to the bundled binary when unset.
        if config.claude_agent_cli_path:
            sdk_options.cli_path = config.claude_agent_cli_path

        adapter = SDKResponseAdapter(
            message_id=message_id,
            session_id=session_id,
            render_reasoning_in_ui=config.render_reasoning_in_ui,
        )

        # Propagate user_id/session_id as OTEL context attributes so the
        # langsmith tracing integration attaches them to every span.  This
        # is what Langfuse (or any OTEL backend) maps to its native
        # user/session fields.
        _user_tier = await get_user_tier(user_id) if user_id else None
        _otel_metadata: dict[str, str] = {
            "resume": str(use_resume),
            "conversation_turn": str(turn),
        }
        if _user_tier:
            _otel_metadata["subscription_tier"] = _user_tier.value

        # Open a Langfuse parent span so the trace_id is observable from
        # the finally block — ``propagate_attributes`` only annotates an
        # existing span, it does not create one.
        try:
            _lf_span = get_client().start_as_current_span(name="copilot-sdk-turn")
            _lf_span.__enter__()
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
        # them).
        _pre_drain_msg_count = len(session.messages)

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
        # include the pending texts, avoiding a duplicate INSERT that
        # would occur if we used insert_pending_before_last +
        # persist_session_safe (routes.py has already saved the user
        # message at sequence N before the executor runs, so an
        # incremental upsert would write a second copy at N+1).
        pending_messages = await drain_pending_safe(session_id, log_prefix)
        if pending_messages:
            logger.info(
                "%s Draining %d pending message(s) at turn start",
                log_prefix,
                len(pending_messages),
            )
            # NOTE: combining and per-row persistence both happen *after*
            # ``inject_user_context`` below — see the comment near that
            # call.  At this point ``current_message`` is still the
            # original turn-starting send (no pending text yet).

        if not current_message.strip():
            yield StreamError(
                errorText="Message cannot be empty.",
                code="empty_prompt",
            )
            return

        # Strip any user-injected <user_context> tags from current_message.
        # On --resume, current_message may come from session history which was
        # already sanitized on the original turn; strip again as defence-in-depth.
        current_message = strip_user_context_tags(current_message)

        # On the first turn inject user context into the message before building
        # the query so that _build_query_message sees the full prefixed content.
        # The system prompt is now static (same for all users) so the LLM can
        # cache it across sessions.
        #
        # On resume (has_history=True) we intentionally skip re-injection: the
        # transcript already contains the <user_context> and <memory_context>
        # prefixes from the original turn (persisted to the DB via
        # inject_user_context), so the SDK replay carries context continuity
        # without us prepending them again.
        if not has_history:
            # Build env_ctx for the working directory and pass it into
            # inject_user_context so it is prepended AFTER
            # sanitize_user_supplied_context runs — preventing the trusted
            # <env_context> block from being stripped by the sanitizer.
            env_ctx_content = ""
            if not use_e2b and sdk_cwd:
                env_ctx_content = f"working_dir: {sdk_cwd}"
            # Pass warm_ctx and env_ctx to inject_user_context so they are
            # prepended AFTER sanitize_user_supplied_context runs — preventing
            # trusted server-injected blocks from being stripped by the sanitizer.
            # inject_user_context persists the fully prefixed message to DB.
            prefixed_message = await inject_user_context(
                understanding,
                current_message,
                session_id,
                session.messages,
                warm_ctx=warm_ctx,
                env_ctx=env_ctx_content,
                user_id=user_id,
            )
            if prefixed_message is not None:
                current_message = prefixed_message

        # Now that ``inject_user_context`` has wrapped + persisted the
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
                current_message = combine_pending_with_current(
                    pending_messages,
                    current_message,
                    request_arrival_at=request_arrival_at,
                )

        query_message, was_compacted = await _build_query_message(
            current_message,
            session,
            use_resume,
            transcript_msg_count,
            session_id,
            session_msg_ceiling=_pre_drain_msg_count,
            prior_messages=restore_context_messages,
        )
        # If files are attached, prepare them: images become vision
        # content blocks in the user message, other files go to sdk_cwd.
        attachments = await _prepare_file_attachments(
            file_ids or [], user_id or "", session_id, sdk_cwd
        )
        if attachments.hint:
            query_message = f"{query_message}\n\n{attachments.hint}"

        # warm_ctx is injected via inject_user_context above (warm_ctx= kwarg).
        # No separate injection needed here.

        # Inject per-turn builder context when the session is bound to a
        # graph via ``metadata.builder_graph_id``.  Runs on EVERY user turn
        # (including resumes) so the LLM always sees the live graph snapshot
        # — if the user edits the graph between turns, the next turn carries
        # the updated nodes/links.  The block also carries the full
        # agent-building guide, replacing the per-turn
        # ``get_agent_building_guide`` round-trip.  Not persisted to the
        # transcript: the snapshot is stale-by-definition after the turn ends.
        query_message = await _maybe_prepend_builder_context(
            session, user_id, is_user_message, query_message
        )

        # When running without --resume and no prior transcript in storage,
        # seed the transcript builder from compressed DB messages so that
        # upload_transcript saves a compact version for future turns.
        if not use_resume and not transcript_content and not skip_transcript_upload:
            (
                transcript_content,
                transcript_covers_prefix,
                transcript_msg_count,
            ) = await _seed_transcript(
                session,
                transcript_builder,
                transcript_covers_prefix,
                transcript_msg_count,
                log_prefix,
            )

        tried_compaction = False

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
        )

        # ---------------------------------------------------------------
        # Retry loop: original → compacted → no transcript
        # ---------------------------------------------------------------
        attempts_exhausted = False
        transient_exhausted = False
        stream_err: Exception | None = None

        transient_retries = 0
        max_transient_retries = config.claude_agent_max_transient_retries
        # Preserved across transient retries so the fallback-model notification
        # is not lost when a retry resets local per-attempt variables.  Reset
        # only on context-level attempt changes (same guard as transient_retries).
        fallback_model_activated_per_attempt = False
        fallback_notified_per_attempt = False

        state = _RetryState(
            options=sdk_options,
            query_message=query_message,
            was_compacted=was_compacted,
            use_resume=use_resume,
            resume_file=resume_file,
            transcript_msg_count=transcript_msg_count,
            adapter=adapter,
            transcript_builder=transcript_builder,
            usage=_TokenUsage(),
        )

        attempt = 0
        _last_reset_attempt = -1
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
                fallback_model_activated_per_attempt = False
                fallback_notified_per_attempt = False
                _last_reset_attempt = attempt
            # Clear any stale stash signal from the previous attempt so
            # wait_for_stash() doesn't fire prematurely on a leftover event.
            reset_stash_event()
            # Reset tool-level circuit breaker so failures from a previous
            # (rolled-back) attempt don't carry over to the fresh attempt.
            reset_tool_failure_counters()
            if attempt > 0:
                logger.info(
                    "%s Retrying with reduced context (%d/%d)",
                    log_prefix,
                    attempt + 1,
                    _MAX_STREAM_ATTEMPTS,
                )
                yield StreamStatus(message="Optimizing conversation context\u2026")

                ctx = await _reduce_context(
                    transcript_content,
                    tried_compaction,
                    session_id,
                    sdk_cwd,
                    log_prefix,
                    attempt=attempt,
                    runtime_model=sdk_model,
                )
                state.transcript_builder = ctx.builder
                state.use_resume = ctx.use_resume
                state.resume_file = ctx.resume_file
                tried_compaction = ctx.tried_compaction
                state.transcript_msg_count = 0
                state.target_tokens = ctx.target_tokens
                if ctx.transcript_lost:
                    skip_transcript_upload = True

                # Rebuild SDK options and query for the reduced context
                sdk_options_retry = copy(sdk_options)
                if ctx.use_resume and ctx.resume_file:
                    sdk_options_retry.resume = ctx.resume_file
                    sdk_options_retry.session_id = None
                else:
                    # No --resume on this retry. Whether we entered with
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
                # Recompute system_prompt for retry — the preset is safe on
                # every turn (requires CLI ≥ 2.1.98, bundled in
                # claude-agent-sdk >= 0.1.64).
                sdk_options_retry.system_prompt = _build_system_prompt_value(
                    system_prompt,
                    cross_user_cache=config.claude_agent_cross_user_prompt_cache,
                )
                state.options = sdk_options_retry
                # Retry intentionally omits prior_messages (transcript+gap context) and
                # falls back to full session.messages[:-1] from DB — the authoritative
                # source.  transcript+gap is an optimisation for the first attempt only;
                # on retry the extra overhead of full-DB context is acceptable.
                state.query_message, state.was_compacted = await _build_query_message(
                    current_message,
                    session,
                    state.use_resume,
                    state.transcript_msg_count,
                    session_id,
                    session_msg_ceiling=_pre_drain_msg_count,
                    target_tokens=state.target_tokens,
                )
                if attachments.hint:
                    state.query_message = f"{state.query_message}\n\n{attachments.hint}"
                # warm_ctx is already baked into current_message via
                # inject_user_context — no separate injection needed.
                # Re-inject per-turn builder context so retries carry the
                # same live graph snapshot + guide as the initial attempt.
                state.query_message = await _maybe_prepend_builder_context(
                    session, user_id, is_user_message, state.query_message
                )
                state.adapter = SDKResponseAdapter(
                    message_id=message_id,
                    session_id=session_id,
                    render_reasoning_in_ui=config.render_reasoning_in_ui,
                )
                # Carry the per-turn re-prompt cap forward so a transient
                # retry mid-turn does not unlock another re-prompt round.
                state.adapter.thinking_only_reprompted = state.thinking_only_reprompted
                # Reset token accumulators so a failed attempt's partial
                # usage is not double-counted in the successful attempt.
                state.usage.reset()

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
                    # Emit a one-time StreamStatus when the SDK switches
                    # to the fallback model (detected via stderr).  The flag
                    # is preserved across transient retries (reset only on
                    # context-level attempt change) so the notification is
                    # not lost if the activation occurs during a failed sub-
                    # attempt that later retries successfully.
                    if (
                        fallback_model_activated_per_attempt
                        and not fallback_notified_per_attempt
                    ):
                        fallback_notified_per_attempt = True
                        yield StreamStatus(
                            message="Primary model overloaded — "
                            "using fallback model for this request"
                        )
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
                            backoff, state, message_id, session_id
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
                # transcript_builder still contains entries from the aborted
                # attempt that no longer match session.messages.  Skip upload
                # so a future --resume doesn't replay rolled-back content.
                skip_transcript_upload = True
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
                    skip_transcript_upload = True
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
                            backoff, state, message_id, session_id
                        ):
                            yield evt
                        continue  # retry same context-level attempt
                    # Retries exhausted — persist retryable marker so the
                    # frontend shows "Try again" after refresh.
                    # Mirrors the _HandledStreamError exhausted-retry path
                    # at line ~2310.
                    transient_exhausted = True
                    skip_transcript_upload = True
                    ended_with_stream_error = True
                    break

                if not is_context_error:
                    # Non-context, non-transient errors (auth, fatal)
                    # should not trigger compaction — surface immediately.
                    skip_transcript_upload = True
                    ended_with_stream_error = True
                    break
                attempt += 1  # advance to next context-level attempt
                continue
        else:
            # while condition became False — all attempts exhausted without
            # break.  skip_transcript_upload is already set by _reduce_context
            # when the transcript was dropped (transcript_lost=True).
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
                interrupted, attempts_exhausted, transient_exhausted, stream_err
            )
            if failure is not None:
                cleanup_events: list[StreamBaseResponse] = []
                if state is not None:
                    state.adapter._end_text_if_open(cleanup_events)
                cleanup_events.extend(
                    interrupted.finalize(
                        session,
                        state,
                        failure.display_msg,
                        retryable=failure.retryable,
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
            interrupted.finalize(session, state, display_msg, retryable=is_transient)
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
            yield StreamError(errorText=display_msg, code=code)

        raise
    finally:
        # Pending messages are drained atomically at the start of each
        # turn (see drain_pending_messages call above), so there's
        # nothing to clean up here — any message pushed after that
        # point belongs to the next turn.

        # --- Close OTEL context (with cost attributes) ---
        # Captured before __exit__ so the reconcile task (launched below,
        # after the span closes) can attach a backfill event to this turn's
        # Langfuse trace.  Without it, Langfuse shows the rate-card estimate
        # only — for non-Anthropic OpenRouter routes that's wildly wrong.
        langfuse_trace_id: str | None = None
        if _otel_ctx is not None:
            try:
                span = otel_trace.get_current_span()
                if span and span.is_recording():
                    span.set_attribute("gen_ai.usage.prompt_tokens", turn_prompt_tokens)
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
                logger.warning("Langfuse parent span teardown failed", exc_info=True)

        # --- Persist token usage to session + rate-limit counters ---
        # Both must live in finally so they stay consistent even when an
        # exception interrupts the try block after StreamUsage was yielded.
        effective_model = sdk_model or config.thinking_standard_model
        # ``state`` is populated lazily inside the retry loop; when the
        # turn exits before the first attempt runs (e.g. very early
        # validation error) it's still None, so ``generation_ids`` is
        # empty by definition.
        collected_gen_ids: list[str] = (
            list(state.generation_ids) if state is not None else []
        )
        _use_openrouter_reconcile = bool(
            config.openrouter_active
            and config.sdk_reconcile_openrouter_cost
            and collected_gen_ids
        )

        # CLI project dir — used by the reconcile task to sweep for
        # compaction subagents' gen-IDs.  ``sdk_cwd`` is the per-session
        # CLI working directory; the CLI encodes it into the project-dir
        # name the same way ``encode_cwd_for_cli`` does, and writes
        # the main transcript + any ``subagents/`` alongside it under
        # ``~/.claude/projects/<encoded>/``.  Empty when sdk_cwd isn't
        # set (shouldn't happen in practice for SDK turns).
        cli_project_dir: str | None = None
        if sdk_cwd:
            cli_project_dir = os.path.join(
                os.path.expanduser("~/.claude/projects"),
                encode_cwd_for_cli(sdk_cwd),
            )

        if _use_openrouter_reconcile:
            # Defer the single cost-and-rate-limit write to a background
            # task that queries OpenRouter's authoritative
            # ``/generation?id=`` for every round in this turn.  Covers
            # all vendors:
            #
            # * Non-Anthropic (Kimi et al): the CLI's ``total_cost_usd``
            #   is computed from a static Anthropic rate table that
            #   doesn't know the model — silently over-bills by ~5x.
            #   The reconcile replaces it with OpenRouter's real bill.
            # * Anthropic via OpenRouter: the CLI's number matches
            #   Anthropic's own rates penny-for-penny in the common
            #   case, but the reconcile catches any rate change the
            #   CLI binary hasn't picked up and any OpenRouter-side
            #   divergence (cache-discount accounting, promo pricing).
            #
            # The task calls ``persist_and_record_usage`` exactly once
            # per turn — same method as the sync path, so append-only
            # cost-log + rate-limit counter update together.  The sync
            # path below is skipped entirely when the reconcile fires,
            # so no double-counting.  Kill-switch:
            # ``CHAT_SDK_RECONCILE_OPENROUTER_COST=false``.
            #
            # Brief window (~0.5-2s) where the rate-limit counter is
            # unaware of this turn — back-to-back turns in that window
            # see a stale counter.
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
                    cli_project_dir=cli_project_dir,
                    cli_session_id=session_id,
                    turn_start_ts=turn_start_ts,
                    fallback_cost_usd=turn_cost_usd,
                    api_key=config.api_key,
                    log_prefix=log_prefix,
                    langfuse_trace_id=langfuse_trace_id,
                )
            )
            _background_tasks.add(cost_reconcile_task)
            cost_reconcile_task.add_done_callback(_background_tasks.discard)
        else:
            # Reconcile disabled, OpenRouter inactive, or subscription
            # path (no gen-IDs).  Record the SDK CLI's
            # ``total_cost_usd`` synchronously: accurate for Anthropic
            # (same rate card as billing); for non-Anthropic it's the
            # rate-card estimate that ``_override_cost_for_non_anthropic``
            # caps (still 1.5-2x off vs real OpenRouter bill, but much
            # closer than the ~5x Sonnet-rate fallback).
            await persist_and_record_usage(
                session=session,
                user_id=user_id,
                prompt_tokens=turn_prompt_tokens,
                completion_tokens=turn_completion_tokens,
                cache_read_tokens=turn_cache_read_tokens,
                cache_creation_tokens=turn_cache_creation_tokens,
                log_prefix=log_prefix,
                cost_usd=turn_cost_usd,
                model=effective_model,
                # ``provider`` labels the cost-analytics row; the cost
                # value still comes from the SDK-reported number.
                # Tracks the actual upstream so the row matches reality:
                # OpenRouter when ``openrouter_active``, Anthropic
                # otherwise.
                provider=("open_router" if config.openrouter_active else "anthropic"),
            )

        # --- Persist session messages ---
        # This MUST run in finally to persist messages even when the generator
        # is stopped early (e.g., user clicks stop, processor breaks stream loop).
        # Without this, messages disappear after refresh because they were never
        # saved to the database.
        if session is not None:
            try:
                await asyncio.shield(upsert_chat_session(session))
                logger.info(
                    "%s Session persisted in finally with %d messages",
                    log_prefix,
                    len(session.messages),
                )
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
            task = asyncio.create_task(pause_sandbox_direct(e2b_sandbox, session_id))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)

        # --- Graphiti: ingest conversation turn for temporal memory ---
        if graphiti_enabled and user_id and message and is_user_message:
            from ..graphiti.ingest import enqueue_conversation_turn

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
                enqueue_conversation_turn(
                    user_id, session_id, message, assistant_msg=_last_assistant
                )
            )
            _background_tasks.add(_ingest_task)
            _ingest_task.add_done_callback(_background_tasks.discard)

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
        # Intentionally NOT gated on skip_transcript_upload: that flag is set
        # when our custom JSONL transcript is dropped (transcript_lost=True on
        # reduced-context retries) but the CLI's native session file is written
        # independently.  Blocking CLI upload on transcript_lost would prevent
        # T1 prompt-too-long retries from uploading their valid session file,
        # breaking --resume on the next pod.  The ended_with_stream_error gate
        # above already covers actual turn failures.
        if (
            config.claude_agent_use_resume
            and user_id
            and sdk_cwd
            and session is not None
            and state is not None
            and not ended_with_stream_error
        ):
            logger.info(
                "%s Attempting CLI session upload"
                " (use_resume=%s, has_history=%s, skip_transcript=%s)",
                log_prefix,
                state.use_resume,
                has_history,
                skip_transcript_upload,
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
                    # Watermark = number of DB messages this transcript covers.
                    # len(session.messages) is accurate: the CLI session file
                    # was just written after the turn completed, so it covers
                    # all messages through this turn.  Any gap from a prior
                    # missed upload was already detected by detect_gap and
                    # injected as context, so the model has the full history.
                    #
                    # Previously this used _final_tmsg_count + 2, which
                    # under-counted for tool-use turns (delta = 2 + 2*N_tool_calls),
                    # causing persistent spurious gap-fills on every subsequent turn.
                    # That concern was addressed by the inflated-watermark fix
                    # (using the GCS watermark as the anchor for gap detection),
                    # which makes len(session.messages) safe to use here.
                    #
                    # Mid-turn follow-up user rows (persisted via the
                    # StreamToolOutputAvailable handler) are NOT part of the CLI
                    # JSONL — the CLI only knows them as embedded text inside a
                    # tool_result, and even that embedding can be stripped by
                    # the CLI's internal tool_result size cap.  Deduct them
                    # from the watermark so detect_gap on the next turn
                    # treats them as gap-fill entries and the model sees them
                    # as real user messages instead of missing text.
                    _midturn_offset = (
                        state.midturn_user_rows if state is not None else 0
                    )
                    # ``role="reasoning"`` rows are persisted to session.messages
                    # for frontend replay but never appear in the CLI JSONL
                    # (extended_thinking lives embedded in assistant entries, not
                    # as standalone rows).  Exclude them from the watermark so
                    # ``detect_gap`` on the next turn doesn't skip real
                    # user/assistant rows.  See sentry comment 3106186683.
                    _non_reasoning_count = sum(
                        1 for m in session.messages if m.role != "reasoning"
                    )
                    _jsonl_covered = _non_reasoning_count - _midturn_offset
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
            except Exception as cli_upload_err:
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

    # -------------------------------------------------------------------------
    # Auto-continue: drain any messages the user queued AFTER the turn-start
    # drain window and process them as a new turn automatically.
    #
    # This code only executes on NORMAL turn completion.  GeneratorExit and
    # BaseException both re-raise inside their except blocks, so the generator
    # closes before reaching here — messages queued during a cancelled turn are
    # preserved in Redis for the next manual turn.
    # -------------------------------------------------------------------------
    if not ended_with_stream_error:
        _auto_pending_messages = await drain_pending_safe(session_id, log_prefix)
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
                    file_ids=None,
                    permissions=permissions,
                    mode=mode,
                    model=model,
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

    from ..graphiti.context import fetch_warm_context

    ctx = await fetch_warm_context(user_id, message or "") or ""
    return True, ctx
