"""Baseline LLM fallback — OpenAI-compatible streaming with tool calling.

Used when ``CHAT_USE_CLAUDE_AGENT_SDK=false``, e.g. as a fallback when the
Claude Agent SDK / Anthropic API is unavailable.  Routes through any
OpenAI-compatible provider (OpenRouter by default) and reuses the same
shared tool registry as the SDK path.
"""

import asyncio
import base64
import logging
import math
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import AsyncGenerator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import orjson
from langfuse import propagate_attributes
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from opentelemetry import trace as otel_trace

from backend.copilot.anthropic_rate_card import (
    compute_anthropic_cost_usd,
    get_max_output_tokens,
)
from backend.copilot.baseline.reasoning import (
    BaselineReasoningEmitter,
    anthropic_thinking_extra_body,
    reasoning_extra_body,
)
from backend.copilot.builder_context import (
    build_builder_context_turn_prefix,
    build_builder_system_prompt_suffix,
)
from backend.copilot.config import CopilotLlmModel, CopilotMode
from backend.copilot.context import get_workspace_manager, set_execution_context
from backend.copilot.graphiti.config import is_enabled_for_user
from backend.copilot.model import (
    ChatMessage,
    ChatSession,
    get_chat_session,
    maybe_append_user_message,
    upsert_chat_session,
)
from backend.copilot.model_normalize import normalize_model_for_transport
from backend.copilot.model_router import resolve_model
from backend.copilot.moonshot import is_moonshot_model
from backend.copilot.pending_message_helpers import (
    combine_pending_with_current,
    drain_pending_safe,
    persist_pending_as_user_rows,
    persist_session_safe,
)
from backend.copilot.pending_messages import (
    drain_pending_messages,
    format_pending_as_user_message,
)
from backend.copilot.prompting import SHARED_TOOL_NOTES, get_graphiti_supplement
from backend.copilot.rate_limit import build_budget_ctx
from backend.copilot.response_model import (
    StreamBaseResponse,
    StreamError,
    StreamFinish,
    StreamFinishStep,
    StreamStart,
    StreamStartStep,
    StreamTextDelta,
    StreamTextEnd,
    StreamTextStart,
    StreamToolInputAvailable,
    StreamToolInputStart,
    StreamToolOutputAvailable,
    StreamUsage,
)
from backend.copilot.service import (
    _build_system_prompt,
    _get_main_client,
    _update_title_async,
    config,
    inject_user_context,
    strip_user_context_tags,
)
from backend.copilot.session_cleanup import prune_orphan_tool_calls
from backend.copilot.thinking_stripper import ThinkingStripper as _ThinkingStripper
from backend.copilot.token_tracking import (
    _extract_cache_creation_tokens,
    persist_and_record_usage,
)
from backend.copilot.tools import ToolGroup, execute_tool, get_available_tools
from backend.copilot.tracking import track_user_message
from backend.copilot.transcript import (
    STOP_REASON_END_TURN,
    STOP_REASON_TOOL_USE,
    TranscriptDownload,
    detect_gap,
    download_transcript,
    extract_context_messages,
    fill_hole_between_transcript_and_gap,
    next_uncovered_sequence,
    strip_for_upload,
    upload_transcript,
    validate_transcript,
)
from backend.copilot.transcript_builder import TranscriptBuilder
from backend.util import json as util_json
from backend.util.exceptions import NotFoundError
from backend.util.prompt import (
    compress_context,
    estimate_token_count,
    estimate_token_count_str,
)
from backend.util.tool_call_loop import (
    LLMLoopResponse,
    LLMToolCall,
    ToolCallResult,
    tool_call_loop,
)

if TYPE_CHECKING:
    from backend.copilot.permissions import CopilotPermissions

logger = logging.getLogger(__name__)

# Set to hold background tasks to prevent garbage collection
_background_tasks: set[asyncio.Task[Any]] = set()

# Hint appended on the last tool round so the model wraps up with a summary
# instead of issuing another tool call that gets cut off cold. The shared
# ``tool_call_loop`` drops ``tools`` on the last iteration (see util/tool_call_loop.py),
# so the model is forced to produce text and always finishes naturally.
_LAST_ITERATION_HINT = (
    "You have reached the tool-call budget for this turn. Do not call any "
    "more tools — produce a final text response summarizing what you did, "
    "what remains, and how the user can continue the work in the next turn."
)

# Fallback surfaced when the tool-round budget is exhausted *and* the forced-
# text last round left the user with zero visible response.
_BUDGET_EXHAUSTED_FALLBACK_TEXT = (
    "Reached the tool-call budget for this turn. "
    "Send a follow-up message to continue from here."
)


def _budget_exhausted_notice_text(terminal_round_text: str) -> str | None:
    """Return the fallback notice when a budget-exhausted turn produced no
    visible text, or ``None`` when the model already summarised itself.

    ``terminal_round_text`` is the text added by the *final* round only —
    earlier-round chatter shouldn't mask a silent final round.
    """
    if terminal_round_text.strip():
        return None
    return _BUDGET_EXHAUSTED_FALLBACK_TEXT


def _build_budget_exhausted_fallback_events(
    terminal_round_text: str,
) -> tuple[list[StreamBaseResponse], str]:
    """Build the fallback stream events surfaced when a budget-exhausted
    turn left the terminal round with no visible text.

    Returns ``(events, text_to_append)``.  Empty list + empty string when
    no fallback is needed.  Split out of the async generator so it's unit-
    testable without the surrounding streaming machinery.
    """
    notice = _budget_exhausted_notice_text(terminal_round_text)
    if notice is None:
        return [], ""
    block_id = str(uuid.uuid4())
    events: list[StreamBaseResponse] = [
        StreamTextStart(id=block_id),
        StreamTextDelta(id=block_id, delta=notice),
        StreamTextEnd(id=block_id),
    ]
    return events, notice


# Max seconds to wait for transcript upload in the finally block before
# letting it continue as a background task (tracked in _background_tasks).
_TRANSCRIPT_UPLOAD_TIMEOUT_S = 5

# MIME types that can be embedded as vision content blocks (OpenAI format).
_VISION_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})


# Max size for embedding images directly in the user message (20 MiB raw).
_MAX_INLINE_IMAGE_BYTES = 20 * 1024 * 1024

# Matches characters unsafe for filenames.
_UNSAFE_FILENAME = re.compile(r"[^\w.\-]")

# OpenRouter-specific extra_body flag that embeds the real generation cost
# into the final usage chunk. Module-level constant so we don't reallocate
# an identical dict on every streaming call.
_OPENROUTER_INCLUDE_USAGE_COST = {"usage": {"include": True}}


def _extract_usage_cost(usage: CompletionUsage) -> float | None:
    """Return the provider-reported USD cost on a streaming usage chunk.

    OpenRouter piggybacks a ``cost`` field on the OpenAI-compatible usage
    object when the request body includes ``usage: {"include": True}``.
    The OpenAI SDK's typed ``CompletionUsage`` does not declare it, so we
    read it off ``model_extra`` (the pydantic v2 container for extras) to
    keep the access fully typed — no ``getattr``.

    Returns ``None`` when the field is absent, explicitly null,
    non-numeric, non-finite, or negative. Invalid values (including
    present-but-null) are logged here — they indicate a provider bug
    worth chasing; plain absences are silent so the caller can dedupe
    the "missing cost" warning per stream.
    """
    extras = usage.model_extra or {}
    if "cost" not in extras:
        return None
    raw = extras["cost"]
    if raw is None:
        logger.error("[Baseline] usage.cost is present but null")
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        logger.error("[Baseline] usage.cost is not numeric: %r", raw)
        return None
    if not math.isfinite(val) or val < 0:
        logger.error("[Baseline] usage.cost is non-finite or negative: %r", val)
        return None
    return val


async def _prepare_baseline_attachments(
    file_ids: list[str],
    user_id: str,
    session_id: str,
    working_dir: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Download workspace files and prepare them for the baseline LLM.

    Images become OpenAI-format vision content blocks.  Non-image files are
    saved to *working_dir* so tool handlers can access them.

    Returns ``(hint_text, image_blocks)``.
    """
    if not file_ids or not user_id:
        return "", []

    try:
        manager = await get_workspace_manager(user_id, session_id)
    except Exception:
        logger.warning(
            "Failed to create workspace manager for file attachments",
            exc_info=True,
        )
        return "", []

    image_blocks: list[dict[str, Any]] = []
    file_descriptions: list[str] = []

    for fid in file_ids:
        try:
            file_info = await manager.get_file_info(fid)
            if file_info is None:
                continue
            content = await manager.read_file_by_id(fid)
            mime = (file_info.mime_type or "").split(";")[0].strip().lower()

            if mime in _VISION_MIME_TYPES and len(content) <= _MAX_INLINE_IMAGE_BYTES:
                b64 = base64.b64encode(content).decode("ascii")
                image_blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": mime, "data": b64},
                    }
                )
                file_descriptions.append(
                    f"- {file_info.name} ({mime}, "
                    f"{file_info.size_bytes:,} bytes) [embedded as image]"
                )
            else:
                safe = _UNSAFE_FILENAME.sub("_", file_info.name) or "file"
                candidate = os.path.join(working_dir, safe)
                if os.path.exists(candidate):
                    stem, ext = os.path.splitext(safe)
                    idx = 1
                    while os.path.exists(candidate):
                        candidate = os.path.join(working_dir, f"{stem}_{idx}{ext}")
                        idx += 1
                with open(candidate, "wb") as f:
                    f.write(content)
                file_descriptions.append(
                    f"- {file_info.name} ({mime}, "
                    f"{file_info.size_bytes:,} bytes) saved to "
                    f"{os.path.basename(candidate)}"
                )
        except Exception:
            logger.warning("Failed to prepare file %s", fid[:12], exc_info=True)

    if not file_descriptions:
        return "", []

    noun = "file" if len(file_descriptions) == 1 else "files"
    has_non_images = len(file_descriptions) > len(image_blocks)
    read_hint = (
        " Use the read_workspace_file tool to view non-image files."
        if has_non_images
        else ""
    )
    hint = (
        f"\n[The user attached {len(file_descriptions)} {noun}.{read_hint}\n"
        + "\n".join(file_descriptions)
        + "]"
    )
    return hint, image_blocks


def _filter_tools_by_permissions(
    tools: list[ChatCompletionToolParam],
    permissions: "CopilotPermissions",
) -> list[ChatCompletionToolParam]:
    """Filter OpenAI-format tools based on CopilotPermissions.

    Uses short tool names (the ``function.name`` field) to compute the
    effective allowed set, then keeps only matching tools.
    """
    from backend.copilot.permissions import all_known_tool_names

    if permissions.is_empty():
        return tools

    all_tools = all_known_tool_names()
    effective = permissions.effective_allowed_tools(all_tools)

    return [
        t
        for t in tools
        if t.get("function", {}).get("name") in effective  # type: ignore[union-attr]
    ]


async def _resolve_baseline_model(
    tier: CopilotLlmModel | None, user_id: str | None
) -> str:
    """Pick the model for the baseline path based on the per-request tier.

    Delegates to :func:`copilot.model_router.resolve_model` so the
    ``(fast, tier)`` cell is LD-overridable per user.  ``None`` tier
    maps to ``"standard"``.
    """
    tier_name = "advanced" if tier == "advanced" else "standard"
    return await resolve_model("fast", tier_name, user_id, config=config)


@dataclass
class _BaselineStreamState:
    """Mutable state shared between the tool-call loop callbacks.

    Extracted from ``stream_chat_completion_baseline`` so that the callbacks
    can be module-level functions instead of deeply nested closures.
    """

    model: str = ""
    # Live delivery channel drained concurrently by ``stream_chat_completion_baseline``
    # so reasoning / text / tool events reach the SSE wire **during** the upstream
    # LLM stream, not after ``_baseline_llm_caller`` returns.  Before this was a
    # ``list`` drained per ``tool_call_loop`` iteration, so any model with
    # extended thinking (Anthropic via OpenRouter, Moonshot, future reasoning
    # routes) froze the UI for the entire duration of each LLM round before
    # flushing the backlog in one burst.  The queue is single-producer (the
    # streaming loop) / single-consumer (the outer async-gen yield loop);
    # ``None`` is the close sentinel.
    pending_events: asyncio.Queue[StreamBaseResponse | None] = field(
        default_factory=asyncio.Queue
    )
    # Mirror of every event put on ``pending_events`` — kept for unit tests that
    # inspect post-hoc what was emitted.  Not consumed by production code.
    emitted_events: list[StreamBaseResponse] = field(default_factory=list)
    assistant_text: str = ""
    text_block_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text_started: bool = False
    reasoning_emitter: BaselineReasoningEmitter = field(init=False)
    turn_prompt_tokens: int = 0
    turn_completion_tokens: int = 0
    turn_cache_read_tokens: int = 0
    turn_cache_creation_tokens: int = 0
    cost_usd: float | None = None
    # Tracks whether we've already warned about a missing `cost` field in
    # the usage chunk this stream, so non-OpenRouter providers don't
    # generate one warning per streaming call.
    cost_missing_logged: bool = False
    thinking_stripper: _ThinkingStripper = field(default_factory=_ThinkingStripper)
    # MUTATE in place only — ``__post_init__`` hands this list reference to
    # ``BaselineReasoningEmitter`` so reasoning rows can be appended as
    # deltas stream in.  Reassigning (``state.session_messages = [...]``)
    # would silently detach the emitter from the new list.
    session_messages: list[ChatMessage] = field(default_factory=list)
    # Tracks how much of ``assistant_text`` has already been flushed to
    # ``session.messages`` via mid-loop pending drains, so the ``finally``
    # block only appends the *new* assistant text (avoiding duplication of
    # round-1 text when round-1 entries were cleared from session_messages).
    _flushed_assistant_text_len: int = 0
    # Memoised system-message dict with cache_control applied.  The system
    # prompt is static within a session, so we build it once on the first
    # LLM round and reuse the same dict on subsequent rounds — avoiding
    # an O(N) dict-copy of the growing ``messages`` list on every tool-call
    # iteration.  ``None`` means "not yet computed" (or the first message
    # wasn't a system role, so no marking applies).
    cached_system_message: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        # Wire the reasoning emitter to ``session_messages`` so it can
        # append ``role="reasoning"`` rows as reasoning streams in — the
        # frontend's ``convertChatSessionToUiMessages`` relies on these
        # rows to render the Reasoning collapse after the AI SDK's
        # stream-end hydrate swaps in the DB-backed message list.
        # ``render_in_ui`` is sourced from ``config.render_reasoning_in_ui``
        # so the operator can silence the reasoning collapse globally
        # without dropping the persisted audit trail.
        self.reasoning_emitter = BaselineReasoningEmitter(
            self.session_messages,
            render_in_ui=config.render_reasoning_in_ui,
        )


def _emit(state: "_BaselineStreamState", event: StreamBaseResponse) -> None:
    """Queue *event* for the live SSE wire AND mirror into ``emitted_events``.

    Single helper so every streaming producer (LLM stream loop, tool executor,
    conversation updater) posts to the same single-consumer queue.  The mirror
    list is read-only from production code — it exists so unit tests can assert
    on the full sequence emitted during one call.
    """
    state.pending_events.put_nowait(event)
    state.emitted_events.append(event)


def _emit_all(
    state: "_BaselineStreamState", events: Iterable[StreamBaseResponse]
) -> None:
    """Queue *events* in order — convenience for emitter batches."""
    for event in events:
        _emit(state, event)


def _is_anthropic_model(model: str) -> bool:
    """Return True if *model* routes to Anthropic (native or via OpenRouter).

    Examples that return True:
      - ``anthropic/claude-sonnet-4-6`` (OpenRouter route)
      - ``claude-3-5-sonnet-20241022`` (direct Anthropic API)
      - ``anthropic.claude-3-5-sonnet`` (Bedrock-style)

    False for ``openai/gpt-4o``, ``google/gemini-2.5-pro``, ``xai/grok-4``
    etc.  Moonshot is False here too even though Moonshot's
    Anthropic-compat endpoint honours ``cache_control`` — use
    :func:`_supports_prompt_cache_markers` for the cache-gating decision,
    which also allows Moonshot routes.  This function stays scoped to
    "genuinely Anthropic" so callers that need the stricter check (e.g.
    ``anthropic-beta`` header emission) keep their existing semantics.
    """
    lowered = model.lower()
    return "claude" in lowered or lowered.startswith("anthropic")


def _supports_prompt_cache_markers(model: str) -> bool:
    """Return True when *model* accepts Anthropic-style ``cache_control``.

    Superset of :func:`_is_anthropic_model` — also allows Moonshot
    (``moonshotai/*``), whose OpenRouter Anthropic-compat endpoint
    honours the marker and empirically lifts cache hit rate on
    continuation turns from near-zero (Moonshot's own automatic prefix
    cache, which drifts readily) to the 60-95% Anthropic ballpark.

    OpenAI / Grok / Gemini still 400 on ``cache_control``, so this
    function returns False for those providers — add new vendors here
    only after verifying their endpoint accepts the field.
    """
    return _is_anthropic_model(model) or is_moonshot_model(model)


def _fresh_ephemeral_cache_control() -> dict[str, str]:
    """Return a FRESH ephemeral ``cache_control`` dict each call.

    The ``ttl`` is sourced from :attr:`ChatConfig.baseline_prompt_cache_ttl`
    (default ``1h``) so the static prefix stays warm across many users'
    requests in the same workspace cache.  Anthropic caches are keyed
    per-workspace, so every copilot user reading the same system prompt
    hits the same cached entry.

    Using a shared module-level dict would let any downstream mutation
    (e.g. the OpenAI SDK normalising fields in-place) poison every future
    request's marker.  Construction is O(1) so the safety margin is free.
    """
    return {"type": "ephemeral", "ttl": config.baseline_prompt_cache_ttl}


def _fresh_anthropic_caching_headers() -> dict[str, str]:
    """Return a FRESH ``extra_headers`` dict requesting the Anthropic
    prompt-caching beta.

    Same reasoning as :func:`_fresh_ephemeral_cache_control`: never hand a
    shared module-level dict to third-party SDKs.  OpenRouter auto-forwards
    cache_control for Anthropic routes without this header, but passing it
    makes the intent unambiguous on-wire and is a no-op for non-Anthropic
    providers (unknown headers are dropped).
    """
    return {"anthropic-beta": "prompt-caching-2024-07-31"}


def _mark_tools_with_cache_control(
    tools: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return a copy of *tools* with ``cache_control`` on the last entry.

    Marking the last tool is a cache breakpoint that covers the whole tool
    schema block as a cacheable prefix segment.  Extracted from
    :func:`_mark_system_message_with_cache_control` so callers can precompute
    the marked tool list once per session — the tool set is static within a
    request and the ~43 dict-copies would otherwise run on every LLM round
    in the tool-call loop.

    **Only call this for Anthropic model routes.**  Non-Anthropic providers
    (OpenAI, Grok, Gemini) reject the unknown ``cache_control`` field with
    a 400 schema validation error.  Gate via :func:`_is_anthropic_model`.
    """
    cached: list[dict[str, Any]] = [dict(t) for t in tools]
    if cached:
        cached[-1] = {
            **cached[-1],
            "cache_control": _fresh_ephemeral_cache_control(),
        }
    return cached


def _build_cached_system_message(
    system_message: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a copy of *system_message* with ``cache_control`` applied.

    Anthropic's cache uses prefix-match with up to 4 explicit breakpoints.
    Combined with the last-tool marker this gives two cache segments — the
    system block alone, and system+all-tools — so requests that share only
    the system prefix still get a partial cache hit.

    The system message is rebuilt via spread (``{**original, ...}``) so any
    unknown fields the caller set (e.g. ``name``) survive the transformation.
    Non-Anthropic models silently ignore the markers.

    Returns the original dict (shallow-copied) unchanged when the content
    shape is unsupported (missing / non-string / empty) — callers should
    splice it into the message list as-is in that case.
    """
    sys_copy = dict(system_message)
    sys_content = sys_copy.get("content")
    if isinstance(sys_content, str) and sys_content:
        sys_copy["content"] = [
            {
                "type": "text",
                "text": sys_content,
                "cache_control": _fresh_ephemeral_cache_control(),
            }
        ]
    return sys_copy


def _mark_system_message_with_cache_control(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return a copy of *messages* with ``cache_control`` on the system block.

    Thin wrapper around :func:`_build_cached_system_message` that preserves
    the original list shape.  Prefer the memoised path in
    ``_baseline_llm_caller`` (which builds the cached system dict once per
    session) for hot-loop callers; this function is retained for call sites
    outside the tool-call loop where per-call copying is acceptable.
    """
    cached_messages: list[dict[str, Any]] = [dict(m) for m in messages]
    if cached_messages and cached_messages[0].get("role") == "system":
        cached_messages[0] = _build_cached_system_message(cached_messages[0])
    return cached_messages


async def _baseline_llm_caller(
    messages: list[dict[str, Any]],
    tools: Sequence[Any],
    *,
    state: _BaselineStreamState,
) -> LLMLoopResponse:
    """Stream an OpenAI-compatible response and collect results.

    Extracted from ``stream_chat_completion_baseline`` for readability.
    """
    _emit(state, StreamStartStep())
    # Fresh thinking-strip state per round so a malformed unclosed
    # block in one LLM call cannot silently drop content in the next.
    state.thinking_stripper = _ThinkingStripper()

    round_text = ""
    try:
        client = _get_main_client()
        supports_cache = _supports_prompt_cache_markers(state.model)
        if supports_cache:
            # Build cached system message once; splice on each round to avoid O(n) list copy.
            if (
                state.cached_system_message is None
                and messages
                and messages[0].get("role") == "system"
            ):
                state.cached_system_message = _build_cached_system_message(messages[0])
            if state.cached_system_message is not None and messages:
                final_messages = [state.cached_system_message, *messages[1:]]
            else:
                final_messages = messages
            extra_headers = (
                _fresh_anthropic_caching_headers()
                if _is_anthropic_model(state.model)
                else None
            )
        else:
            final_messages = messages
            extra_headers = None
        typed_messages = cast(list[ChatCompletionMessageParam], final_messages)
        # OR-only: Anthropic's compat endpoint 400s on usage.include + reasoning; use native thinking param in direct mode.
        if config.openrouter_active:
            extra_body: dict[str, Any] = dict(_OPENROUTER_INCLUDE_USAGE_COST)
            reasoning_param = reasoning_extra_body(
                state.model, config.claude_agent_max_thinking_tokens
            )
            if reasoning_param:
                extra_body.update(reasoning_param)
        else:
            extra_body = {}
            thinking_param = anthropic_thinking_extra_body(
                state.model, config.claude_agent_max_thinking_tokens
            )
            if thinking_param:
                extra_body.update(thinking_param)
        create_kwargs: dict[str, Any] = {
            "model": state.model,
            "messages": typed_messages,
            "stream": True,
            "extra_body": extra_body,
        }
        # OR-only: Anthropic's compat endpoint 400s on stream_options; OR embeds cost via it.
        if config.openrouter_active:
            create_kwargs["stream_options"] = {"include_usage": True}
        # Direct: Anthropic requires max_tokens > budget_tokens explicitly; OR injects a default.
        if not config.openrouter_active and "thinking" in extra_body:
            model_max = get_max_output_tokens(state.model)
            budget = min(config.claude_agent_max_thinking_tokens, model_max - 1)
            extra_body["thinking"]["budget_tokens"] = budget
            create_kwargs["max_tokens"] = min(budget + 4096, model_max)
        if extra_headers:
            create_kwargs["extra_headers"] = extra_headers
        if tools:
            create_kwargs["tools"] = cast(list[ChatCompletionToolParam], list(tools))
        response = await client.chat.completions.create(**create_kwargs)
        tool_calls_by_index: dict[int, dict[str, str]] = {}

        # Iterate under an inner try/finally so early exits (cancel, tool-call
        # break, exception) always release the underlying httpx connection.
        # Without this, openai.AsyncStream leaks the streaming response and
        # the TCP socket ends up in CLOSE_WAIT until the process exits.
        try:
            async for chunk in response:
                if chunk.usage:
                    state.turn_prompt_tokens += chunk.usage.prompt_tokens or 0
                    state.turn_completion_tokens += chunk.usage.completion_tokens or 0
                    ptd = chunk.usage.prompt_tokens_details
                    if ptd:
                        state.turn_cache_read_tokens += ptd.cached_tokens or 0
                        state.turn_cache_creation_tokens += (
                            _extract_cache_creation_tokens(ptd)
                        )
                    cost = _extract_usage_cost(chunk.usage)
                    direct_mode = not config.openrouter_active
                    if cost is None and direct_mode:
                        # Direct mode: no usage.cost field (OR extension); compute from rate card.
                        ptd = chunk.usage.prompt_tokens_details
                        cost = compute_anthropic_cost_usd(
                            model=state.model,
                            prompt_tokens=chunk.usage.prompt_tokens or 0,
                            completion_tokens=chunk.usage.completion_tokens or 0,
                            cache_read_tokens=(ptd.cached_tokens or 0) if ptd else 0,
                            cache_creation_tokens=(
                                _extract_cache_creation_tokens(ptd) if ptd else 0
                            ),
                            cache_ttl=config.baseline_prompt_cache_ttl,
                        )
                    if cost is not None:
                        state.cost_usd = (state.cost_usd or 0.0) + cost
                    elif (
                        not direct_mode
                        and "cost" not in (chunk.usage.model_extra or {})
                        and not state.cost_missing_logged
                    ):
                        # OR with cost field absent — warn once so monitoring catches persistent misses.
                        logger.warning(
                            "[Baseline] usage chunk missing cost (model=%s, "
                            "prompt=%s, completion=%s) — rate-limit will "
                            "skip this call",
                            state.model,
                            chunk.usage.prompt_tokens,
                            chunk.usage.completion_tokens,
                        )
                        state.cost_missing_logged = True

                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue

                _emit_all(state, state.reasoning_emitter.on_delta(delta))

                if delta.content:
                    # Text and reasoning must not interleave on the wire — the
                    # AI SDK maps distinct start/end pairs to distinct UI
                    # parts.  Close any open reasoning block before emitting
                    # the first text delta of this run.
                    _emit_all(state, state.reasoning_emitter.close())
                    emit = state.thinking_stripper.process(delta.content)
                    if emit:
                        if not state.text_started:
                            _emit(state, StreamTextStart(id=state.text_block_id))
                            state.text_started = True
                        round_text += emit
                        _emit(
                            state,
                            StreamTextDelta(id=state.text_block_id, delta=emit),
                        )

                if delta.tool_calls:
                    # Same rule as the text branch: close any open reasoning
                    # block before a tool_use starts so the AI SDK treats
                    # reasoning and tool-use as distinct parts.
                    _emit_all(state, state.reasoning_emitter.close())
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_by_index:
                            tool_calls_by_index[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                        entry = tool_calls_by_index[idx]
                        if tc.id:
                            entry["id"] = tc.id
                        if tc.function and tc.function.name:
                            entry["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            entry["arguments"] += tc.function.arguments
        finally:
            # Release the streaming httpx connection back to the pool on every
            # exit path (normal completion, break, exception). openai.AsyncStream
            # does not auto-close when the async-for loop exits early.
            try:
                await response.close()
            except Exception:
                pass

    finally:
        # Close open blocks on both normal and exception paths so the
        # frontend always sees matched start/end pairs.  An exception mid
        # ``async for chunk in response`` would otherwise leave reasoning
        # and/or text unterminated and only ``StreamFinishStep`` emitted —
        # the Reasoning / Text collapses would never finalise.
        _emit_all(state, state.reasoning_emitter.close())
        # Flush any buffered text held back by the thinking stripper.
        tail = state.thinking_stripper.flush()
        if tail:
            if not state.text_started:
                _emit(state, StreamTextStart(id=state.text_block_id))
                state.text_started = True
            round_text += tail
            _emit(state, StreamTextDelta(id=state.text_block_id, delta=tail))
        if state.text_started:
            _emit(state, StreamTextEnd(id=state.text_block_id))
            state.text_started = False
            state.text_block_id = str(uuid.uuid4())
        # Always persist partial text so the session history stays consistent,
        # even when the stream is interrupted by an exception.
        state.assistant_text += round_text
        # Always emit StreamFinishStep to match the StreamStartStep,
        # even if an exception occurred during streaming.
        _emit(state, StreamFinishStep())

    # Convert to shared format
    llm_tool_calls = [
        LLMToolCall(
            id=tc["id"],
            name=tc["name"],
            arguments=tc["arguments"] or "{}",
        )
        for tc in tool_calls_by_index.values()
    ]

    return LLMLoopResponse(
        response_text=round_text or None,
        tool_calls=llm_tool_calls,
        raw_response=None,  # Not needed for baseline conversation updater
        prompt_tokens=0,  # Tracked via state accumulators
        completion_tokens=0,
    )


async def _baseline_tool_executor(
    tool_call: LLMToolCall,
    tools: Sequence[Any],
    *,
    state: _BaselineStreamState,
    user_id: str | None,
    session: ChatSession,
) -> ToolCallResult:
    """Execute a tool via the copilot tool registry.

    Extracted from ``stream_chat_completion_baseline`` for readability.
    """
    tool_call_id = tool_call.id
    tool_name = tool_call.name
    raw_args = tool_call.arguments or "{}"

    try:
        tool_args = orjson.loads(raw_args)
    except orjson.JSONDecodeError as parse_err:
        parse_error = f"Invalid JSON arguments for tool '{tool_name}': {parse_err}"
        logger.warning("[Baseline] %s", parse_error)
        _emit(
            state,
            StreamToolOutputAvailable(
                toolCallId=tool_call_id,
                toolName=tool_name,
                output=parse_error,
                success=False,
            ),
        )
        return ToolCallResult(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=parse_error,
            is_error=True,
        )

    _emit(
        state,
        StreamToolInputStart(toolCallId=tool_call_id, toolName=tool_name),
    )
    _emit(
        state,
        StreamToolInputAvailable(
            toolCallId=tool_call_id,
            toolName=tool_name,
            input=tool_args,
        ),
    )

    # Announce the tool call to the session so in-turn guards like
    # ``require_guide_read`` can see it *right now*, before the tool
    # actually runs.  Without this, the tool_call row lives only in
    # ``state.session_messages`` until the ``finally`` block flushes it
    # into ``session.messages`` at turn end — so a second tool in the
    # same turn (e.g. ``create_agent`` after ``get_agent_building_guide``)
    # scans a stale ``session.messages`` and the guard re-fires despite
    # the guide having been called.  The announce-set is cleared at turn
    # end; we deliberately don't touch ``session.messages`` here to avoid
    # duplicating the assistant row that ``_baseline_conversation_updater``
    # will append at round end.
    session.announce_inflight_tool_call(tool_name)

    try:
        result: StreamToolOutputAvailable = await execute_tool(
            tool_name=tool_name,
            parameters=tool_args,
            user_id=user_id,
            session=session,
            tool_call_id=tool_call_id,
        )
        _emit(state, result)
        tool_output = (
            result.output if isinstance(result.output, str) else str(result.output)
        )
        return ToolCallResult(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=tool_output,
        )
    except Exception as e:
        error_output = f"Tool execution error: {e}"
        logger.error(
            "[Baseline] Tool %s failed: %s",
            tool_name,
            error_output,
            exc_info=True,
        )
        _emit(
            state,
            StreamToolOutputAvailable(
                toolCallId=tool_call_id,
                toolName=tool_name,
                output=error_output,
                success=False,
            ),
        )
        return ToolCallResult(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=error_output,
            is_error=True,
        )


def _mutate_openai_messages(
    messages: list[dict[str, Any]],
    response: LLMLoopResponse,
    tool_results: list[ToolCallResult] | None,
) -> None:
    """Append assistant / tool-result entries to the OpenAI message list.

    This is the side-effect boundary for the next LLM call — no transcript
    mutation happens here.
    """
    if tool_results:
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if response.response_text:
            assistant_msg["content"] = response.response_text
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in response.tool_calls
        ]
        messages.append(assistant_msg)
        for tr in tool_results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tr.tool_call_id,
                    "content": tr.content,
                }
            )
    elif response.response_text:
        messages.append({"role": "assistant", "content": response.response_text})


def _record_turn_to_transcript(
    response: LLMLoopResponse,
    tool_results: list[ToolCallResult] | None,
    *,
    transcript_builder: TranscriptBuilder,
    model: str,
) -> None:
    """Append assistant + tool-result entries to the transcript builder.

    Kept separate from :func:`_mutate_openai_messages` so the two
    concerns (next-LLM-call payload vs. durable conversation log) can
    evolve independently.
    """
    if tool_results:
        content_blocks: list[dict[str, Any]] = []
        if response.response_text:
            content_blocks.append({"type": "text", "text": response.response_text})
        for tc in response.tool_calls:
            try:
                args = orjson.loads(tc.arguments) if tc.arguments else {}
            except (ValueError, TypeError, orjson.JSONDecodeError) as parse_err:
                logger.debug(
                    "[Baseline] Failed to parse tool_call arguments "
                    "(tool=%s, id=%s): %s",
                    tc.name,
                    tc.id,
                    parse_err,
                )
                args = {}
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": args,
                }
            )
        if content_blocks:
            transcript_builder.append_assistant(
                content_blocks=content_blocks,
                model=model,
                stop_reason=STOP_REASON_TOOL_USE,
            )
        for tr in tool_results:
            # Record tool result to transcript AFTER the assistant tool_use
            # block to maintain correct Anthropic API ordering:
            # assistant(tool_use) → user(tool_result)
            transcript_builder.append_tool_result(
                tool_use_id=tr.tool_call_id,
                content=tr.content,
            )
    elif response.response_text:
        transcript_builder.append_assistant(
            content_blocks=[{"type": "text", "text": response.response_text}],
            model=model,
            stop_reason=STOP_REASON_END_TURN,
        )


def _baseline_conversation_updater(
    messages: list[dict[str, Any]],
    response: LLMLoopResponse,
    tool_results: list[ToolCallResult] | None = None,
    *,
    transcript_builder: TranscriptBuilder,
    model: str = "",
    state: _BaselineStreamState | None = None,
) -> None:
    """Update OpenAI message list with assistant response + tool results.

    Also records structured ChatMessage entries in ``state.session_messages``
    so the full tool-call history is persisted to the session (not just the
    concatenated assistant text).
    """
    _mutate_openai_messages(messages, response, tool_results)
    _record_turn_to_transcript(
        response,
        tool_results,
        transcript_builder=transcript_builder,
        model=model,
    )
    # Record structured messages for session persistence so tool calls
    # and tool results survive across turns and mode switches.
    if state is not None and tool_results:
        assistant_msg = ChatMessage(
            role="assistant",
            content=response.response_text or "",
            tool_calls=[
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in response.tool_calls
            ],
        )
        state.session_messages.append(assistant_msg)
        for tr in tool_results:
            state.session_messages.append(
                ChatMessage(
                    role="tool",
                    content=tr.content,
                    tool_call_id=tr.tool_call_id,
                )
            )


async def _compress_session_messages(
    messages: list[ChatMessage],
    model: str,
) -> list[ChatMessage]:
    """Compress session messages if they exceed the model's token limit.

    Uses the shared compress_context() utility which supports LLM-based
    summarization of older messages while keeping recent ones intact,
    with progressive truncation and middle-out deletion as fallbacks.
    """
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
        result = await compress_context(
            messages=messages_dict,
            model=model,
            client=_get_main_client(),
        )
    except Exception as e:
        logger.warning("[Baseline] Context compression with LLM failed: %s", e)
        result = await compress_context(
            messages=messages_dict,
            model=model,
            client=None,
        )

    if result.was_compacted:
        logger.info(
            "[Baseline] Context compacted: %d -> %d tokens (%d summarized, %d dropped)",
            result.original_token_count,
            result.token_count,
            result.messages_summarized,
            result.messages_dropped,
        )
        return [
            ChatMessage(
                role=m["role"],
                content=m.get("content"),
                tool_calls=m.get("tool_calls"),
                tool_call_id=m.get("tool_call_id"),
            )
            for m in result.messages
        ]

    return messages


def should_upload_transcript(user_id: str | None, upload_safe: bool) -> bool:
    """Return ``True`` when the caller should upload the final transcript.

    Uploads require a logged-in user (for the storage key) *and* a safe
    upload signal from ``_load_prior_transcript`` — i.e. GCS does not hold a
    newer version that we'd be overwriting.
    """
    return bool(user_id) and upload_safe


def _append_gap_to_builder(
    gap: list[ChatMessage],
    builder: TranscriptBuilder,
) -> None:
    """Append gap messages from chat-db into the TranscriptBuilder.

    Converts ChatMessage (OpenAI format) to TranscriptBuilder entries
    (Claude CLI JSONL format) so the uploaded transcript covers all turns.

    Pre-condition: ``gap`` always starts at a user or assistant boundary
    (never mid-turn at a ``tool`` role), because ``detect_gap`` enforces
    ``session_messages[wm-1].role == 'assistant'`` before returning a non-empty
    gap.  Any ``tool`` role messages within the gap always follow an assistant
    entry that already exists in the builder or in the gap itself.
    """
    for msg in gap:
        if msg.role == "user":
            builder.append_user(msg.content or "")
        elif msg.role == "assistant":
            content_blocks: list[dict] = []
            if msg.content:
                content_blocks.append({"type": "text", "text": msg.content})
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    input_data = util_json.loads(fn.get("arguments", "{}"), fallback={})
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", "") if isinstance(tc, dict) else "",
                            "name": fn.get("name", "unknown"),
                            "input": input_data,
                        }
                    )
            if not content_blocks:
                # Fallback: ensure every assistant gap message produces an entry
                # so the builder's entry count matches the gap length.
                content_blocks.append({"type": "text", "text": ""})
            builder.append_assistant(content_blocks=content_blocks)
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
                logger.warning(
                    "[Baseline] Skipping tool gap message with no tool_call_id"
                )


async def _load_prior_transcript(
    user_id: str,
    session_id: str,
    session_messages: list[ChatMessage],
    transcript_builder: TranscriptBuilder,
) -> tuple[bool, "TranscriptDownload | None"]:
    """Download and load the prior CLI session into ``transcript_builder``.

    Returns a tuple of (upload_safe, transcript_download):
    - ``upload_safe`` is ``True`` when it is safe to upload at the end of this
      turn.  Upload is suppressed only for **download errors** (unknown GCS
      state) — missing and invalid files return ``True`` because there is
      nothing in GCS worth protecting against overwriting.
    - ``transcript_download`` is a ``TranscriptDownload`` with str content
      (pre-decoded and stripped) when available, or ``None`` when no valid
      transcript could be loaded.  Callers pass this to
      ``extract_context_messages`` to build the LLM context.
    """
    try:
        restore = await download_transcript(
            user_id, session_id, log_prefix="[Baseline]"
        )
    except Exception as e:
        logger.warning("[Baseline] Session restore failed: %s", e)
        # Unknown GCS state — be conservative, skip upload.
        return False, None

    if restore is None:
        logger.debug("[Baseline] No CLI session available — will upload fresh")
        # Nothing in GCS to protect; allow upload so the first baseline turn
        # writes the initial transcript snapshot.
        return True, None

    content_bytes = restore.content
    try:
        raw_str = (
            content_bytes.decode("utf-8")
            if isinstance(content_bytes, bytes)
            else content_bytes
        )
    except UnicodeDecodeError:
        logger.warning("[Baseline] CLI session content is not valid UTF-8")
        # Corrupt file in GCS; overwriting with a valid one is better.
        return True, None

    stripped = strip_for_upload(raw_str)
    if not validate_transcript(stripped):
        logger.warning("[Baseline] CLI session content invalid after strip")
        # Corrupt file in GCS; overwriting with a valid one is better.
        return True, None

    transcript_builder.load_previous(stripped, log_prefix="[Baseline]")
    logger.info(
        "[Baseline] Loaded CLI session: %dB, msg_count=%d",
        len(content_bytes) if isinstance(content_bytes, bytes) else len(raw_str),
        restore.message_count,
    )

    gap = detect_gap(restore, session_messages)
    # Hole-fill: when the cap engaged AND the windowed view starts above the
    # transcript watermark, the sequences in between live in DB but neither
    # the transcript nor the loaded gap.  Persisting them to the builder here
    # (so the next-turn upload includes them) closes the loop — without this,
    # detect_gap on every subsequent turn would re-discover the same hole.
    hole: list[ChatMessage] = []
    if restore.message_count > 0:
        hole = await fill_hole_between_transcript_and_gap(
            session_id, restore.message_count, gap
        )
    if hole or gap:
        _append_gap_to_builder(hole + gap, transcript_builder)
        logger.info(
            "[Baseline] Filled gap: loaded %d transcript msgs + %d hole msgs "
            "+ %d gap msgs from DB",
            restore.message_count,
            len(hole),
            len(gap),
        )

    # Return a str-content version so extract_context_messages receives a
    # pre-decoded, stripped transcript (avoids redundant decode + strip).
    # TranscriptDownload.content is typed as bytes | str; we pass str here
    # to avoid a redundant encode + decode round-trip.
    str_restore = TranscriptDownload(
        content=stripped,
        message_count=restore.message_count,
        mode=restore.mode,
    )
    return True, str_restore


async def _upload_final_transcript(
    user_id: str,
    session_id: str,
    transcript_builder: TranscriptBuilder,
    session_msg_count: int,
) -> None:
    """Serialize and upload the transcript for next-turn continuity.

    Uses the builder's own invariants to decide whether to upload,
    avoiding a JSONL re-parse.  A builder that ends with an assistant
    entry is structurally complete; a builder that doesn't (empty, or
    ends mid-turn) is skipped.
    """
    try:
        if transcript_builder.last_entry_type != "assistant":
            logger.debug(
                "[Baseline] No complete assistant turn to upload (last_entry=%s)",
                transcript_builder.last_entry_type,
            )
            return
        content = transcript_builder.to_jsonl()
        if not content:
            logger.debug("[Baseline] Empty transcript content, skipping upload")
            return
        # Track the upload as a background task so a timeout doesn't leak an
        # orphaned coroutine; shield it so cancellation of this caller doesn't
        # abort the in-flight GCS write.
        upload_task = asyncio.create_task(
            upload_transcript(
                user_id=user_id,
                session_id=session_id,
                content=content.encode("utf-8"),
                message_count=session_msg_count,
                mode="baseline",
                log_prefix="[Baseline]",
            )
        )
        _background_tasks.add(upload_task)
        upload_task.add_done_callback(_background_tasks.discard)
        # Bound the wait: a hung storage backend must not block the response
        # from finishing. The task keeps running in _background_tasks on
        # timeout and will be cleaned up when it resolves.
        await asyncio.wait_for(
            asyncio.shield(upload_task), timeout=_TRANSCRIPT_UPLOAD_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        # Upload is still running in _background_tasks; we just stopped waiting.
        logger.info(
            "[Baseline] Transcript upload exceeded %ss wait — continuing as background task",
            _TRANSCRIPT_UPLOAD_TIMEOUT_S,
        )
    except Exception as upload_err:
        logger.error("[Baseline] Transcript upload failed: %s", upload_err)


async def stream_chat_completion_baseline(
    session_id: str,
    message: str | None = None,
    is_user_message: bool = True,
    user_id: str | None = None,
    session: ChatSession | None = None,
    file_ids: list[str] | None = None,
    permissions: "CopilotPermissions | None" = None,
    context: dict[str, str] | None = None,
    mode: CopilotMode | None = None,
    model: CopilotLlmModel | None = None,
    request_arrival_at: float = 0.0,
    **_kwargs: Any,
) -> AsyncGenerator[StreamBaseResponse, None]:
    """Baseline LLM with tool calling via OpenAI-compatible API.

    Designed as a fallback when the Claude Agent SDK is unavailable.
    Uses the same tool registry as the SDK path but routes through any
    OpenAI-compatible provider (e.g. OpenRouter).

    Flow: stream response -> if tool_calls, execute them -> feed results back -> repeat.
    """
    if session is None:
        session = await get_chat_session(session_id, user_id)

    if not session:
        raise NotFoundError(
            f"Session {session_id} not found. Please create a new session first."
        )

    # Drop orphan tool_use + trailing stop-marker rows left by a previous
    # Stop mid-tool-call so the new turn starts from a well-formed message list.
    prune_orphan_tool_calls(
        session.messages, log_prefix=f"[Baseline] [{session_id[:12]}]"
    )

    # Strip any user-injected <user_context> tags on every turn.
    # Only the server-injected prefix on the first message is trusted.
    if message:
        message = strip_user_context_tags(message)

    if maybe_append_user_message(session, message, is_user_message):
        if is_user_message:
            track_user_message(
                user_id=user_id,
                session_id=session_id,
                message_length=len(message or ""),
            )

    # Capture count *before* the pending drain so is_first_turn and the
    # transcript staleness check are not skewed by queued messages.
    _pre_drain_msg_count = len(session.messages)

    # Drain any messages the user queued via POST /messages/pending
    # while this session was idle (or during a previous turn whose
    # mid-loop drains missed them).
    # The drained content is appended after ``message`` so the user's submitted
    # message remains the leading context (better UX: the user sent their primary
    # message first, queued follow-ups second).  The already-saved user message
    # in the DB is updated via update_message_content_by_sequence rather than
    # inserting a new row, because routes.py has already saved the user message
    # before the executor picks up the turn (using insert_pending_before_last +
    # persist_session_safe would add a duplicate row at sequence N+1).
    drained_at_start_pending = await drain_pending_safe(session_id, "[Baseline]")
    if drained_at_start_pending:
        logger.info(
            "[Baseline] Draining %d pending message(s) at turn start for session %s",
            len(drained_at_start_pending),
            session_id,
        )
        # NOTE: combine + per-row persist both happen *after*
        # ``inject_user_context`` below — see the comment near that
        # call for the ordering rationale.  At this point ``message``
        # is still the original turn-starting send.

    # Select model based on the per-request tier toggle (standard / advanced).
    # The path (fast vs extended_thinking) is already decided — we're in the
    # baseline (fast) path; ``mode`` is accepted for logging parity only.
    # Normalize immediately so EVERY downstream helper (E2B context,
    # ``_compress_session_messages``, the streaming caller) sees the
    # transport-correct slug — otherwise compaction in direct-Anthropic
    # mode silently falls back to the non-LLM path because the OR slug
    # would be rejected by the direct client.  Pass the baseline-side
    # ``config`` so monkeypatch fixtures targeting this module's
    # ``config`` symbol drive the decision.
    resolved_model = await _resolve_baseline_model(model, user_id)
    try:
        active_model = normalize_model_for_transport(resolved_model, config)
    except ValueError as exc:
        # Mirror SDK's LD-reject soft-fallback (see ``copilot.sdk.service``):
        # a per-user LD ``copilot-model-routing`` cell can pin a non-Anthropic
        # slug (e.g. ``moonshotai/kimi-*``) on a direct-Anthropic deployment
        # where the baseline transport rejects it.  Fall back to the
        # TIER-SPECIFIC config default so the request still streams instead
        # of erroring at the user.  Re-raises the original error if the
        # config default is also invalid (deployment-level misconfig caught
        # by the ``model_validator`` at startup).
        tier_default = (
            config.fast_advanced_model
            if model == "advanced"
            else config.fast_standard_model
        )
        try:
            active_model = normalize_model_for_transport(tier_default, config)
        except ValueError:
            raise exc
        logger.warning(
            "[Baseline] [%s] LD model %r rejected for tier=%s (%s); falling "
            "back to tier default %s",
            session_id[:12] if session_id else "?",
            resolved_model,
            "advanced" if model == "advanced" else "standard",
            exc,
            active_model,
        )

    # --- E2B sandbox setup (feature parity with SDK path) ---
    e2b_sandbox = None
    e2b_api_key = config.active_e2b_api_key
    if e2b_api_key:
        try:
            from backend.copilot.tools.e2b_sandbox import get_or_create_sandbox

            e2b_sandbox = await get_or_create_sandbox(
                session_id,
                api_key=e2b_api_key,
                template=config.e2b_sandbox_template,
                timeout=config.e2b_sandbox_timeout,
                on_timeout=config.e2b_sandbox_on_timeout,
            )
        except Exception:
            logger.warning("[Baseline] E2B sandbox setup failed", exc_info=True)

    # --- Transcript support (feature parity with SDK path) ---
    transcript_builder = TranscriptBuilder()
    transcript_upload_safe = True

    # Build system prompt only on the first turn to avoid mid-conversation
    # changes from concurrent chats updating business understanding.
    # Use the pre-drain count so queued pending messages don't incorrectly
    # flip is_first_turn to False on an actual first turn.
    is_first_turn = _pre_drain_msg_count <= 1
    # Gate context fetch on both first turn AND user message so that assistant-
    # role calls (e.g. tool-result submissions) on the first turn don't trigger
    # a needless DB lookup for user understanding.
    should_inject_user_context = is_first_turn and is_user_message

    if should_inject_user_context:
        prompt_task = _build_system_prompt(user_id)
    else:
        prompt_task = _build_system_prompt(None)

    # Run download + prompt build concurrently — both are independent I/O
    # on the request critical path.  Use the pre-drain count so pending
    # messages drained at turn start don't spuriously trigger a transcript
    # load on an actual first turn.
    transcript_download: TranscriptDownload | None = None
    if user_id and _pre_drain_msg_count > 1:
        (
            (transcript_upload_safe, transcript_download),
            (base_system_prompt, understanding),
        ) = await asyncio.gather(
            _load_prior_transcript(
                user_id=user_id,
                session_id=session_id,
                session_messages=session.messages,
                transcript_builder=transcript_builder,
            ),
            prompt_task,
        )
    else:
        base_system_prompt, understanding = await prompt_task

    # Append user message to transcript after context injection below so the
    # transcript receives the prefixed message when user context is available.

    # NOTE: drained pending messages are folded into the current user
    # message's content (see the turn-start drain above), so the single
    # ``transcript_builder.append_user`` call below (covered by the
    # ``if message and is_user_message`` branch that appends
    # ``user_message_for_transcript or message``) already records the
    # combined text in the transcript. Do NOT also append drained items
    # individually here — on the ``transcript_download is None`` path
    # that would produce N separate pending entries plus the combined
    # entry, duplicating the pending content in the JSONL uploaded for
    # the next turn's ``--resume``.

    # Generate title for new sessions
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

    # Append tool documentation, technical notes, and Graphiti memory instructions
    graphiti_enabled = await is_enabled_for_user(user_id)

    graphiti_supplement = get_graphiti_supplement() if graphiti_enabled else ""
    # Append the builder-session block (graph id+name + full building guide)
    # AFTER the shared supplements so the system prompt is byte-identical
    # across turns of the same builder session — Claude's prompt cache keeps
    # the ~20KB guide warm for the whole session.  Empty string for
    # non-builder sessions keeps the cross-user cache hot.
    builder_session_suffix = await build_builder_system_prompt_suffix(session)
    system_prompt = (
        base_system_prompt
        + SHARED_TOOL_NOTES
        + graphiti_supplement
        + builder_session_suffix
    )

    # Warm context: pre-load relevant facts from Graphiti on first turn.
    # Use the pre-drain count so pending messages drained at turn start
    # don't prevent warm context injection on an actual first turn.
    # Stored here but injected into the user message (not the system prompt)
    # after openai_messages is built — keeps system prompt static for caching.
    warm_ctx: str | None = None
    if graphiti_enabled and user_id and _pre_drain_msg_count <= 1:
        from backend.copilot.graphiti.context import fetch_warm_context

        warm_ctx = await fetch_warm_context(user_id, message or "")

    # Context path: transcript content (compacted, isCompactSummary preserved) +
    # gap (DB messages after watermark) + current user turn.
    # This avoids re-reading the full session history from DB on every turn.
    # See extract_context_messages() in transcript.py for the shared primitive.
    prior_context = await extract_context_messages(
        transcript_download, session.messages, session_id=session.session_id
    )
    messages_for_context = await _compress_session_messages(
        prior_context + ([session.messages[-1]] if session.messages else []),
        model=active_model,
    )

    # Build OpenAI message list from session history.
    # Include tool_calls on assistant messages and tool-role results so the
    # model retains full context of what tools were invoked and their outcomes.
    openai_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt}
    ]
    for msg in messages_for_context:
        if msg.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant"}
            if msg.content:
                entry["content"] = msg.content
            if msg.tool_calls:
                entry["tool_calls"] = msg.tool_calls
            if msg.content or msg.tool_calls:
                openai_messages.append(entry)
        elif msg.role == "tool" and msg.tool_call_id:
            openai_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content or "",
                }
            )
        elif msg.role == "user" and msg.content:
            openai_messages.append({"role": msg.role, "content": msg.content})

    # Inject user context into the first user message on first turn.
    # Done before attachment/URL injection so the context prefix lands at
    # the very start of the message content.
    user_message_for_transcript = message
    if should_inject_user_context:
        # Mirror the SDK path's "model knows about budget" behaviour by
        # injecting a per-turn ``<budget_context>`` block — the baseline
        # path has no equivalent of the SDK CLI's native
        # ``max_budget_usd`` running-cost reminder, so this is the only
        # signal the model gets about how much USD spend headroom the
        # user has left.
        budget_ctx = await build_budget_ctx(
            user_id=user_id,
            default_daily_cost_limit=config.daily_cost_limit_microdollars,
            default_weekly_cost_limit=config.weekly_cost_limit_microdollars,
        )
        prefixed = await inject_user_context(
            understanding,
            message or "",
            session_id,
            session.messages,
            budget_ctx=budget_ctx,
            user_id=user_id,
        )
        if prefixed is not None:
            # Reverse scan so we update the current turn's user message, not
            # the first (oldest) one when pending messages were drained.
            for msg in reversed(openai_messages):
                if msg["role"] == "user":
                    msg["content"] = prefixed
                    break
            user_message_for_transcript = prefixed
        else:
            logger.warning("[Baseline] No user message found for context injection")

    # Now that ``inject_user_context`` has wrapped + persisted the
    # original turn-starting send into its row, fold pending into the
    # model's current-turn input AND persist each pending message as
    # its own raw-text user row.  Three things to keep in sync:
    #
    #  1. ``openai_messages`` (what the model sees this round) — append
    #     each pending as a separate user entry, matching the mid-turn
    #     drain pattern below.
    #  2. ``message`` (used for transcript/title fallbacks) — combine.
    #  3. ``session.messages`` + DB — one raw row per pending via
    #     ``persist_pending_as_user_rows(transcript_builder=None)``.
    #
    # Order matters: persist must run after inject so inject targets
    # the routes.py-saved row, not a pending row.
    if drained_at_start_pending:
        # Persist FIRST.  Only fold pending into the model's prompt
        # (``message`` for transcript fallback + ``openai_messages`` for
        # the live LLM call) when persistence succeeded — if the helper
        # rolled back and re-queued the pending into Redis, leaving
        # those untouched ensures the NEXT turn's drain doesn't
        # double-combine (re-queued pending + combined-from-this-turn)
        # into the model's context.
        persisted_ok = await persist_pending_as_user_rows(
            session,
            None,
            drained_at_start_pending,
            log_prefix="[Baseline]",
        )
        if persisted_ok:
            message = combine_pending_with_current(
                drained_at_start_pending,
                message,
                request_arrival_at=request_arrival_at,
            )
            for pm in drained_at_start_pending:
                openai_messages.append(format_pending_as_user_message(pm))

    # Inject Graphiti warm context into the current turn's user message (not
    # the system prompt) so the system prompt stays static and cacheable.
    # warm_ctx is already wrapped in <temporal_context>.
    # Appended AFTER user_context so <user_context> stays at the very start.
    # Reverse scan so we update the current turn's user message, not the
    # oldest one when pending messages were drained.
    if warm_ctx:
        for msg in reversed(openai_messages):
            if msg["role"] == "user":
                existing = msg.get("content", "")
                if isinstance(existing, str):
                    msg["content"] = f"{existing}\n\n{warm_ctx}"
                break
        # Do NOT append warm_ctx to user_message_for_transcript — it would
        # persist stale temporal context into the transcript for future turns.

    # Inject the per-turn ``<builder_context>`` prefix when the session is
    # bound to a graph via ``metadata.builder_graph_id``.  Runs on every
    # user turn (not just the first) so the LLM always sees the live graph
    # snapshot — if the user edits the graph between turns, the next turn
    # carries the updated nodes/links. Only version + nodes + links here;
    # the static guide + graph id live in the system prompt via
    # ``build_builder_system_prompt_suffix`` (session-stable, prompt-cached).
    # Prepended AFTER any <user_context>/<memory_context>/<env_context> blocks
    # — same trust tier as those server-injected prefixes. Not persisted to
    # the transcript: the snapshot is stale-by-definition after the turn ends.
    if is_user_message and session.metadata.builder_graph_id:
        builder_block = await build_builder_context_turn_prefix(session, user_id)
        if builder_block:
            for msg in reversed(openai_messages):
                if msg["role"] == "user":
                    existing = msg.get("content", "")
                    if isinstance(existing, str):
                        msg["content"] = builder_block + existing
                    break

    # Append user message to transcript.
    # Always append when the message is present and is from the user,
    # even on duplicate-suppressed retries (is_new_message=False).
    # The loaded transcript may be stale (uploaded before the previous
    # attempt stored this message), so skipping it would leave the
    # transcript without the user turn, creating a malformed
    # assistant-after-assistant structure when the LLM reply is added.
    if message and is_user_message:
        transcript_builder.append_user(content=user_message_for_transcript or message)

    # --- File attachments (feature parity with SDK path) ---
    working_dir: str | None = None
    attachment_hint = ""
    image_blocks: list[dict[str, Any]] = []
    if file_ids and user_id:
        working_dir = tempfile.mkdtemp(prefix=f"copilot-baseline-{session_id[:8]}-")
        attachment_hint, image_blocks = await _prepare_baseline_attachments(
            file_ids, user_id, session_id, working_dir
        )

    # --- URL context ---
    context_hint = ""
    if context and context.get("url"):
        url = context["url"]
        content_text = context.get("content", "")
        if content_text:
            context_hint = (
                f"\n[The user shared a URL: {url}\nContent:\n{content_text[:8000]}]"
            )
        else:
            context_hint = f"\n[The user shared a URL: {url}]"

    # Append attachment + context hints and image blocks to the last user
    # message in a single reverse scan.
    extra_hint = attachment_hint + context_hint
    if extra_hint or image_blocks:
        for i in range(len(openai_messages) - 1, -1, -1):
            if openai_messages[i].get("role") == "user":
                existing = openai_messages[i].get("content", "")
                if isinstance(existing, str):
                    text = existing + "\n" + extra_hint if extra_hint else existing
                    if image_blocks:
                        parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
                        for img in image_blocks:
                            parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": (
                                            f"data:{img['source']['media_type']};"
                                            f"base64,{img['source']['data']}"
                                        )
                                    },
                                }
                            )
                        openai_messages[i]["content"] = parts
                    else:
                        openai_messages[i]["content"] = text
                break

    disabled_tool_groups: list[ToolGroup] = []
    if not graphiti_enabled:
        disabled_tool_groups.append("graphiti")
    tools = get_available_tools(disabled_groups=disabled_tool_groups)

    # --- Permission filtering ---
    if permissions is not None:
        tools = _filter_tools_by_permissions(tools, permissions)

    # Pre-mark cache_control on the last tool schema once per session.  The
    # tool set is static within a request, so doing this here (instead of in
    # _baseline_llm_caller) avoids re-copying ~43 tool dicts on every LLM
    # round of the tool-call loop.
    #
    # Applies to Anthropic AND Moonshot routes — OpenAI/Grok/Gemini 400
    # on the unknown ``cache_control`` field inside tool definitions, so
    # the gate stays narrow (see :func:`_supports_prompt_cache_markers`).
    if _supports_prompt_cache_markers(active_model):
        tools = cast(
            list[ChatCompletionToolParam], _mark_tools_with_cache_control(tools)
        )

    # Propagate execution context so tool handlers can read session-level flags.
    set_execution_context(
        user_id,
        session,
        sandbox=e2b_sandbox,
        sdk_cwd=working_dir,
        permissions=permissions,
    )

    yield StreamStart(messageId=message_id, sessionId=session_id)

    # Propagate user/session context to Langfuse so all LLM calls within
    # this request are grouped under a single trace with proper attribution.
    _trace_ctx: Any = None
    try:
        _trace_ctx = propagate_attributes(
            user_id=user_id,
            session_id=session_id,
            trace_name="copilot-baseline",
            tags=["baseline"],
        )
        _trace_ctx.__enter__()
    except Exception:
        logger.warning("[Baseline] Langfuse trace context setup failed")

    _stream_error = False  # Track whether an error occurred during streaming
    state = _BaselineStreamState(model=active_model)

    # Bind extracted module-level callbacks to this request's state/session
    # using functools.partial so they satisfy the Protocol signatures.
    _bound_llm_caller = partial(_baseline_llm_caller, state=state)

    # ``session`` is reassigned after each mid-turn ``persist_session_safe``
    # call (``upsert_chat_session`` returns a fresh ``model_copy``).  Holding
    # the object via ``partial(session=session)`` would pin tool executions
    # to the *original* object — any post-persist ``session.successful_agent_runs``
    # mutation from a run_agent tool call would then land on the stale copy
    # and be lost on the final persist.  Wrap in a 1-element holder and read
    # the current binding lazily so the executor always sees the latest session.
    _session_holder: list[ChatSession] = [session]

    async def _bound_tool_executor(
        tool_call: LLMToolCall, tools: Sequence[Any]
    ) -> ToolCallResult:
        return await _baseline_tool_executor(
            tool_call,
            tools,
            state=state,
            user_id=user_id,
            session=_session_holder[0],
        )

    _bound_conversation_updater = partial(
        _baseline_conversation_updater,
        transcript_builder=transcript_builder,
        model=active_model,
        state=state,
    )

    # Run the tool-call loop concurrently with the event consumer so
    # ``StreamReasoning*`` / ``StreamText*`` deltas emitted inside
    # ``_baseline_llm_caller`` reach the SSE wire DURING the upstream LLM
    # stream instead of only at iteration boundaries.  Any reasoning route
    # that streams for several minutes per round (extended thinking on
    # Anthropic / Moonshot / future providers) would otherwise freeze the
    # UI for the whole window before flushing the backlog in one burst.
    loop_result_holder: list[Any] = [None]
    loop_task: asyncio.Task[None] | None = None
    # Length of ``state.assistant_text`` at the end of the last non-final
    # yield — used as an anchor by the budget-exhausted fallback to check
    # whether the *terminal* round produced any visible text, not the whole
    # turn. Without this, earlier-round chatter would suppress a fallback
    # that should fire.
    text_len_before_final_round: list[int] = [0]

    async def _run_tool_call_loop() -> None:
        # Read/write the current session via ``_session_holder`` so this
        # closure doesn't need to ``nonlocal session`` — pyright can't narrow
        # the outer ``session: ChatSession | None`` through a nested scope,
        # but the holder is typed non-optional after the preflight guard
        # above.
        try:
            max_tool_rounds = config.agent_max_turns
            async for loop_result in tool_call_loop(
                messages=openai_messages,
                tools=tools,
                llm_call=_bound_llm_caller,
                execute_tool=_bound_tool_executor,
                update_conversation=_bound_conversation_updater,
                max_iterations=max_tool_rounds,
                last_iteration_message=_LAST_ITERATION_HINT,
            ):
                loop_result_holder[0] = loop_result
                # Inject any messages the user queued while the turn was
                # running.  ``tool_call_loop`` mutates ``openai_messages``
                # in-place, so appending here means the model sees the new
                # messages on its next LLM call.
                #
                # IMPORTANT: skip when the loop has already finished (no
                # more LLM calls are coming).  ``tool_call_loop`` yields
                # a final ``ToolCallLoopResult`` on both paths:
                #   - natural finish: ``finished_naturally=True``
                #   - hit max_iterations: ``finished_naturally=False``
                #                         and ``iterations >= max_iterations``
                # In either case the loop is about to return on the next
                # ``async for`` step, so draining here would silently
                # lose the message (the user sees 202 but the model never
                # reads the text).  Those messages stay in the buffer and
                # get picked up at the start of the next turn.
                is_final_yield = (
                    loop_result.finished_naturally
                    or loop_result.iterations >= max_tool_rounds
                )
                if is_final_yield:
                    continue
                # Non-final yield: the next round may be the last one, so
                # record where ``assistant_text`` ends now.  If that next
                # round hits the budget without adding any text, the outer
                # fallback uses this anchor to detect a silent finish.
                text_len_before_final_round[0] = len(state.assistant_text)
                try:
                    pending = await drain_pending_messages(session_id)
                except Exception:
                    logger.warning(
                        "[Baseline] mid-loop drain_pending_messages failed for "
                        "session %s",
                        session_id,
                        exc_info=True,
                    )
                    pending = []
                if pending:
                    # Flush any buffered assistant/tool messages from completed
                    # rounds into session.messages BEFORE appending the pending
                    # user message.  ``_baseline_conversation_updater`` only
                    # records assistant+tool rounds into ``state.session_messages``
                    # — they are normally batch-flushed in the finally block.
                    # Without this in-order flush, the mid-loop pending user
                    # message lands before the preceding round's assistant/tool
                    # entries, producing chronologically-wrong session.messages
                    # on persist (user interposed between an assistant tool_call
                    # and its tool-result), which breaks OpenAI tool-call ordering
                    # invariants on the next turn's replay.
                    #
                    # Also persist any assistant text from text-only rounds (rounds
                    # with no tool calls, which ``_baseline_conversation_updater``
                    # does NOT record in session_messages).  If we only update
                    # ``_flushed_assistant_text_len`` without persisting the text,
                    # that text is silently lost: the finally block only appends
                    # assistant_text[_flushed_assistant_text_len:], so text generated
                    # before this drain never reaches session.messages.
                    recorded_text = "".join(
                        m.content or ""
                        for m in state.session_messages
                        if m.role == "assistant"
                    )
                    unflushed_text = state.assistant_text[
                        state._flushed_assistant_text_len :
                    ]
                    text_only_text = (
                        unflushed_text[len(recorded_text) :]
                        if unflushed_text.startswith(recorded_text)
                        else unflushed_text
                    )
                    current_session = _session_holder[0]
                    if text_only_text.strip():
                        current_session.messages.append(
                            ChatMessage(role="assistant", content=text_only_text)
                        )
                    for _buffered in state.session_messages:
                        current_session.messages.append(_buffered)
                    state.session_messages.clear()
                    # Record how much assistant_text has been covered by the
                    # structured entries just flushed, so the finally block's
                    # final-text dedup doesn't re-append rounds already persisted.
                    state._flushed_assistant_text_len = len(state.assistant_text)

                    # Persist the assistant/tool flush BEFORE the pending append
                    # so a later pending-persist failure can roll back the
                    # pending rows without also discarding LLM output.
                    current_session = await persist_session_safe(
                        current_session, "[Baseline]"
                    )
                    # ``upsert_chat_session`` may return a *new* ``ChatSession``
                    # instance (e.g. when a concurrent title update has written a
                    # newer title to Redis, it returns ``session.model_copy``).
                    # Keep ``_session_holder`` in sync so subsequent tool rounds
                    # executed via ``_bound_tool_executor`` see the fresh session
                    # — any tool-side mutations on the stale object would be
                    # discarded when the new one is persisted in the ``finally``.
                    _session_holder[0] = current_session

                    # ``format_pending_as_user_message`` embeds file attachments
                    # and context URL/page content into the content string so
                    # the in-session transcript is a faithful copy of what the
                    # model actually saw.  We also mirror each push into
                    # ``openai_messages`` so the model's next LLM round sees it.
                    #
                    # Pre-compute the formatted dicts once so both the openai
                    # messages append and the content_of lookup inside the
                    # shared helper use the same string — and so ``on_rollback``
                    # can trim ``openai_messages`` to the recorded anchor.
                    formatted_by_pm = {
                        id(pm): format_pending_as_user_message(pm) for pm in pending
                    }
                    _openai_anchor = len(openai_messages)
                    for pm in pending:
                        openai_messages.append(formatted_by_pm[id(pm)])

                    def _trim_openai_on_rollback(_session_anchor: int) -> None:
                        del openai_messages[_openai_anchor:]

                    await persist_pending_as_user_rows(
                        current_session,
                        transcript_builder,
                        pending,
                        log_prefix="[Baseline]",
                        content_of=lambda pm: formatted_by_pm[id(pm)]["content"],
                        on_rollback=_trim_openai_on_rollback,
                    )
        finally:
            # Always post the sentinel so the outer consumer exits — even if
            # ``tool_call_loop`` raised.  ``_baseline_llm_caller``'s own
            # finally block has already pushed ``StreamReasoningEnd`` /
            # ``StreamTextEnd`` / ``StreamFinishStep`` at this point, so the
            # sentinel only terminates the consumer; it does not suppress
            # any still-unflushed events.
            state.pending_events.put_nowait(None)

    loop_task = asyncio.create_task(_run_tool_call_loop())
    try:
        while True:
            evt = await state.pending_events.get()
            if evt is None:
                break
            yield evt
        # Sentinel received — surface any exception the inner task hit.
        await loop_task
        loop_result = loop_result_holder[0]
        # Budget was reached when iterations hit the configured cap. This
        # covers both exit paths out of ``tool_call_loop``:
        #   - ``finished_naturally=True``: the last iteration ran with
        #     ``tools=[]`` and the model returned text (may be empty)
        #   - ``finished_naturally=False``: a non-compliant model still
        #     emitted tool calls despite the empty tool list, so the loop
        #     fell through the ``while`` guard
        # Either way, we check the terminal round's text contribution — an
        # empty one means the user got no explanation and we need to emit
        # the fallback notice.
        budget_reached = bool(
            loop_result and loop_result.iterations >= config.agent_max_turns
        )
        if budget_reached:
            if loop_result and not loop_result.finished_naturally:
                logger.warning(
                    "[Baseline] Hit %d-round tool budget without natural finish; "
                    "ending turn gracefully",
                    loop_result.iterations,
                )
            terminal_round_text = state.assistant_text[text_len_before_final_round[0] :]
            fallback_events, fallback_text = _build_budget_exhausted_fallback_events(
                terminal_round_text
            )
            for evt in fallback_events:
                yield evt
            state.assistant_text += fallback_text
    except Exception as e:
        _stream_error = True
        error_msg = str(e) or type(e).__name__
        logger.error("[Baseline] Streaming error: %s", error_msg, exc_info=True)
        # Drain any queued tail events (reasoning/text close + finish step)
        # that ``_baseline_llm_caller``'s finally block pushed before the
        # sentinel arrived — without this the frontend would be missing the
        # matching end / finish parts for the partial round.
        while not state.pending_events.empty():
            evt = state.pending_events.get_nowait()
            if evt is not None:
                yield evt
        yield StreamError(errorText=error_msg, code="baseline_error")
        # Still persist whatever we got
    finally:
        # Cancel the inner task if we're unwinding early (client disconnect,
        # unexpected error in the consumer) so it doesn't keep streaming
        # tokens into a dead queue.
        if loop_task is not None and not loop_task.done():
            loop_task.cancel()
            try:
                await loop_task
            except (asyncio.CancelledError, Exception):
                pass
        # Re-sync the outer ``session`` binding in case the inner task
        # reassigned it via a mid-loop ``persist_session_safe`` call.
        session = _session_holder[0]

        # In-flight tool-call announcements are only meaningful for the
        # current turn; clear at the top of the outer finally so the next
        # turn starts with a clean scratch buffer even if one of the
        # awaited cleanup steps below (usage persistence, session upsert,
        # transcript upload) raises.  The buffer is a process-local scratch
        # set — if we leak it into the next turn the guide-read guard would
        # observe a phantom in-flight call and skip its gate, so this must
        # run unconditionally.
        session.clear_inflight_tool_calls()

        # Pending messages are drained atomically at turn start and
        # between tool rounds, so there's nothing to clear in finally.
        # Any message pushed after the final drain window stays in the
        # buffer and gets picked up at the start of the next turn.

        # Set cost attributes on OTEL span before closing
        if _trace_ctx is not None:
            try:
                span = otel_trace.get_current_span()
                if span and span.is_recording():
                    span.set_attribute(
                        "gen_ai.usage.prompt_tokens", state.turn_prompt_tokens
                    )
                    span.set_attribute(
                        "gen_ai.usage.completion_tokens",
                        state.turn_completion_tokens,
                    )
                    if state.cost_usd is not None:
                        span.set_attribute("gen_ai.usage.cost_usd", state.cost_usd)
            except Exception:
                logger.debug("[Baseline] Failed to set OTEL cost attributes")
            try:
                _trace_ctx.__exit__(None, None, None)
            except Exception:
                logger.warning("[Baseline] Langfuse trace context teardown failed")

        # Fallback: estimate tokens via tiktoken when the provider does
        # not honour stream_options={"include_usage": True}.
        # ``openai_messages`` is mutated in-place by tool_call_loop so it
        # contains the full accumulated conversation (system + history + all
        # tool-call rounds) by this point — the estimate covers the complete
        # prompt, not just the final round.
        # Skip fallback when an error occurred and no output was produced —
        # charging rate-limit tokens for completely failed requests is unfair.
        if (
            state.turn_prompt_tokens == 0
            and state.turn_completion_tokens == 0
            and not (_stream_error and not state.assistant_text)
        ):
            state.turn_prompt_tokens = max(
                estimate_token_count(openai_messages, model=active_model), 1
            )
            state.turn_completion_tokens = estimate_token_count_str(
                state.assistant_text, model=active_model
            )
            logger.info(
                "[Baseline] No streaming usage reported; estimated tokens: "
                "prompt=%d, completion=%d",
                state.turn_prompt_tokens,
                state.turn_completion_tokens,
            )
        # Safety net: recover cost from rate card if usage chunk was dropped (truncated SSE).
        # OR mode skips recovery — OR's markup differs from raw Anthropic pricing.
        if state.cost_usd is None and not config.openrouter_active:
            recovered = compute_anthropic_cost_usd(
                model=active_model,
                prompt_tokens=state.turn_prompt_tokens,
                completion_tokens=state.turn_completion_tokens,
                cache_read_tokens=state.turn_cache_read_tokens,
                cache_creation_tokens=state.turn_cache_creation_tokens,
                cache_ttl=config.baseline_prompt_cache_ttl,
            )
            if recovered is not None:
                state.cost_usd = recovered

        # OAI-compat prompt_tokens includes cache buckets; subtract to keep the three buckets disjoint.
        uncached_prompt = max(
            0,
            state.turn_prompt_tokens
            - state.turn_cache_read_tokens
            - state.turn_cache_creation_tokens,
        )
        await persist_and_record_usage(
            session=session,
            user_id=user_id,
            prompt_tokens=uncached_prompt,
            completion_tokens=state.turn_completion_tokens,
            cache_read_tokens=state.turn_cache_read_tokens,
            cache_creation_tokens=state.turn_cache_creation_tokens,
            log_prefix="[Baseline]",
            cost_usd=state.cost_usd,
            model=active_model,
            provider="open_router" if config.openrouter_active else "anthropic",
        )

        # Persist structured tool-call history (assistant + tool messages)
        # collected by the conversation updater, then the final text response.
        for msg in state.session_messages:
            session.messages.append(msg)
        # Append the final assistant text (from the last LLM call that had
        # no tool calls, i.e. the natural finish).  Only add it if the
        # conversation updater didn't already record it as part of a
        # tool-call round (which would have empty response_text).
        # Only consider assistant text produced AFTER the last mid-loop
        # flush.  ``_flushed_assistant_text_len`` tracks the prefix already
        # persisted via structured session_messages during mid-loop pending
        # drains; including it here would duplicate those rounds.
        final_text = state.assistant_text[state._flushed_assistant_text_len :]
        if state.session_messages:
            # Strip text already captured in tool-call round messages
            recorded = "".join(
                m.content or "" for m in state.session_messages if m.role == "assistant"
            )
            if final_text.startswith(recorded):
                final_text = final_text[len(recorded) :]
        if final_text.strip():
            session.messages.append(ChatMessage(role="assistant", content=final_text))
        try:
            await upsert_chat_session(session)
        except Exception as persist_err:
            logger.error("[Baseline] Failed to persist session: %s", persist_err)

        # --- Graphiti: ingest conversation turn for temporal memory ---
        if graphiti_enabled and user_id and message and is_user_message:
            from backend.copilot.graphiti.ingest import enqueue_conversation_turn

            # Pass only the final assistant reply (after stripping tool-loop
            # chatter) so derived-finding distillation sees the substantive
            # response, not intermediate tool-planning text.
            _ingest_task = asyncio.create_task(
                enqueue_conversation_turn(
                    user_id,
                    session_id,
                    message,
                    assistant_msg=final_text if state else "",
                )
            )
            _background_tasks.add(_ingest_task)
            _ingest_task.add_done_callback(_background_tasks.discard)

        # --- Upload transcript for next-turn continuity ---
        # Backfill partial assistant text that wasn't recorded by the
        # conversation updater (e.g. when the stream aborted mid-round).
        # Without this, mode-switching after a failed turn would lose
        # the partial assistant response from the transcript.
        if _stream_error and state.assistant_text:
            if transcript_builder.last_entry_type != "assistant":
                transcript_builder.append_assistant(
                    content_blocks=[{"type": "text", "text": state.assistant_text}],
                    model=active_model,
                    stop_reason=STOP_REASON_END_TURN,
                )

        if user_id and should_upload_transcript(user_id, transcript_upload_safe):
            # Watermark = the next uncovered DB sequence (= max non-reasoning
            # sequence + 1).  Stays accurate on cap-engaged sessions because
            # the loaded window always includes the most-recent rows; the
            # max(sequence) is the last assistant just appended this turn.
            await _upload_final_transcript(
                user_id=user_id,
                session_id=session_id,
                transcript_builder=transcript_builder,
                session_msg_count=next_uncovered_sequence(session.messages),
            )

        # Clean up the ephemeral working directory used for file attachments.
        if working_dir is not None:
            shutil.rmtree(working_dir, ignore_errors=True)

    # Yield usage and finish AFTER try/finally (not inside finally).
    # PEP 525 prohibits yielding from finally in async generators during
    # aclose() — doing so raises RuntimeError on client disconnect.
    # On GeneratorExit the client is already gone, so unreachable yields
    # are harmless; on normal completion they reach the SSE stream.
    if state.turn_prompt_tokens > 0 or state.turn_completion_tokens > 0:
        # Report uncached prompt tokens to match what was billed — both
        # cache_read and cache_creation are excluded so the three
        # buckets emitted on ``StreamUsage`` are disjoint and the
        # frontend can sum them without double-counting cache writes.
        billed_prompt = max(
            0,
            state.turn_prompt_tokens
            - state.turn_cache_read_tokens
            - state.turn_cache_creation_tokens,
        )
        yield StreamUsage(
            prompt_tokens=billed_prompt,
            completion_tokens=state.turn_completion_tokens,
            total_tokens=billed_prompt + state.turn_completion_tokens,
            cache_read_tokens=state.turn_cache_read_tokens,
            cache_creation_tokens=state.turn_cache_creation_tokens,
        )

    yield StreamFinish()
