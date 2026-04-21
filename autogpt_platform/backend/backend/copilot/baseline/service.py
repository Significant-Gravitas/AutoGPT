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
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import orjson
from langfuse import propagate_attributes
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.completion_usage import PromptTokensDetails
from opentelemetry import trace as otel_trace

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
    _get_openai_client,
    _update_title_async,
    config,
    inject_user_context,
    strip_user_context_tags,
)
from backend.copilot.session_cleanup import prune_orphan_tool_calls
from backend.copilot.thinking_stripper import ThinkingStripper as _ThinkingStripper
from backend.copilot.token_tracking import persist_and_record_usage
from backend.copilot.tools import execute_tool, get_available_tools
from backend.copilot.tracking import track_user_message
from backend.copilot.transcript import (
    STOP_REASON_END_TURN,
    STOP_REASON_TOOL_USE,
    TranscriptDownload,
    detect_gap,
    download_transcript,
    extract_context_messages,
    strip_for_upload,
    upload_transcript,
    validate_transcript,
)
from backend.copilot.transcript_builder import TranscriptBuilder
from backend.data.db_accessors import chat_db
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

# Maximum number of tool-call rounds before forcing a text response.
_MAX_TOOL_ROUNDS = 30

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


def _extract_cache_creation_tokens(ptd: PromptTokensDetails) -> int:
    """Return cache-write token count from an OpenAI-compatible
    ``PromptTokensDetails``, handling provider-specific field names and
    SDK-version shape differences.

    Two shapes we care about:

    - **OpenRouter** (our primary baseline provider) streams the cache-write
      count as ``cache_write_tokens``.  Newer ``openai-python`` versions
      declare this as a typed attribute on ``PromptTokensDetails``; older
      versions expose it only in ``model_extra``.  Verified empirically:
      cold-cache request returns ``cache_write_tokens`` > 0, warm-cache
      request returns ``cached_tokens`` > 0 and ``cache_write_tokens`` = 0.
    - **Direct Anthropic API** uses ``cache_creation_input_tokens`` —
      never a typed attribute on the OpenAI SDK, always lives in
      ``model_extra``.

    Lookup order: typed attr → ``model_extra`` (OpenRouter) → ``model_extra``
    (Anthropic-native).  ``getattr`` handles both the typed-attr case
    (newer SDK) and the no-such-attr case (older SDK) — we can't only use
    ``model_extra`` because when the field is typed it's filtered out of
    ``model_extra``, leaving us at 0 on the modern happy path.
    """
    typed_val = getattr(ptd, "cache_write_tokens", None)
    if typed_val:
        return int(typed_val)
    extras = ptd.model_extra or {}
    return int(
        extras.get("cache_write_tokens")
        or extras.get("cache_creation_input_tokens")
        or 0
    )


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


def _resolve_baseline_model(tier: CopilotLlmModel | None) -> str:
    """Pick the model for the baseline path based on the per-request tier.

    The baseline (fast) and SDK (extended thinking) paths now share the
    same tier-based model resolution — only the *path* differs between
    "fast" and "extended_thinking".  ``'advanced'`` → Opus;
    ``'standard'`` / ``None`` → the config default (Sonnet).
    """
    from backend.copilot.service import resolve_chat_model

    return resolve_chat_model(tier)


@dataclass
class _BaselineStreamState:
    """Mutable state shared between the tool-call loop callbacks.

    Extracted from ``stream_chat_completion_baseline`` so that the callbacks
    can be module-level functions instead of deeply nested closures.
    """

    model: str = ""
    pending_events: list[StreamBaseResponse] = field(default_factory=list)
    assistant_text: str = ""
    text_block_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text_started: bool = False
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


def _is_anthropic_model(model: str) -> bool:
    """Return True if *model* routes to Anthropic (native or via OpenRouter).

    Cache-control markers on message content + the ``anthropic-beta`` header
    are Anthropic-specific.  OpenAI rejects the unknown ``cache_control``
    field with a 400 ("Extra inputs are not permitted") and Grok / other
    providers behave similarly.  OpenRouter strips unknown headers but
    passes through ``cache_control`` on the body regardless of provider —
    which would also fail when OpenRouter routes to a non-Anthropic model.

    Examples that return True:
      - ``anthropic/claude-sonnet-4-6`` (OpenRouter route)
      - ``claude-3-5-sonnet-20241022`` (direct Anthropic API)
      - ``anthropic.claude-3-5-sonnet`` (Bedrock-style)

    False for ``openai/gpt-4o``, ``google/gemini-2.5-pro``, ``xai/grok-4``
    etc.
    """
    lowered = model.lower()
    return "claude" in lowered or lowered.startswith("anthropic")


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
    state.pending_events.append(StreamStartStep())
    # Fresh thinking-strip state per round so a malformed unclosed
    # block in one LLM call cannot silently drop content in the next.
    state.thinking_stripper = _ThinkingStripper()

    round_text = ""
    try:
        client = _get_openai_client()
        # Cache markers are Anthropic-specific.  For OpenAI/Grok/other
        # providers, leaving them on would trigger a 400 ("Extra inputs
        # are not permitted" on cache_control).  Tools were precomputed
        # in stream_chat_completion_baseline via _mark_tools_with_cache_control
        # (only when the model was Anthropic), so on non-Anthropic routes
        # tools ship without cache_control on the last entry too.
        #
        # `extra_body` `usage.include=true` asks OpenRouter to embed the real
        # generation cost into the final usage chunk — required by the
        # cost-based rate limiter in routes.py.  Separate from the Anthropic
        # caching headers, always sent.
        is_anthropic = _is_anthropic_model(state.model)
        if is_anthropic:
            # Build the cached system dict once per session and splice it in
            # on each round.  The full ``messages`` list grows with every
            # tool call, so copying the entire list just to mutate index 0
            # scales with conversation length (sentry flagged this); this
            # splice touches only list slots, not message contents.
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
            extra_headers = _fresh_anthropic_caching_headers()
        else:
            final_messages = messages
            extra_headers = None
        typed_messages = cast(list[ChatCompletionMessageParam], final_messages)
        create_kwargs: dict[str, Any] = {
            "model": state.model,
            "messages": typed_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": _OPENROUTER_INCLUDE_USAGE_COST,
        }
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
                    if cost is not None:
                        state.cost_usd = (state.cost_usd or 0.0) + cost
                    elif (
                        "cost" not in (chunk.usage.model_extra or {})
                        and not state.cost_missing_logged
                    ):
                        # Field absent (non-OpenRouter route, or OpenRouter
                        # misconfigured) — warn once per stream so error
                        # monitoring picks up persistent misses without
                        # flooding. Invalid values already logged inside
                        # _extract_usage_cost, so no duplicate warning here.
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

                if delta.content:
                    emit = state.thinking_stripper.process(delta.content)
                    if emit:
                        if not state.text_started:
                            state.pending_events.append(
                                StreamTextStart(id=state.text_block_id)
                            )
                            state.text_started = True
                        round_text += emit
                        state.pending_events.append(
                            StreamTextDelta(id=state.text_block_id, delta=emit)
                        )

                if delta.tool_calls:
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

        # Flush any buffered text held back by the thinking stripper.
        tail = state.thinking_stripper.flush()
        if tail:
            if not state.text_started:
                state.pending_events.append(StreamTextStart(id=state.text_block_id))
                state.text_started = True
            round_text += tail
            state.pending_events.append(
                StreamTextDelta(id=state.text_block_id, delta=tail)
            )
        # Close text block
        if state.text_started:
            state.pending_events.append(StreamTextEnd(id=state.text_block_id))
            state.text_started = False
            state.text_block_id = str(uuid.uuid4())
    finally:
        # Always persist partial text so the session history stays consistent,
        # even when the stream is interrupted by an exception.
        state.assistant_text += round_text
        # Always emit StreamFinishStep to match the StreamStartStep,
        # even if an exception occurred during streaming.
        state.pending_events.append(StreamFinishStep())

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
        state.pending_events.append(
            StreamToolOutputAvailable(
                toolCallId=tool_call_id,
                toolName=tool_name,
                output=parse_error,
                success=False,
            )
        )
        return ToolCallResult(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=parse_error,
            is_error=True,
        )

    state.pending_events.append(
        StreamToolInputStart(toolCallId=tool_call_id, toolName=tool_name)
    )
    state.pending_events.append(
        StreamToolInputAvailable(
            toolCallId=tool_call_id,
            toolName=tool_name,
            input=tool_args,
        )
    )

    try:
        result: StreamToolOutputAvailable = await execute_tool(
            tool_name=tool_name,
            parameters=tool_args,
            user_id=user_id,
            session=session,
            tool_call_id=tool_call_id,
        )
        state.pending_events.append(result)
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
        state.pending_events.append(
            StreamToolOutputAvailable(
                toolCallId=tool_call_id,
                toolName=tool_name,
                output=error_output,
                success=False,
            )
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
            client=_get_openai_client(),
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
    if gap:
        _append_gap_to_builder(gap, transcript_builder)
        logger.info(
            "[Baseline] Filled gap: loaded %d transcript msgs + %d gap msgs from DB",
            restore.message_count,
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
        # Chronological combine: pending typed BEFORE this /stream
        # request's arrival go ahead of ``message``; race-path follow-ups
        # typed AFTER (queued while /stream was still processing) go
        # after.  See ``combine_pending_with_current`` for details.
        message = combine_pending_with_current(
            drained_at_start_pending,
            message,
            request_arrival_at=request_arrival_at,
        )
        # Update the in-memory content of the already-saved user message
        # and persist that update by sequence number.
        last_user_msg = next(
            (m for m in reversed(session.messages) if m.role == "user"), None
        )
        if last_user_msg is None or last_user_msg.sequence is None:
            # Defensive: routes.py always pre-saves the user message with a
            # sequence before dispatch, so this is unreachable under normal
            # flow. Raising instead of a warning-and-continue avoids silent
            # data loss (in-memory message diverges from the DB row, so the
            # queued chip would disappear from the UI after refresh without
            # a corresponding bubble).
            raise RuntimeError(
                f"[Baseline] Cannot persist turn-start pending injection: "
                f"last_user_msg={'missing' if last_user_msg is None else 'has no sequence'}"
            )
        last_user_msg.content = message
        await chat_db().update_message_content_by_sequence(
            session_id, last_user_msg.sequence, message
        )

    # Select model based on the per-request tier toggle (standard / advanced).
    # The path (fast vs extended_thinking) is already decided — we're in the
    # baseline (fast) path; ``mode`` is accepted for logging parity only.
    active_model = _resolve_baseline_model(model)

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
    system_prompt = base_system_prompt + SHARED_TOOL_NOTES + graphiti_supplement

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
    prior_context = extract_context_messages(transcript_download, session.messages)
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
        prefixed = await inject_user_context(
            understanding, message or "", session_id, session.messages
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

    tools = get_available_tools()

    # --- Permission filtering ---
    if permissions is not None:
        tools = _filter_tools_by_permissions(tools, permissions)

    # Pre-mark cache_control on the last tool schema once per session.  The
    # tool set is static within a request, so doing this here (instead of in
    # _baseline_llm_caller) avoids re-copying ~43 tool dicts on every LLM
    # round of the tool-call loop.
    #
    # Only apply to Anthropic routes — OpenAI/Grok/other providers would
    # 400 on the unknown ``cache_control`` field inside tool definitions.
    if _is_anthropic_model(active_model):
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

    try:
        loop_result = None
        async for loop_result in tool_call_loop(
            messages=openai_messages,
            tools=tools,
            llm_call=_bound_llm_caller,
            execute_tool=_bound_tool_executor,
            update_conversation=_bound_conversation_updater,
            max_iterations=_MAX_TOOL_ROUNDS,
        ):
            # Drain buffered events after each iteration (real-time streaming)
            for evt in state.pending_events:
                yield evt
            state.pending_events.clear()

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
                or loop_result.iterations >= _MAX_TOOL_ROUNDS
            )
            if is_final_yield:
                continue
            try:
                pending = await drain_pending_messages(session_id)
            except Exception:
                logger.warning(
                    "[Baseline] mid-loop drain_pending_messages failed for session %s",
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
                if text_only_text.strip():
                    session.messages.append(
                        ChatMessage(role="assistant", content=text_only_text)
                    )
                for _buffered in state.session_messages:
                    session.messages.append(_buffered)
                state.session_messages.clear()
                # Record how much assistant_text has been covered by the
                # structured entries just flushed, so the finally block's
                # final-text dedup doesn't re-append rounds already persisted.
                state._flushed_assistant_text_len = len(state.assistant_text)

                # Persist the assistant/tool flush BEFORE the pending append
                # so a later pending-persist failure can roll back the
                # pending rows without also discarding LLM output.
                session = await persist_session_safe(session, "[Baseline]")
                # ``upsert_chat_session`` may return a *new* ``ChatSession``
                # instance (e.g. when a concurrent title update has written a
                # newer title to Redis, it returns ``session.model_copy``).
                # Keep ``_session_holder`` in sync so subsequent tool rounds
                # executed via ``_bound_tool_executor`` see the fresh session
                # — any tool-side mutations on the stale object would be
                # discarded when the new one is persisted in the ``finally``.
                _session_holder[0] = session

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
                    session,
                    transcript_builder,
                    pending,
                    log_prefix="[Baseline]",
                    content_of=lambda pm: formatted_by_pm[id(pm)]["content"],
                    on_rollback=_trim_openai_on_rollback,
                )

        if loop_result and not loop_result.finished_naturally:
            limit_msg = (
                f"Exceeded {_MAX_TOOL_ROUNDS} tool-call rounds "
                "without a final response."
            )
            logger.error("[Baseline] %s", limit_msg)
            yield StreamError(
                errorText=limit_msg,
                code="baseline_tool_round_limit",
            )

    except Exception as e:
        _stream_error = True
        error_msg = str(e) or type(e).__name__
        logger.error("[Baseline] Streaming error: %s", error_msg, exc_info=True)
        # Close any open text block.  The llm_caller's finally block
        # already appended StreamFinishStep to pending_events, so we must
        # insert StreamTextEnd *before* StreamFinishStep to preserve the
        # protocol ordering:
        #   StreamStartStep -> StreamTextStart -> ...deltas... ->
        #   StreamTextEnd -> StreamFinishStep
        # Appending (or yielding directly) would place it after
        # StreamFinishStep, violating the protocol.
        if state.text_started:
            # Find the last StreamFinishStep and insert before it.
            insert_pos = len(state.pending_events)
            for i in range(len(state.pending_events) - 1, -1, -1):
                if isinstance(state.pending_events[i], StreamFinishStep):
                    insert_pos = i
                    break
            state.pending_events.insert(
                insert_pos, StreamTextEnd(id=state.text_block_id)
            )
        # Drain pending events in correct order
        for evt in state.pending_events:
            yield evt
        state.pending_events.clear()
        yield StreamError(errorText=error_msg, code="baseline_error")
        # Still persist whatever we got
    finally:
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
        # Count the full message list (system + history + turn) since
        # each API call sends the complete context window.
        # NOTE: This estimates one round's prompt tokens. Multi-round tool-calling
        # turns consume prompt tokens on each API call, so the total is underestimated.
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
        # Persist token usage to session and record for rate limiting.
        # When prompt_tokens_details.cached_tokens is reported, subtract
        # them from prompt_tokens to get the uncached count so the cost
        # breakdown stays accurate.
        uncached_prompt = state.turn_prompt_tokens
        if state.turn_cache_read_tokens > 0:
            uncached_prompt = max(
                0, state.turn_prompt_tokens - state.turn_cache_read_tokens
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
            await _upload_final_transcript(
                user_id=user_id,
                session_id=session_id,
                transcript_builder=transcript_builder,
                session_msg_count=len(session.messages),
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
        # Report uncached prompt tokens to match what was billed — cached tokens
        # are excluded so the frontend display is consistent with cost_usd.
        billed_prompt = max(0, state.turn_prompt_tokens - state.turn_cache_read_tokens)
        yield StreamUsage(
            prompt_tokens=billed_prompt,
            completion_tokens=state.turn_completion_tokens,
            total_tokens=billed_prompt + state.turn_completion_tokens,
            cache_read_tokens=state.turn_cache_read_tokens,
            cache_creation_tokens=state.turn_cache_creation_tokens,
        )

    yield StreamFinish()
