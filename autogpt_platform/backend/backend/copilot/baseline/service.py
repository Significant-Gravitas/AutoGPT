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
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import orjson
from langfuse import propagate_attributes
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from opentelemetry import trace as otel_trace

from backend.copilot.config import CopilotMode
from backend.copilot.context import get_workspace_manager, set_execution_context
from backend.copilot.db import update_message_content_by_sequence
from backend.copilot.graphiti.config import is_enabled_for_user
from backend.copilot.model import (
    ChatMessage,
    ChatSession,
    get_chat_session,
    maybe_append_user_message,
    upsert_chat_session,
)
from backend.copilot.pending_messages import (
    drain_pending_messages,
    format_pending_as_user_message,
)
from backend.copilot.prompting import get_baseline_supplement, get_graphiti_supplement
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
    _build_cacheable_system_prompt,
    _get_openai_client,
    _update_title_async,
    config,
)
from backend.copilot.token_tracking import persist_and_record_usage
from backend.copilot.tools import execute_tool, get_available_tools
from backend.copilot.tracking import track_user_message
from backend.copilot.transcript import (
    STOP_REASON_END_TURN,
    STOP_REASON_TOOL_USE,
    TranscriptDownload,
    download_transcript,
    upload_transcript,
    validate_transcript,
)
from backend.copilot.transcript_builder import TranscriptBuilder
from backend.data.understanding import format_understanding_for_prompt
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


def _resolve_baseline_model(mode: CopilotMode | None) -> str:
    """Pick the model for the baseline path based on the per-request mode.

    Only ``mode='fast'`` downgrades to the cheaper/faster model.  Any other
    value (including ``None`` and ``'extended_thinking'``) preserves the
    default model so that users who never select a mode don't get
    silently moved to the cheaper tier.
    """
    if mode == "fast":
        return config.fast_model
    return config.model


# Tag pairs to strip from baseline streaming output.  Different models use
# different tag names for their internal reasoning (Claude uses <thinking>,
# Gemini uses <internal_reasoning>, etc.).
_REASONING_TAG_PAIRS: list[tuple[str, str]] = [
    ("<thinking>", "</thinking>"),
    ("<internal_reasoning>", "</internal_reasoning>"),
]

# Longest opener — used to size the partial-tag buffer.
_MAX_OPEN_TAG_LEN = max(len(o) for o, _ in _REASONING_TAG_PAIRS)


class _ThinkingStripper:
    """Strip reasoning blocks from a stream of text deltas.

    Handles multiple tag patterns (``<thinking>``, ``<internal_reasoning>``,
    etc.) so the same stripper works across Claude, Gemini, and other models.

    Buffers just enough characters to detect a tag that may be split
    across chunks; emits text immediately when no tag is in-flight.
    Robust to single chunks that open and close a block, multiple
    blocks per stream, and tags that straddle chunk boundaries.
    """

    def __init__(self) -> None:
        self._buffer: str = ""
        self._in_thinking: bool = False
        self._close_tag: str = ""  # closing tag for the currently open block

    def _find_open_tag(self) -> tuple[int, str, str]:
        """Find the earliest opening tag in the buffer.

        Returns (position, open_tag, close_tag) or (-1, "", "") if none.
        """
        best_pos = -1
        best_open = ""
        best_close = ""
        for open_tag, close_tag in _REASONING_TAG_PAIRS:
            pos = self._buffer.find(open_tag)
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos = pos
                best_open = open_tag
                best_close = close_tag
        return best_pos, best_open, best_close

    def process(self, chunk: str) -> str:
        """Feed a chunk and return the text that is safe to emit now."""
        self._buffer += chunk
        out: list[str] = []
        while self._buffer:
            if self._in_thinking:
                end = self._buffer.find(self._close_tag)
                if end == -1:
                    keep = len(self._close_tag) - 1
                    self._buffer = self._buffer[-keep:] if keep else ""
                    return "".join(out)
                self._buffer = self._buffer[end + len(self._close_tag) :]
                self._in_thinking = False
                self._close_tag = ""
            else:
                start, open_tag, close_tag = self._find_open_tag()
                if start == -1:
                    # No opening tag; emit everything except a tail that
                    # could start a partial opener on the next chunk.
                    safe_end = len(self._buffer)
                    for keep in range(
                        min(_MAX_OPEN_TAG_LEN - 1, len(self._buffer)), 0, -1
                    ):
                        tail = self._buffer[-keep:]
                        if any(o[:keep] == tail for o, _ in _REASONING_TAG_PAIRS):
                            safe_end = len(self._buffer) - keep
                            break
                    out.append(self._buffer[:safe_end])
                    self._buffer = self._buffer[safe_end:]
                    return "".join(out)
                out.append(self._buffer[:start])
                self._buffer = self._buffer[start + len(open_tag) :]
                self._in_thinking = True
                self._close_tag = close_tag
        return "".join(out)

    def flush(self) -> str:
        """Return any remaining emittable text when the stream ends."""
        if self._in_thinking:
            # Unclosed thinking block — discard the buffered reasoning.
            self._buffer = ""
            return ""
        out = self._buffer
        self._buffer = ""
        return out


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
    cost_usd: float | None = None
    thinking_stripper: _ThinkingStripper = field(default_factory=_ThinkingStripper)
    session_messages: list[ChatMessage] = field(default_factory=list)
    # Tracks how much of ``assistant_text`` has already been flushed to
    # ``session.messages`` via mid-loop pending drains, so the ``finally``
    # block only appends the *new* assistant text (avoiding duplication of
    # round-1 text when round-1 entries were cleared from session_messages).
    _flushed_assistant_text_len: int = 0


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
    response = None  # initialized before try so finally block can access it
    try:
        client = _get_openai_client()
        typed_messages = cast(list[ChatCompletionMessageParam], messages)
        if tools:
            typed_tools = cast(list[ChatCompletionToolParam], tools)
            response = await client.chat.completions.create(
                model=state.model,
                messages=typed_messages,
                tools=typed_tools,
                stream=True,
                stream_options={"include_usage": True},
            )
        else:
            response = await client.chat.completions.create(
                model=state.model,
                messages=typed_messages,
                stream=True,
                stream_options={"include_usage": True},
            )
        tool_calls_by_index: dict[int, dict[str, str]] = {}

        async for chunk in response:
            if chunk.usage:
                state.turn_prompt_tokens += chunk.usage.prompt_tokens or 0
                state.turn_completion_tokens += chunk.usage.completion_tokens or 0

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
        # Extract OpenRouter cost from response headers (in finally so we
        # capture cost even when the stream errors mid-way — we already paid).
        # Accumulate across multi-round tool-calling turns.
        try:
            # Access undocumented _response attribute — same pattern as
            # extract_openrouter_cost() in blocks/llm.py.
            cost_header = response._response.headers.get("x-total-cost")  # type: ignore[attr-defined]
            if cost_header:
                cost = float(cost_header)
                if math.isfinite(cost) and cost >= 0:
                    state.cost_usd = (state.cost_usd or 0.0) + cost
        except (AttributeError, ValueError):
            pass

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


def is_transcript_stale(dl: TranscriptDownload | None, session_msg_count: int) -> bool:
    """Return ``True`` when a download doesn't cover the current session.

    A transcript is stale when it has a known ``message_count`` and that
    count doesn't reach ``session_msg_count - 1`` (i.e. the session has
    already advanced beyond what the stored transcript captures).
    Loading a stale transcript would silently drop intermediate turns,
    so callers should treat stale as "skip load, skip upload".

    An unknown ``message_count`` (``0``) is treated as **not stale**
    because older transcripts uploaded before msg_count tracking
    existed must still be usable.
    """
    if dl is None:
        return False
    if not dl.message_count:
        return False
    return dl.message_count < session_msg_count - 1


def should_upload_transcript(
    user_id: str | None, transcript_covers_prefix: bool
) -> bool:
    """Return ``True`` when the caller should upload the final transcript.

    Uploads require a logged-in user (for the storage key) *and* a
    transcript that covered the session prefix when loaded — otherwise
    we'd be overwriting a more complete version in storage with a
    partial one built from just the current turn.
    """
    return bool(user_id) and transcript_covers_prefix


async def _load_prior_transcript(
    user_id: str,
    session_id: str,
    session_msg_count: int,
    transcript_builder: TranscriptBuilder,
) -> bool:
    """Download and load the prior transcript into ``transcript_builder``.

    Returns ``True`` when the loaded transcript fully covers the session
    prefix; ``False`` otherwise (stale, missing, invalid, or download
    error).  Callers should suppress uploads when this returns ``False``
    to avoid overwriting a more complete version in storage.
    """
    try:
        dl = await download_transcript(user_id, session_id, log_prefix="[Baseline]")
    except Exception as e:
        logger.warning("[Baseline] Transcript download failed: %s", e)
        return False

    if dl is None:
        logger.debug("[Baseline] No transcript available")
        return False

    if not validate_transcript(dl.content):
        logger.warning("[Baseline] Downloaded transcript but invalid")
        return False

    if is_transcript_stale(dl, session_msg_count):
        logger.warning(
            "[Baseline] Transcript stale: covers %d of %d messages, skipping",
            dl.message_count,
            session_msg_count,
        )
        return False

    transcript_builder.load_previous(dl.content, log_prefix="[Baseline]")
    logger.info(
        "[Baseline] Loaded transcript: %dB, msg_count=%d",
        len(dl.content),
        dl.message_count,
    )
    return True


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
                content=content,
                message_count=session_msg_count,
                log_prefix="[Baseline]",
                skip_strip=True,
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
    # mid-loop drains missed them).  Atomic LPOP guarantees that a
    # concurrent push lands *after* the drain and stays queued for the
    # next turn instead of being lost.
    drained_at_start = await drain_pending_messages(session_id)
    if drained_at_start:
        logger.info(
            "[Baseline] Draining %d pending message(s) at turn start for session %s",
            len(drained_at_start),
            session_id,
        )
        for pm in drained_at_start:
            content = format_pending_as_user_message(pm)["content"]
            # Append directly — pending messages are atomically-popped from
            # Redis and are never stale-cache duplicates, so the
            # maybe_append_user_message dedup is wrong here.
            session.messages.append(ChatMessage(role="user", content=content))

    # Persist the drained pending messages (if any) plus the current user
    # message.  Wrap in try/except so a transient DB failure here does not
    # silently discard messages that were already popped from Redis — the
    # turn can still proceed using the in-memory session.messages, and a
    # later resume/replay will backfill from the DB on the next turn.
    try:
        session = await upsert_chat_session(session)
    except Exception as _persist_err:
        logger.warning(
            "[Baseline] Failed to persist session at turn start "
            "(pending drain may not be durable): %s",
            _persist_err,
        )

    # Select model based on the per-request mode.  'fast' downgrades to
    # the cheaper/faster model; everything else keeps the default.
    active_model = _resolve_baseline_model(mode)

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
    transcript_covers_prefix = True

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
        prompt_task = _build_cacheable_system_prompt(user_id)
    else:
        prompt_task = _build_cacheable_system_prompt(None)

    # Run download + prompt build concurrently — both are independent I/O
    # on the request critical path.  Use the pre-drain count so pending
    # messages drained at turn start don't spuriously trigger a transcript
    # load on an actual first turn.
    if user_id and _pre_drain_msg_count > 1:
        transcript_covers_prefix, (base_system_prompt, understanding) = (
            await asyncio.gather(
                _load_prior_transcript(
                    user_id=user_id,
                    session_id=session_id,
                    # Use pre-drain count so pending messages don't falsely
                    # mark the stored transcript as stale and prevent upload.
                    session_msg_count=_pre_drain_msg_count,
                    transcript_builder=transcript_builder,
                ),
                prompt_task,
            )
        )
    else:
        base_system_prompt, understanding = await prompt_task

    # Append user message to transcript after context injection below so the
    # transcript receives the prefixed message when user context is available.

    # Mirror any messages drained at turn start (see above) into the
    # transcript — otherwise the loaded prior transcript would be
    # missing them and a mid-turn upload could leave a malformed
    # assistant-after-assistant structure on the next turn.
    if drained_at_start:
        for pm in drained_at_start:
            transcript_builder.append_user(
                content=format_pending_as_user_message(pm)["content"]
            )

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
    system_prompt = base_system_prompt + get_baseline_supplement() + graphiti_supplement

    # Warm context: pre-load relevant facts from Graphiti on first turn.
    # Use the pre-drain count so pending messages drained at turn start
    # don't prevent warm context injection on an actual first turn.
    if graphiti_enabled and user_id and _pre_drain_msg_count <= 1:
        from backend.copilot.graphiti.context import fetch_warm_context

        warm_ctx = await fetch_warm_context(user_id, message or "")
        if warm_ctx:
            system_prompt += f"\n\n{warm_ctx}"

    # Compress context if approaching the model's token limit
    messages_for_context = await _compress_session_messages(
        session.messages, model=active_model
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
    # The prefixed content is also stored back into session.messages and the
    # transcript so that resumed sessions and the transcript both carry the
    # personalisation beyond the first request.
    user_message_for_transcript = message
    if should_inject_user_context and understanding:
        user_ctx = format_understanding_for_prompt(understanding)
        prefixed: str | None = None
        for msg in openai_messages:
            if msg["role"] == "user":
                prefixed = (
                    f"<user_context>\n{user_ctx}\n</user_context>\n\n{msg['content']}"
                )
                msg["content"] = prefixed
                break
        if prefixed is not None:
            # Persist the prefixed content so subsequent turns and --resume
            # retain the user context.
            # The user message was already saved to DB before context injection
            # (at ~line 932); update the DB record so the prefixed content
            # survives page reload.
            for idx, session_msg in enumerate(session.messages):
                if session_msg.role == "user":
                    session_msg.content = prefixed
                    await update_message_content_by_sequence(session_id, idx, prefixed)
                    break
            user_message_for_transcript = prefixed
        else:
            logger.warning("[Baseline] No user message found for context injection")

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
                f"\n[The user shared a URL: {url}\n" f"Content:\n{content_text[:8000]}]"
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
    _bound_tool_executor = partial(
        _baseline_tool_executor, state=state, user_id=user_id, session=session
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
            if loop_result is None:
                continue
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
                    "Mid-loop drain_pending_messages failed for session %s",
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
                for _buffered in state.session_messages:
                    session.messages.append(_buffered)
                state.session_messages.clear()
                # Record how much assistant_text has been covered by the
                # structured entries just flushed, so the finally block's
                # final-text dedup doesn't re-append rounds already persisted.
                state._flushed_assistant_text_len = len(state.assistant_text)

                for pm in pending:
                    # ``format_pending_as_user_message`` embeds file
                    # attachments and context URL/page content into the
                    # content string so the in-session transcript is
                    # a faithful copy of what the model actually saw.
                    formatted = format_pending_as_user_message(pm)
                    content_for_db = formatted["content"]
                    # Append directly — pending messages are atomically-popped
                    # from Redis and are never stale-cache duplicates, so the
                    # maybe_append_user_message dedup is wrong here and would
                    # cause openai_messages/transcript to diverge from session.
                    session.messages.append(
                        ChatMessage(role="user", content=content_for_db)
                    )
                    openai_messages.append(formatted)
                    transcript_builder.append_user(content=content_for_db)
                try:
                    await upsert_chat_session(session)
                except Exception as persist_err:
                    logger.warning(
                        "[Baseline] Failed to persist pending messages for "
                        "session %s: %s",
                        session_id,
                        persist_err,
                    )
                logger.info(
                    "[Baseline] Injected %d pending message(s) into "
                    "session %s mid-turn",
                    len(pending),
                    session_id,
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
        # NOTE: OpenRouter folds cached tokens into prompt_tokens, so we
        # cannot break out cache_read/cache_creation weights. Users on the
        # baseline path may be slightly over-counted vs the SDK path.
        await persist_and_record_usage(
            session=session,
            user_id=user_id,
            prompt_tokens=state.turn_prompt_tokens,
            completion_tokens=state.turn_completion_tokens,
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

            _ingest_task = asyncio.create_task(
                enqueue_conversation_turn(user_id, session_id, message)
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

        if user_id and should_upload_transcript(user_id, transcript_covers_prefix):
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
        yield StreamUsage(
            prompt_tokens=state.turn_prompt_tokens,
            completion_tokens=state.turn_completion_tokens,
            total_tokens=state.turn_prompt_tokens + state.turn_completion_tokens,
        )

    yield StreamFinish()
