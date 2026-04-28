"""Response adapter for converting Claude Agent SDK messages to Vercel AI SDK format.

This module provides the adapter layer that converts streaming messages from
the Claude Agent SDK into the Vercel AI SDK UI Stream Protocol format that
the frontend expects.
"""

import json
import logging
import time
import uuid
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    Message,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from backend.copilot.constants import FRIENDLY_TRANSIENT_MSG, is_transient_api_error
from backend.copilot.response_model import (
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
)

from .tool_adapter import MCP_TOOL_PREFIX, pop_pending_tool_output

logger = logging.getLogger(__name__)


# Coalescing thresholds for ``thinking_delta`` events on the SDK partial
# stream — matches the baseline window (see
# ``baseline/reasoning.py::_COALESCE_MIN_CHARS``).  Anthropic's extended-
# thinking channel emits ~1 event per token (~4,700 per Kimi K2.6 turn);
# a 64-char / 50 ms window halves the event rate vs 32/40 while staying
# well under the ~100 ms perceptual threshold.
_THINKING_COALESCE_MIN_CHARS = 64
_THINKING_COALESCE_MAX_INTERVAL_MS = 50.0


class SDKResponseAdapter:
    """Adapter for converting Claude Agent SDK messages to Vercel AI SDK format.

    This class maintains state during a streaming session to properly track
    text blocks, tool calls, and message lifecycle.
    """

    def __init__(
        self,
        message_id: str | None = None,
        session_id: str | None = None,
        *,
        render_reasoning_in_ui: bool = True,
    ):
        self.message_id = message_id or str(uuid.uuid4())
        self.session_id = session_id
        self.text_block_id = str(uuid.uuid4())
        self.has_started_text = False
        self.has_ended_text = False
        self.reasoning_block_id = str(uuid.uuid4())
        self.has_started_reasoning = False
        self.has_ended_reasoning = True
        self.render_reasoning_in_ui = render_reasoning_in_ui
        self.current_tool_calls: dict[str, dict[str, str]] = {}
        self.resolved_tool_calls: set[str] = set()
        self.step_open = False
        # Track whether any ``TextBlock`` was emitted after the most recent
        # tool_result.  Used at ``ResultMessage`` time to detect the
        # "thinking-only final turn" case — when Claude's last LLM call
        # produced only a ``ThinkingBlock`` (no text, no tool_use) the UI
        # hangs on the last tool result with a "Thought for Xs" label and
        # no response text.  We synthesize a short closing line in that
        # case so the turn renders as cleanly complete.
        self._text_since_last_tool_result = False
        self._any_tool_results_seen = False
        # --- Partial-message streaming state (CHAT_SDK_INCLUDE_PARTIAL_MESSAGES)
        # When ``include_partial_messages=True`` is set on
        # ``ClaudeAgentOptions``, the CLI emits raw Anthropic streaming
        # events (``content_block_start`` / ``content_block_delta`` /
        # ``content_block_stop``) as ``StreamEvent`` messages ahead of
        # each summary ``AssistantMessage``.  We consume those for
        # per-token wire emission and reconcile against the summary to
        # catch any tail content the partial stream missed (short blocks
        # the CLI emits summary-only, OpenRouter proxy quirks, encrypted
        # thinking).
        #
        self._block_types_by_index: dict[int, str] = {}
        # Running partial-stream buffers.  Summary AssistantMessages can
        # arrive *before* the corresponding ``content_block_stop`` event
        # (the CLI flushes the summary as soon as the block is complete
        # on the provider side, with the stop event following as a
        # separate frame).  Reconcile-by-index therefore can't rely on
        # completed-block queues — instead we maintain running buffers
        # of all partial output of each type, and each summary block of
        # that type consumes its prefix.  This also trivially handles
        # Kimi K2.6's pattern of emitting each content block as its own
        # summary AssistantMessage: Python list indices don't align
        # with Anthropic content_block indices, but per-type order does.
        self._partial_text_buffer: str = ""
        self._partial_thinking_buffer: str = ""
        # Coalescing buffer for ``thinking_delta`` — text_delta is
        # naturally coarser so we let it through unbuffered.
        self._pending_thinking_delta: str = ""
        self._pending_thinking_index: int | None = None
        self._last_thinking_flush_monotonic: float = 0.0

    @property
    def has_unresolved_tool_calls(self) -> bool:
        """True when there are tool calls that haven't received output yet."""
        return bool(self.current_tool_calls.keys() - self.resolved_tool_calls)

    def convert_message(self, sdk_message: Message) -> list[StreamBaseResponse]:
        """Convert a single SDK message to Vercel AI SDK format."""
        responses: list[StreamBaseResponse] = []

        if isinstance(sdk_message, SystemMessage):
            if sdk_message.subtype == "init":
                responses.append(
                    StreamStart(messageId=self.message_id, sessionId=self.session_id)
                )
                # Open the first step (matches non-SDK: StreamStart then StreamStartStep)
                responses.append(StreamStartStep())
                self.step_open = True
            elif sdk_message.subtype == "task_progress":
                # Emit a heartbeat so publish_chunk is called during long
                # sub-agent runs. Without this, the Redis stream and meta
                # key TTLs expire during gaps where no real chunks are
                # produced (task_progress events were previously silent).
                responses.append(StreamHeartbeat())

        elif isinstance(sdk_message, StreamEvent):
            # Raw Anthropic streaming events — only delivered when
            # ``include_partial_messages=True`` is set on
            # ``ClaudeAgentOptions`` (gated by
            # ``config.sdk_include_partial_messages``).  Drives per-token
            # emission of text + thinking; tool_use and other structural
            # events stay on the ``AssistantMessage`` path.
            self._handle_stream_event(sdk_message, responses)

        elif isinstance(sdk_message, AssistantMessage):
            # Flush any SDK built-in tool calls that didn't get a UserMessage
            # result (e.g. WebSearch, Read handled internally by the CLI).
            # BUT skip flush when this AssistantMessage is a parallel tool
            # continuation (contains only ToolUseBlocks) — the prior tools
            # are still executing concurrently and haven't finished yet.
            is_tool_only = all(isinstance(b, ToolUseBlock) for b in sdk_message.content)
            if not is_tool_only:
                self.flush_unresolved_tool_calls(responses)

            # After tool results, the SDK sends a new AssistantMessage for the
            # next LLM turn. Open a new step if the previous one was closed.
            if not self.step_open:
                responses.append(StreamStartStep())
                self.step_open = True

            # Hoist ThinkingBlocks to the front of the iteration so the UI
            # sees reasoning *before* the answer it produced — that's the
            # natural reading order and the way Anthropic models emit them.
            # OpenRouter passthrough providers (Moonshot/Kimi, DeepSeek)
            # often place ``reasoning`` after the visible text in the
            # response, which would make ``ReasoningCollapse`` render under
            # the assistant message instead of above it.  ToolUse and other
            # block types stay in their original relative order so tool
            # call sequences remain coherent.
            #
            # Note: when ``include_partial_messages=True`` is active the
            # per-token stream already emitted reasoning + text in their
            # natural on-the-wire order via ``_handle_stream_event``.  The
            # summary walk below falls through to ``_emit_text_tail`` /
            # ``_emit_thinking_tail`` which emit only the diff, preserving
            # that ordering without duplicating content.
            blocks_with_idx = sorted(
                enumerate(sdk_message.content),
                key=lambda pair: 0 if isinstance(pair[1], ThinkingBlock) else 1,
            )

            for block_index, block in blocks_with_idx:
                if isinstance(block, TextBlock):
                    # Reasoning and text are distinct UI parts; close any
                    # open reasoning block before opening text so the AI
                    # SDK transport doesn't merge them.
                    tail = self._text_tail_for_summary_block(block.text)
                    if tail:
                        self._end_reasoning_if_open(responses)
                        self._ensure_text_started(responses)
                        responses.append(
                            StreamTextDelta(id=self.text_block_id, delta=tail)
                        )
                        self._text_since_last_tool_result = True
                    elif block.text:
                        # Partial stream already emitted the full text.
                        self._text_since_last_tool_result = True

                elif isinstance(block, ThinkingBlock):
                    # Stream extended_thinking content as a reasoning
                    # block.  The Vercel AI SDK's ``useChat`` transport
                    # recognises ``reasoning-start`` / ``reasoning-delta``
                    # / ``reasoning-end`` events and accumulates them into
                    # a ``type: 'reasoning'`` UIMessage part the frontend
                    # renders via ``ReasoningCollapse`` (collapsed by
                    # default).  We also persist the text as a
                    # ``type: 'thinking'`` part in ``session.messages`` via
                    # ``_format_sdk_content_blocks``, so shared / reloaded
                    # sessions see the same reasoning.  Without streaming
                    # it live, extended_thinking turns that end
                    # thinking-only left the UI stuck on "Thought for Xs"
                    # with nothing rendered until a page refresh.
                    #
                    # When ``render_reasoning_in_ui=False`` the three
                    # reasoning helpers below (and the append) no-op, so
                    # the frontend sees a text-only stream AND no
                    # ``ChatMessage(role='reasoning')`` row is persisted
                    # (the row is only created by ``_dispatch_response``
                    # when ``StreamReasoningStart`` arrives, which is
                    # suppressed here).  Persistence of the thinking text
                    # into the SDK transcript via
                    # ``_format_sdk_content_blocks`` is unaffected — that
                    # feeds ``--resume`` continuity, not the UI.
                    #
                    # Flush any pending coalesce buffer to the wire BEFORE
                    # computing the tail — otherwise a summary that
                    # arrives between the last partial delta and the
                    # ``content_block_stop`` event (race: summary is
                    # flushed by the CLI as soon as the block is complete
                    # provider-side, with stop lagging as a separate
                    # frame) would see ``_partial_thinking_buffer``
                    # missing the pending prefix, and
                    # ``_thinking_tail_for_summary_block`` would emit the
                    # full block — duplicating the tail that
                    # ``_end_reasoning_if_open`` still drains on stop.
                    self._flush_pending_thinking(responses)
                    tail = self._thinking_tail_for_summary_block(block.thinking)
                    if tail:
                        self._end_text_if_open(responses)
                        self._ensure_reasoning_started(responses)
                        responses.append(
                            StreamReasoningDelta(
                                id=self.reasoning_block_id,
                                delta=tail,
                            )
                        )

                elif isinstance(block, ToolUseBlock):
                    self._end_text_if_open(responses)
                    self._end_reasoning_if_open(responses)

                    # Strip MCP prefix so frontend sees "find_block"
                    # instead of "mcp__copilot__find_block".
                    tool_name = block.name.strip().removeprefix(MCP_TOOL_PREFIX)

                    responses.append(
                        StreamToolInputStart(toolCallId=block.id, toolName=tool_name)
                    )
                    responses.append(
                        StreamToolInputAvailable(
                            toolCallId=block.id,
                            toolName=tool_name,
                            input=block.input,
                        )
                    )
                    self.current_tool_calls[block.id] = {"name": tool_name}

        elif isinstance(sdk_message, UserMessage):
            # UserMessage carries tool results back from tool execution.
            content = sdk_message.content
            blocks = content if isinstance(content, list) else []
            resolved_in_blocks: set[str] = set()

            sid = (self.session_id or "?")[:12]
            parent_id_preview = getattr(sdk_message, "parent_tool_use_id", None)
            logger.info(
                "[SDK] [%s] UserMessage: %d blocks, content_type=%s, "
                "parent_tool_use_id=%s",
                sid,
                len(blocks),
                type(content).__name__,
                parent_id_preview[:12] if parent_id_preview else "None",
            )

            for block in blocks:
                if isinstance(block, ToolResultBlock) and block.tool_use_id:
                    # Skip if already resolved (e.g. by flush) — the real
                    # result supersedes the empty flush, but re-emitting
                    # would confuse the frontend's state machine.
                    if block.tool_use_id in self.resolved_tool_calls:
                        continue
                    tool_info = self.current_tool_calls.get(block.tool_use_id, {})
                    tool_name = tool_info.get("name", "unknown")

                    # Prefer the stashed full output over the SDK's
                    # (potentially truncated) ToolResultBlock content.
                    # The SDK truncates large results, writing them to disk,
                    # which breaks frontend widget parsing.
                    output = pop_pending_tool_output(tool_name) or (
                        _extract_tool_output(block.content)
                    )

                    responses.append(
                        StreamToolOutputAvailable(
                            toolCallId=block.tool_use_id,
                            toolName=tool_name,
                            output=output,
                            success=not (block.is_error or False),
                        )
                    )
                    resolved_in_blocks.add(block.tool_use_id)

            # Handle SDK built-in tool results carried via parent_tool_use_id
            # instead of (or in addition to) ToolResultBlock content.
            parent_id = sdk_message.parent_tool_use_id
            if (
                parent_id
                and parent_id not in resolved_in_blocks
                and parent_id not in self.resolved_tool_calls
            ):
                tool_info = self.current_tool_calls.get(parent_id, {})
                tool_name = tool_info.get("name", "unknown")

                # Try stashed output first (from PostToolUse hook),
                # then tool_use_result dict, then string content.
                output = pop_pending_tool_output(tool_name)
                if not output:
                    tur = sdk_message.tool_use_result
                    if tur is not None:
                        output = _extract_tool_use_result(tur)
                if not output and isinstance(content, str) and content.strip():
                    output = content.strip()

                if output:
                    responses.append(
                        StreamToolOutputAvailable(
                            toolCallId=parent_id,
                            toolName=tool_name,
                            output=output,
                            success=True,
                        )
                    )
                    resolved_in_blocks.add(parent_id)

            self.resolved_tool_calls.update(resolved_in_blocks)
            if resolved_in_blocks:
                # A new tool_result just landed — reset the
                # "has the model emitted text since the last tool result?"
                # tracker so the thinking-only-final-turn guard at
                # ``ResultMessage`` time stays accurate.
                self._text_since_last_tool_result = False
                self._any_tool_results_seen = True

            # Close the current step after tool results — the next
            # AssistantMessage will open a new step for the continuation.
            if self.step_open:
                self._end_reasoning_if_open(responses)
                responses.append(StreamFinishStep())
                self.step_open = False

            # Narrate the gap between "tool returned" and "model emits its
            # next chunk". Usually sub-second, but with large tool outputs
            # or complex continuations it can stretch long enough for the
            # generic "Thinking…" copy to feel dead. The frontend replaces
            # it with actual content as soon as the next chunk lands.
            if resolved_in_blocks:
                responses.append(StreamStatus(message="Analyzing result\u2026"))

        elif isinstance(sdk_message, ResultMessage):
            self.flush_unresolved_tool_calls(responses)
            # SECRT-2252: surface ghost-finished sessions as errors instead of silent finishes.
            if sdk_message.subtype == "success" and self._is_empty_completion(
                sdk_message
            ):
                if self.step_open:
                    responses.append(StreamFinishStep())
                    self.step_open = False
                responses.append(
                    StreamError(
                        errorText="The model returned an empty response.",
                        code="empty_completion",
                    )
                )
                # Pair with StreamFinish so ``acc.stream_completed`` flips True
                # in ``_dispatch_response`` — without it the service-layer
                # post-stream branch mis-classifies the turn as "stopped by
                # user" and appends a STOPPED_BY_USER_MARKER on top of the
                # error marker.
                responses.append(StreamFinish())
                logger.warning(
                    "[SDK] [%s] Empty-success ResultMessage detected — "
                    "emitting stream error instead of silent finish",
                    (self.session_id or "?")[:12],
                )
                return responses
            # Thinking-only final turn guard: when the model's last LLM
            # call after a tool result produced only a ``ThinkingBlock``
            # (no ``TextBlock``, no ``ToolUseBlock``) the UI has nothing
            # to render after the tool output — it hangs on "Thought for
            # Xs" with no response text.  Synthesise a short closing line
            # so the turn visibly completes.  Condition: we've seen at
            # least one tool_result AND zero TextBlocks since.  The
            # prompt rule (``_USER_FOLLOW_UP_NOTE``'s closing clause)
            # asks the model to always end with text, but we can't rely
            # on it for extended_thinking / edge cases.
            if (
                self._any_tool_results_seen
                and not self._text_since_last_tool_result
                and sdk_message.subtype == "success"
            ):
                # UserMessage (tool_result) closed the last step, so we must
                # open a fresh one before emitting any text — the AI SDK v5
                # transport rejects text-delta chunks that aren't wrapped in
                # start-step / finish-step.
                if not self.step_open:
                    responses.append(StreamStartStep())
                    self.step_open = True
                # Close any open reasoning block first — text and reasoning
                # must not interleave on the wire (AI SDK v5 maps distinct
                # start/end events to distinct UI parts).
                self._end_reasoning_if_open(responses)
                self._ensure_text_started(responses)
                responses.append(
                    StreamTextDelta(
                        id=self.text_block_id,
                        delta="(Done — no further commentary.)",
                    )
                )
            self._end_text_if_open(responses)
            self._end_reasoning_if_open(responses)
            # Close the step before finishing.
            if self.step_open:
                responses.append(StreamFinishStep())
                self.step_open = False

            if sdk_message.subtype == "success":
                responses.append(StreamFinish())
            elif sdk_message.subtype in ("error", "error_during_execution"):
                raw_error = str(sdk_message.result or "Unknown error")
                if is_transient_api_error(raw_error):
                    error_text, code = FRIENDLY_TRANSIENT_MSG, "transient_api_error"
                else:
                    error_text, code = raw_error, "sdk_error"
                responses.append(StreamError(errorText=error_text, code=code))
                responses.append(StreamFinish())
            else:
                logger.warning(
                    f"Unexpected ResultMessage subtype: {sdk_message.subtype}"
                )
                responses.append(StreamFinish())

        else:
            logger.debug(f"Unhandled SDK message type: {type(sdk_message).__name__}")

        return responses

    def _is_empty_completion(self, msg: ResultMessage) -> bool:
        """True when a success ResultMessage carries no content at all.

        Detects the SDK's ghost-finished session: empty ``result``, zero
        ``output_tokens``, and nothing emitted on the wire this turn (no
        text, no reasoning, no tool calls).
        """
        if msg.result:
            return False
        if self.has_started_text or self.has_started_reasoning:
            return False
        if self.current_tool_calls:
            return False
        if self._any_tool_results_seen:
            return False
        usage = msg.usage or {}
        output_tokens = usage.get("output_tokens") or 0
        return output_tokens == 0

    def _ensure_text_started(self, responses: list[StreamBaseResponse]) -> None:
        """Start (or restart) a text block if needed."""
        if not self.has_started_text or self.has_ended_text:
            if self.has_ended_text:
                self.text_block_id = str(uuid.uuid4())
                self.has_ended_text = False
            responses.append(StreamTextStart(id=self.text_block_id))
            self.has_started_text = True

    def _end_text_if_open(self, responses: list[StreamBaseResponse]) -> None:
        """End the current text block if one is open."""
        if self.has_started_text and not self.has_ended_text:
            responses.append(StreamTextEnd(id=self.text_block_id))
            self.has_ended_text = True

    def _ensure_reasoning_started(self, responses: list[StreamBaseResponse]) -> None:
        """Start (or restart) a reasoning block if needed.

        Each ``ThinkingBlock`` the SDK emits gets its own streaming block
        so the frontend can render a new ``Reasoning`` part per LLM turn
        (rather than concatenating across the whole session).  Events
        are emitted unconditionally — the caller filters them out of the
        SSE wire when ``render_reasoning_in_ui=False`` but still feeds
        them through ``_dispatch_response`` so the session transcript
        keeps a ``role='reasoning'`` row.
        """
        if not self.has_started_reasoning or self.has_ended_reasoning:
            if self.has_ended_reasoning:
                self.reasoning_block_id = str(uuid.uuid4())
                self.has_ended_reasoning = False
            responses.append(StreamReasoningStart(id=self.reasoning_block_id))
            self.has_started_reasoning = True

    def _end_reasoning_if_open(self, responses: list[StreamBaseResponse]) -> None:
        """End the current reasoning block if one is open.

        Drains any buffered thinking_delta text so the tail isn't lost
        when the block closes before the coalesce window elapses.
        """
        if self.has_started_reasoning and not self.has_ended_reasoning:
            if self._pending_thinking_delta:
                responses.append(
                    StreamReasoningDelta(
                        id=self.reasoning_block_id,
                        delta=self._pending_thinking_delta,
                    )
                )
                self._partial_thinking_buffer += self._pending_thinking_delta
                self._pending_thinking_delta = ""
                self._pending_thinking_index = None
            responses.append(StreamReasoningEnd(id=self.reasoning_block_id))
            self.has_ended_reasoning = True

    # ------------------------------------------------------------------
    # Partial-message streaming (CHAT_SDK_INCLUDE_PARTIAL_MESSAGES)
    # ------------------------------------------------------------------

    def _reset_partial_stream_state(self) -> None:
        """Clear per-message partial-stream state.

        Anthropic's ``content_block`` indices are scoped to a single
        message — when a fresh ``message_start`` event arrives (new
        ``AssistantMessage`` turn) the maps must reset so indices from
        the previous message don't suppress genuine content in the new
        one.

        Also clears ``_partial_*_buffer``: multi-round turns (tool use)
        emit a ``message_start`` per LLM round, and leftover prefix
        content from round N would cause the summary walk in round N+1
        to either match the wrong prefix (silently dropping new content)
        or diverge and fall back to re-emitting the whole block.
        """
        self._block_types_by_index = {}
        self._partial_text_buffer = ""
        self._partial_thinking_buffer = ""
        self._pending_thinking_delta = ""
        self._pending_thinking_index = None

    def _text_tail_for_summary_block(self, full_text: str) -> str:
        """Reconcile the next summary ``TextBlock`` against the running
        partial-stream buffer.

        The CLI can emit the summary ``AssistantMessage`` before the
        matching ``content_block_stop`` event, so we can't rely on a
        queue of completed blocks.  Instead we maintain
        ``_partial_text_buffer`` — the concatenation of every
        ``text_delta`` chunk that hasn't been claimed by a summary
        block yet — and consume ``full_text`` as a prefix from it.
        Summary blocks that have no partial backing (buffer empty)
        emit their full text; blocks that partial covered wholly are
        silent; blocks with a partial prefix + a summary tail emit
        only the tail.  Kimi K2.6's pattern of emitting each content
        block as its own summary ``AssistantMessage`` is handled
        automatically because block order is preserved across both
        streams.
        """
        if not full_text:
            return ""
        if not self._partial_text_buffer:
            return full_text
        if full_text.startswith(self._partial_text_buffer):
            tail = full_text[len(self._partial_text_buffer) :]
            self._partial_text_buffer = ""
            return tail
        if self._partial_text_buffer.startswith(full_text):
            # Partial already emitted this whole block plus more — the
            # "more" belongs to a later summary block.  Consume only the
            # prefix matching this block and leave the rest buffered.
            self._partial_text_buffer = self._partial_text_buffer[len(full_text) :]
            return ""
        logger.warning(
            "SDK partial/summary text diverged "
            "(partial_buf=%d chars, summary=%d chars) — emitting summary, "
            "clearing partial buffer to recover",
            len(self._partial_text_buffer),
            len(full_text),
        )
        self._partial_text_buffer = ""
        return full_text

    def _thinking_tail_for_summary_block(self, full_thinking: str) -> str:
        """Same as :meth:`_text_tail_for_summary_block` for reasoning."""
        if not full_thinking:
            return ""
        if not self._partial_thinking_buffer:
            return full_thinking
        if full_thinking.startswith(self._partial_thinking_buffer):
            tail = full_thinking[len(self._partial_thinking_buffer) :]
            self._partial_thinking_buffer = ""
            return tail
        if self._partial_thinking_buffer.startswith(full_thinking):
            self._partial_thinking_buffer = self._partial_thinking_buffer[
                len(full_thinking) :
            ]
            return ""
        logger.warning(
            "SDK partial/summary thinking diverged "
            "(partial_buf=%d chars, summary=%d chars) — emitting summary, "
            "clearing partial buffer to recover",
            len(self._partial_thinking_buffer),
            len(full_thinking),
        )
        self._partial_thinking_buffer = ""
        return full_thinking

    def _handle_stream_event(
        self, evt: StreamEvent, responses: list[StreamBaseResponse]
    ) -> None:
        """Translate raw Anthropic streaming events into wire events.

        Handles four event types; everything else (``message_delta``
        stop reasons, ``signature_delta``, ``input_json_delta``,
        ``ping``, ...) is ignored because the summary ``AssistantMessage``
        carries their effects.

        * ``message_start`` — new message boundary, reset per-index maps
        * ``content_block_start`` — open text / reasoning block on the
          wire and remember the block type at that index
        * ``content_block_delta`` — forward ``text_delta`` immediately
          and coalesce ``thinking_delta`` (64-char / 50 ms window)
        * ``content_block_stop`` — drain any buffered thinking and close
          the corresponding wire block
        """
        raw: dict[str, Any] = evt.event or {}
        event_type = raw.get("type")

        if event_type == "message_start":
            self._reset_partial_stream_state()
            return

        if event_type == "content_block_start":
            block = raw.get("content_block") or {}
            index = raw.get("index")
            block_type = block.get("type")
            if not isinstance(index, int) or not isinstance(block_type, str):
                return
            self._block_types_by_index[index] = block_type
            if block_type == "text":
                self._end_reasoning_if_open(responses)
                self._ensure_text_started(responses)
                # Seed any preamble the block_start carries.
                seed = block.get("text") or ""
                if seed:
                    responses.append(StreamTextDelta(id=self.text_block_id, delta=seed))
                    self._partial_text_buffer += seed
                    self._text_since_last_tool_result = True
            elif block_type == "thinking":
                self._end_text_if_open(responses)
                self._ensure_reasoning_started(responses)
                self._last_thinking_flush_monotonic = time.monotonic()
            # tool_use / server_tool_use / redacted_thinking blocks stay
            # on the ``AssistantMessage`` path — the frontend widgets
            # need the final ``input`` payload which only arrives in the
            # summary.
            return

        if event_type == "content_block_delta":
            index = raw.get("index")
            if not isinstance(index, int):
                return
            delta = raw.get("delta") or {}
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                chunk = delta.get("text") or ""
                if not chunk:
                    return
                self._ensure_text_started(responses)
                responses.append(StreamTextDelta(id=self.text_block_id, delta=chunk))
                self._partial_text_buffer += chunk
                self._text_since_last_tool_result = True
            elif delta_type == "thinking_delta":
                chunk = delta.get("thinking") or ""
                if not chunk:
                    return
                self._ensure_reasoning_started(responses)
                # Flush the coalesce buffer if the index changed — shouldn't
                # happen in practice but guard against interleaved indices.
                if (
                    self._pending_thinking_index is not None
                    and self._pending_thinking_index != index
                ):
                    self._flush_pending_thinking(responses)
                self._pending_thinking_delta += chunk
                self._pending_thinking_index = index
                now = time.monotonic()
                elapsed_ms = (now - self._last_thinking_flush_monotonic) * 1000.0
                if (
                    len(self._pending_thinking_delta) >= _THINKING_COALESCE_MIN_CHARS
                    or elapsed_ms >= _THINKING_COALESCE_MAX_INTERVAL_MS
                ):
                    self._flush_pending_thinking(responses)
                    self._last_thinking_flush_monotonic = now
            # Other delta types (``signature_delta``, ``input_json_delta``)
            # are CLI / tool-dispatch plumbing — not surfaced on the wire.
            return

        if event_type == "content_block_stop":
            index = raw.get("index")
            if not isinstance(index, int):
                return
            block_type = self._block_types_by_index.pop(index, None)
            if block_type == "text":
                self._end_text_if_open(responses)
            elif block_type == "thinking":
                self._end_reasoning_if_open(responses)
            return

    def _flush_pending_thinking(self, responses: list[StreamBaseResponse]) -> None:
        """Drain the coalesce buffer into a ``StreamReasoningDelta``.

        Separate from ``_end_reasoning_if_open`` because the coalesce
        window can flush mid-block (threshold hit) without closing the
        reasoning block.
        """
        if not self._pending_thinking_delta:
            return
        responses.append(
            StreamReasoningDelta(
                id=self.reasoning_block_id,
                delta=self._pending_thinking_delta,
            )
        )
        self._partial_thinking_buffer += self._pending_thinking_delta
        self._pending_thinking_delta = ""
        self._pending_thinking_index = None

    def flush_unresolved_tool_calls(self, responses: list[StreamBaseResponse]) -> None:
        """Emit outputs for tool calls that didn't receive a UserMessage result.

        SDK built-in tools (WebSearch, Read, etc.) may be executed by the CLI
        internally without surfacing a separate ``UserMessage`` with
        ``ToolResultBlock`` content.  The ``PostToolUse`` hook stashes their
        output, which we pop and emit here before the next ``AssistantMessage``
        starts.

        Callers that need to both record synthetic tool_results in history AND
        yield the same events to the client should call this exactly once and
        share the resulting list — the method mutates ``resolved_tool_calls``,
        so a second call returns nothing and ``has_unresolved_tool_calls``
        flips to ``False`` after the first invocation.
        """
        unresolved = [
            (tid, info.get("name", "unknown"))
            for tid, info in self.current_tool_calls.items()
            if tid not in self.resolved_tool_calls
        ]
        sid = (self.session_id or "?")[:12]
        if not unresolved:
            logger.info(
                "[SDK] [%s] Flush called but all %d tool(s) already resolved",
                sid,
                len(self.current_tool_calls),
            )
            return
        logger.info(
            "[SDK] [%s] Flushing %d unresolved tool call(s): %s",
            sid,
            len(unresolved),
            ", ".join(f"{name}({tid[:12]})" for tid, name in unresolved),
        )

        flushed = False
        for tool_id, tool_name in unresolved:
            output = pop_pending_tool_output(tool_name)
            if output is not None:
                responses.append(
                    StreamToolOutputAvailable(
                        toolCallId=tool_id,
                        toolName=tool_name,
                        output=output,
                        success=True,
                    )
                )
                self.resolved_tool_calls.add(tool_id)
                flushed = True
                logger.info(
                    "[SDK] [%s] Flushed stashed output for %s (call %s, %d chars)",
                    sid,
                    tool_name,
                    tool_id[:12],
                    len(output),
                )
            else:
                # No output available — emit an empty output so the frontend
                # transitions the tool from input-available to output-available
                # (stops the spinner).
                responses.append(
                    StreamToolOutputAvailable(
                        toolCallId=tool_id,
                        toolName=tool_name,
                        output="",
                        success=True,
                    )
                )
                self.resolved_tool_calls.add(tool_id)
                flushed = True
                logger.warning(
                    "[SDK] [%s] Flushed EMPTY output for unresolved tool %s "
                    "(call %s) — stash was empty (likely SDK hook race "
                    "condition: PostToolUse hook hadn't completed before "
                    "flush was triggered)",
                    sid,
                    tool_name,
                    tool_id[:12],
                )

        if flushed:
            # Mirror the UserMessage tool_result path: a flushed tool output is
            # still a tool_result as far as the thinking-only-final-turn guard
            # is concerned.  Without this, a turn whose ONLY tool outputs come
            # from the flush path (SDK built-ins like WebSearch) would miss
            # the fallback synthesis if the model then produced no text.
            self._text_since_last_tool_result = False
            self._any_tool_results_seen = True
            if self.step_open:
                responses.append(StreamFinishStep())
                self.step_open = False


def _extract_tool_output(content: str | list[dict[str, str]] | None) -> str:
    """Extract a string output from a ToolResultBlock's content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if item.get("type") == "text"]
        if parts:
            return "".join(parts)
        try:
            return json.dumps(content)
        except (TypeError, ValueError):
            return str(content)
    if content is None:
        return ""
    try:
        return json.dumps(content)
    except (TypeError, ValueError):
        return str(content)


def _extract_tool_use_result(result: object) -> str:
    """Extract a string from a UserMessage's ``tool_use_result`` dict.

    SDK built-in tools may store their result in ``tool_use_result``
    instead of (or in addition to) ``ToolResultBlock`` content blocks.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        # Try common result keys
        for key in ("content", "text", "output", "stdout", "result"):
            val = result.get(key)
            if isinstance(val, str) and val:
                return val
        # Fall back to JSON serialization of the whole dict
        try:
            return json.dumps(result)
        except (TypeError, ValueError):
            return str(result)
    if result is None:
        return ""
    try:
        return json.dumps(result)
    except (TypeError, ValueError):
        return str(result)
