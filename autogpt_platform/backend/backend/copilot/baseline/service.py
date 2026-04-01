"""Baseline LLM fallback — OpenAI-compatible streaming with tool calling.

Used when ``CHAT_USE_CLAUDE_AGENT_SDK=false``, e.g. as a fallback when the
Claude Agent SDK / Anthropic API is unavailable.  Routes through any
OpenAI-compatible provider (OpenRouter by default) and reuses the same
shared tool registry as the SDK path.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, cast

import orjson
from langfuse import propagate_attributes
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from backend.copilot.context import set_execution_context
from backend.copilot.model import (
    ChatMessage,
    ChatSession,
    get_chat_session,
    update_session_title,
    upsert_chat_session,
)
from backend.copilot.prompting import get_baseline_supplement
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
    _generate_session_title,
    _get_openai_client,
    config,
)
from backend.copilot.token_tracking import persist_and_record_usage
from backend.copilot.tools import execute_tool, get_available_tools
from backend.copilot.tracking import track_user_message
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

logger = logging.getLogger(__name__)

# Set to hold background tasks to prevent garbage collection
_background_tasks: set[asyncio.Task[Any]] = set()

# Maximum number of tool-call rounds before forcing a text response.
_MAX_TOOL_ROUNDS = 30


@dataclass
class _BaselineStreamState:
    """Mutable state shared between the tool-call loop callbacks.

    Extracted from ``stream_chat_completion_baseline`` so that the callbacks
    can be module-level functions instead of deeply nested closures.
    """

    pending_events: list[StreamBaseResponse] = field(default_factory=list)
    assistant_text: str = ""
    text_block_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text_started: bool = False
    turn_prompt_tokens: int = 0
    turn_completion_tokens: int = 0


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

    round_text = ""
    try:
        client = _get_openai_client()
        typed_messages = cast(list[ChatCompletionMessageParam], messages)
        if tools:
            typed_tools = cast(list[ChatCompletionToolParam], tools)
            response = await client.chat.completions.create(
                model=config.model,
                messages=typed_messages,
                tools=typed_tools,
                stream=True,
                stream_options={"include_usage": True},
            )
        else:
            response = await client.chat.completions.create(
                model=config.model,
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
                if not state.text_started:
                    state.pending_events.append(StreamTextStart(id=state.text_block_id))
                    state.text_started = True
                round_text += delta.content
                state.pending_events.append(
                    StreamTextDelta(id=state.text_block_id, delta=delta.content)
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


def _baseline_conversation_updater(
    messages: list[dict[str, Any]],
    response: LLMLoopResponse,
    tool_results: list[ToolCallResult] | None = None,
) -> None:
    """Update OpenAI message list with assistant response + tool results.

    Extracted from ``stream_chat_completion_baseline`` for readability.
    """
    if tool_results:
        # Build assistant message with tool_calls
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
    else:
        if response.response_text:
            messages.append({"role": "assistant", "content": response.response_text})


async def _update_title_async(
    session_id: str, message: str, user_id: str | None
) -> None:
    """Generate and persist a session title in the background."""
    try:
        title = await _generate_session_title(message, user_id, session_id)
        if title and user_id:
            await update_session_title(session_id, user_id, title, only_if_empty=True)
    except Exception as e:
        logger.warning("[Baseline] Failed to update session title: %s", e)


async def _compress_session_messages(
    messages: list[ChatMessage],
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
        messages_dict.append(msg_dict)

    try:
        result = await compress_context(
            messages=messages_dict,
            model=config.model,
            client=_get_openai_client(),
        )
    except Exception as e:
        logger.warning("[Baseline] Context compression with LLM failed: %s", e)
        result = await compress_context(
            messages=messages_dict,
            model=config.model,
            client=None,
        )

    if result.was_compacted:
        logger.info(
            "[Baseline] Context compacted: %d -> %d tokens "
            "(%d summarized, %d dropped)",
            result.original_token_count,
            result.token_count,
            result.messages_summarized,
            result.messages_dropped,
        )
        return [
            ChatMessage(role=m["role"], content=m.get("content"))
            for m in result.messages
        ]

    return messages


async def stream_chat_completion_baseline(
    session_id: str,
    message: str | None = None,
    is_user_message: bool = True,
    user_id: str | None = None,
    session: ChatSession | None = None,
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

    # Append user message
    new_role = "user" if is_user_message else "assistant"
    if message and (
        len(session.messages) == 0
        or not (
            session.messages[-1].role == new_role
            and session.messages[-1].content == message
        )
    ):
        session.messages.append(ChatMessage(role=new_role, content=message))
        if is_user_message:
            track_user_message(
                user_id=user_id,
                session_id=session_id,
                message_length=len(message),
            )

    session = await upsert_chat_session(session)

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

    # Build system prompt only on the first turn to avoid mid-conversation
    # changes from concurrent chats updating business understanding.
    is_first_turn = len(session.messages) <= 1
    if is_first_turn:
        base_system_prompt, _ = await _build_system_prompt(
            user_id, has_conversation_history=False
        )
    else:
        base_system_prompt, _ = await _build_system_prompt(
            user_id=None, has_conversation_history=True
        )

    # Append tool documentation and technical notes
    system_prompt = base_system_prompt + get_baseline_supplement()

    # Compress context if approaching the model's token limit
    messages_for_context = await _compress_session_messages(session.messages)

    # Build OpenAI message list from session history
    openai_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt}
    ]
    for msg in messages_for_context:
        if msg.role in ("user", "assistant") and msg.content:
            openai_messages.append({"role": msg.role, "content": msg.content})

    tools = get_available_tools()

    # Propagate execution context so tool handlers can read session-level flags.
    set_execution_context(user_id, session)

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
    state = _BaselineStreamState()

    # Bind extracted module-level callbacks to this request's state/session
    # using functools.partial so they satisfy the Protocol signatures.
    _bound_llm_caller = partial(_baseline_llm_caller, state=state)
    _bound_tool_executor = partial(
        _baseline_tool_executor, state=state, user_id=user_id, session=session
    )

    try:
        loop_result = None
        async for loop_result in tool_call_loop(
            messages=openai_messages,
            tools=tools,
            llm_call=_bound_llm_caller,
            execute_tool=_bound_tool_executor,
            update_conversation=_baseline_conversation_updater,
            max_iterations=_MAX_TOOL_ROUNDS,
        ):
            # Drain buffered events after each iteration (real-time streaming)
            for evt in state.pending_events:
                yield evt
            state.pending_events.clear()

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
        # Close Langfuse trace context
        if _trace_ctx is not None:
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
                estimate_token_count(openai_messages, model=config.model), 1
            )
            state.turn_completion_tokens = estimate_token_count_str(
                state.assistant_text, model=config.model
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
        )

        # Persist assistant response
        if state.assistant_text:
            session.messages.append(
                ChatMessage(role="assistant", content=state.assistant_text)
            )
        try:
            await upsert_chat_session(session)
        except Exception as persist_err:
            logger.error("[Baseline] Failed to persist session: %s", persist_err)

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
