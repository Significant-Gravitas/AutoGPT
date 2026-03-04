"""Baseline LLM fallback — OpenAI-compatible streaming with tool calling.

Used when ``CHAT_USE_CLAUDE_AGENT_SDK=false``, e.g. as a fallback when the
Claude Agent SDK / Anthropic API is unavailable.  Routes through any
OpenAI-compatible provider (OpenRouter by default) and reuses the same
shared tool registry as the SDK path.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import orjson

from backend.copilot.model import (
    ChatMessage,
    ChatSession,
    get_chat_session,
    update_session_title,
    upsert_chat_session,
)
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
)
from backend.copilot.service import (
    _build_system_prompt,
    _generate_session_title,
    client,
    config,
)
from backend.copilot.tools import execute_tool, get_available_tools
from backend.copilot.tracking import track_user_message
from backend.util.exceptions import NotFoundError
from backend.util.prompt import compress_context

logger = logging.getLogger(__name__)

# Set to hold background tasks to prevent garbage collection
_background_tasks: set[asyncio.Task[Any]] = set()

# Maximum number of tool-call rounds before forcing a text response.
_MAX_TOOL_ROUNDS = 30


async def _update_title_async(
    session_id: str, message: str, user_id: str | None
) -> None:
    """Generate and persist a session title in the background."""
    try:
        title = await _generate_session_title(message, user_id, session_id)
        if title:
            await update_session_title(session_id, title)
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
            client=client,
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
        system_prompt, _ = await _build_system_prompt(
            user_id, has_conversation_history=False
        )
    else:
        system_prompt, _ = await _build_system_prompt(
            user_id=None, has_conversation_history=True
        )

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

    yield StreamStart(messageId=message_id, sessionId=session_id)

    assistant_text = ""
    text_block_id = str(uuid.uuid4())
    text_started = False
    step_open = False
    try:
        for _round in range(_MAX_TOOL_ROUNDS):
            # Open a new step for each LLM round
            yield StreamStartStep()
            step_open = True

            # Stream a response from the model
            create_kwargs: dict[str, Any] = dict(
                model=config.model,
                messages=openai_messages,
                stream=True,
            )
            if tools:
                create_kwargs["tools"] = tools
            response = await client.chat.completions.create(**create_kwargs)  # type: ignore[arg-type]  # dynamic kwargs

            # Accumulate streamed response (text + tool calls)
            round_text = ""
            tool_calls_by_index: dict[int, dict[str, str]] = {}

            async for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue

                # Text content
                if delta.content:
                    if not text_started:
                        yield StreamTextStart(id=text_block_id)
                        text_started = True
                    round_text += delta.content
                    yield StreamTextDelta(id=text_block_id, delta=delta.content)

                # Tool call fragments (streamed incrementally)
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

            # Close text block if we had one this round
            if text_started:
                yield StreamTextEnd(id=text_block_id)
                text_started = False
                text_block_id = str(uuid.uuid4())

            # Accumulate text for session persistence
            assistant_text += round_text

            # No tool calls -> model is done
            if not tool_calls_by_index:
                yield StreamFinishStep()
                step_open = False
                break

            # Close step before tool execution
            yield StreamFinishStep()
            step_open = False

            # Append the assistant message with tool_calls to context
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if round_text:
                assistant_msg["content"] = round_text
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for tc in tool_calls_by_index.values()
            ]
            openai_messages.append(assistant_msg)

            # Execute each tool call and stream events
            for tc in tool_calls_by_index.values():
                tool_call_id = tc["id"]
                tool_name = tc["name"]
                raw_args = tc["arguments"] or "{}"
                try:
                    tool_args = orjson.loads(raw_args)
                except orjson.JSONDecodeError as parse_err:
                    parse_error = (
                        f"Invalid JSON arguments for tool '{tool_name}': {parse_err}"
                    )
                    logger.warning("[Baseline] %s", parse_error)
                    yield StreamToolOutputAvailable(
                        toolCallId=tool_call_id,
                        toolName=tool_name,
                        output=parse_error,
                        success=False,
                    )
                    openai_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": parse_error,
                        }
                    )
                    continue

                yield StreamToolInputStart(toolCallId=tool_call_id, toolName=tool_name)
                yield StreamToolInputAvailable(
                    toolCallId=tool_call_id,
                    toolName=tool_name,
                    input=tool_args,
                )

                # Execute via shared tool registry
                try:
                    result: StreamToolOutputAvailable = await execute_tool(
                        tool_name=tool_name,
                        parameters=tool_args,
                        user_id=user_id,
                        session=session,
                        tool_call_id=tool_call_id,
                    )
                    yield result
                    tool_output = (
                        result.output
                        if isinstance(result.output, str)
                        else str(result.output)
                    )
                except Exception as e:
                    error_output = f"Tool execution error: {e}"
                    logger.error(
                        "[Baseline] Tool %s failed: %s",
                        tool_name,
                        error_output,
                        exc_info=True,
                    )
                    yield StreamToolOutputAvailable(
                        toolCallId=tool_call_id,
                        toolName=tool_name,
                        output=error_output,
                        success=False,
                    )
                    tool_output = error_output

                # Append tool result to context for next round
                openai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_output,
                    }
                )
        else:
            # for-loop exhausted without break -> tool-round limit hit
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
        error_msg = str(e) or type(e).__name__
        logger.error("[Baseline] Streaming error: %s", error_msg, exc_info=True)
        # Close any open text/step before emitting error
        if text_started:
            yield StreamTextEnd(id=text_block_id)
        if step_open:
            yield StreamFinishStep()
        yield StreamError(errorText=error_msg, code="baseline_error")
        # Still persist whatever we got
    finally:
        # Persist assistant response
        if assistant_text:
            session.messages.append(
                ChatMessage(role="assistant", content=assistant_text)
            )
        try:
            await upsert_chat_session(session)
        except Exception as persist_err:
            logger.error("[Baseline] Failed to persist session: %s", persist_err)

    yield StreamFinish()
