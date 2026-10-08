"""Pydantic AI message history <-> copilot chat rows, and held-call resumption.

The engine keeps its own history (``ModelMessagesTypeAdapter`` JSON, see
:mod:`.history_store`) because it holds what chat rows cannot: thinking
signatures, provider ids, and tool calls still waiting on the user. Chat rows
stay the source of truth for everything else, so a session that ran on
another engine (or whose stored history is missing) is rebuilt from them, and
rows written past the stored history's watermark are folded in on load.
"""

import dataclasses
from collections.abc import Iterable, Sequence

from pydantic import BaseModel
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
)

from backend.copilot.model import ChatMessage

_LOST_RESULT = (
    "Nothing ran: this call's result was lost between turns. Tell the user, "
    "and ask before trying it again."
)


class HeldToolCall(BaseModel):
    """A call the gate parked, as the run that made it ended on it."""

    tool_call_id: str
    tool_name: str
    review_id: str
    # What the model would have read had the call not ended the run.
    output: str


def chat_rows_to_messages(rows: Iterable[ChatMessage]) -> list[ModelMessage]:
    """Rebuild a model history from chat rows (reasoning rows skipped).

    Tool calls without a result in the very next request are dropped, as are
    results whose call is gone: providers reject either half of a broken pair.
    """
    messages: list[ModelMessage] = []
    names: dict[str, str] = {}
    for row in rows:
        if row.role == "user" and row.content:
            _append_request(messages, [UserPromptPart(content=row.content)])
        elif row.role == "assistant":
            parts = _assistant_parts(row, names)
            if parts:
                _append_response(messages, parts)
        elif row.role == "tool" and row.tool_call_id:
            part = ToolReturnPart(
                tool_name=names.get(row.tool_call_id, "unknown"),
                content=row.content or "",
                tool_call_id=row.tool_call_id,
            )
            _append_request(messages, [part])
    return repair_tool_pairs(messages)


def messages_to_chat_rows(messages: Sequence[ModelMessage]) -> list[ChatMessage]:
    """Flatten a model history into OpenAI-shaped rows (for compaction)."""
    rows: list[ChatMessage] = []
    for message in messages:
        if message.kind == "request":
            rows.extend(_request_rows(message))
        else:
            row = _response_row(message)
            if row is not None:
                rows.append(row)
    return rows


def pending_tool_calls(messages: Sequence[ModelMessage]) -> list[ToolCallPart]:
    """Tool calls in the last response that no later request answers."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.kind != "response":
            continue
        answered = {
            call_id
            for later in messages[index + 1 :]
            if later.kind == "request"
            for part in later.parts
            if (call_id := _answered_call_id(part))
        }
        return [
            call for call in message.tool_calls if call.tool_call_id not in answered
        ]
    return []


def resume_results(
    pending: Sequence[ToolCallPart],
    held: Sequence[HeldToolCall],
    delivered: dict[str, str],
) -> dict[str, str]:
    """The result each pending call resumes with.

    The user's answer (``delivered``, by tool call id) when it came in; the
    gate's refusal the model would have read when it has not yet; a "lost"
    note when the call's record is gone, so the history is never left broken.
    """
    by_id = {record.tool_call_id: record.output for record in held}
    return {
        call.tool_call_id: delivered.get(call.tool_call_id)
        or by_id.get(call.tool_call_id)
        or _LOST_RESULT
        for call in pending
    }


def close_pending_calls(
    messages: list[ModelMessage], results: dict[str, str]
) -> list[ModelMessage]:
    """Answer pending calls in the history itself (when rows from another
    engine have to follow them, so ``DeferredToolResults`` cannot be used)."""
    pending = pending_tool_calls(messages)
    if not pending:
        return messages
    parts: list[ModelRequestPart] = [
        ToolReturnPart(
            tool_name=call.tool_name,
            content=results.get(call.tool_call_id, _LOST_RESULT),
            tool_call_id=call.tool_call_id,
        )
        for call in pending
    ]
    closed = list(messages)
    _append_request(closed, parts)
    return closed


def sanitize_for_storage(messages: Sequence[ModelMessage]) -> list[ModelMessage]:
    """Drop per-request instructions (rebuilt every turn, and the bulk of the
    file) and inline binary content (images live in the workspace)."""
    sanitized: list[ModelMessage] = []
    for message in messages:
        if message.kind == "response":
            sanitized.append(message)
            continue
        parts = [_strip_binary(part) for part in message.parts]
        sanitized.append(dataclasses.replace(message, parts=parts, instructions=None))
    return sanitized


def repair_tool_pairs(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Keep only tool calls answered by the next request, and vice versa."""
    repaired: list[ModelMessage] = []
    for index, message in enumerate(messages):
        if message.kind == "response":
            following = messages[index + 1] if index + 1 < len(messages) else None
            answered = _answered_ids(following)
            parts = [
                part
                for part in message.parts
                if part.part_kind != "tool-call" or part.tool_call_id in answered
            ]
            if parts:
                repaired.append(dataclasses.replace(message, parts=parts))
            continue
        previous = repaired[-1] if repaired else None
        called = (
            {call.tool_call_id for call in previous.tool_calls}
            if previous is not None and previous.kind == "response"
            else set()
        )
        parts = [
            part
            for part in message.parts
            if part.part_kind != "tool-return" or part.tool_call_id in called
        ]
        if parts:
            repaired.append(dataclasses.replace(message, parts=parts))
    return repaired


def _answered_call_id(part: ModelRequestPart) -> str | None:
    if part.part_kind == "tool-return":
        return part.tool_call_id
    if part.part_kind == "retry-prompt" and part.tool_name:
        return part.tool_call_id
    return None


def _answered_ids(message: ModelMessage | None) -> set[str]:
    if message is None or message.kind != "request":
        return set()
    return {
        part.tool_call_id for part in message.parts if part.part_kind == "tool-return"
    }


def _assistant_parts(
    row: ChatMessage, names: dict[str, str]
) -> list[ModelResponsePart]:
    parts: list[ModelResponsePart] = []
    if row.content:
        parts.append(TextPart(content=row.content))
    for call in row.tool_calls or []:
        function = call.get("function") or {}
        call_id = str(call.get("id") or "")
        name = str(function.get("name") or "unknown")
        if not call_id:
            continue
        names[call_id] = name
        parts.append(
            ToolCallPart(
                tool_name=name,
                args=str(function.get("arguments") or "{}"),
                tool_call_id=call_id,
            )
        )
    return parts


def _append_request(
    messages: list[ModelMessage], parts: list[ModelRequestPart]
) -> None:
    last = messages[-1] if messages else None
    if last is not None and last.kind == "request":
        messages[-1] = dataclasses.replace(last, parts=[*last.parts, *parts])
        return
    messages.append(ModelRequest(parts=parts))


def _append_response(
    messages: list[ModelMessage], parts: list[ModelResponsePart]
) -> None:
    last = messages[-1] if messages else None
    if last is not None and last.kind == "response":
        messages[-1] = dataclasses.replace(last, parts=[*last.parts, *parts])
        return
    messages.append(ModelResponse(parts=parts))


def _request_rows(message: ModelRequest) -> list[ChatMessage]:
    rows: list[ChatMessage] = []
    for part in message.parts:
        if part.part_kind == "user-prompt":
            rows.append(ChatMessage(role="user", content=_prompt_text(part.content)))
        elif part.part_kind == "tool-return":
            rows.append(
                ChatMessage(
                    role="tool",
                    content=part.model_response_str(),
                    tool_call_id=part.tool_call_id,
                )
            )
        elif part.part_kind == "retry-prompt" and part.tool_name:
            rows.append(
                ChatMessage(
                    role="tool",
                    content=part.model_response(),
                    tool_call_id=part.tool_call_id,
                )
            )
    return rows


def _response_row(message: ModelResponse) -> ChatMessage | None:
    text = "".join(part.content for part in message.parts if part.part_kind == "text")
    calls = [
        {
            "id": call.tool_call_id,
            "type": "function",
            "function": {"name": call.tool_name, "arguments": call.args_as_json_str()},
        }
        for call in message.tool_calls
    ]
    if not text and not calls:
        return None
    return ChatMessage(role="assistant", content=text or None, tool_calls=calls or None)


def _prompt_text(content: str | Sequence[UserContent]) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        item if isinstance(item, str) else "[attachment]" for item in content
    )


def _strip_binary(part: ModelRequestPart) -> ModelRequestPart:
    if part.part_kind != "user-prompt" or isinstance(part.content, str):
        return part
    content: list[UserContent] = [
        _binary_note(item) if isinstance(item, BinaryContent) else item
        for item in part.content
    ]
    return dataclasses.replace(part, content=content)


def _binary_note(item: BinaryContent) -> str:
    return (
        f"[{item.media_type} attachment, {len(item.data):,} bytes, not kept in history]"
    )
