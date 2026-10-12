"""Anthropic Messages API, the dialect the Claude Agent SDK's CLI speaks."""

import uuid
from collections.abc import Iterator
from typing import Any

from .invoice import requested_cache_ttl
from .planner import Conversation, Reply, ToolResult, bare_name
from .streaming import Event, json_chunks, text_chunks


def conversation(body: dict[str, Any]) -> Conversation:
    messages = body.get("messages") or []
    last_assistant = max(
        (i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1
    )
    answered = messages[last_assistant] if last_assistant >= 0 else {}
    tool_names = {
        block.get("id"): bare_name(str(block.get("name")))
        for block in _blocks(answered)
        if block.get("type") == "tool_use"
    }
    return Conversation(
        model=str(body.get("model", "")),
        stream=bool(body.get("stream")),
        user_text="\n".join(
            _text(m.get("content")) for m in messages if m.get("role") == "user"
        ),
        assistant_messages=sum(1 for m in messages if m.get("role") == "assistant"),
        offered_tools=tuple(t["name"] for t in body.get("tools") or [] if "name" in t),
        tool_results=tuple(
            ToolResult(
                tool=tool_names.get(block.get("tool_use_id"), "<unknown>"),
                content=_text(block.get("content")),
                is_error=bool(block.get("is_error")),
            )
            for message in messages[last_assistant + 1 :]
            for block in _blocks(message)
            if block.get("type") == "tool_result"
        ),
        cache_ttl=requested_cache_ttl(body),
        latest_user_text="\n".join(
            _text(m.get("content"))
            for m in messages[last_assistant + 1 :]
            if m.get("role") == "user"
        ),
    )


def events(reply: Reply, model: str) -> Iterator[Event]:
    """The reply as a Messages stream: text, then one block per tool call."""
    yield Event("message_start", _message_start(reply, model), delay_ms=0)
    index = 0
    if reply.text:
        yield _block_start(index, {"type": "text", "text": ""})
        for piece in text_chunks(reply.text, reply.chunk_words):
            delta = {"type": "text_delta", "text": piece}
            yield Event(
                "content_block_delta",
                {"type": "content_block_delta", "index": index, "delta": delta},
                reply.delay_ms,
            )
        yield _block_stop(index)
        index += 1
    for name, arguments in reply.tool_calls:
        block = {"type": "tool_use", "id": _id("toolu_"), "name": name, "input": {}}
        yield _block_start(index, block)
        for piece in json_chunks(arguments):
            delta = {"type": "input_json_delta", "partial_json": piece}
            yield Event(
                "content_block_delta",
                {"type": "content_block_delta", "index": index, "delta": delta},
                reply.delay_ms,
            )
        yield _block_stop(index)
        index += 1
    yield Event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": _stop_reason(reply), "stop_sequence": None},
            "usage": {"output_tokens": reply.usage.output_tokens},
        },
        delay_ms=0,
    )
    yield Event("message_stop", {"type": "message_stop"}, delay_ms=0)


def message(reply: Reply, model: str) -> dict[str, Any]:
    """The reply as one non-streamed Messages response."""
    content: list[dict[str, Any]] = (
        [{"type": "text", "text": reply.text}] if reply.text else []
    )
    content += [
        {"type": "tool_use", "id": _id("toolu_"), "name": name, "input": arguments}
        for name, arguments in reply.tool_calls
    ]
    return {
        **_message_start(reply, model)["message"],
        "content": content,
        "stop_reason": _stop_reason(reply),
        "usage": _usage(reply),
    }


def error(message_text: str) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": f"scripted-model: {message_text}",
        },
    }


def _message_start(reply: Reply, model: str) -> dict[str, Any]:
    return {
        "type": "message_start",
        "message": {
            "id": _id("msg_"),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {**_usage(reply), "output_tokens": 1},
        },
    }


def _usage(reply: Reply) -> dict[str, Any]:
    usage = reply.usage
    written = usage.cache_creation_input_tokens
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
        "cache_creation_input_tokens": written,
        # The CLI prices a cache write by the lifetime it bought.
        "cache_creation": {
            "ephemeral_5m_input_tokens": 0 if reply.cache_ttl == "1h" else written,
            "ephemeral_1h_input_tokens": written if reply.cache_ttl == "1h" else 0,
        },
    }


def _block_start(index: int, block: dict[str, Any]) -> Event:
    return Event(
        "content_block_start",
        {"type": "content_block_start", "index": index, "content_block": block},
        delay_ms=0,
    )


def _block_stop(index: int) -> Event:
    return Event(
        "content_block_stop", {"type": "content_block_stop", "index": index}, delay_ms=0
    )


def _stop_reason(reply: Reply) -> str:
    return "tool_use" if reply.tool_calls else "end_turn"


def _blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    return (
        [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []
    )


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in content
        )
    return ""


def _id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex[:24]
