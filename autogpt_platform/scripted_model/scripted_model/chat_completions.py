"""OpenAI chat completions, the dialect of the baseline engine and side calls.

Usage is reported the way OpenRouter reports it: ``prompt_tokens`` counts the
cache buckets too, which ``prompt_tokens_details`` breaks out.
"""

import json
import time
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
        call.get("id"): bare_name(str(call.get("function", {}).get("name")))
        for call in answered.get("tool_calls") or []
    }
    return Conversation(
        model=str(body.get("model", "")),
        stream=bool(body.get("stream")),
        user_text="\n".join(
            _text(m.get("content")) for m in messages if m.get("role") == "user"
        ),
        assistant_messages=sum(1 for m in messages if m.get("role") == "assistant"),
        offered_tools=tuple(
            t["function"]["name"]
            for t in body.get("tools") or []
            if isinstance(t.get("function"), dict) and "name" in t["function"]
        ),
        tool_results=tuple(
            ToolResult(
                tool=tool_names.get(m.get("tool_call_id"), "<unknown>"),
                content=_text(m.get("content")),
                # Chat completions has no error flag on a tool message.
                is_error=False,
            )
            for m in messages[last_assistant + 1 :]
            if m.get("role") == "tool"
        ),
        cache_ttl=requested_cache_ttl(body),
        latest_user_text="\n".join(
            _text(m.get("content"))
            for m in messages[last_assistant + 1 :]
            if m.get("role") == "user"
        ),
    )


def events(reply: Reply, model: str) -> Iterator[Event]:
    """The reply as chat-completion chunks, ending in a usage-only chunk."""
    base = {
        "id": _id("chatcmpl-"),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
    }

    def chunk(
        delta: dict[str, Any], finish: str | None = None, delay_ms: int = 0
    ) -> Event:
        choice = {"index": 0, "delta": delta, "finish_reason": finish}
        return Event(None, {**base, "choices": [choice]}, delay_ms)

    yield chunk({"role": "assistant", "content": ""})
    for piece in text_chunks(reply.text, reply.chunk_words):
        yield chunk({"content": piece}, delay_ms=reply.delay_ms)
    for index, (name, arguments) in enumerate(reply.tool_calls):
        call = {"index": index, "id": _id("call_"), "type": "function"}
        yield chunk(
            {"tool_calls": [{**call, "function": {"name": name, "arguments": ""}}]}
        )
        for piece in json_chunks(arguments):
            delta = {"tool_calls": [{"index": index, "function": {"arguments": piece}}]}
            yield chunk(delta, delay_ms=reply.delay_ms)
    yield chunk({}, finish=_finish_reason(reply))
    yield Event(None, {**base, "choices": [], "usage": _usage(reply)}, delay_ms=0)
    yield Event(None, "[DONE]", delay_ms=0)


def completion(reply: Reply, model: str) -> dict[str, Any]:
    """The reply as one non-streamed chat completion."""
    message: dict[str, Any] = {"role": "assistant", "content": reply.text or None}
    if reply.tool_calls:
        message["tool_calls"] = [
            {
                "id": _id("call_"),
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
            for name, arguments in reply.tool_calls
        ]
    return {
        "id": _id("chatcmpl-"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": _finish_reason(reply)}
        ],
        "usage": _usage(reply),
    }


def error(message_text: str) -> dict[str, Any]:
    return {
        "error": {
            "message": f"scripted-model: {message_text}",
            "type": "invalid_request_error",
            "code": "scripted_model",
        }
    }


def _usage(reply: Reply) -> dict[str, Any]:
    usage = reply.usage
    prompt = (
        usage.input_tokens
        + usage.cache_read_input_tokens
        + usage.cache_creation_input_tokens
    )
    body: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": usage.output_tokens,
        "total_tokens": prompt + usage.output_tokens,
        "prompt_tokens_details": {
            "cached_tokens": usage.cache_read_input_tokens,
            "cache_write_tokens": usage.cache_creation_input_tokens,
        },
    }
    if reply.cost_usd is not None:
        # OpenRouter's extension: what the call was billed, in US dollars.
        body["cost"] = reply.cost_usd
    return body


def _finish_reason(reply: Reply) -> str:
    return "tool_calls" if reply.tool_calls else "stop"


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return ""


def _id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex[:24]
