"""Streamed response bodies in each upstream's own format."""

import json
import uuid
from typing import Any, Literal

from pydantic import BaseModel

from .invoice import CacheTtl, Usage

# A tool no engine has, so every engine answers it with an error result and
# asks the model again: a second billed call in the same turn.
TOOL_NAME = "lookup_fact"
TOOL_ARGUMENTS = {"query": "billing"}


class Reply(BaseModel):
    kind: Literal["text", "tool"]
    text: str = ""


def anthropic_messages(
    *, message_id: str, model: str, reply: Reply, usage: Usage, ttl: CacheTtl | None
) -> str:
    """Anthropic Messages SSE, as Anthropic and OpenRouter's /messages stream it."""
    start_usage = {
        "input_tokens": usage.input_tokens,
        "output_tokens": 0,
        "cache_read_input_tokens": usage.cache_read_tokens,
        "cache_creation_input_tokens": usage.cache_write_tokens,
        "cache_creation": {
            "ephemeral_5m_input_tokens": usage.cache_write_tokens if ttl == "5m" else 0,
            "ephemeral_1h_input_tokens": usage.cache_write_tokens if ttl == "1h" else 0,
        },
    }
    if reply.kind == "tool":
        block: dict[str, Any] = {
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex[:24]}",
            "name": TOOL_NAME,
            "input": {},
        }
        delta = {"type": "input_json_delta", "partial_json": json.dumps(TOOL_ARGUMENTS)}
    else:
        block = {"type": "text", "text": ""}
        delta = {"type": "text_delta", "text": reply.text}
    events = [
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": start_usage,
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": block},
        {"type": "content_block_delta", "index": 0, "delta": delta},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": "tool_use" if reply.kind == "tool" else "end_turn",
                "stop_sequence": None,
            },
            "usage": {"output_tokens": usage.output_tokens},
        },
        {"type": "message_stop"},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def chat_completion_chunks(
    *,
    model: str,
    reply: Reply,
    usage: Usage,
    cost: float | None,
    cache_write_field: str | None,
) -> str:
    """OpenAI chat-completion SSE, ending in a usage-only chunk.

    ``prompt_tokens`` includes the cache buckets, as the OpenAI shape does;
    ``cost`` is OpenRouter's extension and ``cache_write_field`` names the
    cache-write count the way that upstream spells it.
    """
    base = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
    }
    if reply.kind == "tool":
        delta: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": [
                {
                    "index": 0,
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": TOOL_NAME,
                        "arguments": json.dumps(TOOL_ARGUMENTS),
                    },
                }
            ],
        }
    else:
        delta = {"role": "assistant", "content": reply.text}
    prompt_tokens = (
        usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
    )
    usage_body: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": prompt_tokens + usage.output_tokens,
    }
    if cache_write_field is not None:
        usage_body["prompt_tokens_details"] = {
            "cached_tokens": usage.cache_read_tokens,
            cache_write_field: usage.cache_write_tokens,
        }
    if cost is not None:
        usage_body["cost"] = cost
    finish = "tool_calls" if reply.kind == "tool" else "stop"
    chunks = [
        {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
        {**base, "choices": [], "usage": usage_body},
    ]
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def responses_events(
    *, response_id: str, model: str, reply: Reply, usage: Usage
) -> str:
    """OpenAI Responses SSE, as ChatGPT's Codex backend streams it.

    OpenAI caching has no write bucket: ``input_tokens`` counts every input
    token and ``cached_tokens`` the cached subset of them.
    """
    if reply.kind == "tool":
        item: dict[str, Any] = {
            "type": "function_call",
            "id": f"fc_{uuid.uuid4().hex}",
            "call_id": f"call_{uuid.uuid4().hex[:24]}",
            "name": TOOL_NAME,
            "arguments": json.dumps(TOOL_ARGUMENTS),
            "status": "completed",
        }
        deltas: list[dict[str, Any]] = []
    else:
        item = {
            "type": "message",
            "id": f"msg_{uuid.uuid4().hex}",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": reply.text, "annotations": []}],
        }
        deltas = [
            {
                "type": "response.output_text.delta",
                "item_id": item["id"],
                "output_index": 0,
                "content_index": 0,
                "delta": reply.text,
            }
        ]
    response = {
        "id": response_id,
        "object": "response",
        "created_at": 0,
        "model": model,
        "status": "completed",
        "output": [item],
        "usage": {
            "input_tokens": usage.input_tokens + usage.cache_read_tokens,
            "input_tokens_details": {"cached_tokens": usage.cache_read_tokens},
            "output_tokens": usage.output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": usage.input_tokens
            + usage.cache_read_tokens
            + usage.output_tokens,
        },
    }
    events = [
        {
            "type": "response.created",
            "response": {**response, "status": "in_progress", "output": []},
        },
        *deltas,
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {"type": "response.completed", "response": response},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def graph_chat_snapshots(*, conversation_id: str, prompt: str, answer: str) -> str:
    """Microsoft Graph's chatOverStream: whole-conversation snapshots, the
    reply growing a word at a time."""
    words = answer.split(" ")
    snapshots = [
        {
            "id": conversation_id,
            "state": "active",
            "messages": [
                {"id": f"{conversation_id}-prompt", "text": prompt},
                {"id": f"{conversation_id}-reply", "text": " ".join(words[:count])},
            ],
        }
        for count in range(1, len(words) + 1)
    ]
    return "".join(f"data: {json.dumps(s)}\n\n" for s in snapshots)
