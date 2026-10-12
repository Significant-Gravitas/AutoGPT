"""The scripted model's answers in each upstream's own wire format.

Each one exercises the parts of its protocol the gateway has to handle: the
Chat Completions answer streams reasoning, splits a tool call's arguments
across chunks and reports a cached share of the prompt; the Responses answer
leads with a reasoning item carrying encrypted content; the overflows are
the bodies llama.cpp, vLLM and the Codex backend actually send.
"""

import json
from typing import Any

from backend.copilot.sdk.model_gateway.testing.script import Decision

LLAMA_CPP_OVERFLOW = {
    "error": {
        "code": 400,
        "message": "request (40000 tokens) exceeds the available context size"
        " (32768 tokens), try increasing it",
        "type": "exceed_context_size_error",
        "n_prompt_tokens": 40000,
        "n_ctx": 32768,
    }
}
VLLM_OVERFLOW = {
    "object": "error",
    "message": "This model's maximum context length is 32768 tokens. However, your"
    " request has 40000 input tokens. Please reduce the length of the input"
    " messages.",
    "type": "BadRequestError",
    "param": None,
    "code": 400,
}
CODEX_OVERFLOW = {
    "code": "context_length_exceeded",
    "message": "Your input exceeds the context window of this model. Please adjust"
    " your input and try again.",
}


def sse(events: list[dict[str, Any]], *, named: bool = True) -> bytes:
    if not named:
        return b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events)
    return b"".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events
    )


def anthropic_message(
    decision: Decision, *, model: str, tool: str, n: int, input_tokens: int
) -> dict[str, Any]:
    if decision.kind == "tool":
        block: dict[str, Any] = {
            "type": "tool_use",
            "id": f"toolu_stub_{n}",
            "name": tool,
            "input": {"page": decision.page},
        }
    else:
        block = {"type": "text", "text": decision.text}
    return {
        "id": f"msg_stub_{n}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [block],
        "stop_reason": "tool_use" if decision.kind == "tool" else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": 9},
    }


def anthropic_events(message: dict[str, Any]) -> list[dict[str, Any]]:
    block = message["content"][0]
    if block["type"] == "tool_use":
        start = {**block, "input": {}}
        delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
    else:
        start = {"type": "text", "text": ""}
        delta = {"type": "text_delta", "text": block["text"]}
    return [
        {"type": "message_start", "message": {**message, "content": []}},
        {"type": "content_block_start", "index": 0, "content_block": start},
        {"type": "content_block_delta", "index": 0, "delta": delta},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
            "usage": {"output_tokens": 9},
        },
        {"type": "message_stop"},
    ]


def chat_chunks(
    decision: Decision,
    *,
    model: str,
    tool: str,
    n: int,
    prompt_tokens: int,
    fully_cached: bool = False,
) -> list[dict[str, Any]]:
    base = {"id": f"chatcmpl-{n}", "object": "chat.completion.chunk", "model": model}

    def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
        return {
            **base,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    chunks = [chunk({"role": "assistant", "reasoning_content": "Let me think."})]
    if decision.kind == "tool":
        arguments = json.dumps({"page": decision.page})
        call = {"index": 0, "id": f"call_stub_{n}", "type": "function"}
        chunks += [
            chunk(
                {"tool_calls": [{**call, "function": {"name": tool, "arguments": ""}}]}
            ),
            chunk(
                {"tool_calls": [{"index": 0, "function": {"arguments": arguments[:5]}}]}
            ),
            chunk(
                {"tool_calls": [{"index": 0, "function": {"arguments": arguments[5:]}}]}
            ),
            chunk({}, "tool_calls"),
        ]
    else:
        half = len(decision.text) // 2
        chunks += [
            chunk({"content": decision.text[:half]}),
            chunk({"content": decision.text[half:]}),
            chunk({}, "stop"),
        ]
    cached = prompt_tokens if fully_cached else prompt_tokens // 3
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 9,
        "total_tokens": prompt_tokens + 9,
        "prompt_tokens_details": {"cached_tokens": cached},
    }
    return chunks + [{**base, "choices": [], "usage": usage}]


def responses_events(
    decision: Decision,
    *,
    model: str,
    tool: str,
    n: int,
    input_tokens: int,
    encrypted: str,
    summarize: bool = True,
    fully_cached: bool = False,
) -> list[dict[str, Any]]:
    response = {"id": f"resp_{n}", "object": "response", "model": model}
    created = {
        "type": "response.created",
        "response": {**response, "status": "in_progress"},
    }
    if decision.kind == "overflow":
        failed = {**response, "status": "failed", "error": CODEX_OVERFLOW}
        return [created, {"type": "response.failed", "response": failed}]
    events = [created, *_reasoning_item(n, encrypted, summarize)]
    events += (
        _function_call_item(n, tool, decision.page)
        if decision.kind == "tool"
        else _message_item(n, decision.text)
    )
    usage = {
        "input_tokens": input_tokens,
        "input_tokens_details": {
            "cached_tokens": input_tokens if fully_cached else input_tokens // 4
        },
        "output_tokens": 12,
        "output_tokens_details": {"reasoning_tokens": 3},
        "total_tokens": input_tokens + 12,
    }
    done = {**response, "status": "completed", "usage": usage}
    return events + [{"type": "response.completed", "response": done}]


def _reasoning_item(n: int, encrypted: str, summarize: bool) -> list[dict[str, Any]]:
    """A reasoning item; without a summary it is encrypted content alone."""
    item_id = f"rs_{n}"
    text = "Planning the next step."
    summary = [{"type": "summary_text", "text": text}] if summarize else []
    added = {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {"type": "reasoning", "id": item_id, "summary": []},
    }
    done = {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "type": "reasoning",
            "id": item_id,
            "summary": summary,
            "encrypted_content": encrypted,
        },
    }
    if not summarize:
        return [added, done]
    part = {"item_id": item_id, "summary_index": 0}
    return [
        added,
        {"type": "response.reasoning_summary_part.added", **part},
        {"type": "response.reasoning_summary_text.delta", **part, "delta": text},
        done,
    ]


def _function_call_item(n: int, tool: str, page: int) -> list[dict[str, Any]]:
    arguments = json.dumps({"page": page})
    item = {
        "type": "function_call",
        "id": f"fc_{n}",
        "call_id": f"call_{n}",
        "name": tool,
    }
    return [
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {**item, "arguments": ""},
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": item["id"],
            "delta": arguments,
        },
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {**item, "arguments": arguments, "status": "completed"},
        },
    ]


def _message_item(n: int, text: str) -> list[dict[str, Any]]:
    item = {"type": "message", "id": f"msg_{n}", "role": "assistant"}
    content = [{"type": "output_text", "text": text, "annotations": []}]
    return [
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {**item, "content": []},
        },
        {"type": "response.output_text.delta", "item_id": item["id"], "delta": text},
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {**item, "content": content},
        },
    ]
