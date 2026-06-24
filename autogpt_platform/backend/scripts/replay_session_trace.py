#!/usr/bin/env python3
"""Replay a langfuse-captured copilot session trace through the response adapter.

Local debugging utility for investigating "empty response" / spurious-overlay
incidents on dev or prod. Pulls the trace by session ID, reconstructs the SDK
message stream the adapter would have seen (AssistantMessage / UserMessage /
ResultMessage), and prints whether ``StreamError(code="empty_completion")``
would fire — with the current adapter code in this checkout.

Usage (must be run as a module so package imports resolve):
    LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... LANGFUSE_HOST=... \
        poetry run python -m scripts.replay_session_trace <session_id> [<session_id> ...]

Optional flags:
    --subtype <subtype>   ResultMessage subtype to cap the stream with
                          (default: success). Use error_max_budget_usd /
                          error_max_turns / error / error_during_execution
                          when investigating those failure modes.

Does NOT make any modifications. Read-only against langfuse + the local
adapter code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from claude_agent_sdk import (
    AssistantMessage,
    ContentBlock,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from backend.copilot.response_model import StreamError
from backend.copilot.sdk.response_adapter import SDKResponseAdapter


def _block_from_dict(b: dict) -> ContentBlock | None:
    t = b.get("type")
    if t == "text":
        return TextBlock(text=b.get("text", ""))
    if t == "thinking":
        return ThinkingBlock(
            thinking=b.get("thinking", ""),
            signature=b.get("signature", ""),
        )
    if t == "tool_use":
        return ToolUseBlock(
            id=b.get("id", ""),
            name=b.get("name", "unknown"),
            input=b.get("input") or {},
        )
    return None


def _fetch_observations(session_id: str) -> list[dict]:
    """Pull the largest trace for the session and return its observations
    sorted by start_time.
    """
    from langfuse import Langfuse

    lf = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ["LANGFUSE_HOST"],
    )
    traces = lf.api.trace.list(session_id=session_id, limit=20).data
    if not traces:
        return []
    best = max(traces, key=lambda t: len(t.observations or []))
    trace = lf.api.trace.get(best.id)
    obs = sorted(trace.observations or [], key=lambda o: o.start_time)
    out: list[dict] = []
    for o in obs:
        if o.type == "GENERATION" and o.name == "claude.assistant.turn":
            if o.output:
                content = (
                    o.output.get("content", []) if isinstance(o.output, dict) else []
                )
                out.append({"kind": "assistant", "content": content})
        elif o.type == "TOOL":
            output = o.output
            if not isinstance(output, str):
                output = json.dumps(output) if output is not None else ""
            # Capture the input so the replay can match this tool_result to
            # the right pending ToolUseBlock when multiple same-name calls
            # are outstanding (e.g. two parallel ``find_block`` calls).
            inp = o.input if isinstance(o.input, dict) else {}
            out.append(
                {"kind": "tool_result", "name": o.name, "input": inp, "output": output}
            )
    return out


def replay_session(session_id: str, result_subtype: str = "success") -> dict:
    """Replay one session through a fresh adapter; return summary dict."""
    sequence = _fetch_observations(session_id)
    if not sequence:
        return {"session_id": session_id, "error": "no traces found"}

    adapter = SDKResponseAdapter(session_id=session_id)
    events: list = []
    events.extend(adapter.convert_message(SystemMessage(subtype="init", data={})))

    # Map name -> list of (tool_use_id, input_dict) for outstanding calls.
    # Match tool_results by (name, input) when possible — same-name parallel
    # calls (e.g. two ``find_block`` with different queries) would otherwise
    # be replayed against the wrong ToolUseBlock under FIFO-by-name.
    unresolved: dict[str, list[tuple[str, dict]]] = {}
    for step in sequence:
        if step["kind"] == "assistant":
            blocks: list[ContentBlock] = []
            for raw in step.get("content", []):
                block = _block_from_dict(raw)
                if block is None:
                    continue
                if isinstance(block, ToolUseBlock):
                    unresolved.setdefault(block.name, []).append(
                        (block.id, block.input or {})
                    )
                blocks.append(block)
            events.extend(
                adapter.convert_message(AssistantMessage(content=blocks, model="test"))
            )
        elif step["kind"] == "tool_result":
            queue = unresolved.get(step["name"]) or []
            if not queue:
                continue
            # Prefer matching the queued call whose input matches the
            # tool_result's input; fall back to FIFO if no input match.
            target_input = step.get("input") or {}
            match_idx = next(
                (i for i, (_, inp) in enumerate(queue) if inp == target_input),
                0,
            )
            tool_use_id, _ = queue.pop(match_idx)
            events.extend(
                adapter.convert_message(
                    UserMessage(
                        content=[
                            ToolResultBlock(
                                tool_use_id=tool_use_id,
                                content=step.get("output") or "",
                            )
                        ],
                    )
                )
            )

    events.extend(
        adapter.convert_message(
            ResultMessage(
                subtype=result_subtype,
                duration_ms=100,
                duration_api_ms=50,
                is_error=result_subtype != "success",
                num_turns=1,
                session_id=session_id,
                result="",
                usage={"output_tokens": 0},
            )
        )
    )

    stream_errors = [
        {"code": e.code, "text": e.errorText[:120]}
        for e in events
        if isinstance(e, StreamError)
    ]
    return {
        "session_id": session_id,
        "subtype": result_subtype,
        "steps": len(sequence),
        "any_real_tool_result_seen": adapter._any_real_tool_result_seen,
        "any_orphan_flush_seen": adapter._any_orphan_flush_seen,
        "has_started_text": adapter.has_started_text,
        "emitted_real_content_to_wire": adapter.emitted_real_content_to_wire,
        "stream_errors": stream_errors,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("session_ids", nargs="+")
    p.add_argument("--subtype", default="success")
    args = p.parse_args()

    for sid in args.session_ids:
        result = replay_session(sid, result_subtype=args.subtype)
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
