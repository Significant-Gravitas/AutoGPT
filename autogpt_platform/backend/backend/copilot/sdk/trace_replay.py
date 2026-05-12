"""Replay langfuse-captured SDK message sequences through the response adapter.

Used by regression tests to verify the adapter behaves correctly on real
production failure traces (e.g. the empty-completion false-positive that
PR #13090 fixes). Fixtures live under ``_test_fixtures/`` and are produced
by ``scripts/dump_langfuse_traces.py``.

The fixture shape:

```
{
  "session_id": "...",
  "trace_id": "...",
  "expected_outcome": "no_overlay" | "specific_error",
  "sequence": [
    {"kind": "assistant", "content": [<TextBlock|ThinkingBlock|ToolUseBlock>]},
    {"kind": "tool_result", "name": "mcp__...", "input": {...}, "output": "..."},
    ...
  ]
}
```

Each ``assistant`` entry becomes one ``AssistantMessage``; each ``tool_result``
entry becomes one ``UserMessage`` with a ``ToolResultBlock``. A synthetic
``ResultMessage`` of the requested subtype is emitted at the end.
"""

import json
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ContentBlock,
    Message,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

FIXTURES_DIR = Path(__file__).parent / "_test_fixtures"


def load_fixture(name: str) -> dict:
    """Load a fixture file by short name (e.g. ``trace_b0e8ca83``)."""
    path = FIXTURES_DIR / f"{name}.json"
    return json.loads(path.read_text())


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


def build_messages(
    sequence: list[dict],
    result_subtype: str = "success",
    result_text: str = "",
) -> list[Message]:
    """Convert a fixture sequence into the SDK message stream the adapter
    consumes, capped by a synthetic ``ResultMessage``.

    Tool-result entries are matched to the most recent unresolved
    ``ToolUseBlock`` of the same name (best-effort FIFO — langfuse drops
    the original tool_use_id linkage, so we re-derive it from order).
    """
    # Init message — adapter expects this first.
    msgs: list[Message] = [SystemMessage(subtype="init", data={})]

    unresolved_by_name: dict[str, list[str]] = {}

    for step in sequence:
        kind = step.get("kind")
        if kind == "assistant":
            blocks: list[ContentBlock] = []
            for raw in step.get("content", []):
                block = _block_from_dict(raw)
                if block is None:
                    continue
                if isinstance(block, ToolUseBlock):
                    unresolved_by_name.setdefault(block.name, []).append(block.id)
                blocks.append(block)
            msgs.append(AssistantMessage(content=blocks, model="test"))
        elif kind == "tool_result":
            tool_name = step.get("name") or "unknown"
            queue = unresolved_by_name.get(tool_name) or []
            if not queue:
                # Result for an unknown tool — synthesise an ID; the
                # adapter ignores tool_results without a matching tool_use.
                continue
            tool_use_id = queue.pop(0)
            output = step.get("output") or ""
            if not isinstance(output, str):
                output = json.dumps(output)
            msgs.append(
                UserMessage(
                    content=[
                        ToolResultBlock(
                            tool_use_id=tool_use_id,
                            content=output,
                        )
                    ],
                )
            )
    msgs.append(
        ResultMessage(
            subtype=result_subtype,
            duration_ms=100,
            duration_api_ms=50,
            is_error=result_subtype != "success",
            num_turns=len([m for m in msgs if isinstance(m, AssistantMessage)]),
            session_id="replay",
            result=result_text,
            usage={"output_tokens": 0},
        )
    )
    return msgs


def replay(
    fixture_name: str,
    adapter,
    result_subtype: str = "success",
    result_text: str = "",
) -> list:
    """Drive the supplied adapter through the fixture and return all
    ``StreamBaseResponse`` events the adapter emitted (concatenated).
    """
    fixture = load_fixture(fixture_name)
    msgs = build_messages(
        fixture["sequence"],
        result_subtype=result_subtype,
        result_text=result_text,
    )
    out: list = []
    for m in msgs:
        out.extend(adapter.convert_message(m))
    return out
