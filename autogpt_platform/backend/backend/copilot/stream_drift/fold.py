"""The rows a turn's stream folds to, by the design's row rules.

A reference for the fixture checks, in the canonical shape of
``stream_checkpoint.canonical_rows``: at every checkpoint, and at the end, the
fold of the entries so far must equal the rows the backend persisted.
"""

from collections.abc import Iterable
from typing import Any


def fold_rows(chunks: Iterable[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    assistant: int | None = None  # the row text and tool calls land in
    blocks: dict[str, int] = {}
    for chunk in chunks:
        kind = chunk["type"]
        if kind == "text-start":
            if assistant is None:
                rows.append(["assistant", "", []])
                assistant = len(rows) - 1
            blocks[chunk["id"]] = assistant
        elif kind == "reasoning-start":
            rows.append(["reasoning", "", []])
            blocks[chunk["id"]] = len(rows) - 1
        elif kind in ("text-delta", "reasoning-delta"):
            rows[blocks[chunk["id"]]][1] += chunk["delta"]
        elif kind == "tool-input-available":
            if assistant is None:
                rows.append(["assistant", "", []])
                assistant = len(rows) - 1
            rows[assistant][2].append([chunk["toolCallId"], chunk["toolName"]])
        elif kind == "tool-output-available":
            rows.append(["tool", chunk["toolCallId"], []])
            assistant = None
        elif kind == "data-pending-drained":
            rows.extend(["user", m["content"], []] for m in chunk["messages"])
            assistant = None
    return rows
