"""Cutting a reply into the deltas a streamed response sends."""

import json
import re
from dataclasses import dataclass
from typing import Any

_WORD = re.compile(r"\s*\S+")


@dataclass(frozen=True)
class Event:
    """One server-sent event, sent after waiting ``delay_ms``."""

    name: str | None
    data: Any
    delay_ms: int


def text_chunks(text: str, words: int) -> list[str]:
    """``text`` in pieces of ``words`` words that join back to exactly ``text``."""
    tokens = _WORD.findall(text)
    if not tokens:
        return [text] if text else []
    tokens[-1] += text[sum(len(t) for t in tokens) :]
    return ["".join(tokens[i : i + words]) for i in range(0, len(tokens), words)]


def json_chunks(value: Any, pieces: int = 3) -> list[str]:
    """``value`` as JSON text, cut into a few partial strings as a model streams
    tool arguments."""
    text = json.dumps(value)
    size = max(1, -(-len(text) // pieces))
    return [text[i : i + size] for i in range(0, len(text), size)]


def sse(event: Event) -> bytes:
    prefix = f"event: {event.name}\n" if event.name else ""
    data = event.data if isinstance(event.data, str) else json.dumps(event.data)
    return f"{prefix}data: {data}\n\n".encode()
