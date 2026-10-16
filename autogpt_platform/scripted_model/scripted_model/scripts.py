"""Scripts: what the fake model answers, one model turn at a time.

A journey picks a script with a marker in its chat message,
``[[script:<name>]]``, or ``[[script:<name>#<nonce>]]`` to find its own
requests in the journal later. The script is ``<scripts dir>/<name>.json``.
It is read on every request, so an edit applies without a restart.

Turn ``n`` answers the model request whose conversation already holds ``n``
assistant messages. Every request carries the whole conversation, so this
needs no state: concurrent runs of one script, a platform retry and a resumed
session all land on the right turn.
"""

import json
import re
from collections.abc import Set
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MARKER = re.compile(r"\[\[script:([a-z0-9][a-z0-9-]*)(?:#([A-Za-z0-9_.-]+))?\]\]")

DEFAULT_TITLE = "Scripted chat"


class ScriptError(Exception):
    """A model request the script can't answer. The message says why."""


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True)
class ToolCall:
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class Expectation:
    """A tool result the turn expects to find in the request. Only logged."""

    tool: str
    contains: str | None = None
    is_error: bool = False


@dataclass(frozen=True)
class Turn:
    text: str
    tool_calls: tuple[ToolCall, ...]
    usage: Usage
    expect_tool_results: tuple[Expectation, ...]
    # Text the user's newest message must contain, e.g. a queued follow-up.
    expect_user_text: str | None
    # How the text streams: this many words per delta, this long between them.
    chunk_words: int
    delay_ms: int


@dataclass(frozen=True)
class Script:
    name: str
    title: str
    turns: tuple[Turn, ...]


@dataclass(frozen=True)
class Marker:
    script: str
    nonce: str | None


def find_marker(text: str) -> Marker | None:
    match = MARKER.search(text)
    if match is None:
        return None
    return Marker(script=match.group(1), nonce=match.group(2))


def load_script(scripts_dir: Path, name: str) -> Script:
    path = scripts_dir / f"{name}.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        known = sorted(p.stem for p in scripts_dir.glob("*.json"))
        raise ScriptError(f"no script named {name!r}; known scripts: {known}")
    except json.JSONDecodeError as error:
        raise ScriptError(f"script {name!r} is not valid JSON: {error}")
    return parse_script(name, raw)


def parse_script(name: str, raw: Any) -> Script:
    data = _object(
        raw, f"script {name!r}", required={"turns"}, optional={"title", "description"}
    )
    turns = data["turns"]
    if not isinstance(turns, list) or not turns:
        raise ScriptError(f"script {name!r}: 'turns' must be a non-empty list")
    return Script(
        name=name,
        title=str(data.get("title") or DEFAULT_TITLE),
        turns=tuple(_turn(f"script {name!r} turn {i}", t) for i, t in enumerate(turns)),
    )


def _turn(where: str, raw: Any) -> Turn:
    data = _object(
        raw,
        where,
        required={"usage"},
        optional={
            "text",
            "tool_calls",
            "expect_tool_results",
            "expect_user_text",
            "stream",
        },
    )
    text = data.get("text", "")
    calls = [
        _tool_call(f"{where} tool call {i}", c)
        for i, c in enumerate(data.get("tool_calls", []))
    ]
    if not isinstance(text, str) or not (text or calls):
        raise ScriptError(f"{where}: needs 'text', 'tool_calls' or both")
    stream = _object(
        data.get("stream", {}), f"{where} stream", optional={"chunk_words", "delay_ms"}
    )
    return Turn(
        text=text,
        tool_calls=tuple(calls),
        usage=_usage(f"{where} usage", data["usage"]),
        expect_tool_results=tuple(
            _expectation(f"{where} expectation {i}", e)
            for i, e in enumerate(data.get("expect_tool_results", []))
        ),
        expect_user_text=(
            None
            if data.get("expect_user_text") is None
            else str(data["expect_user_text"])
        ),
        chunk_words=_count(
            stream.get("chunk_words", 3), f"{where} chunk_words", minimum=1
        ),
        delay_ms=_count(stream.get("delay_ms", 15), f"{where} delay_ms"),
    )


def _tool_call(where: str, raw: Any) -> ToolCall:
    data = _object(raw, where, required={"name", "input"})
    if not isinstance(data["input"], dict):
        raise ScriptError(f"{where}: 'input' must be a JSON object")
    return ToolCall(name=str(data["name"]), input=data["input"])


def _usage(where: str, raw: Any) -> Usage:
    data = _object(
        raw,
        where,
        required={"input_tokens", "output_tokens"},
        optional={"cache_read_input_tokens", "cache_creation_input_tokens"},
    )
    return Usage(
        **{key: _count(value, f"{where} {key}") for key, value in data.items()}
    )


def _expectation(where: str, raw: Any) -> Expectation:
    data = _object(raw, where, required={"tool"}, optional={"contains", "is_error"})
    return Expectation(
        tool=str(data["tool"]),
        contains=None if data.get("contains") is None else str(data["contains"]),
        is_error=bool(data.get("is_error", False)),
    )


def _object(
    raw: Any,
    where: str,
    required: Set[str] = frozenset(),
    optional: Set[str] = frozenset(),
) -> dict[str, Any]:
    """``raw`` as a JSON object with exactly the allowed keys: a typo fails
    loudly instead of being ignored."""
    if not isinstance(raw, dict):
        raise ScriptError(f"{where}: must be a JSON object")
    if missing := required - raw.keys():
        raise ScriptError(f"{where}: missing {sorted(missing)}")
    if unknown := raw.keys() - required - optional:
        raise ScriptError(f"{where}: unknown keys {sorted(unknown)}")
    return raw


def _count(value: Any, where: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ScriptError(f"{where}: must be an integer >= {minimum}")
    return value
