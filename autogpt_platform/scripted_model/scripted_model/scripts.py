"""Scripts: what the fake model answers, one model turn at a time.

A journey picks a script with a marker in its chat message,
``[[script:<name>]]``, or ``[[script:<name>#<nonce>]]`` to find its own
requests in the journal later. The script is ``<scripts dir>/<name>.json``.
It is read on every request, so an edit applies without a restart.

Turn ``n`` answers the model request whose conversation already holds ``n``
assistant messages after the user message that carries the marker. Every
request carries the whole conversation, so this needs no state: concurrent
runs of one script, a platform retry and a resumed session all land on the
right turn.

When several user messages carry a marker, the newest one picks the script,
so a message the user sends with a marker of its own starts that script. The
platform may resume the earlier conversation or pass it on as text inside the
new message (a turn after a stopped one can do either); the new marker comes
after that text, and no assistant message follows it yet, so both land on the
new script's first turn.

That newest marker is read from the user's own words only. The platform adds
tagged blocks to a user message, such as recalled memories
(``<memory_context>``) or the earlier chat passed on as text
(``<conversation_history>``), and these can quote earlier messages, markers
and all, from this chat or the user's others. A request with no marker in
the user's own words, such as a title request that quotes the chat inside a
``<conversation>`` tag, takes the first marker in its user messages.

A tool call's input can read a value from a result the platform sent back,
as a model reads an id out of what a tool returned: in place of the value,
``{"$from_tool_result": {"tool": "<tool>", "pattern": "<regex>"}}`` takes the
first group the pattern matches in a result of that tool from the last turn:
the last such result with a match whose first group took part (an optional
group can match empty-handed).
"""

import json
import re
from collections.abc import Set
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .streaming import held_chunk, text_chunks

MARKER = re.compile(r"\[\[script:([a-z0-9][a-z0-9-]*)(?:#([A-Za-z0-9_.-]+))?\]\]")

# An opening or closing tag of the blocks the platform adds to a user
# message, e.g. <memory_context> and </memory_context>.
TAG = re.compile(r"<(/?)([A-Za-z][\w-]*)(?:\s[^>]*)?>")

DEFAULT_TITLE = "Scripted chat"

FROM_TOOL_RESULT = "$from_tool_result"


class ScriptError(Exception):
    """A model request the script can't answer. The message says why."""


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True)
class Capture:
    """A tool call argument read from a result the platform sent back: the
    first group ``pattern`` matches in the last turn's results of ``tool``, from
    the last one with a match whose first group took part."""

    tool: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class ToolCall:
    name: str
    # Values may be Captures, filled in when the turn is answered.
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
    # After this many words the reply waits until the journey releases its run
    # (holds.py); None streams the reply straight through.
    hold_after_words: int | None = None


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


def typed_marker(text: str) -> Marker | None:
    """The newest marker in the user's own words in ``text``: the last one
    outside the tagged blocks the platform adds."""
    matches = list(MARKER.finditer(own_words(text)))
    if not matches:
        return None
    return Marker(script=matches[-1].group(1), nonce=matches[-1].group(2))


def own_words(text: str) -> str:
    """``text`` without its tagged blocks. Each closed block goes whole, with
    whatever it nests: a recalled message can hold the very tag that recalls
    it. A tag that never closes stays, as text."""
    blocks: list[tuple[int, int]] = []
    open_tags: list[tuple[str, int]] = []
    for tag in TAG.finditer(text):
        closing, name = tag.group(1), tag.group(2)
        if not closing:
            open_tags.append((name, tag.start()))
            continue
        # It closes the innermost open block of its name, and any block left
        # open inside that one; a closing tag with none open is text.
        depth = next(
            (i for i in reversed(range(len(open_tags))) if open_tags[i][0] == name),
            None,
        )
        if depth is None:
            continue
        start = open_tags[depth][1]
        del open_tags[depth:]
        while blocks and blocks[-1][0] >= start:
            blocks.pop()
        blocks.append((start, tag.end()))
    kept, last = [], 0
    for start, end in blocks:
        kept.append(text[last:start])
        last = end
    kept.append(text[last:])
    return "".join(kept)


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
        data.get("stream", {}),
        f"{where} stream",
        optional={"chunk_words", "delay_ms", "hold_after_words"},
    )
    chunk_words = _count(
        stream.get("chunk_words", 3), f"{where} chunk_words", minimum=1
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
        chunk_words=chunk_words,
        delay_ms=_count(stream.get("delay_ms", 15), f"{where} delay_ms"),
        hold_after_words=_hold_after_words(where, stream, text, chunk_words),
    )


def _hold_after_words(
    where: str, stream: dict[str, Any], text: str, chunk_words: int
) -> int | None:
    if "hold_after_words" not in stream:
        return None
    words = _count(stream["hold_after_words"], f"{where} hold_after_words", minimum=1)
    if held_chunk(chunk_words, words) >= len(text_chunks(text, chunk_words)):
        raise ScriptError(
            f"{where} hold_after_words: the text ends within its first {words} "
            "words, so the reply would never hold"
        )
    return words


def _tool_call(where: str, raw: Any) -> ToolCall:
    data = _object(raw, where, required={"name", "input"})
    if not isinstance(data["input"], dict):
        raise ScriptError(f"{where}: 'input' must be a JSON object")
    return ToolCall(name=str(data["name"]), input=_arguments(where, data["input"]))


def _arguments(where: str, value: Any) -> Any:
    """``value`` with each ``{"$from_tool_result": ...}`` read as a Capture."""
    if isinstance(value, dict):
        if FROM_TOOL_RESULT in value:
            return _capture(where, value)
        return {key: _arguments(f"{where}.{key}", v) for key, v in value.items()}
    if isinstance(value, list):
        return [_arguments(f"{where}[{i}]", v) for i, v in enumerate(value)]
    return value


def _capture(where: str, raw: dict[str, Any]) -> Capture:
    spec = _object(
        _object(raw, where, required={FROM_TOOL_RESULT})[FROM_TOOL_RESULT],
        f"{where} {FROM_TOOL_RESULT}",
        required={"tool", "pattern"},
    )
    try:
        pattern = re.compile(str(spec["pattern"]))
    except re.error as error:
        raise ScriptError(f"{where}: 'pattern' is not a regular expression: {error}")
    if pattern.groups < 1:
        raise ScriptError(f"{where}: 'pattern' needs a group to read the value from")
    return Capture(tool=str(spec["tool"]), pattern=pattern)


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
