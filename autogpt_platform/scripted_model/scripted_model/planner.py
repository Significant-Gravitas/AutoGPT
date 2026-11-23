"""Decide what to answer a model request, whichever wire format it came in."""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .invoice import cost_usd
from .scripts import (
    Capture,
    Expectation,
    Marker,
    Script,
    ScriptError,
    Turn,
    Usage,
    find_marker,
    load_script,
    typed_marker,
)

# What a side call (a request offering no tools: a session title, a summary)
# costs when no script names it.
SIDE_CALL_USAGE = Usage(input_tokens=200, output_tokens=10)


@dataclass(frozen=True)
class ToolResult:
    tool: str
    content: str
    is_error: bool


@dataclass(frozen=True)
class Conversation:
    """The parts of a model request a script cares about."""

    model: str
    stream: bool
    # Every user message's text, in order; where the marker is looked for.
    user_text: str
    assistant_messages: int
    # Tool names the request offers the model, as the model must call them.
    offered_tools: tuple[str, ...]
    # Results the platform sent back since the last assistant message.
    tool_results: tuple[ToolResult, ...]
    # What the user said since the last assistant message.
    latest_user_text: str = ""
    # The prompt-cache lifetime the request asked for, if any.
    cache_ttl: str | None = None
    # The system prompt's text.
    system_text: str = ""
    # Assistant messages before the user message that carries the marker; the
    # script's turns count from that message.
    assistant_messages_before_marker: int = 0

    def marker(self) -> Marker | None:
        """The marker that picks the script: the newest one in the user's own
        words; with none there, the first in the user's messages (a title
        request quotes the chat inside a ``<conversation>`` tag), else one in
        the system prompt. A turn the platform sends itself (an expert's
        kickoff) has no user-typed marker, so a journey puts one in text the
        user controls that the system prompt carries (an expert's writing
        sample)."""
        return (
            typed_marker(self.user_text)
            or find_marker(self.user_text)
            or find_marker(self.system_text)
        )


def assistant_messages_before_marker(messages: Sequence[tuple[str, str]]) -> int:
    """How many assistant messages come before the newest user message whose
    own words carry a marker, given each message's role and text. 0 when
    none does: a marker in the system prompt counts every turn."""
    marked = max(
        (
            index
            for index, (role, text) in enumerate(messages)
            if role == "user" and typed_marker(text)
        ),
        default=0,
    )
    return sum(1 for role, _ in messages[:marked] if role == "assistant")


@dataclass(frozen=True)
class Check:
    what: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class Reply:
    text: str
    # (name as offered to the model, input)
    tool_calls: tuple[tuple[str, dict], ...]
    usage: Usage
    chunk_words: int = 3
    delay_ms: int = 15
    hold_after_words: int | None = None
    # Which bucket cache writes are reported in, and what the call costs.
    cache_ttl: str | None = None
    cost_usd: float | None = None


@dataclass(frozen=True)
class Plan:
    kind: str  # "turn" or "side_call"
    reply: Reply
    marker: Marker | None = None
    turn_index: int | None = None
    turn_count: int | None = None
    checks: tuple[Check, ...] = field(default_factory=tuple)


def plan(conversation: Conversation, scripts_dir: Path) -> Plan:
    """The priced reply for this request, or ScriptError when the script can't
    give one."""
    planned = _plan(conversation, scripts_dir)
    reply = planned.reply
    priced = replace(
        reply,
        cache_ttl=conversation.cache_ttl,
        cost_usd=cost_usd(conversation.model, reply.usage, conversation.cache_ttl),
    )
    return replace(planned, reply=priced)


def _plan(conversation: Conversation, scripts_dir: Path) -> Plan:
    """The reply for this request, or ScriptError when the script can't give one.

    A request offering no tools is a side call the platform makes around a
    turn (a session title, say): it gets the script's title, or a fixed
    answer. A request offering tools is a model turn and must belong to a
    script.
    """
    marker = conversation.marker()
    script = load_script(scripts_dir, marker.script) if marker else None
    if not conversation.offered_tools:
        title = script.title if script else "Scripted reply"
        return Plan(
            kind="side_call",
            marker=marker,
            reply=Reply(text=title, tool_calls=(), usage=SIDE_CALL_USAGE, delay_ms=0),
        )
    if script is None:
        raise ScriptError(
            "this conversation has no [[script:<name>]] marker, so the scripted "
            "model has nothing to answer with. Put the marker in the first message."
        )
    index = (
        conversation.assistant_messages - conversation.assistant_messages_before_marker
    )
    if index >= len(script.turns):
        raise ScriptError(
            f"script {script.name!r} has {len(script.turns)} turn(s), but the "
            f"platform asked for turn {index + 1}: the conversation went further "
            "than the script expected (an extra tool round, retry or message)."
        )
    turn = script.turns[index]
    if turn.hold_after_words is not None and not (marker and marker.nonce):
        raise ScriptError(
            f"script {script.name!r} turn {index + 1} holds its reply until the "
            "journey releases the run, which it names by nonce: put "
            f"[[script:{script.name}#<nonce>]] in the first message."
        )
    return Plan(
        kind="turn",
        marker=marker,
        turn_index=index,
        turn_count=len(script.turns),
        reply=_reply(script, index, turn, conversation),
        checks=_checks(turn, conversation),
    )


def _reply(script: Script, index: int, turn: Turn, conversation: Conversation) -> Reply:
    return Reply(
        text=turn.text,
        tool_calls=tuple(
            (
                _offered_name(script, index, call.name, conversation.offered_tools),
                _arguments(script, index, call.input, conversation.tool_results),
            )
            for call in turn.tool_calls
        ),
        usage=turn.usage,
        chunk_words=turn.chunk_words,
        delay_ms=turn.delay_ms,
        hold_after_words=turn.hold_after_words,
    )


def _arguments(
    script: Script, index: int, value: Any, results: tuple[ToolResult, ...]
) -> Any:
    """The tool call's input, with each value the script reads from a tool
    result (a ``Capture``) filled in from this request's results."""
    if isinstance(value, Capture):
        return _captured(script, index, value, results)
    if isinstance(value, dict):
        return {k: _arguments(script, index, v, results) for k, v in value.items()}
    if isinstance(value, list):
        return [_arguments(script, index, v, results) for v in value]
    return value


def _captured(
    script: Script, index: int, capture: Capture, results: tuple[ToolResult, ...]
) -> str:
    """What the capture's pattern reads from a result of its tool, the way a
    model reads an id out of what a tool just returned. *results* are only
    the ones this request sends back, for the last turn's calls, so an
    earlier turn's result (and its possibly stale id) is never read. Of
    several calls to the tool in that turn, the last one with a match whose
    first group took part wins: an optional group can match empty-handed."""
    for result in reversed(results):
        if result.tool != capture.tool:
            continue
        for match in capture.pattern.finditer(result.content):
            # An optional group can match without taking part: no value.
            if (value := match.group(1)) is not None:
                return value
    raise ScriptError(
        f"script {script.name!r} turn {index + 1} reads an argument from a "
        f"{capture.tool!r} result, but none in this request matches "
        f"{capture.pattern.pattern!r}. Results: "
        f"{[(r.tool, r.content[:200]) for r in results]}"
    )


def _offered_name(
    script: Script, index: int, name: str, offered: tuple[str, ...]
) -> str:
    """The tool as the platform offers it: scripts name ``create_agent``, the
    SDK engine offers it as ``mcp__<server>__create_agent``."""
    for candidate in offered:
        if candidate == name or candidate.endswith(f"__{name}"):
            return candidate
    raise ScriptError(
        f"script {script.name!r} turn {index + 1} calls tool {name!r}, which the "
        f"platform did not offer. Offered: {sorted(bare_name(t) for t in offered)}"
    )


def _checks(turn: Turn, conversation: Conversation) -> tuple[Check, ...]:
    checks = [_check(e, conversation.tool_results) for e in turn.expect_tool_results]
    if turn.expect_user_text is not None:
        said = conversation.latest_user_text
        checks.append(
            Check(
                "user message",
                turn.expect_user_text in said,
                (
                    "matched"
                    if turn.expect_user_text in said
                    else f"the user said: {said[-500:]!r}"
                ),
            )
        )
    return tuple(checks)


def _check(expectation: Expectation, results: tuple[ToolResult, ...]) -> Check:
    what = f"{expectation.tool} result"
    matches = [r for r in results if r.tool == expectation.tool]
    if not matches:
        seen = [r.tool for r in results]
        return Check(what, False, f"no result from {expectation.tool!r}; got {seen}")
    for result in matches:
        if result.is_error != expectation.is_error:
            continue
        if expectation.contains is None or expectation.contains in result.content:
            return Check(what, True, "matched")
    first = matches[0]
    return Check(
        what,
        False,
        f"{expectation.tool!r} returned is_error={first.is_error}: {first.content[:500]}",
    )


def bare_name(tool: str) -> str:
    """``mcp__copilot__create_agent`` -> ``create_agent``."""
    return tool.rsplit("__", 1)[-1]
