"""Decide what to answer a model request, whichever wire format it came in."""

from dataclasses import dataclass, field, replace
from pathlib import Path

from .invoice import cost_usd
from .scripts import (
    Expectation,
    Marker,
    Script,
    ScriptError,
    Turn,
    Usage,
    find_marker,
    load_script,
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
    marker = find_marker(conversation.user_text)
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
    index = conversation.assistant_messages
    if index >= len(script.turns):
        raise ScriptError(
            f"script {script.name!r} has {len(script.turns)} turn(s), but the "
            f"platform asked for turn {index + 1}: the conversation went further "
            "than the script expected (an extra tool round, retry or message)."
        )
    turn = script.turns[index]
    return Plan(
        kind="turn",
        marker=marker,
        turn_index=index,
        turn_count=len(script.turns),
        reply=_reply(script, index, turn, conversation.offered_tools),
        checks=_checks(turn, conversation),
    )


def _reply(script: Script, index: int, turn: Turn, offered: tuple[str, ...]) -> Reply:
    return Reply(
        text=turn.text,
        tool_calls=tuple(
            (_offered_name(script, index, call.name, offered), call.input)
            for call in turn.tool_calls
        ),
        usage=turn.usage,
        chunk_words=turn.chunk_words,
        delay_ms=turn.delay_ms,
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
