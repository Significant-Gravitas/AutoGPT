"""What one scheduled run came to, in the terms a failure streak counts.

A scheduled chat that keeps failing the same way is paused on the third run
and its owner told once (SECRT-2879). "The same way" has to be precise enough
to pick the fix: which connection, the run's spending limit, or one tool's
error. It has to be stable without storing anything a tool's error text says,
because that can quote the user's data, so an uncoded error is compared by a
hash of its text.

Two distinctions decide what the streak does with a failure:

* A failed step is not a failed run. A morning briefing whose Twitter step is
  refused still goes out without Twitter; pausing it would stop the rest too.
  A run fails outright when it ends in an error, or nothing it tried worked.
* A run that calls no tool proves nothing either way. The every-minute monitor
  that could not reach Discord spent its later runs repeating "Discord still
  needs connecting" without trying: those runs must not clear the streak.
"""

import hashlib
import re
from typing import Literal

import orjson
from pydantic import BaseModel

from backend.copilot.tools.models import ResponseType

# An error code such as ``credential_rejected``; anything else is free text.
# The same rule as the scheduled-turn alert's.
_ERROR_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
UNCLASSIFIED = "unclassified"
# A word with a digit in it: a count, an id, a time, a duration. Two runs that
# fail the same way rarely agree on one.
_VOLATILE_WORD = re.compile(r"\w*\d\w*")

# A connection the owner fixes on the integrations page.
CONNECTION_ERRORS = frozenset(
    {
        "missing_credentials",
        "credential_rejected",
        "pinned_credential_missing",
        "pinned_credential_unusable",
    }
)
# A chat platform (Discord, Slack, ...) not linked to the account yet.
CHAT_LINK_ERRORS = frozenset({"no_linked_servers", "no_dm_link"})
# Turn errors that are this schedule's own doing: it asks for more than one
# run may spend or take. Every other turn error (an API blip, an idle timeout)
# is the platform's weather, which our own alerts cover, and neither counts
# toward a pause nor clears one.
COUNTED_TURN_ERRORS = frozenset({"max_budget_exhausted", "max_turns_exhausted"})
# A run that reaches either limit lands as a pause (``turn_pause``), and a
# pause at either counts as that limit's error. A pause because the account's
# daily, weekly or trial allowance ran out is the owner's state, not this
# schedule's, and counts as neither.
PAUSE_ERROR_CODES = {"task": "max_budget_exhausted", "steps": "max_turns_exhausted"}

FailureKind = Literal["connection", "chat_link", "budget", "error"]


class RunFailure(BaseModel):
    # What "failed the same way" compares across runs.
    key: str
    kind: FailureKind
    provider: str | None = None
    # The tool that failed, or None when the run itself errored.
    tool: str | None = None


class RunOutcome(BaseModel):
    # None when the run was fine.
    failure: RunFailure | None = None
    # The run failed outright, rather than one of its steps.
    whole_run: bool = False
    # Called no platform tool: it neither counts nor clears a failure of its
    # own, but continues a connection failure that is still open.
    idle: bool = False
    # Cancelled, or our outage: neither counts nor clears anything.
    neutral: bool = False


class ToolFailureDetail(BaseModel):
    error_type: str
    provider: str | None = None
    # For an uncoded error: a short hash of its text, never the text.
    text_hash: str | None = None


def classify_run(
    *,
    cancelled: bool,
    turn_error_code: str | None,
    turn_errored: bool,
    tool_failures: list[tuple[str, ToolFailureDetail]],
    tool_successes: int,
) -> RunOutcome:
    """*tool_failures* pairs each failing platform tool with what it said;
    *tool_successes* counts the platform tool calls that worked, leaving out
    the lookups that only got a run ready to work."""
    if cancelled:
        return RunOutcome(neutral=True)
    if turn_errored:
        if turn_error_code not in COUNTED_TURN_ERRORS:
            return RunOutcome(neutral=True)
        return RunOutcome(
            failure=RunFailure(key=f"turn:{turn_error_code}", kind="budget"),
            whole_run=True,
        )
    if tool_failures:
        tool, detail = _leading_failure(tool_failures)
        return RunOutcome(
            failure=_failure_of(tool, detail), whole_run=tool_successes == 0
        )
    return RunOutcome(idle=tool_successes == 0)


def _leading_failure(
    failures: list[tuple[str, ToolFailureDetail]],
) -> tuple[str, ToolFailureDetail]:
    """A connection the owner can fix explains a run better than whatever
    else failed. Of several, the least key: tools called together finish in
    any order, and a run failing the same ways must fail the same way."""
    fixable = CONNECTION_ERRORS | CHAT_LINK_ERRORS
    candidates = [f for f in failures if f[1].error_type in fixable] or failures
    return min(candidates, key=lambda failure: _failure_of(*failure).key)


def _failure_of(tool: str, detail: ToolFailureDetail) -> RunFailure:
    if detail.error_type in CONNECTION_ERRORS:
        provider = detail.provider or tool
        return RunFailure(
            key=f"connection:{provider}",
            kind="connection",
            provider=detail.provider,
            tool=tool,
        )
    if detail.error_type in CHAT_LINK_ERRORS:
        return RunFailure(
            key=f"chat_link:{detail.provider or tool}",
            kind="chat_link",
            provider=detail.provider,
            tool=tool,
        )
    suffix = detail.text_hash or detail.error_type
    return RunFailure(key=f"tool:{tool}:{suffix}", kind="error", tool=tool)


def tool_failure_detail(output: str | dict) -> ToolFailureDetail | None:
    """What a copilot tool's ``ErrorResponse`` said, or None when the output
    is not one. Tools report failures as a payload rather than by raising."""
    payload = output
    if isinstance(output, str):
        try:
            payload = orjson.loads(output)
        except orjson.JSONDecodeError:
            return None
    if not isinstance(payload, dict) or payload.get("type") != ResponseType.ERROR:
        return None
    error = payload.get("error")
    details = payload.get("details")
    provider = details.get("provider") if isinstance(details, dict) else None
    provider = provider if isinstance(provider, str) and provider else None
    if isinstance(error, str) and _ERROR_CODE.fullmatch(error):
        return ToolFailureDetail(error_type=error, provider=provider)
    return ToolFailureDetail(
        error_type=UNCLASSIFIED,
        provider=provider,
        text_hash=text_hash(error or payload.get("message")),
    )


def text_hash(text: object) -> str:
    """Compares two uncoded errors without keeping what either says. Words
    with a digit in them don't count, so a timeout after 30 s and one after
    31 s, each with its own request id, fail the same way."""
    raw = text if isinstance(text, str) else repr(text)
    return hashlib.sha256(_VOLATILE_WORD.sub("#", raw).encode()).hexdigest()[:12]
