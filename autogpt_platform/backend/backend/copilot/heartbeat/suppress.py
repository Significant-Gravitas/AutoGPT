"""Which heartbeat answers reach the user.

OpenClaw's rules: the silent token is stripped wherever it sits; what is left
is dropped when it is short (300 characters by default) unless the model
alerted on purpose, through ``heartbeat_respond``.
"""

import re
from typing import Any, Literal

from pydantic import BaseModel

from .prompt import RESPOND_TOOL, SILENT_TOKENS

# A reply shorter than this, once the tokens are gone, is an acknowledgement.
ACK_MAX_CHARS = 300
_TOKENS = re.compile(
    r"\**\b(" + "|".join(re.escape(t) for t in SILENT_TOKENS) + r")\b\**[.!]?",
    re.IGNORECASE,
)


class Verdict(BaseModel):
    deliver: bool
    text: str = ""
    reason: Literal["explicit_alert", "long_reply", "silent", "short_reply", "declined"]


class ExplicitResponse(BaseModel):
    notify: bool
    notification_text: str = ""


def strip_tokens(text: str) -> str:
    return _TOKENS.sub("", text or "").strip()


def decide(reply_text: str, explicit: ExplicitResponse | None) -> Verdict:
    """What, if anything, this heartbeat says to the user.

    An explicit ``heartbeat_respond`` decides outright: notify with its text,
    or stay quiet whatever the reply says. Without one, a reply that is only
    a silent token, or short once the tokens are stripped, says nothing.
    """
    if explicit is not None:
        text = strip_tokens(explicit.notification_text)
        if explicit.notify and text:
            return Verdict(deliver=True, text=text, reason="explicit_alert")
        return Verdict(deliver=False, reason="declined")
    text = strip_tokens(reply_text)
    if not text:
        return Verdict(deliver=False, reason="silent")
    if len(text) < ACK_MAX_CHARS:
        return Verdict(deliver=False, reason="short_reply")
    return Verdict(deliver=True, text=text, reason="long_reply")


def explicit_from_tool_calls(tool_calls: list[Any]) -> ExplicitResponse | None:
    """The last ``heartbeat_respond`` call in the turn, named directly or
    through ``run_capability``. Engines prefix MCP tool names, hence the
    suffix match."""
    found: ExplicitResponse | None = None
    for call in tool_calls:
        name = str(_field(call, "tool_name") or "")
        args = _field(call, "input")
        if not isinstance(args, dict):
            continue
        if name.endswith("run_capability"):
            if str(args.get("id") or "").removeprefix("tool:") != RESPOND_TOOL:
                continue
            args = args.get("input")
            if not isinstance(args, dict):
                continue
        elif not name.endswith(RESPOND_TOOL):
            continue
        found = ExplicitResponse(
            notify=bool(args.get("notify")),
            notification_text=str(args.get("notification_text") or ""),
        )
    return found


def _field(call: Any, name: str) -> Any:
    return call.get(name) if isinstance(call, dict) else getattr(call, name, None)
