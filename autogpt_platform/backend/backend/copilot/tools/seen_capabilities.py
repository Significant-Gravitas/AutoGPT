"""Per-turn ``<seen_capabilities>`` block: what this session already knows.

The discovery rule in the prompt says "describe an id before its first use
in this session".  Nothing backed that rule: after a ``--resume`` or a
compaction the model has no reliable memory of which ids it already
described or which skills it already loaded, so it re-describes every
block and re-loads every skill at the top of each turn to be safe
(SECRT-2791: 5-8 wasted calls per turn).

The durable record of those calls is the session's message history, so
this module derives the answer from it rather than keeping new state:
every ``describe_capability`` / ``run_capability`` / ``read_skill`` call
persisted on an assistant row whose tool result is a non-error response.  The engines prepend the rendered block to the current turn's
model input only — like ``<skills_update>`` it is never persisted, so it
is re-derived from the same history on every turn.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from backend.copilot.capabilities.sources import skill_name
from backend.copilot.model import ChatSession
from backend.copilot.service import SEEN_CAPABILITIES_TAG

logger = logging.getLogger(__name__)

DESCRIBE_TOOL = "describe_capability"
RUN_TOOL = "run_capability"
READ_SKILL_TOOL = "read_skill"

# The block rides in every later turn's model input, so bound it.  Most
# recently used ids win: those are the ones the next turn reaches for.
MAX_LISTED_IDS = 40
MAX_LISTED_SKILLS = 20


class SeenCapabilities(BaseModel):
    """Ids already described or run, and skills already loaded, this session."""

    described: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.described or self.skills)


def _call_name_and_args(tool_call: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Name and argument dict of one persisted tool call.

    The SDK engine stores ``{"function": {"name", "arguments": <json str>}}``;
    some rows carry a bare ``name`` and a dict.  Anything unparsable is an
    empty dict so a malformed row costs one skipped call, not the turn.
    """
    function = tool_call.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        raw_args = function.get("arguments")
    else:
        name = tool_call.get("name")
        raw_args = tool_call.get("arguments", tool_call.get("input"))
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except ValueError:
            raw_args = None
    args = dict(raw_args) if isinstance(raw_args, Mapping) else {}
    return (str(name or "").strip(), args)


def _answered_tool_call_ids(session: ChatSession) -> set[str]:
    """Ids of tool calls whose persisted result is a non-error tool response.

    Only an answer that parses as a tool response object with a non-error
    ``type`` proves the model learned something about the id.  Anything
    else — an ``error`` response, an ``input_validation_error``, a
    plain-text interrupted marker, a result row that never landed — is
    treated as unanswered so the id is not marked as seen; the worst case
    is one more describe, which is today's behaviour.
    """
    answered: set[str] = set()
    for msg in session.messages:
        if msg.role != "tool" or not msg.tool_call_id or not msg.content:
            continue
        content = msg.content.lstrip()
        if not content.startswith("{"):
            continue
        try:
            payload = json.loads(content)
        except ValueError:
            continue
        if not isinstance(payload, Mapping):
            continue
        kind = payload.get("type")
        if not isinstance(kind, str) or _is_error_type(kind):
            continue
        answered.add(msg.tool_call_id)
    return answered


def _is_error_type(kind: str) -> bool:
    return kind == "error" or kind.endswith("_error")


def _capability_id(args: Mapping[str, Any]) -> str:
    value = args.get("id")
    return str(value).strip() if isinstance(value, str) else ""


def seen_capabilities(session: ChatSession) -> SeenCapabilities:
    """What the session history shows the model already described or loaded.

    Walks the persisted assistant rows newest-first so the lists favour the
    ids the model used most recently, dedupes, and keeps only calls whose
    persisted result is a non-error tool response.  ``run_capability`` on a ``skill:<name>`` id is a skill
    load (the dispatcher turns it into ``read_skill``; the baseline engine
    persists the call as made); on any other id it counts as described —
    a run that went through had the schema, and a run with bad input was
    answered with it.
    """
    answered = _answered_tool_call_ids(session)
    described: dict[str, None] = {}
    skills: dict[str, None] = {}
    for msg in reversed(session.messages):
        if msg.role != "assistant" or not msg.tool_calls:
            continue
        for tool_call in reversed(msg.tool_calls):
            if not isinstance(tool_call, Mapping):
                continue
            if str(tool_call.get("id") or "") not in answered:
                continue
            name, args = _call_name_and_args(tool_call)
            if name == READ_SKILL_TOOL:
                skill = str(args.get("name") or "").strip().lower()
                if skill:
                    skills.setdefault(skill, None)
                continue
            if name not in (DESCRIBE_TOOL, RUN_TOOL):
                continue
            capability_id = _capability_id(args)
            if not capability_id:
                continue
            skill = skill_name(capability_id)
            if skill is not None and name == RUN_TOOL:
                skills.setdefault(skill, None)
                continue
            if skill is not None:
                capability_id = f"skill:{skill}"
            described.setdefault(capability_id, None)
    return SeenCapabilities(
        described=list(described)[:MAX_LISTED_IDS],
        skills=list(skills)[:MAX_LISTED_SKILLS],
    )


def render_seen_capabilities(seen: SeenCapabilities) -> str:
    """The ``<seen_capabilities>`` block for *seen*, or ``""`` when empty.

    Includes the trailing ``\\n\\n`` separator every server-injected block
    carries so it can be prepended straight onto the turn's user text.
    """
    if not seen:
        return ""
    lines: list[str] = []
    if seen.described:
        lines.append(
            "Already described or run this session (do not call "
            "describe_capability on these again; call run_capability "
            "directly — an input error returns the schema): "
            + ", ".join(seen.described)
        )
    if seen.skills:
        lines.append(
            "Already loaded skills (their bodies are earlier in this "
            "conversation; do not load them again with tool:read_skill or "
            "run_capability unless the body is no longer visible to you): "
            + ", ".join(seen.skills)
        )
    body = "\n".join(lines)
    return f"<{SEEN_CAPABILITIES_TAG}>\n{body}\n</{SEEN_CAPABILITIES_TAG}>\n\n"


def build_seen_capabilities_notice(session: ChatSession) -> str:
    """Per-turn notice for the engines to prepend, or ``""`` on a fresh
    session.  Never raises: a malformed history row degrades to no block,
    which is just today's behaviour."""
    try:
        return render_seen_capabilities(seen_capabilities(session))
    except Exception:
        logger.exception("[capabilities] failed to build seen_capabilities notice")
        return ""
