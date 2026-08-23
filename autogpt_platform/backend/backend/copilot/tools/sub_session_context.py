"""ETA, definition-of-done, and phase timeline for delegated work.

Three small concerns shared by ``run_sub_session``, ``delegate_to_expert``
and ``get_sub_session_result``, kept together here rather than bolted onto
any one of those already-long modules:

* **Estimate** — the delegating model's own ``estimated_minutes`` guess. It
  is recorded on the sub's session metadata so a later poll (or a wake-up
  turn that never saw the original tool call) can still state it.
* **Success criteria** — the caller's testable definition of done. It goes
  two places: into the delegated prompt, so the sub knows what it is being
  measured against, and onto the sub's metadata, so the parent can re-read
  it on completion instead of taking "done" on faith.
* **Phases** — a progress timeline for the parent's card. ``TodoWrite`` is
  stateless: the model's most recent ``todos`` argument in the transcript
  *is* the canonical list (see :mod:`todo_write`), and it is persisted like
  any other tool call. So the latest snapshot is mined by walking the sub's
  messages backwards — no new table, no new cache, and it works for both
  the baseline MCP tool and the SDK's CLI-native one, which write the same
  arguments under the same name.

Every value here originates in model-authored JSON, so each one is
normalised at this boundary rather than trusted.
"""

import json
import logging
from typing import Any

from backend.copilot.model import ChatMessage, get_chat_session

from .models import SubSessionPhase

logger = logging.getLogger(__name__)

# Enough for a real definition of done, short enough that the delegated
# prompt stays readable and the criteria stay concrete.
MAX_SUCCESS_CRITERIA = 8
_MAX_CRITERION_CHARS = 300

# A delegated turn that genuinely needs longer than a day is not something an
# up-front estimate helps with; clamp rather than surface an absurd ETA.
MAX_ESTIMATED_MINUTES = 24 * 60

# Bounds how much of a long plan we echo back into the parent's context.
_MAX_PHASES = 20

_TODO_TOOL_NAME = "todowrite"


ESTIMATED_MINUTES_PARAM: dict[str, Any] = {
    "type": "integer",
    "description": (
        "Your honest estimate in minutes. Always give one for a build or "
        "research task — the user sees it, so you can promise a time instead "
        "of an open-ended wait."
    ),
}

SUCCESS_CRITERIA_PARAM: dict[str, Any] = {
    "type": "array",
    "items": {"type": "string"},
    "description": (
        "2-5 concrete, checkable conditions that define 'done' (max "
        f"{MAX_SUCCESS_CRITERIA}). Given to the worker up front and returned "
        "to you with the result, so you verify it instead of assuming it."
    ),
}


def normalize_estimated_minutes(raw: Any) -> int | None:
    """Coerce a model-supplied estimate to a sane positive minute count."""
    if raw is None or isinstance(raw, bool):
        return None
    try:
        minutes = int(raw)
    except (TypeError, ValueError):
        return None
    if minutes <= 0:
        return None
    return min(minutes, MAX_ESTIMATED_MINUTES)


def normalize_success_criteria(raw: Any) -> list[str] | None:
    """Coerce model-supplied criteria to a bounded list of non-empty strings.

    Returns ``None`` — not ``[]`` — when nothing usable was supplied, so the
    "caller gave no definition of done" case stays distinguishable from
    "caller gave an empty one" all the way down to the stored metadata.
    """
    if not isinstance(raw, list):
        return None
    criteria = [
        item.strip()[:_MAX_CRITERION_CHARS]
        for item in raw
        if isinstance(item, str) and item.strip()
    ]
    return criteria[:MAX_SUCCESS_CRITERIA] or None


def criteria_preamble(success_criteria: list[str] | None) -> str:
    """The block prepended to a delegated prompt stating what 'done' means.

    Empty string when no criteria were given, so callers can concatenate it
    unconditionally.
    """
    if not success_criteria:
        return ""
    lines = "\n".join(f"- {c}" for c in success_criteria)
    return (
        "[Done means ALL of the following are true. Check each one before "
        f"you report back, and say plainly which are unmet:\n{lines}]"
    )


def completion_criteria_reminder(success_criteria: list[str] | None) -> str:
    """The sentence appended to a completed sub-session's tool message.

    The parent may be a fresh turn woken by ``subsession_wake`` that never saw
    the criteria it once set, so they are restated at the point of judgement
    rather than assumed to be in context.
    """
    if not success_criteria:
        return ""
    lines = "; ".join(success_criteria)
    return (
        f" Before telling the user this is done, check the result against the "
        f"{len(success_criteria)} success criteria you set ({lines}) and name "
        "any that is not met instead of declaring success."
    )


async def latest_sub_phases(inner_session_id: str | None) -> list[SubSessionPhase]:
    """The sub's latest plan, loaded by id. Best-effort: ``[]`` on any failure.

    A phase timeline is a nicety on top of the status the caller actually
    asked for, so a lookup problem must never turn a healthy "still running"
    into an error.
    """
    if not inner_session_id:
        return []
    try:
        sub = await get_chat_session(inner_session_id)
    except Exception:
        logger.debug(
            "Phase snapshot unavailable for sub %s",
            inner_session_id[:12],
            exc_info=True,
        )
        return []
    return phases_from_messages(sub.messages) if sub else []


def phases_from_messages(messages: list[ChatMessage]) -> list[SubSessionPhase]:
    """Mine the most recent ``TodoWrite`` task list out of a transcript.

    Walks backwards and stops at the first entry that yields a usable list —
    every ``TodoWrite`` call carries the *whole* plan, so the newest one is
    the current state and everything before it is history.

    Each message is checked two ways because the assistant row carrying the
    ``todos`` argument is not always persisted: ``tool_calls_pending_save`` is
    in-memory only, and the turn's final all-completed update is the one most
    often lost. The baseline tool's *result* row echoes the same list, so it is
    used as a fallback — the frontend history converter compensates for the
    same gap (``orphanTodoWriteResultToPart``).
    """
    for message in reversed(messages):
        for call in reversed(message.tool_calls or []):
            phases = _phases_from_tool_call(call)
            if phases:
                return phases
        phases = _phases_from_result_row(message)
        if phases:
            return phases
    return []


def _phases_from_tool_call(call: Any) -> list[SubSessionPhase]:
    """Parse one persisted tool call into phases, or ``[]`` if it isn't a
    ``TodoWrite`` (or is one we cannot read).

    Tool calls reach us in more than one shape: the persisted OpenAI form
    nests name/arguments under ``function`` with arguments as a JSON string,
    while the live-drain form carries them flat with a dict payload. Both are
    accepted, the same way ``_already_terminal_result`` does.
    """
    if not isinstance(call, dict):
        return []
    function = call.get("function")
    function = function if isinstance(function, dict) else {}
    name = function.get("name") or call.get("name") or ""
    if not isinstance(name, str) or name.lower() != _TODO_TOOL_NAME:
        return []

    payload = _as_mapping(
        function.get("arguments") or call.get("arguments") or call.get("input")
    )
    return _phases_from_todos(payload.get("todos") if payload else None)


def _phases_from_result_row(message: ChatMessage) -> list[SubSessionPhase]:
    """Recover the list from a ``TodoWrite`` result row whose assistant row
    never made it to the database. Baseline serialises the full
    ``TodoWriteResponse`` (including ``todos``) as the row's content; the SDK
    path returns a plain ack, so this simply finds nothing there."""
    if message.role != "tool":
        return []
    payload = _as_mapping(message.content)
    if not payload or payload.get("type") != "todo_write":
        return []
    return _phases_from_todos(payload.get("todos"))


def _phases_from_todos(raw_todos: Any) -> list[SubSessionPhase]:
    if not isinstance(raw_todos, list):
        return []
    phases: list[SubSessionPhase] = []
    for item in raw_todos[:_MAX_PHASES]:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        status = item.get("status")
        phases.append(
            SubSessionPhase(
                content=content.strip(),
                status=(
                    status
                    if status in ("pending", "in_progress", "completed")
                    else "pending"
                ),
            )
        )
    return phases


def _as_mapping(raw: Any) -> dict[str, Any] | None:
    """Coerce a JSON string or dict into a dict, or ``None`` when it is
    neither — every payload here is untrusted transcript content."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None
