"""Why a turn paused, and the chat row that remembers it.

A turn that runs out of budget or steps now stops on purpose: its tools are
refused, the agent writes a checkpoint, and the chat shows a Paused card with
the ways to pick the task back up. The live ``data-turn-paused`` part is gone
once the stream ends (the chat re-reads stored rows), so the pause is also
written as a marker row whose metadata rebuilds the card on reload. Older
clients that don't know the card still render the row as a system note.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from backend.copilot.constants import COPILOT_SYSTEM_PREFIX
from backend.copilot.markers import has_trailing_marker
from backend.copilot.model import ChatMessage, ChatSession

PauseReason = Literal["daily", "weekly", "trial", "task", "steps"]

# Metadata key on the marker row. Namespaced for the same reason as
# ``PROVIDER_FAILURE_KEY``: the bag is shared with other writers.
PAUSE_KEY = "turn_paused"

_SUMMARIES: dict[PauseReason, str] = {
    "daily": "Paused: this task is out of today's budget.",
    "weekly": "Paused: this task is out of this week's budget.",
    "trial": "Paused: the trial allowance is used up.",
    "task": "Paused: this task reached its size limit.",
    "steps": "Paused: checking in after a long run.",
}


class TurnPause(BaseModel):
    """What the Paused card needs to say and offer."""

    reason: PauseReason
    # When the limiting window refills; set for daily and weekly pauses only.
    resets_at: datetime | None = None
    # Model calls the turn made; set for step check-ins.
    steps: int | None = None
    # False when a backstop stopped the turn before the agent could write
    # its own checkpoint, so the card mustn't claim it saved its progress.
    checkpoint_saved: bool = True

    def as_part(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @property
    def summary(self) -> str:
        return _SUMMARIES[self.reason]


def append_pause_marker(session: ChatSession | None, pause: TurnPause) -> bool:
    """Record the pause in the chat. Returns whether a row was added."""
    if session is None or has_trailing_marker(session):
        return False
    session.messages.append(
        ChatMessage(
            role="assistant",
            content=f"{COPILOT_SYSTEM_PREFIX} {pause.summary}",
            metadata={PAUSE_KEY: pause.as_part()},
        )
    )
    return True


def pause_of(message: ChatMessage) -> TurnPause | None:
    """The pause recorded on a marker row, if this row is one."""
    if message.role != "assistant" or not message.metadata:
        return None
    raw = message.metadata.get(PAUSE_KEY)
    if not isinstance(raw, dict):
        return None
    try:
        return TurnPause.model_validate(raw)
    except ValidationError:
        return None
