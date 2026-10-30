"""The bot's note for a turn that paused at its budget or step limit.

The web chat shows a Paused card with a Keep going button. A chat user has
no button, but a reply in the conversation starts a turn that picks the task
up from the checkpoint the paused turn wrote, so the note says to send one.
"""

from datetime import UTC, datetime

from backend.copilot.turn_pause import TurnPause

KEEP_GOING = 'Reply "keep going"'


def pause_note(pause: TurnPause) -> str:
    # A budget pause with no reset time has no plan behind it to refill.
    if pause.reason in ("daily", "weekly") and pause.resets_at is not None:
        return f"{pause.summary} {KEEP_GOING} after it resets{_at(pause.resets_at)}."
    if pause.reason in ("daily", "weekly", "trial"):
        return f"{pause.summary} {KEEP_GOING} once your plan has allowance again."
    return f"{pause.summary} {KEEP_GOING} to pick it up."


def _at(resets_at: datetime) -> str:
    utc = resets_at.astimezone(UTC)
    return f" at {utc:%H:%M} UTC on {utc:%b} {utc.day}"
