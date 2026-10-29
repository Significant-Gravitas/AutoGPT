"""Picking a paused task back up: now ("Keep going") or when the budget resets.

Both run as a turn whose message is a system note rather than a user message:
the chat shows what happened in grey, and the model gets the same instruction
with the note marker taken off (see ``as_model_prompt``). Engines dedupe the
incoming message against the row the route saved, so the stored row and the
dispatched message have to be the same string.
"""

from typing import Any

from backend.copilot.constants import COPILOT_SYSTEM_PREFIX

KEEP_GOING_NOTE = (
    f"{COPILOT_SYSTEM_PREFIX} Keep going: pick the task up from the checkpoint above."
)
RESUMED_AT_RESET_NOTE = (
    f"{COPILOT_SYSTEM_PREFIX} Resumed after the budget reset: pick the task up "
    "from the checkpoint above."
)


# Rows that start a turn rather than end one: not markers, so an error or
# a pause after one still gets its own row.
KICKOFF_NOTES = frozenset({KEEP_GOING_NOTE, RESUMED_AT_RESET_NOTE})


def as_model_prompt(message: str) -> str:
    """The instruction a continuation note gives the model.

    A no-op for any other message, so engines can apply it to every
    non-user message they receive.
    """
    return message.removeprefix(COPILOT_SYSTEM_PREFIX).strip()


def send_trailing_note_as_user(messages: list[dict[str, Any]]) -> None:
    """Send a note that ends a chat-completions request as the user turn.

    The note is stored as the chat's last row, an assistant one, so a request
    built from history ends on it. Sent as is, that is a prefill: Claude 4.6+
    rejects it with a 400, and other models read the note as their own words.
    Only the request changes; the stored row stays a note.
    """
    if not messages or not _is_note(messages[-1]):
        return
    messages[-1] = {"role": "user", "content": as_model_prompt(messages[-1]["content"])}


def _is_note(entry: dict[str, Any]) -> bool:
    if entry.get("role") != "assistant" or entry.get("tool_calls"):
        return False
    return str(entry.get("content") or "").startswith(COPILOT_SYSTEM_PREFIX)
