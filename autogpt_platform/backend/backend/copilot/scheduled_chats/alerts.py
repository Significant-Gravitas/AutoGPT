"""The email side of a paused scheduled chat (SECRT-2879).

A pause raises a ``PAUSED_FAILURES`` alert condition. The alert engine decides
what that becomes: an email where user email is on, a line in a coalesced one,
or a card in the next Briefing. It never sends the same condition twice in a
day. Resuming the chat, reconnecting the provider or a run that works resolves
it, which also cancels a send still in its ten-minute debounce.

Runs in the DatabaseManager, so it writes the condition directly. Best effort
throughout: an alert that cannot be written must not undo the pause.
"""

import logging
from urllib.parse import quote

from backend.copilot.scheduled_chats.failure_notices import provider_name
from backend.copilot.scheduled_chats.forget import alert_key
from backend.copilot.scheduled_chats.outcomes import RunFailure
from backend.data.alerts import raise_alert_condition
from backend.notifications.alert_causes import PausedFailuresCause

logger = logging.getLogger(__name__)

FOLLOWUPS_PATH = "/library/followups"


def paused_cause(
    failure: RunFailure, *, title: str, failures: int, session_id: str | None
) -> PausedFailuresCause:
    """Reads as: Morning briefing paused itself, the step “Twitter connection”
    failed on 3 consecutive runs. Its button opens the chat, where the card
    has the fix."""
    return PausedFailuresCause(
        agent=title,
        step=step_label(failure),
        consecutive_failures=failures,
        cta_path=(
            f"/copilot?sessionId={quote(session_id)}" if session_id else FOLLOWUPS_PATH
        ),
    )


def step_label(failure: RunFailure) -> str:
    """What failed, named as a step of the scheduled chat. The email reads
    "The step “Twitter connection” failed", so an unnamed provider gets a
    label of its own rather than "a connected service connection"."""
    name = provider_name(failure.provider)
    if failure.kind == "connection":
        return f"{name} connection" if failure.provider else "Connected account"
    if failure.kind == "chat_link":
        return f"Posting to {name}" if failure.provider else "Posting to chat"
    if failure.key == "turn:max_turns_exhausted":
        return "Step limit"
    if failure.kind == "budget":
        return "Spending limit"
    if failure.tool:
        return failure.tool.replace("_", " ").capitalize()
    return "One of its steps"


async def raise_paused(
    user_id: str, schedule_key: str, cause: PausedFailuresCause
) -> None:
    try:
        await raise_alert_condition(
            user_id, cause.cause, alert_key(schedule_key), cause.model_dump(mode="json")
        )
    except Exception:
        logger.warning(
            "Could not raise the paused alert for scheduled chat %s",
            schedule_key,
            exc_info=True,
        )
