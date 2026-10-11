"""What a scheduled chat tells its owner when one of its runs was skipped.

The scheduler skips a scheduled chat's run when its owner may not start a
turn: no plan, a lost ChatGPT connection, or a used-up usage limit. A repeating
schedule stays registered and picks up again by itself once that changes.
Skipping is right; skipping without a word is not. So the chat gets one note
per window saying why, with the fix on a button (SECRT-2879).

The note is a card for the owner, not part of the conversation: its metadata
keeps it out of everything the model reads (``hidden_rows``).
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel

from backend.copilot.hidden_rows import HIDDEN_FROM_MODEL_KEY
from backend.copilot.turn_queue import TurnRefusal
from backend.data.db_accessors import chat_db

# The ``kind`` the chat page renders as a card rather than a reply.
NOTICE_METADATA_KIND = "scheduled_chat_notice"

BILLING_PATH = "/settings/billing"
INTEGRATIONS_PATH = "/settings/integrations"

_MESSAGE_NAMESPACE = uuid.UUID("5d0a3a4e-2f3b-5c8e-9a51-6e7c0b1d2f43")
_TITLE_MAX = 60

NoticeReason = Literal["usage", "no_plan", "codex_access"]
# What became of the skipped run: a repeating schedule keeps its next time, a
# one-shot either waits for its usage window to reset or is gone.
SkippedRun = Literal["recurring", "waits_for_reset", "dropped"]


class NoticeLink(BaseModel):
    """A button that opens a page, such as billing."""

    kind: Literal["link"] = "link"
    label: str
    href: str


class ScheduledChatNotice(BaseModel):
    reason: NoticeReason
    headline: str
    body: str
    # When the schedule runs again by itself, where it does.
    resumes_at: datetime | None = None
    actions: list[NoticeLink]
    # Names the window this note covers, so a schedule says it once per
    # window: part of the message id.
    window_key: str

    @property
    def content(self) -> str:
        """The text, for anything that shows the message without the card."""
        return f"{self.headline}. {self.body}"

    def metadata(self, *, schedule_key: str, title: str) -> dict[str, Any]:
        return {
            "kind": NOTICE_METADATA_KIND,
            HIDDEN_FROM_MODEL_KEY: True,
            "reason": self.reason,
            "schedule_key": schedule_key,
            "title": title,
            "headline": self.headline,
            "body": self.body,
            "resumes_at": self.resumes_at.isoformat() if self.resumes_at else None,
            "actions": [action.model_dump() for action in self.actions],
        }


def skip_notice(
    refusal: TurnRefusal,
    *,
    title: str,
    now: datetime,
    run: SkippedRun,
    user_timezone: str | None = None,
) -> ScheduledChatNotice | None:
    """What to tell the owner about a run *refusal* stopped, or None when
    there is nothing for them to do (an outage on our side). Times are in
    *user_timezone*, the one the schedule was made in, else UTC."""
    if refusal.kind == "usage" and refusal.window in ("daily", "weekly"):
        return _usage_notice(refusal, title=title, run=run, user_timezone=user_timezone)
    if refusal.kind == "usage":
        return _plan_notice(
            title=title,
            run=run,
            headline="Skipped: your trial's usage limit is used up",
            window_key=f"trial:{_iso_week(now)}",
        )
    if refusal.kind == "no_plan":
        return _plan_notice(
            title=title,
            run=run,
            headline="Skipped: scheduled chats need a plan",
            window_key=f"no_plan:{_iso_week(now)}",
        )
    if refusal.kind == "codex_access":
        # A one-shot is gone by now; only a repeating schedule comes back.
        after = (
            " Reconnect it in Settings and the chat runs again at its next time."
            if run == "recurring"
            else " Reconnect it in Settings, then ask for it again."
        )
        return ScheduledChatNotice(
            reason="codex_access",
            headline="Skipped: your ChatGPT connection isn't available",
            body=(
                f"“{title}” didn't run: it runs on your ChatGPT plan, which "
                f"AutoPilot can't use right now.{after}"
            ),
            actions=[NoticeLink(label="Open settings", href=INTEGRATIONS_PATH)],
            window_key=f"codex_access:{_iso_week(now)}",
        )
    return None


def _usage_notice(
    refusal: TurnRefusal,
    *,
    title: str,
    run: SkippedRun,
    user_timezone: str | None,
) -> ScheduledChatNotice:
    daily = refusal.window == "daily"
    resets_at = refusal.resets_at
    if resets_at is None:
        when = "when the limit resets"
    else:
        local = _in_timezone(resets_at, user_timezone)
        clock = f"{local:%H:%M} {local.tzname()}"
        # Monday 00:00 UTC is still Sunday for someone in Chicago.
        when = (
            f"when the limit resets at {clock}"
            if daily
            else f"when the limit resets on {local:%A} at {clock}"
        )
    if run == "waits_for_reset":
        body = f"“{title}” will run {when}."
    elif run == "recurring":
        body = f"“{title}” didn't run. Scheduled chats run again {when}."
    else:
        body = f"“{title}” didn't run."
    return ScheduledChatNotice(
        reason="usage",
        headline=(
            "Skipped: daily usage limit reached"
            if daily
            else "Skipped: weekly usage limit reached"
        ),
        body=body,
        resumes_at=resets_at if run != "dropped" else None,
        actions=[NoticeLink(label="Upgrade", href=BILLING_PATH)],
        window_key=(
            f"usage:{refusal.window}:"
            f"{resets_at.isoformat() if resets_at else 'unknown'}"
        ),
    )


def _plan_notice(
    *, title: str, run: SkippedRun, headline: str, window_key: str
) -> ScheduledChatNotice:
    """No window resets these, so a repeating schedule waits for a plan, and
    the note repeats at most weekly while it does."""
    body = (
        f"“{title}” didn't run. Scheduled chats start again once you choose a plan."
        if run == "recurring"
        else f"“{title}” didn't run. Scheduled chats need a plan."
    )
    return ScheduledChatNotice(
        reason="no_plan",
        headline=headline,
        body=body,
        actions=[NoticeLink(label="Choose a plan", href=BILLING_PATH)],
        window_key=window_key,
    )


def schedule_title(*, routine_title: str | None, message: str) -> str:
    """How the note names the scheduled chat: a routine by its title, a
    follow-up by the start of the message it sends."""
    if routine_title:
        return routine_title
    first_line = next(
        (line.strip() for line in message.splitlines() if line.strip()), ""
    )
    if len(first_line) <= _TITLE_MAX:
        return first_line or "Scheduled chat"
    return f"{first_line[: _TITLE_MAX - 1].rstrip()}…"


def notice_message_id(*, user_id: str, schedule_key: str, window_key: str) -> str:
    """Deterministic, so a second skip in the same window writes nothing."""
    return str(uuid.uuid5(_MESSAGE_NAMESPACE, f"{user_id}:{schedule_key}:{window_key}"))


async def post_notice(
    notice: ScheduledChatNotice,
    *,
    user_id: str,
    schedule_key: str,
    title: str,
    session_id: str | None,
    expert_id: str | None,
) -> bool:
    """Post *notice* where its owner will see it, and say whether it posted.

    Into the scheduled chat's own thread when it has one. A run that would
    have opened a fresh chat has none yet, so the note goes to the expert's
    latest thread, or for an AutoPilot schedule to the owner's main chat.
    """
    message_id = notice_message_id(
        user_id=user_id, schedule_key=schedule_key, window_key=notice.window_key
    )
    metadata = notice.metadata(schedule_key=schedule_key, title=title)
    db = chat_db()
    if session_id is not None:
        return await db.append_session_message(
            session_id=session_id,
            user_id=user_id,
            content=notice.content,
            message_id=message_id,
            metadata=metadata,
        )
    if expert_id is not None:
        posted_in = await db.append_expert_run_message(
            user_id=user_id,
            expert_id=expert_id,
            content=notice.content,
            message_id=message_id,
            metadata=metadata,
        )
    else:
        posted_in = await db.append_plain_session_message(
            user_id=user_id,
            content=notice.content,
            message_id=message_id,
            metadata=metadata,
        )
    return posted_in is not None


def _iso_week(now: datetime) -> str:
    year, week, _ = now.isocalendar()
    return f"{year}-W{week:02d}"


def _in_timezone(when: datetime, user_timezone: str | None) -> datetime:
    """*when* on the owner's clock, or UTC when their zone is unknown."""
    if user_timezone:
        try:
            return when.astimezone(ZoneInfo(user_timezone))
        except (ZoneInfoNotFoundError, ValueError):
            pass
    return when.astimezone(timezone.utc)
