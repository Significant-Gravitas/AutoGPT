"""Picking a chat back up when the budget resets: when to, and whether to still.

Out of budget, a user can have a paused task resume when the budget comes back
("Resume at reset"), or write a message now that sends then ("Send at reset").
Each is a scheduled copilot turn bound to the chat, at most one of each kind
per chat (``at_reset_schedule_id``), and kept out of the agent's own schedule
tools. It fires a little after the window that is actually blocking refills,
offset per chat so paused chats don't all fire at midnight UTC, and looks
again when it fires (``fire_verdict``): the user may have picked the chat up
already, or still be out of budget.
"""

import asyncio
import hashlib
import logging
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel

from backend.copilot.config import ChatConfig
from backend.copilot.continuation import KEEP_GOING_NOTE, RESUMED_AT_RESET_NOTE
from backend.copilot.db import PaginatedMessages
from backend.copilot.model import ChatMessage
from backend.copilot.pending_message_helpers import (
    StreamRegistryUnavailable,
    is_turn_in_flight,
)
from backend.copilot.rate_limit import (
    RateLimitExceeded,
    RateLimitUnavailable,
    SubscriptionTier,
    check_rate_limit,
    get_global_rate_limits,
    get_usage_status,
)
from backend.data.db_accessors import chat_db, credit_db
from backend.util.background import spawn_background_task
from backend.util.clients import get_scheduler_client

logger = logging.getLogger(__name__)

AtResetKind = Literal["resume", "send"]
NoResetReason = Literal["not_out_of_budget", "no_reset"]

# What the Scheduled page calls each job.
AT_RESET_JOB_NAMES: dict[AtResetKind, str] = {
    "resume": "Resume when the budget resets",
    "send": "Send when the budget resets",
}

config = ChatConfig()

# Fire a little after the window refills rather than on the dot.
_AFTER_RESET = timedelta(seconds=120)
# Spread over ten minutes, so paused chats don't all fire at 00:00 UTC.
_JITTER_SECONDS = 600
# A send and a resume on one chat would otherwise fire together. The send goes
# first; the resume then finds the user's message newer than itself and stands
# down, as it does for a message sent now.
_RESUME_AFTER_SEND = timedelta(seconds=60)
# How many of the chat's latest rows the fire-time check reads.
_RECENT_ROWS = 50
_PICKUP_NOTES = (KEEP_GOING_NOTE, RESUMED_AT_RESET_NOTE)


class NoResetAhead(Exception):
    """There is no budget reset for an at-reset turn to wait for."""

    def __init__(self, reason: NoResetReason):
        super().__init__(reason)
        self.reason: NoResetReason = reason


class FireVerdict(BaseModel):
    """What a due at-reset turn does now."""

    # fire: dispatch it. drop: let it go. retry: try again in a few minutes.
    # wait: still out of budget, so move it to ``run_at``, the next reset.
    action: Literal["fire", "drop", "retry", "wait"]
    run_at: datetime | None = None
    why: str = ""


def at_reset_schedule_id(kind: AtResetKind, session_id: str) -> str:
    """The job's id: one per kind per chat, so setting it again replaces it."""
    return f"at-reset-{kind}-{session_id}"


async def next_reset_run_at(
    user_id: str, session_id: str, kind: AtResetKind
) -> datetime:
    """When this chat's at-reset turn should fire.

    Just after the exhausted window that refills last: a user out of both
    today's and this week's budget can't start until the week resets. Raises
    ``NoResetAhead`` when nothing is exhausted, or when what is exhausted never
    refills (a trial's lifetime allowance, or a window that allows nothing),
    and ``RateLimitUnavailable`` when usage can't be read.
    """
    offset = timedelta(seconds=reset_jitter_seconds(session_id))
    if kind == "resume":
        offset += _RESUME_AFTER_SEND
    return await _blocking_reset(user_id) + _AFTER_RESET + offset


def reset_jitter_seconds(session_id: str) -> int:
    """This chat's offset into the ten minutes after a reset, in seconds.

    A hash of the chat id rather than a random draw, so moving the turn to a
    later reset keeps it at the same point in that window.
    """
    digest = hashlib.sha256(session_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") % _JITTER_SECONDS


async def load_at_reset_chat(session_id: str, user_id: str) -> PaginatedMessages | None:
    """The chat an at-reset turn is bound to, with its latest rows.

    Read from the database rather than the session cache: the fire-time check
    compares when rows were written, and rows appended in-process and cached
    don't say. None when the chat is gone or isn't the user's.
    """
    return await chat_db().get_chat_messages_paginated(
        session_id, limit=_RECENT_ROWS, user_id=user_id
    )


async def fire_verdict(
    kind: AtResetKind, chat: PaginatedMessages, created_at: datetime | None
) -> FireVerdict:
    """Whether a due at-reset turn fires now, and if not, what becomes of it."""
    session = chat.session
    if kind == "resume" and await _picked_up_since(chat, created_at):
        return FireVerdict(action="drop", why="the chat was picked up after it")
    if kind == "resume" and session.metadata.llm_auth_provider == (
        "microsoft_365_copilot"
    ):
        # Switched since it was set. Microsoft 365 Copilot has none of the
        # task or AutoPilot's tools, and the resume note would reach it as
        # the user's message. A send is the user's own words, so it goes.
        return FireVerdict(
            action="drop", why="the chat now runs on Microsoft 365 Copilot"
        )
    try:
        busy = await is_turn_in_flight(session.session_id)
    except StreamRegistryUnavailable:
        return FireVerdict(action="retry", why="could not tell if the chat is busy")
    if busy:
        return FireVerdict(action="retry", why="a turn is running in the chat")
    if session.metadata.llm_auth_provider != "platform":
        # Only the platform route spends this budget.
        return FireVerdict(action="fire")
    return await _budget_verdict(session.user_id, session.session_id, kind)


def cancel_resume_at_reset_in_background(session_id: str, user_id: str) -> asyncio.Task:
    """Drop the chat's pending "Resume at reset": the user carried it on now.

    A pending send stays. Off the caller's path, so it never fails or slows a
    request; best effort, since the resume also stands down when it fires.
    """
    return spawn_background_task(
        _cancel_resume(session_id, user_id),
        name=f"cancel-resume-at-reset-{session_id[:12]}",
    )


async def _cancel_resume(session_id: str, user_id: str) -> None:
    try:
        # A cancel is safe to repeat if a closed connection loses it.
        await get_scheduler_client(should_retry=True).cancel_at_reset_turn(
            user_id=user_id, session_id=session_id, kind="resume"
        )
    except Exception:
        logger.warning(
            f"Could not cancel the resume-at-reset of session {session_id[:12]}",
            exc_info=True,
        )


async def _blocking_reset(user_id: str) -> datetime:
    daily_limit, weekly_limit, tier = await get_global_rate_limits(
        user_id,
        config.daily_cost_limit_microdollars,
        config.weekly_cost_limit_microdollars,
    )
    # A window that allows nothing (a lapsed trial or plan) never refills, and
    # nor does a trial's lifetime allowance. A trial's own daily or weekly
    # window does, at that window's reset.
    if daily_limit == 0 or weekly_limit == 0:
        raise NoResetAhead("no_reset")
    if tier == SubscriptionTier.TRIAL and await _trial_allowance_spent(user_id):
        raise NoResetAhead("no_reset")
    # No usage, read off a Redis failure, would say the budget is already back.
    status = await get_usage_status(
        user_id, daily_limit, weekly_limit, tier=tier, fail_closed=True
    )
    # As in ``check_rate_limit``: a negative limit means no cap at all.
    exhausted = [w for w in (status.daily, status.weekly) if 0 <= w.limit <= w.used]
    if not exhausted:
        raise NoResetAhead("not_out_of_budget")
    return max(w.resets_at for w in exhausted)


async def _trial_allowance_spent(user_id: str) -> bool:
    """Whether the trial has nothing left to wait for: its lifetime allowance is
    used up, or the trial itself is over (as ``check_rate_limit`` reads it)."""
    trial = await credit_db().get_subscription_trial(user_id)
    return (
        trial is None
        or not trial.active
        or trial.cost_microdollars >= trial.offer.total_cost_limit
    )


async def _budget_verdict(
    user_id: str, session_id: str, kind: AtResetKind
) -> FireVerdict:
    try:
        daily_limit, weekly_limit, tier = await get_global_rate_limits(
            user_id,
            config.daily_cost_limit_microdollars,
            config.weekly_cost_limit_microdollars,
        )
        if daily_limit == 0 and weekly_limit == 0:
            # No plan, or the trial is over. Settled here: an ended
            # trial's usage state still reads TRIAL, which the tier guard
            # below reports as unavailable, so it would only be retried.
            return FireVerdict(action="drop", why="its allowance never refills")
        await check_rate_limit(
            user_id=user_id,
            daily_cost_limit=daily_limit,
            weekly_cost_limit=weekly_limit,
            expected_tier=tier,
        )
    except RateLimitUnavailable:
        return FireVerdict(action="retry", why="the budget could not be read")
    except RateLimitExceeded:
        return await _next_reset_verdict(user_id, session_id, kind)
    return FireVerdict(action="fire")


async def _next_reset_verdict(
    user_id: str, session_id: str, kind: AtResetKind
) -> FireVerdict:
    try:
        run_at = await next_reset_run_at(user_id, session_id, kind)
    except NoResetAhead as e:
        if e.reason == "no_reset":
            return FireVerdict(action="drop", why="its allowance never refills")
        # Over the limit a moment ago and under it now: look again shortly.
        return FireVerdict(action="retry", why="the budget moved while checking")
    except RateLimitUnavailable:
        return FireVerdict(action="retry", why="the budget could not be read")
    return FireVerdict(action="wait", run_at=run_at, why="still out of budget")


async def _picked_up_since(chat: PaginatedMessages, since: datetime | None) -> bool:
    """Whether the user carried the chat on after the resume was set: with a
    message of their own, or a Keep going (or an earlier resume) that ran."""
    if since is None:
        return False
    rows = list(chat.messages)
    # Their newest message can be further back than the latest rows.
    latest_user = await chat_db().get_latest_user_message_in_session(
        chat.session.session_id
    )
    if latest_user is not None:
        rows.append(latest_user)
    return any(_picks_up(row) and _written_after(row, since) for row in rows)


def _picks_up(row: ChatMessage) -> bool:
    return row.role == "user" or row.content in _PICKUP_NOTES


def _written_after(row: ChatMessage, since: datetime) -> bool:
    return row.created_at is not None and _as_utc(row.created_at) > _as_utc(since)


def _as_utc(moment: datetime) -> datetime:
    """A naive time is UTC, as the database writes it."""
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)
