"""Build the ``<returning_context>`` block — the "while you were away" recap.

Sibling of :mod:`backend.copilot.tools.session_context`. Where that block
tells the model what is *scheduled*, this one tells it what *happened* since
the user last typed:

* sub-threads this chat delegated out, and scheduled follow-ups that fired
  into a fresh chat, both touched since the user left;
* work still waiting on the user — open questions and paused/over-budget
  experts — rendered down from the Home "Needs You" composer so the two
  surfaces can never disagree about what counts as needing attention.

The block lands in the **per-turn user message** (after the last
cache_control breakpoint) alongside ``<session_context>``, so injecting it
never busts the prefix cache.

Two entry points, because the engines assemble that message two different
ways and the block must appear **exactly once** per turn:

* :func:`build_returning_context` returns the bare body for the first turn of
  a session, where the engines call ``inject_user_context``;
* :func:`build_returning_context_prefix` returns the wrapped, ready-to-prepend
  block for every *later* turn, where ``inject_user_context`` is skipped
  (SDK ``has_history``, baseline ``is_first_turn``) and the per-turn prefix is
  the only way in. It refuses to build when the first-turn path already ran,
  which is what makes the two mutually exclusive.

The resume path is the one that matters: a user reopening a thread they left
is the whole point of the feature, and it is the only path where the away-gap
is ever non-zero.

Two guards keep it cheap. The whole computation is skipped unless the user
has actually been away (:data:`_RETURNING_GAP`) or the session is brand new,
and it emits **no block at all** when there is nothing to report — the common
case is a user replying within seconds, which costs one timestamp comparison
and zero queries.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from backend.api.features.home.attention import compose_attention_items
from backend.api.features.home.models import HomeAttentionItem
from backend.copilot.briefing.outcome import as_utc
from backend.copilot.model import (
    CHAT_STATUS_IDLE,
    ChatMessage,
    ChatSession,
    ChatSessionInfo,
)
from backend.copilot.service import RETURNING_CONTEXT_TAG
from backend.data.db_accessors import chat_db, experts_db
from backend.util.feature_flag import Flag, is_feature_enabled

logger = logging.getLogger(__name__)

# How long the user must have been silent before a recap is worth its tokens.
# Below this they are mid-conversation and already watched everything happen.
_RETURNING_GAP = timedelta(minutes=30)

# A brand-new session has no prior user message to measure the gap against,
# so background work is looked for over this trailing window instead. Bounded
# rather than unbounded so a first chat after a long holiday doesn't recap a
# fortnight of automation.
_FRESH_SESSION_LOOKBACK = timedelta(hours=24)

# Hard caps — the model needs a survey, not an inventory. Anything beyond is
# collapsed into a ``... +K more`` line.
_MAX_LISTED_SESSIONS = 3
_MAX_LISTED_ATTENTION = 3

# Titles are user-authored and can be arbitrarily long; clip before they eat
# the per-turn budget.
_TITLE_MAX_CHARS = 60

# Only these Home item kinds belong in a chat recap. ``approval`` and
# ``credits`` are excluded deliberately: both need the Home UI (a review
# screen, a top-up flow) that the model cannot drive from inside a turn.
_RECAPPED_ATTENTION_KINDS = frozenset({"question", "paused"})


async def build_returning_context(session: ChatSession, user_id: str | None) -> str:
    """Return the body of the ``<returning_context>`` block, or ``""``.

    ``""`` means "emit no block" and is the expected outcome on the vast
    majority of turns: the flag is off, the user never left, or nothing
    happened while they were gone.

    Every read is best-effort. A recap is a nicety; a failed lookup must
    degrade to no block rather than fail the turn.
    """
    if not user_id:
        return ""
    if not await is_feature_enabled(
        Flag.COPILOT_RETURNING_CONTEXT, user_id, default=False
    ):
        return ""

    now = datetime.now(UTC)
    last_seen = previous_user_activity(session.messages)
    if not should_recap(now=now, last_seen=last_seen):
        return ""

    since = last_seen or now - _FRESH_SESSION_LOOKBACK
    try:
        background, experts, questions = await asyncio.gather(
            chat_db().get_background_sessions_since(
                user_id=user_id,
                parent_session_id=session.session_id,
                since=since,
            ),
            experts_db().list_experts(user_id),
            chat_db().get_sessions_with_pending_question(user_id),
        )
    except Exception as e:
        logger.warning(
            "build_returning_context: recap sources unavailable for session %s (%s); "
            "emitting no block",
            session.session_id,
            e,
        )
        return ""

    attention = compose_attention_items(
        now=now,
        experts=experts,
        reviews=[],
        schedules=[],
        credits_balance=None,
        questions=questions,
    )
    return compose_returning_context(
        now=now,
        last_seen=last_seen,
        parent_session_id=session.session_id,
        background=background,
        attention=attention,
    )


async def build_returning_context_prefix(
    session: ChatSession,
    user_id: str | None,
    *,
    is_user_message: bool,
    already_injected: bool,
) -> str:
    """The wrapped block for a turn that does NOT go through
    ``inject_user_context``, or ``""``.

    ``already_injected`` is the engines' own first-turn predicate (SDK: ``not
    has_history``; baseline: ``should_inject_user_context``). Refusing to
    build when it is true is the single guarantee that a turn can never carry
    two ``<returning_context>`` blocks — the first-turn path and this one are
    exact complements of each other, checked before any work happens.

    Non-user turns (tool continuations, scheduled follow-ups) get nothing:
    there is no returning user to greet.
    """
    if already_injected or not is_user_message:
        return ""
    body = await build_returning_context(session, user_id)
    if not body:
        return ""
    return f"<{RETURNING_CONTEXT_TAG}>\n{body}\n</{RETURNING_CONTEXT_TAG}>\n\n"


def should_recap(*, now: datetime, last_seen: datetime | None) -> bool:
    """Whether the user has been away long enough for a recap to be news.

    ``last_seen is None`` means this session has no earlier user message —
    a fresh chat, where any background work is by definition unseen here.
    """
    if last_seen is None:
        return True
    return now - last_seen >= _RETURNING_GAP


def previous_user_activity(messages: list[ChatMessage]) -> datetime | None:
    """When the user last typed in this session *before* the current turn.

    The current turn is the **trailing run of user rows**, not just the last
    one. By the time the prompt is assembled the turn-starting message is
    already persisted, and the SDK path may have drained queued "pending"
    chips into their own user rows behind it — indexing back a fixed one row
    would read one of those chips (sent seconds ago) and suppress every
    recap. A run ends at the first non-user row, which in a real transcript
    is the assistant reply to the previous message.

    ``None`` when nothing precedes that run (fresh session) or when the rows
    that do precede it carry no timestamp.

    Truncation-safe: ``get_chat_messages_paginated`` fills a session from the
    **newest** end (``MAX_LOADED_CHAT_MESSAGES``), so a capped transcript
    keeps its tail intact and only loses ancient history. Losing older rows
    can only push ``last_seen`` further back, which is the safe direction —
    it never invents recency the user did not have.
    """
    index = len(messages)
    while index > 0 and messages[index - 1].role == "user":
        index -= 1
    for message in reversed(messages[:index]):
        if message.role == "user" and message.created_at:
            return as_utc(message.created_at)
    return None


def compose_returning_context(
    *,
    now: datetime,
    last_seen: datetime | None,
    parent_session_id: str,
    background: list[ChatSessionInfo],
    attention: list[HomeAttentionItem],
) -> str:
    """Render the recap body. ``""`` when there is nothing worth reporting.

    Pure so the shape is testable without a database: the caller owns the
    fetching, this owns the wording and the caps.
    """
    delegated = [
        s for s in background if s.metadata.delegated_by_session_id == parent_session_id
    ]
    # The rest came in on the automation arm of the query — a scheduled
    # follow-up that fired into a chat of its own.
    followups = [s for s in background if s.metadata.delegated_by_session_id is None]
    waiting = [i for i in attention if i.kind in _RECAPPED_ATTENTION_KINDS]

    if not delegated and not followups and not waiting:
        return ""

    lines = [f"away_for: {_away_label(now=now, last_seen=last_seen)}"]
    if delegated:
        lines.append("delegated_work:")
        lines.extend(_session_lines(delegated))
    if followups:
        lines.append("followups_fired:")
        lines.extend(_session_lines(followups))
    if waiting:
        lines.append("waiting_on_you:")
        lines.extend(_attention_lines(waiting))
    return "\n".join(lines)


def _away_label(*, now: datetime, last_seen: datetime | None) -> str:
    """Coarse gap ("2h", "3d") so the model can say "while you were away"
    naturally. ``new chat`` when there is no earlier message to measure from.
    A clock skew that puts ``last_seen`` ahead of ``now`` reads as ``0m``
    rather than a negative duration."""
    if last_seen is None:
        return "new chat"
    minutes = max(0, int((now - last_seen).total_seconds() // 60))
    if minutes < 60:
        return f"{minutes}m"
    if minutes < 60 * 24:
        return f"{minutes // 60}h"
    return f"{minutes // (60 * 24)}d"


def _session_lines(sessions: list[ChatSessionInfo]) -> list[str]:
    visible = sessions[:_MAX_LISTED_SESSIONS]
    lines = [_session_line(s) for s in visible]
    remaining = len(sessions) - len(visible)
    if remaining > 0:
        lines.append(f"... +{remaining} more")
    return lines


def _session_line(session: ChatSessionInfo) -> str:
    """One bullet per background session: ``- "Title" (finished)``.

    ``chat_status`` is the only lifecycle signal persisted on a session, so
    "finished" means idle — the turn is no longer running. It does not claim
    the work succeeded; the model must open the thread to learn that.
    """
    status = "finished" if session.chat_status == CHAT_STATUS_IDLE else "still running"
    return f'- "{_clip_title(session.title)}" ({status}, session {session.session_id})'


def _attention_lines(items: list[HomeAttentionItem]) -> list[str]:
    visible = items[:_MAX_LISTED_ATTENTION]
    lines = [f"- {i.title}: {i.description}" for i in visible]
    remaining = len(items) - len(visible)
    if remaining > 0:
        lines.append(f"... +{remaining} more")
    return lines


def _clip_title(title: str | None) -> str:
    """Collapse whitespace, escape quotes, and clip — the title is
    user-authored text landing inside a quoted prompt field."""
    compact = " ".join((title or "Untitled chat").split()).replace('"', '\\"')
    if len(compact) <= _TITLE_MAX_CHARS:
        return compact
    return f"{compact[:_TITLE_MAX_CHARS - 1]}…"
