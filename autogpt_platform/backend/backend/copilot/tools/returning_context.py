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

    The turn-starting message is already persisted by the time the prompt is
    assembled, so the last user row is the message being answered right now —
    the one before it is the actual "last seen". ``None`` when there is no
    earlier user message (fresh session) or when timestamps are missing.
    """
    stamped = [m.created_at for m in messages if m.role == "user" and m.created_at]
    if len(stamped) < 2:
        return None
    return as_utc(stamped[-2])


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
