"""Hand-offs on Home.

Needs you: a delegated thread paused on a question is answered from the chat
that delegated it, so its row links there; a held hand-off card is titled by
its brief and teammate rather than the generic gate headline. Recent work: a
hand-off that finished this week sits under the teammate who took it.
"""

from datetime import datetime, timedelta
from typing import Literal
from urllib.parse import quote
from zoneinfo import ZoneInfo

from backend.api.features.experts.delegations import DelegationSummary, title_of
from backend.api.features.experts.models import Expert
from backend.copilot.briefing.outcome import as_utc
from backend.copilot.constants import AUTOPILOT_NAME
from backend.copilot.gate.handoff import HandoffCard

from .helpers import to_home_expert
from .models import HomeAction, HomeAttentionItem, HomeDelegationItem, HomeExpert

_FINISHED: dict[str, Literal["completed", "failed", "cancelled"]] = {
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
}
_ENDED_AS = {"completed": "returned", "failed": "failed", "cancelled": "stopped"}


def delegated_question_item(
    delegation: DelegationSummary, now: datetime, expert_by_id: dict[str, Expert]
) -> HomeAttentionItem | None:
    """None when the asking teammate is gone: nothing could clear the row."""
    asker = expert_by_id.get(delegation.expert.id) if delegation.expert else None
    if asker is None or delegation.question is None:
        return None
    delegator = expert_by_id.get(delegation.delegated_by_expert_id or "")
    asked_at = as_utc(delegation.asked_at or delegation.created_at)
    return HomeAttentionItem(
        id=f"question-{delegation.sub_session_id}",
        kind="question",
        priority="normal",
        title=delegation.question,
        description=(
            f"{asker.name}, working for "
            f"{delegator.name if delegator else AUTOPILOT_NAME} on "
            f"“{delegation.title}” · asked {_ago(now - asked_at)}"
        ),
        why_it_matters="The hand-off is paused until you answer.",
        expert=to_home_expert(asker),
        created_at=asked_at,
        primary_action=HomeAction(
            label="Answer", href=_chat_link(delegation.parent_session_id)
        ),
    )


def handoff_title(card: HandoffCard) -> str:
    return f"Hand off “{title_of(card.brief)}” to {card.expert_name}"


def handoff_expert(card: HandoffCard) -> HomeExpert:
    return HomeExpert(
        id=card.expert_id,
        name=card.expert_name,
        role=card.expert_role,
        avatar_url=card.expert_avatar_url,
    )


def finished_delegation_items(
    delegations: list[DelegationSummary], since: datetime, timezone_name: str
) -> list[tuple[str, HomeDelegationItem]]:
    """(expert id, row) for each hand-off that ended since *since*, newest first."""
    zone = ZoneInfo(timezone_name)
    rows = [
        (d.expert.id, _finished_item(d, zone))
        for d in delegations
        if d.expert
        and d.sub_session_id
        and d.status in _FINISHED
        and d.finished_at
        and as_utc(d.finished_at) >= since
    ]
    return sorted(rows, key=lambda row: row[1].occurred_at, reverse=True)


def _finished_item(delegation: DelegationSummary, zone: ZoneInfo) -> HomeDelegationItem:
    finished_at = as_utc(delegation.finished_at or delegation.created_at)
    parts = [
        f"Delegated {_clock(delegation.created_at, zone)}",
        f"{_ENDED_AS[delegation.status]} {_clock(finished_at, zone)}",
    ]
    if delegation.files_count:
        files = delegation.files_count
        parts.append(f"{files} file{'s' if files != 1 else ''}")
    return HomeDelegationItem(
        id=f"delegation-{delegation.sub_session_id}",
        sub_session_id=delegation.sub_session_id or "",
        title=delegation.title,
        description=" · ".join(parts),
        status=_FINISHED[delegation.status],
        expert=(
            HomeExpert(
                id=delegation.expert.id,
                name=delegation.expert.name,
                role=delegation.expert.role,
                avatar_url=delegation.expert.avatar_url,
            )
            if delegation.expert
            else None
        ),
        occurred_at=finished_at,
        files_count=delegation.files_count,
        cost_usd=delegation.cost_usd,
        link=_chat_link(delegation.parent_session_id),
    )


def _chat_link(session_id: str | None) -> str:
    return f"/copilot?sessionId={quote(session_id)}" if session_id else "/copilot"


def _clock(at: datetime, zone: ZoneInfo) -> str:
    return as_utc(at).astimezone(zone).strftime("%H:%M")


def _ago(delta: timedelta) -> str:
    minutes = max(0, int(delta.total_seconds() // 60))
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes}m ago"
    if minutes < 24 * 60:
        return f"{minutes // 60}h ago"
    return f"{minutes // (24 * 60)}d ago"
