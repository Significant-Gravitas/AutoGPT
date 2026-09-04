"""Compose the Home "Recent work" feed from activity events.

Runs are deliberately excluded here: completed runs are the briefing's
job. This card shows the durable things agents produced — files written,
integration actions taken, schedules set up — grouped by who did the
work and in which thread.
"""

from typing import Literal

from backend.api.features.experts.models import Expert
from backend.data.activity_event import ActivityEvent

from .helpers import to_home_expert
from .models import (
    HomeRecentWork,
    HomeRecentWorkGroup,
    HomeRecentWorkItem,
    HomeWorkActor,
)

_MAX_GROUPS = 6
_MAX_ITEMS_PER_GROUP = 4

_ITEM_CATEGORIES: dict[str, Literal["file", "integration", "schedule"]] = {
    "FILE": "file",
    "INTEGRATION": "integration",
    "SCHEDULE": "schedule",
}


def compose_recent_work(
    events: list[ActivityEvent],
    expert_by_id: dict[str, Expert],
    session_titles: dict[str, str | None],
) -> HomeRecentWork:
    """Group displayable events by (actor, thread), newest group first.

    *events* arrive newest-first, so first-seen insertion order already
    ranks groups by their latest event.
    """
    grouped: dict[tuple[str, str], list[ActivityEvent]] = {}
    for event in events:
        if event.category not in _ITEM_CATEGORIES:
            continue
        key = (event.expert_id or "", event.session_id or "")
        grouped.setdefault(key, []).append(event)

    groups = [
        _compose_group(group_events, expert_by_id, session_titles)
        for group_events in grouped.values()
    ]
    total_count = sum(len(group_events) for group_events in grouped.values())
    return HomeRecentWork(groups=groups[:_MAX_GROUPS], total_count=total_count)


def _compose_group(
    events: list[ActivityEvent],
    expert_by_id: dict[str, Expert],
    session_titles: dict[str, str | None],
) -> HomeRecentWorkGroup:
    newest = events[0]
    session_id = newest.session_id
    return HomeRecentWorkGroup(
        actor=_compose_actor(newest, expert_by_id),
        session_id=session_id,
        session_title=session_titles.get(session_id) if session_id else None,
        link=f"/copilot?sessionId={session_id}" if session_id else None,
        latest_at=newest.created_at,
        items=[_compose_item(event) for event in events[:_MAX_ITEMS_PER_GROUP]],
        more_count=max(0, len(events) - _MAX_ITEMS_PER_GROUP),
    )


def _compose_actor(
    event: ActivityEvent, expert_by_id: dict[str, Expert]
) -> HomeWorkActor:
    if event.expert_id:
        expert = expert_by_id.get(event.expert_id)
        if expert:
            return HomeWorkActor(
                kind="expert", name=expert.name, expert=to_home_expert(expert)
            )
        # The expert was archived or removed after the fact; the event and
        # its attribution kind survive, only the profile is gone.
        return HomeWorkActor(kind="expert", name="Expert")
    if event.session_id:
        return HomeWorkActor(kind="autopilot", name="Autopilot")
    return HomeWorkActor(kind="agent", name="Agent run")


def _compose_item(event: ActivityEvent) -> HomeRecentWorkItem:
    mime_type = event.data.get("mime_type")
    return HomeRecentWorkItem(
        id=event.id,
        category=_ITEM_CATEGORIES[event.category],
        event_type=event.event_type,
        title=event.title,
        occurred_at=event.created_at,
        provider=event.provider,
        file_id=event.object_id if event.category == "FILE" else None,
        mime_type=mime_type if isinstance(mime_type, str) else None,
    )
