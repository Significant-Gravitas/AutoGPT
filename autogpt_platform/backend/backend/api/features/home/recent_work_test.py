from datetime import datetime, timedelta, timezone

from backend.api.features.experts.models import Expert
from backend.data.activity_event import ActivityEvent, ActivityEventCategory

from .recent_work import _MAX_ITEMS_PER_GROUP, compose_recent_work

NOW = datetime(2026, 8, 28, 9, 0, tzinfo=timezone.utc)


def _expert(expert_id: str, name: str) -> Expert:
    return Expert(
        id=expert_id,
        name=name,
        avatar_url=None,
        role="Marketing",
        tagline=None,
        bio=None,
        skills=[],
        identity="",
        voice_preferences="",
        boundaries="",
        protected_soul_rules=[],
        is_template=False,
        source_template_id=None,
        is_archived=False,
        workflows=[],
        last_run_status=None,
        schedules_paused_at=None,
    )


def _event(
    *,
    event_id: str,
    category: ActivityEventCategory = "FILE",
    minutes_ago: int = 0,
    expert_id: str | None = None,
    session_id: str | None = None,
    title: str = "draft.md",
    provider: str | None = None,
    object_id: str | None = None,
) -> ActivityEvent:
    return ActivityEvent(
        id=event_id,
        user_id="user-1",
        created_at=NOW - timedelta(minutes=minutes_ago),
        category=category,
        event_type="file.created" if category == "FILE" else "integration.action",
        title=title,
        expert_id=expert_id,
        session_id=session_id,
        provider=provider,
        object_id=object_id,
    )


def test_groups_by_actor_and_thread_newest_first() -> None:
    maria = _expert("maria", "Maria")
    events = [
        _event(event_id="e1", expert_id="maria", session_id="s1", minutes_ago=1),
        _event(event_id="e2", session_id="s2", minutes_ago=5),
        _event(event_id="e3", expert_id="maria", session_id="s1", minutes_ago=10),
    ]

    work = compose_recent_work(events, {"maria": maria}, {"s1": "Blog pipeline"})

    assert work.total_count == 3
    assert [group.actor.name for group in work.groups] == ["Maria", "Autopilot"]
    maria_group = work.groups[0]
    assert maria_group.actor.kind == "expert"
    assert maria_group.actor.expert is not None
    assert maria_group.session_title == "Blog pipeline"
    assert maria_group.link == "/copilot?sessionId=s1"
    assert [item.id for item in maria_group.items] == ["e1", "e3"]
    assert work.groups[1].actor.kind == "autopilot"


def test_runs_are_excluded_and_items_capped() -> None:
    events = [
        _event(
            event_id=f"e{index}",
            expert_id="maria",
            session_id="s1",
            minutes_ago=index,
        )
        for index in range(_MAX_ITEMS_PER_GROUP + 3)
    ] + [_event(event_id="run-1", category="RUN", session_id="s1")]

    work = compose_recent_work(events, {}, {})

    assert work.total_count == _MAX_ITEMS_PER_GROUP + 3
    group = work.groups[0]
    assert len(group.items) == _MAX_ITEMS_PER_GROUP
    assert group.more_count == 3
    # The expert profile is gone but the attribution kind survives.
    assert group.actor.kind == "expert"
    assert group.actor.name == "Expert"


def test_graph_run_integration_event_without_session() -> None:
    events = [
        _event(
            event_id="e1",
            category="INTEGRATION",
            title="Send Email",
            provider="google",
            object_id="block-1",
        )
    ]

    work = compose_recent_work(events, {}, {})

    group = work.groups[0]
    assert group.actor.kind == "agent"
    assert group.link is None
    assert group.items[0].provider == "google"
    assert group.items[0].file_id is None


def test_file_items_carry_file_id_and_mime_type() -> None:
    event = _event(event_id="e1", session_id="s1", object_id="file-9")
    event.data = {"mime_type": "text/markdown", "size_bytes": 10}

    work = compose_recent_work([event], {}, {})

    item = work.groups[0].items[0]
    assert item.file_id == "file-9"
    assert item.mime_type == "text/markdown"
