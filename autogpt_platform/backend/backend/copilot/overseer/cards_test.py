"""Pure-function tests for the briefing nudge and merge composers."""

from datetime import UTC, datetime, timedelta

from backend.api.features.tasks.models import DelegatedTask, TaskAmendment
from backend.copilot.overseer.cards import compose_merge_items, compose_nudge_items

_NOW = datetime(2026, 8, 30, 9, 0, tzinfo=UTC)


def _task(
    *,
    task_id: str,
    title: str = "Draft the weekly report",
    status: str = "WAITING_USER",
    updated_hours_ago: float = 0,
    amendments: list[TaskAmendment] | None = None,
    stale_at: datetime | None = None,
    parent_task_id: str | None = None,
) -> DelegatedTask:
    updated = _NOW - timedelta(hours=updated_hours_ago)
    return DelegatedTask(
        id=task_id,
        title=title,
        spec="spec",
        status=status,  # type: ignore[arg-type]
        acceptance="PENDING",
        created_by_type="USER",
        created_by_id="user-1",
        owner=None,
        parent_task_id=parent_task_id,
        root_task_id=task_id,
        origin_session_id=None,
        ancestor_expert_ids=[],
        handoff_count=0,
        revision_count=0,
        spend_total=0,
        outcome_summary=None,
        amendments=amendments or [],
        stale_at=stale_at,
        created_at=updated,
        updated_at=updated,
    )


def test_nudges_only_tasks_waiting_over_a_day():
    fresh = _task(task_id="fresh", updated_hours_ago=2)
    old = _task(task_id="old", updated_hours_ago=30)
    working = _task(task_id="working", status="WORKING", updated_hours_ago=48)

    items = compose_nudge_items([fresh, old, working], _NOW)

    assert [item.task_id for item in items] == ["old"]


def test_nudge_carries_the_latest_escalation_question_and_stale_flag():
    escalation = TaskAmendment(
        at=_NOW - timedelta(days=2),
        by="expert-1",
        note="Which quarter?",
        kind="escalation",
        question="Which quarter should the report cover?",
    )
    task = _task(
        task_id="old",
        updated_hours_ago=10 * 24,
        amendments=[escalation],
        stale_at=_NOW - timedelta(days=1),
    )

    (item,) = compose_nudge_items([task], _NOW)

    assert item.question == "Which quarter should the report cover?"
    assert item.is_stale is True


def test_similar_titles_suggest_a_merge_once():
    a = _task(task_id="a", title="Write the launch blog post")
    b = _task(task_id="b", title="Write the launch blog post!")
    c = _task(task_id="c", title="Reconcile the Q3 invoices")

    items = compose_merge_items([a, b, c])

    assert len(items) == 1
    assert set(items[0].task_ids) == {"a", "b"}


def test_dissimilar_titles_and_subtasks_never_suggest_merges():
    a = _task(task_id="a", title="Write the launch blog post")
    b = _task(task_id="b", title="Reconcile the Q3 invoices")
    dup_child = _task(
        task_id="child",
        title="Write the launch blog post",
        parent_task_id="a",
    )

    assert compose_merge_items([a, b, dup_child]) == []
