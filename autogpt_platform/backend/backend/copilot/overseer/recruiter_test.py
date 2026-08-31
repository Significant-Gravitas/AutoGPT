"""Pure-function tests for the recruiter's hire recommendation."""

from datetime import UTC, datetime

from backend.api.features.experts.models import Expert
from backend.api.features.tasks.models import DelegatedTask
from backend.copilot.overseer.recruiter import compose_hire_items

_NOW = datetime(2026, 8, 30, 9, 0, tzinfo=UTC)


def _autopilot_task(task_id: str, title: str) -> DelegatedTask:
    return DelegatedTask(
        id=task_id,
        title=title,
        spec="spec",
        status="DONE",
        acceptance="PENDING",
        created_by_type="USER",
        created_by_id="user-1",
        owner=None,
        parent_task_id=None,
        root_task_id=task_id,
        origin_session_id=None,
        ancestor_expert_ids=[],
        handoff_count=0,
        revision_count=0,
        spend_total=0,
        outcome_summary=None,
        amendments=[],
        created_at=_NOW,
        updated_at=_NOW,
    )


def _expert(
    *,
    expert_id: str,
    name: str = "Maria",
    role: str = "Marketing",
    skills: list[str] | None = None,
    is_template: bool = True,
    source_template_id: str | None = None,
) -> Expert:
    return Expert.model_construct(
        id=expert_id,
        name=name,
        role=role,
        skills=skills if skills is not None else ["Marketing campaigns"],
        is_template=is_template,
        source_template_id=source_template_id,
    )


def test_three_matching_tasks_recommend_the_template():
    tasks = [
        _autopilot_task("t1", "Plan the marketing campaign"),
        _autopilot_task("t2", "Draft marketing email"),
        _autopilot_task("t3", "Review marketing budget"),
    ]
    template = _expert(expert_id="tmpl-maria")

    (item,) = compose_hire_items(tasks, [template], hired=[])

    assert item.template_id == "tmpl-maria"
    assert item.task_count == 3
    assert len(item.example_titles) == 3


def test_two_matching_tasks_are_not_enough():
    tasks = [
        _autopilot_task("t1", "Plan the marketing campaign"),
        _autopilot_task("t2", "Draft marketing email"),
    ]

    assert compose_hire_items(tasks, [_expert(expert_id="tmpl")], hired=[]) == []


def test_already_hired_template_is_never_recommended():
    tasks = [
        _autopilot_task("t1", "Plan the marketing campaign"),
        _autopilot_task("t2", "Draft marketing email"),
        _autopilot_task("t3", "Review marketing budget"),
    ]
    template = _expert(expert_id="tmpl-maria")
    hired = [
        _expert(
            expert_id="maria-1",
            is_template=False,
            source_template_id="tmpl-maria",
        )
    ]

    assert compose_hire_items(tasks, [template], hired=hired) == []


def test_best_matching_template_wins():
    tasks = [
        _autopilot_task("t1", "Plan the marketing campaign"),
        _autopilot_task("t2", "Draft marketing email"),
        _autopilot_task("t3", "Review marketing budget"),
        _autopilot_task("t4", "Chase overdue sales invoices"),
        _autopilot_task("t5", "Prepare sales pipeline review"),
        _autopilot_task("t6", "Summarise sales calls"),
        _autopilot_task("t7", "Draft sales outreach sequence"),
    ]
    marketing = _expert(expert_id="tmpl-maria")
    sales = _expert(
        expert_id="tmpl-max", name="Max", role="Sales", skills=["Sales outreach"]
    )

    (item,) = compose_hire_items(tasks, [marketing, sales], hired=[])

    assert item.template_id == "tmpl-max"
    assert item.task_count == 4
