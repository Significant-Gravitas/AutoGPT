"""An approved plan that isn't split runs as one task with its steps as a
checklist."""

from datetime import UTC, datetime

from backend.api.features.tasks.db import _single_plan_task
from backend.api.features.tasks.models import (
    Assignee,
    AssigneeType,
    Plan,
    PlanStatus,
    PlanStep,
    TaskStatus,
)

_WRITER = Assignee(type=AssigneeType.EXPERT, id="exp-1", name="Wren")
_EDITOR = Assignee(type=AssigneeType.EXPERT, id="exp-2", name="Eddie")


def _plan(*steps: PlanStep) -> Plan:
    return Plan(
        id="plan-1",
        status=PlanStatus.PROPOSED,
        goal="Write a 5-part email course",
        budget_usd=3.0,
        steps=list(steps),
        split=False,
        created_at=datetime(2026, 11, 3, tzinfo=UTC),
    )


def test_the_steps_become_the_checklist_of_one_task():
    plan = _plan(
        PlanStep(title="Email 1", description="Welcome", assignee=_WRITER),
        PlanStep(title="Email 2", assignee=_WRITER),
    )

    task = _single_plan_task(plan, plan.steps)

    assert task.title == "Write a 5-part email course"
    assert task.parent_id is None
    assert task.status == TaskStatus.TODO
    assert task.budget_cap_usd == 3.0
    assert task.description == "- [ ] Email 1: Welcome (Wren)\n- [ ] Email 2 (Wren)"


def test_a_shared_assignee_owns_the_task():
    plan = _plan(PlanStep(title="Email 1", assignee=_WRITER))

    task = _single_plan_task(plan, plan.steps)

    assert task.assignee_type == AssigneeType.EXPERT
    assert task.assignee_id == "exp-1"


def test_steps_for_different_people_go_to_otto():
    plan = _plan(
        PlanStep(title="Draft", assignee=_WRITER),
        PlanStep(title="Edit", assignee=_EDITOR),
    )

    task = _single_plan_task(plan, plan.steps)

    assert task.assignee_type == AssigneeType.OTTO
