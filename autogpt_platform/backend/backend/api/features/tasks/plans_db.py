
import prisma.models
from prisma import Json

from backend.api.features.tasks.converters import row_to_plan
from backend.api.features.tasks.models import Plan, PlanStatus, PlanStep


async def create_plan(
    user_id: str,
    goal: str,
    steps: list[PlanStep],
    budget_usd: float | None,
    session_id: str | None,
) -> Plan:
    row = await prisma.models.TaskPlan.prisma().create(
        data={
            "userId": user_id,
            "goal": goal,
            "budgetUsd": budget_usd,
            "steps": Json([s.model_dump(mode="json") for s in steps]),
            "sessionId": session_id,
        }
    )
    return row_to_plan(row)


async def get_plan(user_id: str, plan_id: str) -> Plan | None:
    row = await prisma.models.TaskPlan.prisma().find_first(
        where={"id": plan_id, "userId": user_id}
    )
    return row_to_plan(row) if row else None


async def cancel_plan(user_id: str, plan_id: str) -> Plan | None:
    plan = await get_plan(user_id, plan_id)
    if plan is None:
        return None
    if plan.status != PlanStatus.PROPOSED:
        return plan
    row = await prisma.models.TaskPlan.prisma().update(
        where={"id": plan_id}, data={"status": PlanStatus.CANCELLED.value}
    )
    assert row is not None
    return row_to_plan(row)
