"""A newer plan in a chat replaces the ones still awaiting approval."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.features.tasks import plans_db
from backend.api.features.tasks.models import PlanStatus, PlanStep


@pytest.fixture
def table(monkeypatch):
    actions = MagicMock()
    actions.update_many = AsyncMock()
    actions.create = AsyncMock(return_value=MagicMock())
    monkeypatch.setattr(
        plans_db.prisma.models.TaskPlan, "prisma", MagicMock(return_value=actions)
    )
    monkeypatch.setattr(plans_db, "row_to_plan", lambda row: row)
    return actions


@pytest.mark.asyncio
async def test_a_new_plan_cancels_the_chats_open_plans(table):
    await plans_db.create_plan(
        "user-1", "Launch the pricing page", [PlanStep(title="Copy")], None, "sess-1"
    )

    table.update_many.assert_awaited_once_with(
        where={
            "userId": "user-1",
            "sessionId": "sess-1",
            "status": PlanStatus.PROPOSED.value,
        },
        data={"status": PlanStatus.CANCELLED.value},
    )
    table.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_plan_outside_a_chat_replaces_nothing(table):
    await plans_db.create_plan(
        "user-1", "Launch the pricing page", [PlanStep(title="Copy")], None, None
    )

    table.update_many.assert_not_awaited()
