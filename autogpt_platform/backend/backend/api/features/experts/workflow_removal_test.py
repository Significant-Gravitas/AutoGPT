from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import prisma.models
import pytest

from backend.api.features.experts import experts_db, scheduling
from backend.api.features.experts.errors import ExpertScheduleCleanupError
from backend.util.exceptions import NotFoundError


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcomes,deleted",
    [
        ([RuntimeError("unavailable"), RuntimeError("unavailable")], False),
        ([RuntimeError("transient"), None], True),
        ([NotFoundError("already deleted")], True),
        ([None], True),
    ],
)
async def test_remove_workflow_preserves_row_until_schedule_is_deleted(
    outcomes, deleted
):
    expert_client = SimpleNamespace(find_first=AsyncMock(return_value=object()))
    workflow_client = SimpleNamespace(
        find_first=AsyncMock(
            return_value=SimpleNamespace(id="workflow-1", scheduleId="schedule-1")
        ),
        delete=AsyncMock(),
    )
    scheduler = SimpleNamespace(delete_schedule=AsyncMock(side_effect=outcomes))
    with (
        patch.object(prisma.models.Expert, "prisma", return_value=expert_client),
        patch.object(
            prisma.models.ExpertWorkflow, "prisma", return_value=workflow_client
        ),
        patch.object(scheduling, "get_scheduler_client", return_value=scheduler),
    ):
        if deleted:
            await experts_db.remove_workflow("owner-1", "expert-1", "workflow-1")
        else:
            with pytest.raises(ExpertScheduleCleanupError, match="schedule"):
                await experts_db.remove_workflow("owner-1", "expert-1", "workflow-1")

    assert (
        expert_client.find_first.await_args.kwargs["where"]["ownerUserId"] == "owner-1"
    )
    workflow_client.find_first.assert_awaited_once_with(
        where={"id": "workflow-1", "expertId": "expert-1"}
    )
    assert scheduler.delete_schedule.await_count == len(outcomes)
    scheduler.delete_schedule.assert_awaited_with("schedule-1", user_id="owner-1")
    if deleted:
        workflow_client.delete.assert_awaited_once_with(where={"id": "workflow-1"})
    else:
        workflow_client.delete.assert_not_awaited()
