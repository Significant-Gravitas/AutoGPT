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
            return_value=SimpleNamespace(
                id="workflow-1", scheduleId="schedule-1", LibraryAgent=None
            )
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
        where={"id": "workflow-1", "expertId": "expert-1"},
        include={"LibraryAgent": True},
    )
    assert scheduler.delete_schedule.await_count == len(outcomes)
    scheduler.delete_schedule.assert_awaited_with("schedule-1", user_id="owner-1")
    if deleted:
        workflow_client.delete.assert_awaited_once_with(where={"id": "workflow-1"})
    else:
        workflow_client.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_remove_workflow_stops_the_triggers_the_expert_made_itself():
    """ExpertWorkflow.scheduleId records only the install-time schedule, so a
    cron the expert made through run_agent used to keep firing — and spending —
    after the tool reported the workflow removed."""
    expert_client = SimpleNamespace(find_first=AsyncMock(return_value=object()))
    workflow_client = SimpleNamespace(
        find_first=AsyncMock(
            return_value=SimpleNamespace(
                id="workflow-1",
                scheduleId="install-schedule",
                LibraryAgent=SimpleNamespace(agentGraphId="graph-1"),
            )
        ),
        delete=AsyncMock(),
    )
    preset = SimpleNamespace(name="Inbox webhook")
    preset_client = SimpleNamespace(
        find_many=AsyncMock(return_value=[preset]), update_many=AsyncMock()
    )
    scheduler = SimpleNamespace(
        delete_schedule=AsyncMock(return_value=None),
        pause_schedule=AsyncMock(),
        get_execution_schedules=AsyncMock(
            return_value=[
                SimpleNamespace(
                    id="self-made",
                    kind="graph",
                    expert_id="expert-1",
                    graph_id="graph-1",
                    name="Daily wire transfer",
                    cron="0 9 * * *",
                ),
                SimpleNamespace(
                    id="install-schedule",
                    kind="graph",
                    expert_id="expert-1",
                    graph_id="graph-1",
                    name="Install cadence",
                    cron="0 8 * * *",
                ),
                SimpleNamespace(
                    id="other-graph",
                    kind="graph",
                    expert_id="expert-1",
                    graph_id="graph-2",
                    name="Unrelated",
                    cron="0 7 * * *",
                ),
            ]
        ),
    )
    with (
        patch.object(prisma.models.Expert, "prisma", return_value=expert_client),
        patch.object(
            prisma.models.ExpertWorkflow, "prisma", return_value=workflow_client
        ),
        patch.object(prisma.models.AgentPreset, "prisma", return_value=preset_client),
        patch.object(scheduling, "get_scheduler_client", return_value=scheduler),
    ):
        stopped = await experts_db.remove_workflow("owner-1", "expert-1", "workflow-1")

    # The install-time schedule is deleted, not paused; the expert's own one is
    # paused; another graph's is left alone.
    scheduler.pause_schedule.assert_awaited_once_with("self-made", user_id="owner-1")
    assert stopped == ["Inbox webhook", "Daily wire transfer"]


@pytest.mark.asyncio
async def test_a_nameless_schedule_is_reported_by_id_not_as_none():
    """A one-shot has neither name nor cron, and the caller joins this list
    into its message — appending None raised TypeError there."""
    expert_client = SimpleNamespace(find_first=AsyncMock(return_value=object()))
    workflow_client = SimpleNamespace(
        find_first=AsyncMock(
            return_value=SimpleNamespace(
                id="workflow-1",
                scheduleId=None,
                LibraryAgent=SimpleNamespace(agentGraphId="graph-1"),
            )
        ),
        delete=AsyncMock(),
    )
    preset_client = SimpleNamespace(
        find_many=AsyncMock(return_value=[]), update_many=AsyncMock()
    )
    scheduler = SimpleNamespace(
        pause_schedule=AsyncMock(),
        get_execution_schedules=AsyncMock(
            return_value=[
                SimpleNamespace(
                    id="one-shot",
                    kind="graph",
                    expert_id="expert-1",
                    graph_id="graph-1",
                    name=None,
                    cron=None,
                )
            ]
        ),
    )
    with (
        patch.object(prisma.models.Expert, "prisma", return_value=expert_client),
        patch.object(
            prisma.models.ExpertWorkflow, "prisma", return_value=workflow_client
        ),
        patch.object(prisma.models.AgentPreset, "prisma", return_value=preset_client),
        patch.object(scheduling, "get_scheduler_client", return_value=scheduler),
    ):
        stopped = await experts_db.remove_workflow("owner-1", "expert-1", "workflow-1")

    assert stopped == ["one-shot"]
    assert ", ".join(stopped) == "one-shot"
