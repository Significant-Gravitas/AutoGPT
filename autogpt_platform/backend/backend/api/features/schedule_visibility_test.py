from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autogpt_libs.auth.models import RequestContext

from backend.api.features.schedule_visibility import visible_graph_schedules
from backend.api.features.v1 import (
    list_all_graphs_execution_schedules,
    list_graph_execution_schedules,
)
from backend.executor.scheduler import GraphExecutionJobInfo


@pytest.mark.asyncio
@pytest.mark.parametrize("graph_only", [False, True])
async def test_rest_lists_paused_active_expert_schedules(graph_only):
    schedule = GraphExecutionJobInfo(
        id="paused",
        name="Paused QA",
        next_run_time="",
        user_id="owner",
        graph_id="graph",
        graph_version=1,
        cron="0 9 * * *",
        input_data={},
        expert_id="expert-a",
    )

    async def list_schedules(**kwargs):
        return [schedule] if kwargs.get("include_paused") else []

    with (
        patch(
            "backend.api.features.v1.get_scheduler_client",
            return_value=MagicMock(
                get_graph_execution_schedules=AsyncMock(side_effect=list_schedules)
            ),
        ),
        patch(
            "backend.api.features.v1.experts_db.owns_active_expert",
            AsyncMock(return_value=True),
        ),
    ):
        ctx = RequestContext(
            user_id="owner",
            org_id=None,
            team_id=None,
            is_org_owner=False,
            is_org_admin=False,
            is_org_billing_manager=False,
            is_team_admin=False,
            is_team_billing_manager=False,
            seat_status="ACTIVE",
        )
        result = (
            await list_graph_execution_schedules("owner", ctx, "graph")
            if graph_only
            else await list_all_graphs_execution_schedules("owner", ctx)
        )
    assert result == [schedule]


@pytest.mark.asyncio
async def test_paused_archived_and_finished_one_shot_schedules_stay_hidden():
    base = GraphExecutionJobInfo(
        id="paused",
        name="QA",
        next_run_time="",
        user_id="owner",
        graph_id="graph",
        graph_version=1,
        cron="0 9 * * *",
        input_data={},
    )
    archived = base.model_copy(
        update={"id": "archived", "expert_id": "archived-expert"}
    )
    finished = base.model_copy(update={"id": "finished", "cron": ""})
    with patch(
        "backend.api.features.schedule_visibility.experts_db.owns_active_expert",
        AsyncMock(return_value=False),
    ) as owns:
        result = await visible_graph_schedules([base, archived, finished], "owner")
    assert result == [base]
    owns.assert_awaited_once_with("owner", "archived-expert")
