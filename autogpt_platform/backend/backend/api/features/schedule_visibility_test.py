from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autogpt_libs.auth.models import RequestContext

from backend.api.features.schedule_visibility import (
    is_visible_schedule,
    visible_graph_schedules,
)
from backend.api.features.schedules.routes import (
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
            "backend.api.features.schedules.routes.get_scheduler_client",
            return_value=MagicMock(
                get_graph_execution_schedules=AsyncMock(side_effect=list_schedules)
            ),
        ),
        patch(
            "backend.api.features.experts.experts_db.active_expert_ids",
            AsyncMock(return_value={"expert-a"}),
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
async def test_paused_archived_expert_schedules_stay_hidden():
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
    with patch(
        "backend.api.features.experts.experts_db.active_expert_ids",
        AsyncMock(return_value=set()),
    ) as active:
        result = await visible_graph_schedules([base, archived], "owner")
    assert result == [base]
    # One batched call for the whole listing, not one per archived expert.
    active.assert_awaited_once_with("owner", {"archived-expert"})


@pytest.mark.asyncio
async def test_both_readers_apply_the_same_rule():
    """The REST listing and `list_schedules` must not disagree about a row.

    They read the same jobs and each used to carry its own condition; a row one
    listed and the other hid is what let the model act on what the UI could not
    show. `is_visible_schedule` is now the only rule, so this pins the decision
    rather than either call site.
    """
    live = GraphExecutionJobInfo(
        id="live",
        name="QA",
        next_run_time="2026-01-01T09:00:00Z",
        user_id="owner",
        graph_id="graph",
        graph_version=1,
        cron="0 9 * * *",
        input_data={},
        expert_id="archived-expert",
    )
    paused_archived = live.model_copy(update={"id": "gone", "next_run_time": ""})
    paused_owner = live.model_copy(
        update={"id": "mine", "next_run_time": "", "expert_id": None}
    )
    hidden = {"archived-expert"}
    assert is_visible_schedule(live, hidden) is True
    assert is_visible_schedule(paused_archived, hidden) is False
    # No expert and no next run: paused by the owner, and theirs to resume.
    assert is_visible_schedule(paused_owner, hidden) is True
