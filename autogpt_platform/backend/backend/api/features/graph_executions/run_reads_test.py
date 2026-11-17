"""A workflow's runs page reads the user's onboarding beside the runs: each
stand-in answers only once both are being awaited. A run's page still checks
its workflow before it reads the onboarding, so a missing workflow stays a 404
whatever the onboarding read does."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from backend.api.features.graph_executions import routes
from backend.data import execution as execution_db
from backend.data import graph as graph_db
from backend.data.onboarding import OnboardingStep
from backend.util.models import Pagination
from backend.util.testing import Rendezvous

USER_ID = "user-1"
NO_ORG = SimpleNamespace(org_id=None)


def _onboarding(execution_id: str | None) -> SimpleNamespace:
    return SimpleNamespace(onboardingAgentExecutionId=execution_id, completedSteps=[])


@pytest.mark.asyncio
async def test_a_missing_workflow_is_a_404_even_when_the_onboarding_cannot_be_read() -> (
    None
):
    run = SimpleNamespace(graph_id="g1", graph_version=3)
    onboarding = AsyncMock(side_effect=RuntimeError("onboarding read failed"))
    complete_step = AsyncMock()
    with (
        patch.object(execution_db, "get_graph_execution", AsyncMock(return_value=run)),
        patch.object(graph_db, "get_graph", AsyncMock(return_value=None)),
        patch.object(routes, "get_user_onboarding", onboarding),
        patch.object(routes, "complete_onboarding_step", complete_step),
        pytest.raises(HTTPException) as raised,
    ):
        await routes.get_graph_execution("g1", "x1", USER_ID, NO_ORG)

    assert raised.value.status_code == 404
    onboarding.assert_not_awaited()
    complete_step.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_runs_page_completes_the_results_step_once_its_workflow_is_found() -> (
    None
):
    run = SimpleNamespace(graph_id="g1", graph_version=3)
    get_graph = AsyncMock(return_value=SimpleNamespace(id="g1"))
    complete_step = AsyncMock()
    with (
        patch.object(execution_db, "get_graph_execution", AsyncMock(return_value=run)),
        patch.object(
            routes, "get_user_onboarding", AsyncMock(return_value=_onboarding("x1"))
        ),
        patch.object(graph_db, "get_graph", get_graph),
        patch.object(routes, "complete_onboarding_step", complete_step),
    ):
        result = await routes.get_graph_execution("g1", "x1", USER_ID, NO_ORG)

    assert result is run
    get_graph.assert_awaited_once_with(
        graph_id="g1", version=3, user_id=USER_ID, organization_id=None
    )
    complete_step.assert_awaited_once_with(USER_ID, OnboardingStep.GET_RESULTS)


@pytest.mark.asyncio
async def test_a_run_of_another_workflow_is_refused_before_anything_else_is_read() -> (
    None
):
    get_graph = AsyncMock()
    onboarding = AsyncMock(return_value=_onboarding("x1"))
    complete_step = AsyncMock()
    with (
        patch.object(
            execution_db,
            "get_graph_execution",
            AsyncMock(return_value=SimpleNamespace(graph_id="other", graph_version=1)),
        ),
        patch.object(routes, "get_user_onboarding", onboarding),
        patch.object(graph_db, "get_graph", get_graph),
        patch.object(routes, "complete_onboarding_step", complete_step),
        pytest.raises(HTTPException) as raised,
    ):
        await routes.get_graph_execution("g1", "x1", USER_ID, NO_ORG)

    assert raised.value.status_code == 404
    get_graph.assert_not_awaited()
    onboarding.assert_not_awaited()
    complete_step.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_workflows_runs_and_the_users_onboarding_are_read_side_by_side() -> (
    None
):
    meeting = Rendezvous(2)
    page = execution_db.GraphExecutionsPaginated(
        executions=[],
        pagination=Pagination(
            total_items=0, total_pages=0, current_page=1, page_size=25
        ),
    )
    with (
        patch.object(
            execution_db, "get_graph_executions_paginated", meeting.returning(page)
        ),
        patch.object(
            routes, "get_user_onboarding", meeting.returning(_onboarding(None))
        ),
    ):
        result = await routes.list_graph_executions(
            "g1", USER_ID, NO_ORG, page=1, page_size=25
        )

    assert result.executions == []
    assert result.pagination.page_size == 25
