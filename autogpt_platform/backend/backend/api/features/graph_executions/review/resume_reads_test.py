"""Resuming a reviewed run reads its user and graph settings side by side, and
gets or creates its workspace only once both have answered, as before."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.api.features.graph_executions.review import service as review_service
from backend.util.testing import Rendezvous

USER_ID = "user-1"


@pytest.mark.asyncio
async def test_a_resumed_run_reads_its_user_and_settings_side_by_side_then_its_workspace() -> (
    None
):
    meeting = Rendezvous(2)
    answered: list[str] = []
    answered_before_the_workspace: list[str] = []

    async def read_user(*_args: object, **_kwargs: object) -> SimpleNamespace:
        await meeting.arrive()
        answered.append("user")
        return SimpleNamespace(timezone="Europe/Paris")

    async def read_settings(*_args: object, **_kwargs: object) -> SimpleNamespace:
        await meeting.arrive()
        answered.append("settings")
        return SimpleNamespace(
            human_in_the_loop_safe_mode=False, sensitive_action_safe_mode=True
        )

    async def get_or_create_workspace(*_args: object) -> SimpleNamespace:
        answered_before_the_workspace.extend(answered)
        return SimpleNamespace(id="workspace-1")

    add_execution = AsyncMock()
    with (
        patch.object(review_service, "get_user_by_id", read_user),
        patch.object(review_service, "get_graph_settings", read_settings),
        patch.object(
            review_service, "get_or_create_workspace", get_or_create_workspace
        ),
        patch.object(review_service, "add_graph_execution", add_execution),
    ):
        await review_service._resume(
            USER_ID, "x1", "g1", 3, organization_id="org-1", team_id="team-1"
        )

    assert sorted(answered_before_the_workspace) == ["settings", "user"]
    context = add_execution.await_args.kwargs["execution_context"]
    assert context.user_timezone == "Europe/Paris"
    assert context.workspace_id == "workspace-1"
    assert context.human_in_the_loop_safe_mode is False
    assert context.sensitive_action_safe_mode is True
    assert add_execution.await_args.kwargs["graph_exec_id"] == "x1"


@pytest.mark.asyncio
async def test_a_resume_whose_settings_read_fails_gets_no_workspace_and_restarts_nothing() -> (
    None
):
    workspace = AsyncMock()
    add_execution = AsyncMock()
    with (
        patch.object(
            review_service,
            "get_user_by_id",
            AsyncMock(return_value=SimpleNamespace(timezone="UTC")),
        ),
        patch.object(
            review_service,
            "get_graph_settings",
            AsyncMock(side_effect=RuntimeError("settings read failed")),
        ),
        patch.object(review_service, "get_or_create_workspace", workspace),
        patch.object(review_service, "add_graph_execution", add_execution),
    ):
        await review_service._resume(
            USER_ID, "x1", "g1", 3, organization_id=None, team_id=None
        )

    workspace.assert_not_awaited()
    add_execution.assert_not_awaited()
