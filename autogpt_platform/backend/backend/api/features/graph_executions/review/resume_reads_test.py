"""Resuming a reviewed run reads its user, graph settings and workspace side by
side: each stand-in answers only once all three are being awaited."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.api.features.graph_executions.review import service as review_service
from backend.util.testing import Rendezvous

USER_ID = "user-1"


@pytest.mark.asyncio
async def test_a_resumed_run_reads_its_user_settings_and_workspace_side_by_side() -> (
    None
):
    meeting = Rendezvous(3)
    add_execution = AsyncMock()
    with (
        patch.object(
            review_service,
            "get_user_by_id",
            meeting.returning(SimpleNamespace(timezone="Europe/Paris")),
        ),
        patch.object(
            review_service,
            "get_graph_settings",
            meeting.returning(
                SimpleNamespace(
                    human_in_the_loop_safe_mode=False, sensitive_action_safe_mode=True
                )
            ),
        ),
        patch.object(
            review_service,
            "get_or_create_workspace",
            meeting.returning(SimpleNamespace(id="workspace-1")),
        ),
        patch.object(review_service, "add_graph_execution", add_execution),
    ):
        await review_service._resume(
            USER_ID, "x1", "g1", 3, organization_id="org-1", team_id="team-1"
        )

    context = add_execution.await_args.kwargs["execution_context"]
    assert context.user_timezone == "Europe/Paris"
    assert context.workspace_id == "workspace-1"
    assert context.human_in_the_loop_safe_mode is False
    assert context.sensitive_action_safe_mode is True
    assert add_execution.await_args.kwargs["graph_exec_id"] == "x1"
