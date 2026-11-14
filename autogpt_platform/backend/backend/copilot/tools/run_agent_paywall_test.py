"""run_agent tells a user with no subscription why their run was refused.

``add_graph_execution`` raises ``UserPaywalledError`` for a user without a
plan. An MCP client must read that reason rather than "Failed to process
request", on a direct run and on a preset run alike (SECRT-3350).
"""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.util.exceptions import UserPaywalledError

from ._test_data import make_session
from .models import ErrorResponse
from .run_agent import RunAgentTool

_REFUSAL = "A subscription is required to run workflows."


@pytest.fixture
def paywalled_run(mocker):
    """Every step up to the enqueue passes; the enqueue refuses the user."""
    graph = MagicMock(id="graph-1", version=1, has_external_trigger=False)
    graph.name = "Weekly Report"
    library_agent = MagicMock(graph_id="graph-1", graph_version=1, id="lib-1")
    library_agent.name = "Weekly Report"
    preset = MagicMock(id="preset-1", inputs={}, credentials={})

    module = "backend.copilot.tools.run_agent"
    mocker.patch(
        f"{module}._agent_graph", AsyncMock(return_value=(graph, library_agent))
    )
    mocker.patch(f"{module}._preset_graph", AsyncMock(return_value=(preset, graph)))
    mocker.patch(f"{module}.require_installed_workflow", AsyncMock(return_value=None))
    mocker.patch(f"{module}._provider_refusal", return_value=None)
    mocker.patch.object(
        RunAgentTool, "_check_prerequisites", AsyncMock(return_value=({}, None))
    )
    mocker.patch(
        f"{module}.get_or_create_library_agent",
        AsyncMock(return_value=library_agent),
    )
    mocker.patch(f"{module}.pauses_irreversible_actions", AsyncMock(return_value=False))
    return mocker.patch(
        f"{module}.execution_utils.add_graph_execution",
        AsyncMock(side_effect=UserPaywalledError(_REFUSAL)),
    )


@pytest.mark.parametrize("external_caller", [True, False])
@pytest.mark.parametrize(
    "call",
    [{"username_agent_slug": "someone/weekly-report"}, {"preset_id": "preset-1"}],
    ids=["direct", "preset"],
)
@pytest.mark.asyncio(loop_scope="session")
async def test_a_paywalled_run_says_a_subscription_is_required(
    paywalled_run, caplog, call, external_caller
):
    session = make_session(user_id="user-1")
    session.external_caller = external_caller
    session.organization_id, session.team_id = "org-1", "team-1"

    with caplog.at_level(logging.INFO, logger="backend.copilot.tools.run_agent"):
        response = await RunAgentTool()._execute(
            user_id="user-1", session=session, **call
        )

    paywalled_run.assert_awaited_once()
    assert isinstance(response, ErrorResponse)
    assert response.message == _REFUSAL
    assert response.error == "subscription_required"
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
