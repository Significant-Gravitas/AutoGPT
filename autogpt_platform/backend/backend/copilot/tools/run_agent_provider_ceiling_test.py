"""An agent run is held to the run's ceiling on connected accounts."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.context import _current_permissions
from backend.copilot.permissions import CopilotPermissions
from backend.integrations.providers import ProviderName

from ._test_data import make_session
from .models import ErrorResponse
from .run_agent import RunAgentTool


def _graph(*providers) -> MagicMock:
    graph = MagicMock()
    graph.name = "PR helper"
    graph.aggregate_credentials_inputs.return_value = {
        "credentials": (MagicMock(provider=frozenset(providers)), set(), True)
    }
    return graph


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("scheduled", [False, True], ids=["run", "schedule"])
async def test_an_agent_acting_with_a_denied_provider_is_refused(scheduled):
    session = make_session(user_id="user-1")
    graph = _graph(ProviderName.GITHUB)
    token = _current_permissions.set(
        CopilotPermissions(providers=["github"], providers_exclude=True)
    )
    try:
        tool = RunAgentTool()
        if scheduled:
            response = await tool._schedule_agent(
                "user-1", session, graph, {}, {}, "nightly", "0 3 * * *", None
            )
        else:
            response = await tool._run_agent("user-1", session, graph, {}, {}, False)
    finally:
        _current_permissions.reset(token)
    assert isinstance(response, ErrorResponse)
    assert "GitHub account" in response.message
    assert "not permitted" in response.message


@pytest.mark.asyncio(loop_scope="session")
async def test_the_refusal_comes_before_any_setup_card():
    """A run the ceiling refuses asks the user for no credential or input."""
    session = make_session(user_id="user-1")
    graph = _graph(ProviderName.GITHUB)
    graph.has_external_trigger = False
    prerequisites = AsyncMock()
    token = _current_permissions.set(
        CopilotPermissions(providers=["github"], providers_exclude=True)
    )
    try:
        with (
            patch(
                "backend.copilot.tools.run_agent._agent_graph",
                new=AsyncMock(return_value=(graph, None)),
            ),
            patch(
                "backend.copilot.tools.run_agent._bind_builder_graph", new=AsyncMock()
            ),
            patch(
                "backend.copilot.tools.run_agent.require_installed_workflow",
                new=AsyncMock(return_value=None),
            ),
            patch.object(RunAgentTool, "_check_prerequisites", prerequisites),
        ):
            response = await RunAgentTool()._execute(
                user_id="user-1", session=session, username_agent_slug="ada/pr-helper"
            )
    finally:
        _current_permissions.reset(token)
    assert isinstance(response, ErrorResponse)
    assert "GitHub account" in response.message
    prerequisites.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="session")
async def test_an_agent_with_an_autopilot_block_is_refused_under_a_ceiling():
    """Its nested turn runs in the graph executor, which the ceiling does not
    reach, so it could act with any connected account."""
    from backend.blocks.autopilot import AUTOPILOT_BLOCK_ID

    session = make_session(user_id="user-1")
    graph = _graph(ProviderName.CODEX)
    graph.nodes = [MagicMock(block_id=AUTOPILOT_BLOCK_ID)]
    graph.sub_graphs = []
    token = _current_permissions.set(
        CopilotPermissions(providers=["github"], providers_exclude=True)
    )
    try:
        response = await RunAgentTool()._run_agent(
            "user-1", session, graph, {}, {}, False
        )
    finally:
        _current_permissions.reset(token)
    assert isinstance(response, ErrorResponse)
    assert "GitHub account" in response.message


@pytest.mark.asyncio(loop_scope="session")
async def test_an_autopilot_agent_runs_for_an_unrestricted_turn():
    from backend.blocks.autopilot import AUTOPILOT_BLOCK_ID
    from backend.copilot.permissions import denied_graph_providers

    graph = _graph(ProviderName.CODEX)
    graph.nodes = [MagicMock(block_id=AUTOPILOT_BLOCK_ID)]
    graph.sub_graphs = []
    assert denied_graph_providers(None, graph) == []
    assert denied_graph_providers(CopilotPermissions(), graph) == []
