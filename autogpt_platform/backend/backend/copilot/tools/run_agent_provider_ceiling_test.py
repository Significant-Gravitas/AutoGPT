"""An agent run is held to the run's ceiling on connected accounts."""

from unittest.mock import MagicMock

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
