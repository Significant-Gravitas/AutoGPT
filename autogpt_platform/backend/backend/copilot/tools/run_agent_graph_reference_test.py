from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.run_agent import RunAgentTool

_PATH = "backend.copilot.tools.run_agent"


@pytest.mark.asyncio
@pytest.mark.parametrize("installed", [True, False])
async def test_graph_reference_resolves_before_installed_gate(installed):
    session = ChatSession.new("owner", dry_run=False, expert_id="expert-a")
    library = MagicMock()
    library.get_library_agent = AsyncMock(return_value=None)
    library.get_library_agent_by_graph_id = AsyncMock(
        return_value=MagicMock(id="lib-1", graph_id="graph-1", graph_version=1)
    )
    graph = MagicMock(id="graph-1")
    gate_result = ErrorResponse(
        message="scope reached" if installed else "not installed",
        error="test_stop" if installed else "workflow_not_installed",
    )
    gate = AsyncMock(return_value=gate_result)
    with (
        patch(f"{_PATH}.library_db", return_value=library),
        patch(
            f"{_PATH}.graph_db",
            return_value=MagicMock(get_graph=AsyncMock(return_value=graph)),
        ),
        patch(f"{_PATH}.require_installed_workflow", gate),
    ):
        result = await RunAgentTool()._execute(
            "owner", session, library_agent_id="graph-1"
        )
    assert result == gate_result
    library.get_library_agent_by_graph_id.assert_awaited_once_with("owner", "graph-1")
    assert gate.await_args.kwargs["library_agent_id"] == "lib-1"
    assert gate.await_args.kwargs["graph_id"] == "graph-1"
