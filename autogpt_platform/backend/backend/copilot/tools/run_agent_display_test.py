"""Resolved library names arrive before starting or waiting for a workflow."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.tool_display import tool_display_context
from backend.copilot.tools.run_agent import RunAgentTool


@pytest.mark.asyncio
@pytest.mark.parametrize("wait_for_result", [0, 30])
async def test_canonical_library_name_is_published_before_execution(wait_for_result):
    session = ChatSession.new("user-1", dry_run=False)
    session.organization_id = "org-1"
    library_agent = MagicMock(id="library-id", graph_id="graph-id")
    library_agent.name = "My renamed workflow"
    graph = MagicMock(id="graph-id")
    graph.name = "Original graph name"
    published: list[str] = []

    async def start_execution(**kwargs):
        assert published == ["My renamed workflow"]
        raise RuntimeError("stop before actual execution")

    with (
        patch(
            "backend.copilot.tools.run_agent.get_or_create_library_agent",
            new=AsyncMock(return_value=library_agent),
        ),
        tool_display_context(published.append),
        patch(
            "backend.copilot.tools.run_agent.execution_utils.add_graph_execution",
            new=AsyncMock(side_effect=start_execution),
        ),
        pytest.raises(RuntimeError, match="stop before actual execution"),
    ):
        await RunAgentTool()._run_agent(
            user_id="user-1",
            session=session,
            graph=graph,
            graph_credentials={},
            inputs={},
            dry_run=False,
            wait_for_result=wait_for_result,
        )
    assert published == ["My renamed workflow"]
