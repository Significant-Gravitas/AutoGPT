from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.generic_webhook.triggers import GenericWebhookTriggerBlock
from backend.copilot.tools.agent_generator.core import get_agent_as_json
from backend.data.graph import GraphModel, NodeModel


def _graph_owned_by(owner_id: str) -> GraphModel:
    return GraphModel(
        id="graph-1",
        version=1,
        name="Listed",
        description="",
        user_id=owner_id,
        created_at=datetime(2026, 1, 1),
        nodes=[
            NodeModel(
                id="trigger",
                block_id=GenericWebhookTriggerBlock().id,
                graph_id="graph-1",
                graph_version=1,
                input_default={"secret_token": "owner-token", "constants": {}},
            )
        ],
    )


async def _agent_json_for(user_id: str, graph: GraphModel) -> dict | None:
    db = MagicMock()
    db.get_graph = AsyncMock(return_value=graph)
    with patch("backend.copilot.tools.agent_generator.core.graph_db", return_value=db):
        return await get_agent_as_json("graph-1", user_id)


@pytest.mark.asyncio
async def test_a_library_copy_of_someone_elses_workflow_comes_stripped():
    """A workflow from the marketplace is readable from the caller's library,
    but the JSON they get (written to their workspace, edited) is the one an
    export gives: no secrets the publisher stored on it."""
    agent_json = await _agent_json_for("library-member", _graph_owned_by("publisher"))

    assert agent_json is not None
    [trigger] = agent_json["nodes"]
    assert trigger["input_default"] == {"constants": {}}


@pytest.mark.asyncio
async def test_the_owner_gets_their_workflow_as_saved():
    agent_json = await _agent_json_for("owner", _graph_owned_by("owner"))

    assert agent_json is not None
    [trigger] = agent_json["nodes"]
    assert trigger["input_default"]["secret_token"] == "owner-token"
