from datetime import datetime

import pytest
import pytest_mock

from backend.blocks.generic_webhook.triggers import GenericWebhookTriggerBlock
from backend.data.graph import GraphModel, NodeModel

from .models import ChatRequest
from .service import OttoService


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


async def _graph_data_for(
    mocker: pytest_mock.MockFixture, user_id: str, graph: GraphModel
):
    mocker.patch(
        "backend.api.features.otto.service.graph_db.get_graph",
        mocker.AsyncMock(return_value=graph),
    )
    request = ChatRequest(
        query="What does this do?",
        conversation_history=[],
        message_id="message-1",
        include_graph_data=True,
        graph_id="graph-1",
    )
    return await OttoService._fetch_graph_data(request, user_id)


@pytest.mark.asyncio
async def test_someone_elses_graph_goes_to_otto_without_its_stored_secrets(
    mocker: pytest_mock.MockFixture,
):
    data = await _graph_data_for(mocker, "library-member", _graph_owned_by("publisher"))

    assert data is not None
    [node] = data.nodes
    assert node["data"] == {"constants": {}}


@pytest.mark.asyncio
async def test_the_owners_graph_goes_to_otto_as_saved(
    mocker: pytest_mock.MockFixture,
):
    data = await _graph_data_for(mocker, "owner", _graph_owned_by("owner"))

    assert data is not None
    [node] = data.nodes
    assert node["data"]["secret_token"] == "owner-token"
