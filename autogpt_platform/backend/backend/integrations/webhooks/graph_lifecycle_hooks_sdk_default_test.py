"""Graph save refuses a node that names a refused SDK default credential."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.data.model import APIKeyCredentials
from backend.integrations.webhooks.graph_lifecycle_hooks import (
    GraphActivationError,
    _before_graph_activate,
    credentials_manager,
    on_graph_deactivate,
)

REFUSED = APIKeyCredentials(
    id="linear-default",
    provider="linear",
    api_key=SecretStr("platform-key"),
    title="Linear API Key",
)
RUNNABLE = APIKeyCredentials(
    id="tavily-default",
    provider="tavily",
    api_key=SecretStr("platform-key"),
    title="Tavily API Key",
)


def _node(creds: APIKeyCredentials, *, required: bool = True) -> MagicMock:
    node = MagicMock()
    node.id = "node-1"
    node.credentials_optional = False
    node.input_default = {
        "credentials": {"id": creds.id, "provider": creds.provider, "type": "api_key"}
    }
    node.block.input_schema.get_credentials_fields.return_value = {
        "credentials": object()
    }
    node.block.input_schema.get_required_fields.return_value = (
        {"credentials"} if required else set()
    )
    node.block.name = "SomeBlock"
    node.block.webhook_config = None
    return node


@pytest.fixture(autouse=True)
def _store():
    with patch.object(
        credentials_manager.store,
        "get_all_creds",
        AsyncMock(return_value=[REFUSED, RUNNABLE]),
    ):
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize("required", [True, False])
async def test_graph_save_rejects_refused_sdk_default(required: bool):
    graph = MagicMock(nodes=[_node(REFUSED, required=required)])

    with pytest.raises(GraphActivationError, match="platform's shared Linear"):
        await _before_graph_activate(graph, "user-a")


@pytest.mark.asyncio
async def test_graph_save_accepts_opted_in_sdk_default():
    node = _node(RUNNABLE)
    graph = MagicMock(nodes=[node])

    await _before_graph_activate(graph, "user-a")

    assert node.input_default["credentials"]["id"] == RUNNABLE.id


@pytest.mark.asyncio
async def test_graph_deactivate_still_resolves_refused_sdk_default():
    """Deactivation must reach the stored key to deregister a webhook."""
    node = _node(REFUSED)
    graph = MagicMock(nodes=[node])

    with patch(
        "backend.integrations.webhooks.graph_lifecycle_hooks.on_node_deactivate",
        AsyncMock(side_effect=lambda _user, n, credentials: n),
    ) as deactivate:
        await on_graph_deactivate(graph, "user-a")

    assert deactivate.await_args.kwargs["credentials"] == REFUSED
