"""Graph validation refuses a node that names a refused SDK default credential."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr
from pytest_mock import MockerFixture

from backend.blocks._base import Block
from backend.blocks.linear.issues import LinearSearchIssuesBlock
from backend.blocks.tavily.search import TavilySearchBlock
from backend.data.model import APIKeyCredentials
from backend.executor.utils import (
    CRED_ERR_NOT_AVAILABLE_PREFIX,
    _validate_node_input_credentials,
    is_credential_validation_error_message,
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


def _graph(block: Block, creds: APIKeyCredentials) -> MagicMock:
    node = MagicMock()
    node.id = "node-1"
    node.credentials_optional = False
    node.block = block
    node.input_default = {
        "credentials": {
            "id": creds.id,
            "provider": creds.provider,
            "type": creds.type,
            "title": creds.title,
        }
    }
    return MagicMock(nodes=[node])


@pytest.fixture(autouse=True)
def store(mocker: MockerFixture) -> MagicMock:
    by_id = {c.id: c for c in (REFUSED, RUNNABLE)}
    store = MagicMock()
    store.get_creds_by_id = AsyncMock(side_effect=lambda _user, cid: by_id.get(cid))
    mocker.patch(
        "backend.executor.utils.get_integration_credentials_store",
        return_value=store,
    )
    return store


@pytest.mark.asyncio
async def test_validation_refuses_sdk_default_credentials():
    graph = _graph(LinearSearchIssuesBlock(), REFUSED)

    errors, nodes_to_skip = await _validate_node_input_credentials(graph, "user-a")

    message = errors["node-1"]["credentials"]
    assert message.startswith(CRED_ERR_NOT_AVAILABLE_PREFIX)
    assert "platform's shared Linear key" in message
    assert is_credential_validation_error_message(message)
    assert "node-1" not in nodes_to_skip


@pytest.mark.asyncio
async def test_validation_accepts_opted_in_sdk_default_credentials():
    graph = _graph(TavilySearchBlock(), RUNNABLE)

    errors, _ = await _validate_node_input_credentials(graph, "user-a")

    assert "node-1" not in errors
