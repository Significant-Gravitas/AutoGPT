"""Graph validation refuses the platform's shared keys for a block that sets
``refuses_platform_credentials``, so the run or schedule fails up front instead
of at the node."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr
from pytest_mock import MockerFixture

from backend.blocks._base import Block
from backend.blocks.exa.search import ExaSearchBlock
from backend.blocks.exa.websets import ExaCreateWebsetBlock, ExaPreviewWebsetBlock
from backend.data.model import APIKeyCredentials, Credentials
from backend.executor.utils import (
    CRED_ERR_NOT_AVAILABLE_PREFIX,
    _validate_node_input_credentials,
    is_credential_validation_error_message,
)
from backend.integrations.credentials_store import exa_credentials

USER_OWNED = APIKeyCredentials(
    id="2c0f3f0e-8f43-4b0c-9d1b-0c4c1d2f6a11",
    provider="exa",
    api_key=SecretStr("user-key"),
    title="My Exa key",
)


def _graph(
    block: Block, creds: Credentials, *, masks_inputs: dict[str, Any] | None = None
) -> MagicMock:
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
        },
        **(masks_inputs or {}),
    }
    return MagicMock(nodes=[node])


@pytest.fixture(autouse=True)
def store(mocker: MockerFixture) -> MagicMock:
    store = MagicMock()
    store.get_all_creds = AsyncMock(return_value=[exa_credentials, USER_OWNED])
    mocker.patch(
        "backend.executor.utils.get_integration_credentials_store",
        return_value=store,
    )
    return store


@pytest.mark.asyncio
async def test_a_refusing_block_on_the_system_credential_fails_validation():
    graph = _graph(ExaCreateWebsetBlock(), exa_credentials)

    errors, nodes_to_skip = await _validate_node_input_credentials(graph, "user-a")

    message = errors["node-1"]["credentials"]
    assert message.startswith(CRED_ERR_NOT_AVAILABLE_PREFIX)
    assert "platform's shared Exa key" in message
    assert is_credential_validation_error_message(message)
    assert "node-1" not in nodes_to_skip


@pytest.mark.asyncio
async def test_a_refusing_block_on_the_users_own_key_passes():
    errors, _ = await _validate_node_input_credentials(
        _graph(ExaCreateWebsetBlock(), USER_OWNED), "user-a"
    )

    assert "node-1" not in errors


@pytest.mark.asyncio
async def test_a_stateless_block_on_the_system_credential_passes():
    errors, _ = await _validate_node_input_credentials(
        _graph(ExaSearchBlock(), exa_credentials), "user-a"
    )

    assert "node-1" not in errors


@pytest.mark.asyncio
async def test_preview_is_refused_only_when_it_asks_for_sample_items():
    plain, _ = await _validate_node_input_credentials(
        _graph(ExaPreviewWebsetBlock(), exa_credentials), "user-a"
    )
    sampled, _ = await _validate_node_input_credentials(
        _graph(
            ExaPreviewWebsetBlock(),
            exa_credentials,
            masks_inputs={"include_sample_items": True},
        ),
        "user-a",
    )

    assert "node-1" not in plain
    assert "platform's shared Exa key" in sampled["node-1"]["credentials"]


@pytest.mark.asyncio
async def test_an_input_mask_turning_on_sample_items_is_refused_too():
    graph = _graph(ExaPreviewWebsetBlock(), exa_credentials)

    errors, _ = await _validate_node_input_credentials(
        graph, "user-a", nodes_input_masks={"node-1": {"include_sample_items": True}}
    )

    assert "platform's shared Exa key" in errors["node-1"]["credentials"]


def test_the_builder_schema_marks_the_credentials_field():
    marked = ExaCreateWebsetBlock().get_info().inputSchema["properties"]
    plain = ExaSearchBlock().get_info().inputSchema["properties"]

    assert marked["credentials"]["refuses_platform_credentials"] is True
    assert "refuses_platform_credentials" not in plain["credentials"]
    # The class's cached schema is left untouched.
    assert "refuses_platform_credentials" not in (
        ExaCreateWebsetBlock().input_schema.jsonschema()["properties"]["credentials"]
    )
