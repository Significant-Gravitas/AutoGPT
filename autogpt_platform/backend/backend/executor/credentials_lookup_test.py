"""Validating a run loads the user's credential store once, however many nodes
and credential ids the graph carries."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr
from pytest_mock import MockerFixture

from backend.blocks.tavily.search import TavilySearchBlock
from backend.data.model import APIKeyCredentials
from backend.executor.utils import (
    CRED_ERR_NOT_AVAILABLE_PREFIX,
    CRED_ERR_UNKNOWN_PREFIX,
    _validate_node_input_credentials,
)

STORED = APIKeyCredentials(
    id="user-tavily-key",
    provider="tavily",
    api_key=SecretStr("key"),
    title="Tavily",
)


@pytest.mark.asyncio
async def test_one_store_load_covers_every_node(mocker: MockerFixture):
    store = _patch_store(mocker, AsyncMock(return_value=[STORED]))
    ids = [STORED.id if i % 2 else f"unknown-{i}" for i in range(200)]

    errors, _ = await _validate_node_input_credentials(_graph(ids), "user-a")

    store.get_all_creds.assert_awaited_once_with("user-a")
    assert sorted(errors) == sorted(f"node-{i}" for i in range(0, 200, 2))
    assert errors["node-0"]["credentials"] == f"{CRED_ERR_UNKNOWN_PREFIX}unknown-0"


@pytest.mark.asyncio
async def test_a_failed_load_is_not_retried_per_node(mocker: MockerFixture):
    store = _patch_store(mocker, AsyncMock(side_effect=RuntimeError("store down")))

    errors, _ = await _validate_node_input_credentials(
        _graph([STORED.id] * 50), "user-a"
    )

    store.get_all_creds.assert_awaited_once()
    assert len(errors) == 50
    assert all(
        e["credentials"] == f"{CRED_ERR_NOT_AVAILABLE_PREFIX} store down"
        for e in errors.values()
    )


def _patch_store(mocker: MockerFixture, get_all_creds: AsyncMock) -> MagicMock:
    store = MagicMock()
    store.get_all_creds = get_all_creds
    mocker.patch(
        "backend.executor.utils.get_integration_credentials_store",
        return_value=store,
    )
    return store


def _graph(credential_ids: list[str]) -> MagicMock:
    block = TavilySearchBlock()
    nodes = []
    for i, credentials_id in enumerate(credential_ids):
        node = MagicMock()
        node.id = f"node-{i}"
        node.credentials_optional = False
        node.block = block
        node.input_default = {
            "credentials": {
                "id": credentials_id,
                "provider": "tavily",
                "type": "api_key",
                "title": "Tavily",
            }
        }
        nodes.append(node)
    return MagicMock(nodes=nodes)
