"""Webhook housekeeping still loads a refused SDK default credential.

A webhook registered on a ``<provider>-default`` key before it was refused must
still be pinged, pruned and deregistered with that key. Each housekeeping path
runs against the real ``IntegrationCredentialsManager.get``, so dropping its
``allow_sdk_default=True`` makes the refusal fire and these tests fail.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.api.features.integrations.router import (
    _cleanup_orphaned_webhook_for_graph,
    webhook_ping,
)
from backend.api.features.library.triggers import _prune_dangling_webhook
from backend.data.integrations import unlink_webhook_from_graph
from backend.data.model import APIKeyCredentials
from backend.integrations.credentials_store import IntegrationCredentialsStore

USER_ID = "user-a"
REFUSED = APIKeyCredentials(
    id="linear-default",
    provider="linear",
    api_key=SecretStr("platform-key"),
    title="Linear API Key",
)


@pytest.fixture(autouse=True)
def store():
    with patch.object(
        IntegrationCredentialsStore,
        "get_creds_by_id",
        AsyncMock(
            side_effect=lambda _user, cid: REFUSED if cid == REFUSED.id else None
        ),
    ):
        yield


def _webhook(**fields) -> MagicMock:
    webhook = MagicMock(
        id="wh-1",
        user_id=USER_ID,
        provider="linear",
        credentials_id=REFUSED.id,
        triggered_nodes=[],
        triggered_presets=[],
    )
    for name, value in fields.items():
        setattr(webhook, name, value)
    return webhook


def _webhook_manager() -> MagicMock:
    manager = MagicMock()
    manager.prune_webhook_if_dangling = AsyncMock(return_value=True)
    manager.trigger_ping = AsyncMock()
    return manager


def _pruned_with(manager: MagicMock) -> object:
    manager.prune_webhook_if_dangling.assert_awaited_once()
    return manager.prune_webhook_if_dangling.await_args.args[2]


@pytest.mark.asyncio
async def test_unlinking_a_graph_prunes_with_the_stored_key():
    path = "backend.data.integrations"
    manager = _webhook_manager()
    no_rows = MagicMock(find_many=AsyncMock(return_value=[]))
    with (
        patch(f"{path}.AgentNode.prisma", return_value=no_rows),
        patch(f"{path}.AgentPreset.prisma", return_value=no_rows),
        patch(f"{path}.get_webhook", AsyncMock(return_value=_webhook())),
        patch(f"{path}.get_webhook_manager", return_value=manager),
    ):
        await unlink_webhook_from_graph("wh-1", "graph-1", USER_ID)

    assert _pruned_with(manager) == REFUSED


@pytest.mark.asyncio
async def test_deleting_a_trigger_preset_prunes_with_the_stored_key():
    path = "backend.api.features.library.triggers"
    manager = _webhook_manager()
    with (
        patch(f"{path}.get_webhook", AsyncMock(return_value=_webhook())),
        patch(f"{path}.get_webhook_manager", return_value=manager),
    ):
        await _prune_dangling_webhook(USER_ID, "wh-1")

    assert _pruned_with(manager) == REFUSED


@pytest.mark.asyncio
async def test_ping_uses_the_stored_key():
    path = "backend.api.features.integrations.router"
    manager = _webhook_manager()
    with (
        patch(f"{path}.get_webhook", AsyncMock(return_value=_webhook())),
        patch(f"{path}.get_webhook_manager", return_value=manager),
        patch(f"{path}.wait_for_webhook_event", AsyncMock(return_value=True)),
    ):
        assert await webhook_ping(webhook_id="wh-1", user_id=USER_ID) is True

    manager.trigger_ping.assert_awaited_once()
    assert manager.trigger_ping.await_args.args[1] == REFUSED


@pytest.mark.asyncio
async def test_orphan_cleanup_prunes_with_the_stored_key():
    path = "backend.api.features.integrations.router"
    manager = _webhook_manager()
    linked = _webhook(triggered_nodes=[MagicMock(id="node-1", graph_id="graph-1")])
    with (
        patch(f"{path}.get_webhook", AsyncMock(side_effect=[linked, _webhook()])),
        patch(f"{path}.set_node_webhook", AsyncMock()),
        patch(f"{path}.get_webhook_manager", return_value=manager),
    ):
        await _cleanup_orphaned_webhook_for_graph("graph-1", USER_ID, "wh-1")

    assert _pruned_with(manager) == REFUSED
