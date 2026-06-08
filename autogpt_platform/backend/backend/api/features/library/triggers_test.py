"""Tests for setup_triggered_preset — the shared webhook-trigger preset
creation used by both the POST /presets/setup-trigger route and the copilot
setup_agent_webhook_trigger tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.library.triggers import setup_triggered_preset
from backend.util.exceptions import InvalidInputError, NotFoundError

_USER = "test-user-triggers"
_PATH = "backend.api.features.library.triggers"


def _graph():
    node = MagicMock()
    node.id = "trigger-node"
    graph = MagicMock()
    graph.id = "graph-1"
    graph.version = 1
    graph.webhook_input_node = node
    return graph


def _patches(*, graph, webhook=..., feedback=None, preset=None):
    """Patch triggers.py's collaborators. ``webhook=...`` defaults to a stub
    webhook; pass ``webhook=None`` + ``feedback`` to exercise the rejection."""
    new_webhook = MagicMock(id="wh-1") if webhook is ... else webhook
    return (
        patch(f"{_PATH}.get_graph", new=AsyncMock(return_value=graph)),
        patch(f"{_PATH}.make_node_credentials_input_map", return_value={}),
        patch(
            f"{_PATH}.setup_webhook_for_block",
            new=AsyncMock(return_value=(new_webhook, feedback)),
        ),
        patch(f"{_PATH}.db.create_preset", new=AsyncMock(return_value=preset)),
    )


async def _setup():
    return await setup_triggered_preset(
        user_id=_USER,
        graph_id="graph-1",
        graph_version=1,
        name="My Trigger",
        description="",
        trigger_config={"repo": "owner/repo"},
        agent_credentials={},
    )


@pytest.mark.asyncio
async def test_creates_preset_on_success():
    preset = MagicMock(id="preset-1")
    p_graph, p_creds, p_webhook, p_create = _patches(graph=_graph(), preset=preset)
    with p_graph, p_creds, p_webhook, p_create as create_mock:
        result = await _setup()
    assert result is preset
    create_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_graph_not_found_raises():
    p_graph, p_creds, p_webhook, p_create = _patches(graph=None)
    with p_graph, p_creds, p_webhook, p_create:
        with pytest.raises(NotFoundError):
            await _setup()


@pytest.mark.asyncio
async def test_no_webhook_node_raises():
    graph = _graph()
    graph.webhook_input_node = None
    p_graph, p_creds, p_webhook, p_create = _patches(graph=graph)
    with p_graph, p_creds, p_webhook, p_create:
        with pytest.raises(InvalidInputError):
            await _setup()


@pytest.mark.asyncio
async def test_webhook_setup_rejected_raises():
    p_graph, p_creds, p_webhook, p_create = _patches(
        graph=_graph(), webhook=None, feedback="no enabled events"
    )
    with p_graph, p_creds, p_webhook, p_create as create_mock:
        with pytest.raises(InvalidInputError, match="no enabled events"):
            await _setup()
    create_mock.assert_not_awaited()
