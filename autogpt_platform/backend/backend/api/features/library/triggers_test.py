"""Tests for the shared webhook-preset helpers (setup/update/delete) used by both
the /presets routes and the copilot preset tools."""

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.library.triggers import (
    delete_preset_with_webhook_cleanup,
    setup_triggered_preset,
    update_triggered_preset,
)
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


# ---- update_triggered_preset ----


def _preset(*, webhook_id=None):
    preset = MagicMock()
    preset.id = "preset-1"
    preset.graph_id = "graph-1"
    preset.graph_version = 1
    preset.webhook_id = webhook_id
    return preset


@contextlib.contextmanager
def _update_patches(*, current, graph=..., webhook=..., feedback=None):
    """Patch update_triggered_preset's collaborators; yields the call mocks."""
    new_webhook = MagicMock(id="wh-new") if webhook is ... else webhook
    graph_val = _graph() if graph is ... else graph
    patchers = {
        "get_preset": patch(
            f"{_PATH}.db.get_preset", new=AsyncMock(return_value=current)
        ),
        "get_graph": patch(f"{_PATH}.get_graph", new=AsyncMock(return_value=graph_val)),
        "creds_map": patch(f"{_PATH}.make_node_credentials_input_map", return_value={}),
        "setup": patch(
            f"{_PATH}.setup_webhook_for_block",
            new=AsyncMock(return_value=(new_webhook, feedback)),
        ),
        "update": patch(
            f"{_PATH}.db.update_preset",
            new=AsyncMock(return_value=MagicMock(id="preset-1")),
        ),
        "set_webhook": patch(
            f"{_PATH}.db.set_preset_webhook",
            new=AsyncMock(return_value=MagicMock(id="preset-1")),
        ),
        "prune": patch(f"{_PATH}._prune_dangling_webhook", new=AsyncMock()),
    }
    with contextlib.ExitStack() as stack:
        yield {k: stack.enter_context(p) for k, p in patchers.items()}


@pytest.mark.asyncio
async def test_update_rename_only_skips_webhook():
    with _update_patches(current=_preset(webhook_id="wh-old")) as m:
        await update_triggered_preset(
            user_id=_USER, preset_id="preset-1", name="New name"
        )
    m["update"].assert_awaited_once()
    m["setup"].assert_not_awaited()
    m["set_webhook"].assert_not_awaited()
    m["prune"].assert_not_awaited()


@pytest.mark.asyncio
async def test_update_reconfigure_reregisters_and_prunes_old():
    with _update_patches(current=_preset(webhook_id="wh-old")) as m:
        await update_triggered_preset(
            user_id=_USER,
            preset_id="preset-1",
            inputs={"repo": "owner/repo"},
            credentials={},
        )
    m["setup"].assert_awaited_once()
    m["set_webhook"].assert_awaited_once()
    m["prune"].assert_awaited_once_with(_USER, "wh-old")


@pytest.mark.asyncio
async def test_update_reconfigure_webhook_rejected_raises():
    with _update_patches(current=_preset(), webhook=None, feedback="no events") as m:
        with pytest.raises(InvalidInputError, match="no events"):
            await update_triggered_preset(
                user_id=_USER,
                preset_id="preset-1",
                inputs={"repo": "x"},
                credentials={},
            )
    m["update"].assert_not_awaited()


@pytest.mark.asyncio
async def test_update_preset_not_found_raises():
    with _update_patches(current=None):
        with pytest.raises(NotFoundError):
            await update_triggered_preset(user_id=_USER, preset_id="missing", name="X")


@pytest.mark.asyncio
async def test_update_reconfigure_graph_gone_raises():
    with _update_patches(current=_preset(), graph=None):
        with pytest.raises(NotFoundError):
            await update_triggered_preset(
                user_id=_USER,
                preset_id="preset-1",
                inputs={"repo": "x"},
                credentials={},
            )


# ---- delete_preset_with_webhook_cleanup ----


@contextlib.contextmanager
def _delete_patches(*, preset):
    patchers = {
        "get_preset": patch(
            f"{_PATH}.db.get_preset", new=AsyncMock(return_value=preset)
        ),
        "set_webhook": patch(f"{_PATH}.db.set_preset_webhook", new=AsyncMock()),
        "prune": patch(f"{_PATH}._prune_dangling_webhook", new=AsyncMock()),
        "delete": patch(f"{_PATH}.db.delete_preset", new=AsyncMock()),
    }
    with contextlib.ExitStack() as stack:
        yield {k: stack.enter_context(p) for k, p in patchers.items()}


@pytest.mark.asyncio
async def test_delete_with_webhook_prunes():
    with _delete_patches(preset=_preset(webhook_id="wh-1")) as m:
        await delete_preset_with_webhook_cleanup(user_id=_USER, preset_id="preset-1")
    m["set_webhook"].assert_awaited_once_with(_USER, "preset-1", None)
    m["prune"].assert_awaited_once_with(_USER, "wh-1")
    m["delete"].assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_without_webhook_skips_prune():
    with _delete_patches(preset=_preset(webhook_id=None)) as m:
        await delete_preset_with_webhook_cleanup(user_id=_USER, preset_id="preset-1")
    m["set_webhook"].assert_not_awaited()
    m["prune"].assert_not_awaited()
    m["delete"].assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_not_found_raises():
    with _delete_patches(preset=None):
        with pytest.raises(NotFoundError):
            await delete_preset_with_webhook_cleanup(user_id=_USER, preset_id="missing")


@pytest.mark.asyncio
async def test_prune_dangling_webhook_is_best_effort():
    """Cleanup runs after state is committed, so a prune failure is swallowed
    (logged) rather than raised — the mutation must not fail post-commit."""
    from backend.api.features.library.triggers import _prune_dangling_webhook

    with patch(
        f"{_PATH}.get_webhook",
        new=AsyncMock(side_effect=Exception("provider unreachable")),
    ):
        # Must not raise.
        await _prune_dangling_webhook(_USER, "wh-1")
