"""Tests for ``_execute_webhook_preset_trigger``: the webhook -> preset
execution path that separates the trigger node's input mask (nested under
``_node_input_mask_{node_id}``) from the regular graph inputs."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.integrations.router import _execute_webhook_preset_trigger

_PATH = "backend.api.features.integrations.router"


def _graph_with_trigger(*, triggers_on_event=True):
    block = MagicMock()
    block.is_triggered_by_event_type = MagicMock(return_value=triggers_on_event)
    trigger_node = MagicMock()
    trigger_node.id = "abc123-0"  # node prefix -> "abc123"
    trigger_node.block = block
    graph = MagicMock()
    graph.webhook_input_node = trigger_node
    return graph, trigger_node, block


def _preset(inputs):
    preset = MagicMock()
    preset.id = "preset-1"
    preset.is_active = True
    preset.graph_id = "graph-1"
    preset.graph_version = 1
    preset.user_id = "user-1"
    preset.inputs = inputs
    preset.credentials = {}
    return preset


def _webhook():
    webhook = MagicMock()
    webhook.user_id = "user-1"
    webhook.id = "wh-1"
    return webhook


@pytest.mark.asyncio
async def test_event_check_uses_unwrapped_trigger_config():
    """The event-type filter must run against the unwrapped trigger config, not
    the full preset inputs (which also hold the regular graph inputs)."""
    graph, trigger_node, block = _graph_with_trigger()
    preset = _preset(
        {"regular": "x", "_node_input_mask_abc123": {"events": {"push": True}}}
    )

    with (
        patch(f"{_PATH}.get_graph", AsyncMock(return_value=graph)),
        patch(f"{_PATH}.add_graph_execution", AsyncMock()) as add_exec,
    ):
        await _execute_webhook_preset_trigger(
            preset, _webhook(), "wh-1", "push", {"some": "payload"}
        )

    config = block.is_triggered_by_event_type.call_args.args[0]
    assert config.get("events") == {"push": True}
    assert "regular" not in config
    assert "_node_input_mask_abc123" not in config

    add_exec.assert_awaited_once()
    kwargs = add_exec.await_args.kwargs
    assert kwargs["inputs"] == {"regular": "x"}
    assert kwargs["nodes_input_masks"] == {
        trigger_node.id: {"events": {"push": True}, "payload": {"some": "payload"}}
    }


@pytest.mark.asyncio
async def test_skips_when_event_type_not_matched():
    graph, _trigger_node, _block = _graph_with_trigger(triggers_on_event=False)
    preset = _preset({"_node_input_mask_abc123": {"events": {"push": True}}})

    with (
        patch(f"{_PATH}.get_graph", AsyncMock(return_value=graph)),
        patch(f"{_PATH}.add_graph_execution", AsyncMock()) as add_exec,
    ):
        await _execute_webhook_preset_trigger(preset, _webhook(), "wh-1", "pull", {})

    add_exec.assert_not_awaited()


@pytest.mark.asyncio
async def test_skips_when_input_mask_missing():
    """A preset without the ``_node_input_mask_{node_id}`` key can't be run."""
    graph, _trigger_node, _block = _graph_with_trigger()
    preset = _preset({"regular": "x"})

    with (
        patch(f"{_PATH}.get_graph", AsyncMock(return_value=graph)),
        patch(f"{_PATH}.add_graph_execution", AsyncMock()) as add_exec,
    ):
        await _execute_webhook_preset_trigger(preset, _webhook(), "wh-1", "push", {})

    add_exec.assert_not_awaited()
