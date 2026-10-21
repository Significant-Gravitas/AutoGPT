"""The copilot executor must load the block registry before it takes any turn."""

from unittest.mock import MagicMock, patch

import pytest

from backend.copilot.executor.manager import CoPilotExecutor


class _StopRunLoop(Exception):
    pass


def test_run_loads_blocks_and_freezes_the_heap_before_taking_turns() -> None:
    """Otherwise a turn's first tool call loads every block, and the full
    collections over the grown heap stall every turn on the pod for seconds."""
    executor = CoPilotExecutor()
    order: list[str] = []
    executor._cancel_thread = MagicMock()
    executor._cancel_thread.start.side_effect = lambda: order.append("cancel_thread")
    executor._run_thread = MagicMock()
    executor._run_thread.start.side_effect = lambda: order.append("run_thread")

    def load_blocks() -> dict:
        order.append("load_blocks")
        return {}

    with (
        patch("backend.copilot.executor.manager._load_catalog"),
        patch("backend.copilot.executor.manager.get_blocks", side_effect=load_blocks),
        patch(
            "backend.copilot.executor.manager.freeze_startup_heap",
            side_effect=lambda: order.append("freeze_heap"),
        ),
        patch("backend.copilot.executor.manager.start_http_server"),
        patch.object(CoPilotExecutor, "_update_metrics"),
        patch("backend.copilot.executor.manager.time.sleep", side_effect=_StopRunLoop),
        pytest.raises(_StopRunLoop),
    ):
        executor.run()

    assert order == ["load_blocks", "freeze_heap", "cancel_thread", "run_thread"]
