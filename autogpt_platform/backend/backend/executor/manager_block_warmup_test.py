"""The executor must load the block registry before it takes any run."""

from unittest.mock import MagicMock, patch

import pytest

from backend.executor.manager import ExecutionManager


class _StopRunLoop(Exception):
    pass


def test_run_loads_blocks_before_taking_runs() -> None:
    """Pins the warm-up ahead of the run consumer.

    Without it the first runs on a fresh pod load every block inside
    get_block(), and a stop request for one of them times out while that load
    finishes (the library "start and stop a saved task" e2e flake).
    """
    manager = ExecutionManager()
    order: list[str] = []
    manager._cancel_thread = MagicMock()
    manager._cancel_thread.start.side_effect = lambda: order.append("cancel_thread")
    manager._run_thread = MagicMock()
    manager._run_thread.start.side_effect = lambda: order.append("run_thread")

    def load_blocks() -> dict:
        order.append("load_blocks")
        return {}

    with (
        patch("backend.blocks.load_all_blocks", side_effect=load_blocks),
        patch("backend.executor.manager.start_http_server"),
        patch.object(ExecutionManager, "_update_prompt_metrics"),
        patch("backend.executor.manager.time.sleep", side_effect=_StopRunLoop),
        pytest.raises(_StopRunLoop),
    ):
        manager.run()

    assert order == ["load_blocks", "cancel_thread", "run_thread"]
