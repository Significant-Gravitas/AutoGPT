"""The copilot executor must open its probe port, then load the block registry
before it takes any turn."""

from unittest.mock import MagicMock, patch

import pytest

from backend.copilot.executor.manager import CoPilotExecutor


class _StopRunLoop(Exception):
    pass


def test_run_opens_its_port_before_loading_blocks_and_taking_turns() -> None:
    """Without the load a turn's first tool call loads every block, and the full
    collections over the grown heap stall every turn on the pod for seconds. The
    liveness probe connects to the port, so it must not wait for the loads."""
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
        patch(
            "backend.copilot.executor.manager._load_catalog",
            side_effect=lambda: order.append("load_catalog"),
        ),
        patch(
            "backend.copilot.executor.manager.start_cli_warmup",
            side_effect=lambda: order.append("warm_cli"),
        ) as warm_cli,
        patch("backend.copilot.executor.manager.get_blocks", side_effect=load_blocks),
        patch(
            "backend.copilot.executor.manager.freeze_startup_heap",
            side_effect=lambda: order.append("freeze_heap"),
        ),
        patch(
            "backend.copilot.executor.manager.start_http_server",
            side_effect=lambda *_, **__: order.append("open_port"),
        ),
        patch.object(CoPilotExecutor, "_update_metrics"),
        patch("backend.copilot.executor.manager.time.sleep", side_effect=_StopRunLoop),
        pytest.raises(_StopRunLoop),
    ):
        executor.run()

    warm_cli.assert_called_once_with()
    assert order == [
        "open_port",
        "load_catalog",
        "warm_cli",
        "load_blocks",
        "freeze_heap",
        "cancel_thread",
        "run_thread",
    ]
