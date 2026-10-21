"""The scheduler must answer its liveness probe before it loads the block registry."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from backend.executor import scheduler as scheduler_module
from backend.executor.scheduler import Scheduler


def test_blocks_load_once_the_rpc_server_serves_then_the_heap_freezes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The RPC server starts only after run_service hands over to its event loop,
    so a load inside run_service held the liveness probe for the 10-17 s it takes
    on a pod, and Kubernetes killed the scheduler before it ever answered."""
    for name in ("_event_loop", "_event_loop_thread", "_scheduler_instance"):
        monkeypatch.setattr(scheduler_module, name, getattr(scheduler_module, name))
    service = Scheduler(register_system_tasks=False, throwaway_jobstores=True)
    server = MagicMock(started=False)
    service.http_server = server
    order: list[str] = []
    frozen = threading.Event()

    def load_blocks() -> dict:
        order.append("load_blocks" if server.started else "load_blocks_before_serving")
        return {}

    def serve() -> None:
        order.append("serving")
        server.started = True

    def freeze_heap() -> None:
        order.append("freeze_heap")
        frozen.set()

    with (
        patch.object(scheduler_module, "load_dotenv"),
        patch.object(scheduler_module, "get_blocks", side_effect=load_blocks),
        patch.object(scheduler_module, "freeze_startup_heap", side_effect=freeze_heap),
        patch("backend.util.service.AppService.run_service", side_effect=serve),
        patch("backend.util.service.AppService.cleanup"),
    ):
        service.run_service()
        try:
            frozen.wait(5)
        finally:
            service.cleanup()
            if scheduler_module._event_loop is not None:
                scheduler_module._event_loop.close()

    assert order == ["serving", "load_blocks", "freeze_heap"]
