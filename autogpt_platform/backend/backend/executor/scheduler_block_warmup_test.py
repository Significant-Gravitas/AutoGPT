"""The scheduler must answer its liveness probe before it loads the block registry."""

import asyncio
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from backend.executor import scheduler as scheduler_module
from backend.executor.scheduler import Scheduler
from backend.util.service import UnhealthyServiceError


def test_blocks_load_once_the_rpc_server_serves_then_the_heap_freezes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The RPC server starts only after run_service hands over to its event loop,
    so a load inside run_service held the liveness probe for the 10-17 s it takes
    on a pod, and Kubernetes killed the scheduler before it ever answered."""
    order: list[str] = []
    frozen = threading.Event()

    def load_blocks() -> dict:
        serving = "serving" in order
        order.append("load_blocks" if serving else "load_blocks_before_serving")
        return {}

    def freeze_heap() -> None:
        order.append("freeze_heap")
        frozen.set()

    with _run_service(monkeypatch, order, load_blocks, freeze_heap):
        frozen.wait(5)

    assert order == ["serving", "load_blocks", "freeze_heap"]


async def test_a_failed_block_load_makes_the_scheduler_unhealthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """So the pod restarts, as it did when the load ran before serving, instead of
    staying healthy while every schedule call retries the failing load."""

    def load_blocks() -> dict:
        raise ImportError("a block failed to import")

    with _run_service(monkeypatch, [], load_blocks, MagicMock()) as service:
        for _ in range(100):
            try:
                await service.health_check()
            except UnhealthyServiceError:
                return
            await asyncio.sleep(0.05)

    pytest.fail("the scheduler stayed healthy after its block load failed")


@contextmanager
def _run_service(
    monkeypatch: pytest.MonkeyPatch,
    order: list[str],
    load_blocks: Callable[[], dict],
    freeze_heap: Callable[[], None],
) -> Iterator[Scheduler]:
    """``run_service`` with throwaway jobstores, its RPC server marked serving at
    the hand-over to the event loop, then the service's own cleanup."""
    for name in ("_event_loop", "_event_loop_thread", "_scheduler_instance"):
        monkeypatch.setattr(scheduler_module, name, getattr(scheduler_module, name))
    service = Scheduler(register_system_tasks=False, throwaway_jobstores=True)
    server = MagicMock(started=False)
    service.http_server = server

    def serve() -> None:
        order.append("serving")
        server.started = True

    with (
        patch.object(scheduler_module, "load_dotenv"),
        patch.object(scheduler_module, "get_blocks", side_effect=load_blocks),
        patch.object(scheduler_module, "freeze_startup_heap", side_effect=freeze_heap),
        patch("backend.util.service.AppService.run_service", side_effect=serve),
        patch("backend.util.service.AppService.cleanup"),
    ):
        service.run_service()
        try:
            yield service
        finally:
            service.cleanup()
            if scheduler_module._event_loop is not None:
                scheduler_module._event_loop.close()
