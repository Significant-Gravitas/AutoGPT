import asyncio

import pytest

from backend.api.features.local_executor import websocket as ws_module


@pytest.mark.asyncio
async def test_a_cancelled_teardown_raises_the_handlers_own_cancellation() -> None:
    """Starlette's TestClient cancels the handler as soon as it closes, often
    while the handler is still stopping its relay and token-watch tasks. anyio
    absorbs only its own CancelledError. gather would raise a child's instead:
    the last-listed task's, here one that finished, cancelled without a
    message, while the other was still stopping. That escaped the scope and
    failed the test at teardown."""
    release = asyncio.Event()

    async def slow_to_stop() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await release.wait()
            raise

    slow = asyncio.create_task(slow_to_stop())
    quick = asyncio.create_task(asyncio.sleep(3600))
    await asyncio.sleep(0)

    handler = asyncio.create_task(ws_module._stop_tasks(slow, quick))
    for _ in range(3):
        await asyncio.sleep(0)
    assert quick.done() and not slow.done()
    handler.cancel("cancelled by the server")
    release.set()

    with pytest.raises(asyncio.CancelledError) as raised:
        await handler

    assert raised.value.args == ("cancelled by the server",)
    assert slow.done()


@pytest.mark.asyncio
async def test_stopping_tasks_waits_for_them_and_retrieves_their_errors() -> None:
    async def fails_while_stopping() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise RuntimeError("cleanup failed")

    failing = asyncio.create_task(fails_while_stopping())
    finished = asyncio.create_task(asyncio.sleep(0))
    await asyncio.sleep(0)

    await ws_module._stop_tasks(failing, finished)

    assert failing.done() and finished.done()
    assert isinstance(failing.exception(), RuntimeError)
