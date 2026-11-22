import asyncio

import pytest

from . import deadline


@pytest.mark.asyncio
async def test_deadline_cancels_work(monkeypatch):
    monkeypatch.setattr(deadline, "REGISTRY_CALL_TIMEOUT_SECONDS", 0.01)
    cancelled = asyncio.Event()

    async def work():
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        await deadline.within_deadline(work())
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_nested_deadline_preserves_outer_cancellation(monkeypatch):
    monkeypatch.setattr(deadline, "REGISTRY_CALL_TIMEOUT_SECONDS", 0.01)
    finished = False

    async def nested():
        nonlocal finished
        monkeypatch.setattr(deadline, "REGISTRY_CALL_TIMEOUT_SECONDS", 10)
        await deadline.within_deadline(asyncio.sleep(10))
        finished = True

    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        await deadline.within_deadline(nested())
    assert not finished


@pytest.mark.asyncio
async def test_deadline_preserves_external_cancellation():
    task = asyncio.create_task(deadline.within_deadline(asyncio.sleep(10)))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_deadline_returns_result():
    assert await deadline.within_deadline(asyncio.sleep(0, result=42)) == 42
