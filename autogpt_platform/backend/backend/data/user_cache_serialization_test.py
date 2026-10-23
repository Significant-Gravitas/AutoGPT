"""Concurrency guarantees for the synchronous user-cache clear."""

import asyncio
import threading
from collections.abc import Awaitable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar
from unittest.mock import patch

import pytest

from backend.data import user as user_db

T = TypeVar("T")


class _BlockingClear:
    def __init__(self) -> None:
        self.release = threading.Event()
        self.started = threading.Event()
        self._lock = threading.Lock()
        self._entered = 0
        self._active = 0
        self._peak_active = 0

    def __call__(self) -> None:
        with self._lock:
            self._entered += 1
            self._active += 1
            self._peak_active = max(self._peak_active, self._active)
            self.started.set()
        try:
            assert self.release.wait(5)
        finally:
            with self._lock:
                self._active -= 1

    def snapshot(self) -> tuple[int, int]:
        with self._lock:
            return self._entered, self._peak_active


def _run_in_private_loop(scenario: Awaitable[T]) -> T:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(scenario)
    finally:
        loop.close()


async def _wait_until_set(event: threading.Event) -> None:
    deadline = asyncio.get_running_loop().time() + 2
    while not event.is_set():
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0.01)


def test_concurrent_invalidations_are_serial_and_leave_the_default_executor_free():
    clear = _BlockingClear()

    async def scenario() -> tuple[int, bool, int, str]:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=2))
        ticks = 0

        async def tick() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        invalidations = [
            asyncio.create_task(
                user_db._invalidate_user_caches(f"user-{i}", f"{i}@example.com")
            )
            for i in range(3)
        ]
        ticker = asyncio.create_task(tick())
        probe: asyncio.Task[str] | None = None
        try:
            await _wait_until_set(clear.started)
            probe = asyncio.create_task(asyncio.to_thread(lambda: "probe"))
            await asyncio.sleep(0.1)
            observed_peak = clear.snapshot()[1]
            probe_finished_before_release = probe.done()
            observed_ticks = ticks
        finally:
            clear.release.set()
            ticker.cancel()
            await asyncio.gather(*invalidations, ticker, return_exceptions=True)
            probe_result = await probe if probe is not None else "missing"
            await loop.shutdown_default_executor()
        return (
            observed_peak,
            probe_finished_before_release,
            observed_ticks,
            probe_result,
        )

    with (
        patch.object(user_db.get_or_create_user, "cache_clear", clear),
        patch.object(user_db.get_user_by_id, "cache_delete"),
        patch.object(user_db.get_user_by_email, "cache_delete"),
    ):
        observed = _run_in_private_loop(scenario())

    peak_active, probe_finished, ticks, probe_result = observed
    assert peak_active == 1
    assert probe_finished
    assert ticks >= 5
    assert probe_result == "probe"
    assert clear.snapshot()[0] == 3


def test_cancellation_waits_for_the_submitted_invalidation_to_finish():
    clear = _BlockingClear()

    async def scenario() -> tuple[bool, list[object]]:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        first = asyncio.create_task(
            user_db._invalidate_user_caches("user-1", "1@example.com")
        )
        second: asyncio.Task[None] | None = None
        try:
            await _wait_until_set(clear.started)
            second = asyncio.create_task(
                user_db._invalidate_user_caches("user-2", "2@example.com")
            )
            await asyncio.sleep(0.02)
            second.cancel()
            await asyncio.sleep(0.02)
            second.cancel()
            await asyncio.sleep(0.03)
            returned_before_clear = second.done()
        finally:
            clear.release.set()
            tasks = [first]
            if second is not None:
                tasks.append(second)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            await loop.shutdown_default_executor()
        return returned_before_clear, results

    with (
        patch.object(user_db.get_or_create_user, "cache_clear", clear),
        patch.object(user_db.get_user_by_id, "cache_delete"),
        patch.object(user_db.get_user_by_email, "cache_delete"),
    ):
        returned_before_clear, results = _run_in_private_loop(scenario())

    assert not returned_before_clear
    assert clear.snapshot() == (2, 1)
    assert results[0] is None
    assert isinstance(results[1], asyncio.CancelledError)


@pytest.mark.asyncio
async def test_cache_deletion_order_scope_and_errors_are_preserved():
    events: list[tuple[str, str | None]] = []
    clear_calls = 0

    def delete_id(user_id: str) -> None:
        events.append(("id", user_id))

    def delete_email(email: str) -> None:
        events.append(("email", email))

    def clear() -> None:
        nonlocal clear_calls
        clear_calls += 1
        events.append(("clear", None))
        if clear_calls == 1:
            raise RuntimeError("Redis SCAN failed")

    with (
        patch.object(user_db.get_or_create_user, "cache_clear", clear),
        patch.object(user_db.get_user_by_id, "cache_delete", delete_id),
        patch.object(user_db.get_user_by_email, "cache_delete", delete_email),
    ):
        with pytest.raises(RuntimeError, match="Redis SCAN failed"):
            await user_db._invalidate_user_caches("user-1", "a@example.com")
        await user_db._invalidate_user_caches("user-2", None)

    assert events == [
        ("id", "user-1"),
        ("email", "a@example.com"),
        ("clear", None),
        ("id", "user-2"),
        ("clear", None),
    ]
