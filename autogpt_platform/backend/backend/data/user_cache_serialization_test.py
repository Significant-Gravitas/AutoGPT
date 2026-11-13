"""Concurrency guarantees for the synchronous user-cache clear."""

import asyncio
import contextvars
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import patch

import pytest

from backend.data import user as user_db


class _BlockingClear:
    def __init__(self, *, fail: bool = False) -> None:
        self.release = threading.Event()
        self.started = asyncio.Event()
        self._loop = asyncio.get_running_loop()
        self._fail = fail
        self._lock = threading.Lock()
        self._entered = 0
        self._active = 0
        self._peak_active = 0

    def __call__(self) -> None:
        with self._lock:
            self._entered += 1
            self._active += 1
            self._peak_active = max(self._peak_active, self._active)
        self._loop.call_soon_threadsafe(self.started.set)
        try:
            assert self.release.wait(10), "test did not release the cache clear"
            if self._fail:
                raise RuntimeError("Redis SCAN failed")
        finally:
            with self._lock:
                self._active -= 1

    def snapshot(self) -> tuple[int, int]:
        with self._lock:
            return self._entered, self._peak_active


class _SubmittedClears:
    def __init__(self, expected: int) -> None:
        self.queued = asyncio.Event()
        self.futures: list[Future[None]] = []
        self._expected = expected
        self._submit = user_db._USER_CACHE_CLEAR_EXECUTOR.submit

    def __call__(self, fn: Callable[..., None], *args: object) -> Future[None]:
        future = self._submit(fn, *args)
        self.futures.append(future)
        if len(self.futures) == self._expected:
            self.queued.set()
        return future

    async def drain(self) -> None:
        await asyncio.gather(
            *(asyncio.wrap_future(future) for future in self.futures),
            return_exceptions=True,
        )
        await asyncio.sleep(0)


def test_concurrent_invalidations_are_serial_and_leave_the_default_executor_free():
    async def scenario() -> None:
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=1)
        )
        clear = _BlockingClear()
        submissions = _SubmittedClears(3)
        with (
            patch.object(user_db.get_or_create_user, "cache_clear", clear),
            patch.object(user_db.get_user_by_id, "cache_delete"),
            patch.object(user_db.get_user_by_email, "cache_delete"),
            patch.object(
                user_db._USER_CACHE_CLEAR_EXECUTOR,
                "submit",
                side_effect=submissions,
            ),
        ):
            invalidations = [
                asyncio.create_task(
                    user_db._invalidate_user_caches(f"user-{i}", f"{i}@example.com")
                )
                for i in range(3)
            ]
            try:
                await asyncio.wait_for(submissions.queued.wait(), timeout=5)
                await asyncio.wait_for(clear.started.wait(), timeout=5)
                assert (
                    await asyncio.wait_for(
                        asyncio.to_thread(lambda: "probe"), timeout=5
                    )
                    == "probe"
                )
                assert clear.snapshot() == (1, 1)
                assert all(not task.done() for task in invalidations)
            finally:
                clear.release.set()
                results = await asyncio.gather(*invalidations, return_exceptions=True)
                await submissions.drain()
            assert results == [None, None, None]
            assert clear.snapshot() == (3, 1)

    asyncio.run(scenario())


@pytest.mark.parametrize("queued", [False, True], ids=["running", "queued"])
@pytest.mark.parametrize("worker_fails", [False, True], ids=["success", "worker-error"])
@pytest.mark.asyncio
async def test_cancellation_returns_while_the_submitted_clear_still_runs(
    queued: bool, worker_fails: bool, caplog: pytest.LogCaptureFixture
):
    clear = _BlockingClear(fail=worker_fails)
    count = 2 if queued else 1
    submissions = _SubmittedClears(count)
    error_logged = asyncio.Event()
    loop = asyncio.get_running_loop()
    log_exception = user_db.logger.exception

    def record_error(message: str) -> None:
        log_exception(message)
        loop.call_soon_threadsafe(error_logged.set)

    with (
        patch.object(user_db.get_or_create_user, "cache_clear", clear),
        patch.object(user_db.get_user_by_id, "cache_delete"),
        patch.object(user_db.get_user_by_email, "cache_delete"),
        patch.object(user_db.logger, "exception", side_effect=record_error),
        patch.object(
            user_db._USER_CACHE_CLEAR_EXECUTOR, "submit", side_effect=submissions
        ),
    ):
        tasks = [asyncio.create_task(user_db._invalidate_user_caches("user-1", None))]
        try:
            await asyncio.wait_for(clear.started.wait(), timeout=5)
            if queued:
                tasks.append(
                    asyncio.create_task(user_db._invalidate_user_caches("user-2", None))
                )
            await asyncio.wait_for(submissions.queued.wait(), timeout=5)
            cancelled = tasks[-1]
            cancelled.cancel()
            cancelled.cancel()
            await asyncio.sleep(0)
            cancelled.cancel()
            await asyncio.sleep(0)
            assert cancelled.done(), "cancellation waited for a blocked cache clear"
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            assert not clear.release.is_set()
        finally:
            clear.release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await submissions.drain()
            if worker_fails:
                await asyncio.wait_for(error_logged.wait(), timeout=5)

    assert clear.snapshot() == (count, 1)
    errors = [
        record
        for record in caplog.records
        if record.message == "User cache clear failed after request cancellation"
    ]
    assert len(errors) == int(worker_fails)
    if worker_fails:
        assert errors[0].exc_info is not None
        assert str(errors[0].exc_info[1]) == "Redis SCAN failed"


@pytest.mark.asyncio
async def test_each_clear_preserves_its_request_context():
    request_context = contextvars.ContextVar("cache-clear-request", default="missing")
    observed: list[str] = []

    def clear() -> None:
        observed.append(request_context.get())

    with (
        patch.object(user_db.get_or_create_user, "cache_clear", clear),
        patch.object(user_db.get_user_by_id, "cache_delete"),
        patch.object(user_db.get_user_by_email, "cache_delete"),
    ):
        for value in ("request-one", "request-two"):
            token = request_context.set(value)
            try:
                await user_db._invalidate_user_caches(value, None)
            finally:
                request_context.reset(token)

    assert observed == ["request-one", "request-two"]
    assert request_context.get() == "missing"


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
