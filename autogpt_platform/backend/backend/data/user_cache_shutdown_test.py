"""Cache invalidation failures remain visible after a request loop closes."""

import asyncio
import contextvars
import threading
from unittest.mock import patch

import pytest

from backend.data import user as user_db


def test_cancelled_clear_logs_a_worker_failure_after_its_loop_closes(caplog):
    loop = asyncio.new_event_loop()
    started = asyncio.Event()
    release = threading.Event()
    worker_finished = threading.Event()
    error_logged = threading.Event()
    log_exception = user_db.logger.exception
    request_label = contextvars.ContextVar("request_label", default="missing")
    logged_labels: list[str] = []

    def clear() -> None:
        loop.call_soon_threadsafe(started.set)
        try:
            assert release.wait(10), "test did not release the worker"
            raise RuntimeError("late Redis failure after loop close")
        finally:
            worker_finished.set()

    def record_error(message: str) -> None:
        logged_labels.append(request_label.get())
        log_exception(message)
        error_logged.set()

    async def cancel_request() -> None:
        token = request_label.set("request-17")
        request = asyncio.create_task(user_db._invalidate_user_caches("user-1", None))
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            request.cancel()
            done, _ = await asyncio.wait([request], timeout=5)
            assert request in done
            with pytest.raises(asyncio.CancelledError):
                await request
        finally:
            if not request.done():
                release.set()
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)
            request_label.reset(token)

    with (
        patch.object(user_db.get_or_create_user, "cache_clear", clear),
        patch.object(user_db.get_user_by_id, "cache_delete"),
        patch.object(user_db.logger, "exception", side_effect=record_error),
    ):
        try:
            loop.run_until_complete(cancel_request())
            loop.close()
            release.set()
            assert error_logged.wait(5), "late worker failure was not logged"
        finally:
            release.set()
            assert worker_finished.wait(5)
            loop.close()

    [record] = [
        record
        for record in caplog.records
        if record.message == "User cache clear failed after request cancellation"
    ]
    assert record.exc_info is not None
    assert str(record.exc_info[1]) == "late Redis failure after loop close"
    assert logged_labels == ["request-17"]
