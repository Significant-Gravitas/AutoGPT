"""Bounded ownership of rotating credential exchanges across caller cancellation."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

REFRESH_SECONDS = 20.0
REFRESH_CLEANUP_SECONDS = 5.0
REFRESH_DRAIN_SECONDS = REFRESH_SECONDS + REFRESH_CLEANUP_SECONDS
REFRESH_LEASE_SECONDS = 30.0
REFRESH_PENDING = "_autogpt_refresh_pending"
REFRESH_REVISION = "_autogpt_refresh_revision"

logger = logging.getLogger(__name__)
T = TypeVar("T")
_condition = threading.Condition()
_operations: set[asyncio.Task] = set()
_closing = False


class CredentialRefreshRequired(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "The connection could not be refreshed safely. Reconnect this integration "
            "in Settings > Integrations and try again."
        )


class CredentialRefreshStopping(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "The app is shutting down; credential refresh was not started."
        )


def _finished(task: asyncio.Task) -> None:
    if not task.cancelled():
        task.exception()
    with _condition:
        _operations.discard(task)
        _condition.notify_all()


async def admitted_refresh(
    operation: Callable[[float], Awaitable[T]],
    *,
    cleanup: bool = False,
) -> T:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + (0 if cleanup else REFRESH_SECONDS)
    with _condition:
        if _closing and not cleanup:
            raise CredentialRefreshStopping()
        task = loop.create_task(operation(deadline), name="credential-refresh")
        _operations.add(task)
        task.add_done_callback(_finished)
    cancellation: asyncio.CancelledError | None = None
    end = deadline + REFRESH_CLEANUP_SECONDS
    while not task.done():
        remaining = end - loop.time()
        if remaining <= 0:
            task.cancel()
            logger.error("Credential refresh did not settle before its drain deadline")
            if cancellation is not None:
                raise cancellation
            raise CredentialRefreshRequired()
        try:
            await asyncio.wait({task}, timeout=remaining)
        except asyncio.CancelledError as error:
            cancellation = cancellation or error
    if cancellation is not None:
        if not task.cancelled():
            task.exception()
        raise cancellation
    return task.result()


def drain_credential_refreshes(timeout: float = REFRESH_DRAIN_SECONDS) -> bool:
    """Called off the event-loop thread, before services or loops are stopped."""
    global _closing
    deadline = time.monotonic() + max(0.0, timeout)
    with _condition:
        _closing = True
        while _operations:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.error(
                    "Credential refresh shutdown deadline expired; reconnect may be required"
                )
                return False
            _condition.wait(remaining)
    return True
