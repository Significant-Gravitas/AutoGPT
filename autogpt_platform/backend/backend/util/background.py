"""Fire-and-forget task spawning that survives garbage collection."""

import asyncio
import logging
from typing import Coroutine

logger = logging.getLogger(__name__)

# asyncio only keeps weak references to running tasks, so a task nobody holds
# can be collected mid-flight. This module-level set is the strong reference;
# the done-callback drops it again.
_background_tasks: set[asyncio.Task] = set()


def _on_done(task: asyncio.Task) -> None:
    _background_tasks.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.warning("Background task %s failed", task.get_name(), exc_info=exc)


def spawn_background_task(coro: Coroutine, *, name: str) -> asyncio.Task:
    """Run ``coro`` detached from the caller, keeping a strong ref to it.

    Detached tasks have no awaiter, so their exceptions would otherwise go
    unobserved; failures are logged rather than lost.
    """
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)
    task.add_done_callback(_on_done)
    return task
