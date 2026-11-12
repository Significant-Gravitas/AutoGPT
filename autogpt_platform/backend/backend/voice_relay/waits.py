"""The relay's waits with a deadline, and its way of stopping tasks.

On Python 3.11, ``asyncio.wait_for`` runs a coroutine in a task of its own,
and when the task waiting on it is cancelled as that operation finishes, it
returns the operation's result and the cancellation is lost: the task
carries on. Libraries the relay awaits use ``wait_for`` too, Redis's client
among them.

- Every bounded wait in the relay goes through :func:`bounded`, which awaits
  its operation in the task that waits, inside ``asyncio.timeout``: how
  ``wait_for`` works from Python 3.12. A cancellation then always reaches
  the task it was meant for.
- Every task the relay cancels and waits for, it stops with :func:`stop`,
  which cancels again a task that lost its cancellation inside a library. A
  call's end, which waits for every task of the call's, then always comes.
"""

import asyncio
from collections.abc import Awaitable, Iterable
from typing import TypeVar

T = TypeVar("T")

# A task still running this long after it was cancelled has lost the
# cancellation, and is cancelled again.
STOP_AGAIN_S = 1.0


async def bounded(operation: Awaitable[T], *, timeout: float | None) -> T:
    """*operation*'s result, or ``TimeoutError`` once *timeout* seconds have
    passed; with *timeout* None, it waits as long as the operation takes."""
    async with asyncio.timeout(timeout):
        return await operation


async def stop(
    tasks: Iterable[asyncio.Task], *, again_after_s: float = STOP_AGAIN_S
) -> None:
    """Cancel *tasks* and wait until each has ended, cancelling again one
    still running *again_after_s* after its last cancellation, which must
    outlast the cleanup a task does as it stops. What a task ended with is
    read here, so none is reported as never retrieved. The task that calls
    this is never one it stops."""
    current = asyncio.current_task()
    stopping = [task for task in tasks if task is not current]
    pending = {task for task in stopping if not task.done()}
    while pending:
        for task in pending:
            task.cancel()
        _, pending = await asyncio.wait(pending, timeout=again_after_s)
    await asyncio.gather(*stopping, return_exceptions=True)
