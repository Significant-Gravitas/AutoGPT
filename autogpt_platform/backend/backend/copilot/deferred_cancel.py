"""Hold back a cancel that reaches a copilot turn's cleanup until it ends.

A Stop cancels the turn's task. When the turn hasn't ended within the
worker's grace, the worker cancels it again (``CoPilotProcessor._execute``),
and that second cancel can land in the turn's cleanup. Let through where it
lands, it cuts that step short and skips every step after it (#17286). The
executor's cleanup and a credit-billed turn's charge (``credit_billing``) run
their steps through this.
"""

import asyncio
import logging
from collections.abc import Awaitable, Coroutine
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Once a cancel is held, how much longer a step may run before it is
# cancelled and given up. Well above the longest bound a step has of its own
# today: a credit-billed turn's charge waits up to 45 s for late cost.
HELD_CANCEL_STEP_LIMIT_SECONDS = 120.0


class DeferredCancel:
    """Runs a turn's cleanup steps to their end, holding back any cancel of
    the turn's task that arrives meanwhile. The caller raises ``held`` once
    the last step is done."""

    def __init__(self) -> None:
        self.held: asyncio.CancelledError | None = None

    async def finish(self, step: Coroutine[Any, Any, _T]) -> _T | None:
        """Await *step* to its end on a task of its own, which a cancel of
        this task can't reach. None if that task was cancelled.

        Once a cancel is held, *step* gets ``HELD_CANCEL_STEP_LIMIT_SECONDS``
        more; one still running then is cancelled and given up, so a step
        that hangs can't hold the turn's task for good.

        That task runs in a copy of the turn's context, so a step that resets
        a context variable the turn set must use ``run_here`` instead.
        """
        step_task = asyncio.create_task(step)
        loop = asyncio.get_running_loop()
        give_up_at: float | None = None
        while not step_task.done():
            if self.held is not None and give_up_at is None:
                give_up_at = loop.time() + HELD_CANCEL_STEP_LIMIT_SECONDS
            timeout = None if give_up_at is None else give_up_at - loop.time()
            if timeout is not None and timeout <= 0:
                _give_up(step_task)
                return None
            try:
                await asyncio.wait({step_task}, timeout=timeout)
            except asyncio.CancelledError as cancel:
                self._hold(cancel)
        if step_task.cancelled():
            self._hold(asyncio.CancelledError())
            return None
        return step_task.result()

    async def run_here(self, step: Awaitable[None]) -> None:
        """Await *step* on this task, as a step that resets the turn's context
        variables must run. A cancel can still cut it short, but not the
        steps after it."""
        try:
            await step
        except asyncio.CancelledError as cancel:
            self._hold(cancel)

    def _hold(self, cancel: asyncio.CancelledError) -> None:
        if self.held is None:
            self.held = cancel


def _give_up(step_task: asyncio.Task[Any]) -> None:
    logger.warning(
        f"Cleanup step still running {HELD_CANCEL_STEP_LIMIT_SECONDS:g}s after a "
        f"cancel was held, cancelling it: {step_task!r}"
    )
    step_task.cancel()
    # Nothing awaits it now: its outcome, when it comes, isn't an error left
    # unretrieved.
    step_task.add_done_callback(_retrieve_outcome)


def _retrieve_outcome(step_task: asyncio.Task[Any]) -> None:
    if not step_task.cancelled():
        step_task.exception()
