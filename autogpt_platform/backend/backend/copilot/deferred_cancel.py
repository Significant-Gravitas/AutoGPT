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
from contextvars import ContextVar
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Once a cancel is held, how long the rest of the cleanup may take in all
# before what still runs is cancelled and given up. Well above the longest
# bound a step has of its own today: a credit-billed turn's charge waits up
# to 45 s for late cost.
HELD_CANCEL_LIMIT_SECONDS = 120.0

# The DeferredCancel whose step is running, so that one made inside the step
# (a credit-billed turn's charge, inside the turn's cleanup) shares its
# deadline. Set only while a step starts or runs, so nothing stale outlives it.
_enclosing: ContextVar["DeferredCancel | None"] = ContextVar(
    "enclosing_deferred_cancel", default=None
)


class DeferredCancel:
    """Runs a turn's cleanup steps to their end, holding back any cancel of
    the turn's task that arrives meanwhile. The caller raises ``held`` once
    the last step is done.

    Once a cancel is held, the rest of the cleanup gets
    ``HELD_CANCEL_LIMIT_SECONDS`` in all, shared with any DeferredCancel made
    inside one of its steps. Past that, what still runs is cancelled and
    given up, so steps that hang can't hold the turn's task for good.
    """

    def __init__(self) -> None:
        self.held: asyncio.CancelledError | None = None
        self._root: DeferredCancel = _enclosing.get() or self
        self._give_up_at: float | None = None

    async def finish(self, step: Coroutine[Any, Any, _T]) -> _T | None:
        """Await *step* to its end on a task of its own, which a cancel of
        this task can't reach. None if that task was cancelled or given up.

        That task runs in a copy of the turn's context, so a step that resets
        a context variable the turn set must use ``run_here`` instead.
        """
        token = _enclosing.set(self._root)
        try:
            step_task = asyncio.create_task(step)
        finally:
            _enclosing.reset(token)
        while not step_task.done():
            try:
                done, _ = await asyncio.wait({step_task}, timeout=self._time_left())
            except asyncio.CancelledError as cancel:
                self._hold(cancel)
                continue
            if not done:
                _give_up(step_task)
                return None
        if step_task.cancelled():
            self._hold(asyncio.CancelledError())
            return None
        return step_task.result()

    async def run_here(self, step: Awaitable[None]) -> None:
        """Await *step* on this task, as a step that resets the turn's context
        variables must run. A cancel can still cut it short, but not the
        steps after it."""
        deadline = asyncio.timeout_at(self._root._give_up_at)
        token = _enclosing.set(self._root)
        try:
            async with deadline:
                await step
        except asyncio.CancelledError as cancel:
            self._hold(cancel)
        except TimeoutError:
            if not deadline.expired():
                raise
            logger.warning(
                f"Cleanup still running {HELD_CANCEL_LIMIT_SECONDS:g}s after a "
                "cancel was held, giving up a step on the turn's task"
            )
        finally:
            _enclosing.reset(token)

    def _hold(self, cancel: asyncio.CancelledError) -> None:
        if self.held is None:
            self.held = cancel
        if self._root._give_up_at is None:
            self._root._give_up_at = (
                asyncio.get_running_loop().time() + HELD_CANCEL_LIMIT_SECONDS
            )

    def _time_left(self) -> float | None:
        give_up_at = self._root._give_up_at
        if give_up_at is None:
            return None
        return max(0.0, give_up_at - asyncio.get_running_loop().time())


def _give_up(step_task: asyncio.Task[Any]) -> None:
    logger.warning(
        f"Cleanup still running {HELD_CANCEL_LIMIT_SECONDS:g}s after a cancel "
        f"was held, cancelling a step: {step_task!r}"
    )
    step_task.cancel()
    # Nothing awaits it now: its outcome, when it comes, isn't an error left
    # unretrieved.
    step_task.add_done_callback(_retrieve_outcome)


def _retrieve_outcome(step_task: asyncio.Task[Any]) -> None:
    if not step_task.cancelled():
        step_task.exception()
