"""Hold back a cancel that reaches a copilot turn's cleanup until it ends.

A Stop cancels the turn's task. When the turn hasn't ended within the
worker's grace, the worker cancels it again (``CoPilotProcessor._execute``),
and that second cancel can land in the turn's cleanup. Let through where it
lands, it cuts that step short and skips every step after it (#17286). The
executor's cleanup and a credit-billed turn's charge (``credit_billing``) run
their steps through this.
"""

import asyncio
from collections.abc import Awaitable, Coroutine
from typing import Any, TypeVar

_T = TypeVar("_T")


class DeferredCancel:
    """Runs a turn's cleanup steps to their end, holding back any cancel of
    the turn's task that arrives meanwhile. The caller raises ``held`` once
    the last step is done."""

    def __init__(self) -> None:
        self.held: asyncio.CancelledError | None = None

    async def finish(self, step: Coroutine[Any, Any, _T]) -> _T | None:
        """Await *step* to its end on a task of its own, which a cancel of
        this task can't reach. None if something cancelled that task itself.

        That task runs in a copy of the turn's context, so a step that resets
        a context variable the turn set must use ``run_here`` instead.
        """
        step_task = asyncio.create_task(step)
        while not step_task.done():
            try:
                await asyncio.wait({step_task})
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
