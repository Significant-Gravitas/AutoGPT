"""How long the schedule registry waits on any one call to a dependency.

The registry's calls are best-effort, but its callers are not: a hire, an
archive, a resume route or a cron gate must not hang on them. The
DatabaseManager client retries for minutes and the scheduler client waits up
to 300 s per request, so every registry call runs under this one deadline.
On timeout the call is cancelled and ``TimeoutError`` raised, which each
caller's fail-soft handler treats like any other failure.
"""

import asyncio
from typing import Awaitable, TypeVar

REGISTRY_CALL_TIMEOUT_SECONDS = 10.0

# The scheduler's sync cron bodies reach the registry through ``run_async``;
# its own bound sits just past the registry's so the registry fails soft
# first and the bridge is only a backstop.
BRIDGE_MARGIN_SECONDS = 5.0

T = TypeVar("T")


async def within_deadline(call: Awaitable[T]) -> T:
    """``call`` under the registry deadline; cancelled on timeout.

    ``asyncio.timeout`` rather than ``asyncio.wait_for``: these deadlines
    nest (an API hook bounds a whole change whose calls are bounded too),
    and on Python 3.11 an outer deadline that expires just as an inner
    ``wait_for`` completes has its cancellation swallowed, so the change
    would run on past it.
    """
    async with asyncio.timeout(REGISTRY_CALL_TIMEOUT_SECONDS):
        return await call


def bridge_timeout() -> float:
    """The ``run_async`` bound for a registry call made from a cron body."""
    return REGISTRY_CALL_TIMEOUT_SECONDS + BRIDGE_MARGIN_SECONDS
