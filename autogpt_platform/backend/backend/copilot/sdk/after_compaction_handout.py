"""Hand one compaction's re-sent context to its hook callbacks, a part each.

CLI 2.1.284 starts every ``SessionStart`` callback of a matcher at the same
time and joins their ``additionalContext`` in the order the answers come
back, not the order the callbacks were registered.  So the callbacks cannot
count on which of them runs first, and the order the model reads the parts
in is the order the callbacks return.  Measured with a stub API against the
real CLI 2.1.284.
"""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable

from .after_compaction import AFTER_COMPACTION_MAX_PARTS

# How long a part waits for the part before it to answer.  They all wake on
# the same rebuild, so this only runs out when the CLI never called that
# callback; the part then goes out of order rather than not at all.
_ORDER_WAIT_S = 2.0


class AfterCompactionHandout:
    """Gives each of a compaction's callbacks its part of one rebuild.

    A compaction calls every callback once.  Whichever arrives first starts
    the rebuild and the rest share it; a callback that arrives again belongs
    to the next compaction and starts a new one.  Part ``n`` answers only
    after part ``n - 1`` has, which keeps the parts in order for the model.
    """

    def __init__(
        self,
        rebuild: Callable[[], Awaitable[list[str]]],
        *,
        max_parts: int = AFTER_COMPACTION_MAX_PARTS,
        order_wait_s: float = _ORDER_WAIT_S,
    ) -> None:
        self._rebuild = rebuild
        self._max_parts = max_parts
        self._order_wait_s = order_wait_s
        self._round: _Round | None = None

    async def part(self, index: int) -> str:
        """Part ``index`` of this compaction's context; ``""`` when the
        rebuild has fewer parts."""
        current = self._join(index)
        try:
            # Shielded: the CLI cancelling one callback must not cancel the
            # rebuild the others are waiting on.
            parts = await asyncio.shield(current.parts)
            if index > 0:
                await self._wait_for(current.answered[index - 1])
            return parts[index] if index < len(parts) else ""
        finally:
            # Set right before returning: the SDK takes its write lock for
            # this answer without yielding first, so the next part, which
            # wakes after that, queues its own answer behind this one.
            current.answered[index].set()

    def _join(self, index: int) -> "_Round":
        current = self._round
        if current is None or index in current.joined:
            current = _Round(
                asyncio.ensure_future(self._rebuild()),
                [asyncio.Event() for _ in range(self._max_parts)],
            )
            self._round = current
        current.joined.add(index)
        return current

    async def _wait_for(self, answered: asyncio.Event) -> None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(answered.wait(), self._order_wait_s)


class _Round:
    """One compaction's rebuild and which of its parts have gone out."""

    def __init__(
        self, parts: asyncio.Future[list[str]], answered: list[asyncio.Event]
    ) -> None:
        self.parts = parts
        self.answered = answered
        self.joined: set[int] = set()
