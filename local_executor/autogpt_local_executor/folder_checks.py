"""Looks at a folder that may never return.

A folder on a network share that stops answering (a laptop that left its
network, a server that went away) holds a stat, a realpath or a listing for
as long as the share is gone. The control connection looks at a chat's
folder to verify or issue its grant and to list it for the folder picker,
and it waited for each look without a limit, holding the chat's lock, a
request slot and, for a listing, the folder picker's lock. A few such
requests took every slot, and every chat on the computer was refused until
the shim was restarted.

Here a look runs in a thread of its own and is waited for until a deadline:

- Past the deadline the caller gets FolderCheckTimeout, and the look goes on
  in its thread. Whatever it returns later is dropped.
- While a look nobody waits for any more is still running, another look at
  the same folder fails at once, and so does a request that was waiting its
  turn behind that look when it was given up on: a dead share holds one
  thread for each folder asked about, however many requests come in. Other
  folders are looked at as usual.
- At most MAX_STUCK_FOLDER_CHECKS looks run at once, whatever their
  folders, those still waited for included: counting only the ones given up
  on let requests that came in together each start one, and leave more
  running than the limit when none of them answered. With every one of
  them given up on, a look is refused at once until one returns, a request
  that was waiting for a thread included; while one is still waited for, a
  new look waits for a thread, inside its own time, and goes on as soon as
  any of them returns.
- The threads are daemon threads, not the event loop's worker threads:
  asyncio.run() waits for its workers before it returns, and the
  interpreter waits for them again as it exits, so one stuck in a worker
  kept the shim from stopping, and its supervisor from restarting it.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

_T = TypeVar("_T")

# How long a control request waits for its looks at a folder, in all: half
# of the ten seconds the platform waits for a control request's answer.
FOLDER_CHECK_SECONDS = 5.0
# The most looks running at once, each in a thread, and so the most left
# running once nobody waits for them.
MAX_STUCK_FOLDER_CHECKS = 8


class FolderCheckTimeout(TimeoutError):
    """A look at a folder didn't return in time, or an earlier look at it
    still hasn't. Its text is for the platform."""


@dataclass
class _Look:
    done: concurrent.futures.Future[Any]
    # Done once the look is given up on (abandoned), for the requests
    # waiting behind it, which are answered then and not at their deadlines.
    given_up: asyncio.Future[None]
    # Nobody waits for it any more: its caller's time ran out, or its caller
    # was cancelled.
    abandoned: bool = False


class FolderChecks:
    """The machine's looks at folders that may not answer, shared by its
    control connection and its chats."""

    def __init__(self, *, max_stuck: int = MAX_STUCK_FOLDER_CHECKS) -> None:
        self._max_stuck = max_stuck
        # The look at each folder that is still running, by the folder's path.
        self._looks: dict[str, _Look] = {}

    def stuck(self) -> int:
        """How many looks are still running with nobody waiting for them."""
        return sum(look.abandoned and not look.done.done() for look in self._looks.values())

    async def run(
        self,
        folder: str | os.PathLike[str],
        look: Callable[[], _T],
        *,
        deadline: float,
        on_abandoned: Callable[[], None] | None = None,
    ) -> _T:
        """What `look()` returns or raises, run in a thread of its own, by
        `deadline` (loop time), else FolderCheckTimeout.

        `folder` is what `look` looks at. While an earlier look at it is
        still inside its own caller's wait, this one waits its turn, no
        later than `deadline`; once that look has been given up on, this
        one fails at once, whether it comes then or was already waiting.
        It waits the same way for a thread, while the most that may run
        are running and one of them is still waited for.
        `on_abandoned` is called when this call leaves its own look
        running, by its deadline or by being cancelled."""
        loop = asyncio.get_running_loop()
        # The path as it is given, which is how its callers name a folder
        # each time (a grant's root, the picker's resolved path). Not with
        # its case folded: Windows has folders that differ only in case,
        # where a folder is marked case-sensitive, and one of them not
        # answering would have refused the other without a look.
        key = os.fspath(folder)
        while True:
            for done in [key for key, earlier in self._looks.items() if earlier.done.done()]:
                del self._looks[done]
            # What is left is running, each in its thread.
            earlier = self._looks.get(key)
            if earlier is not None:
                if earlier.abandoned:
                    raise FolderCheckTimeout(
                        "The folder still hasn't answered an earlier request; try again later."
                    )
                # Another request's look at this folder, with its caller
                # still waiting: it ends, or is given up on, within that
                # caller's time.
                wait_for = [earlier]
            elif len(self._looks) >= self._max_stuck:
                # Every thread is taken. One whose caller still waits may
                # well return; with none of those, nothing frees one soon.
                if all(look.abandoned for look in self._looks.values()):
                    raise FolderCheckTimeout(
                        "Too many folders on this computer aren't answering; try again later."
                    )
                # Whichever returns frees its thread, one that was given up
                # on as well: watching only the looks still waited for left
                # this request waiting for them, to its deadline, with a
                # thread free.
                wait_for = list(self._looks.values())
            else:
                break
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise FolderCheckTimeout("The folder didn't answer in time; try again later.")
            # Until one of them returns, or one still waited for is given
            # up on: what this request is told then (a look given up on at
            # its folder, every thread held by one) it is told at once.
            await asyncio.wait(
                {asyncio.wrap_future(look.done) for look in wait_for}
                | {look.given_up for look in wait_for if not look.abandoned},
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise FolderCheckTimeout("The folder didn't answer in time; try again later.")

        mine = _Look(concurrent.futures.Future(), loop.create_future())
        # Running from here on, so that a caller that stops waiting can't
        # cancel it under another's feet.
        mine.done.set_running_or_notify_cancel()

        def work() -> None:
            try:
                result = look()
            except BaseException as exc:
                mine.done.set_exception(exc)
            else:
                mine.done.set_result(result)

        try:
            threading.Thread(target=work, name="autogpt-shim-folder-check", daemon=True).start()
        except RuntimeError as exc:
            raise FolderCheckTimeout("The folder couldn't be looked at; try again later.") from exc
        self._looks[key] = mine
        waiting = asyncio.wrap_future(mine.done)
        waiting.add_done_callback(_retrieved)
        try:
            async with asyncio.timeout(remaining):
                return await asyncio.shield(waiting)
        except TimeoutError:
            if mine.done.done():
                # It returned as the time ran out, or the TimeoutError is
                # the look's own: its answer either way.
                return mine.done.result()
            self._abandon(mine, on_abandoned)
            raise FolderCheckTimeout("The folder didn't answer in time; try again later.") from None
        except asyncio.CancelledError:
            if not mine.done.done():
                self._abandon(mine, on_abandoned)
            raise

    @staticmethod
    def _abandon(look: _Look, on_abandoned: Callable[[], None] | None) -> None:
        look.abandoned = True
        if not look.given_up.done():
            look.given_up.set_result(None)
        if on_abandoned is not None:
            on_abandoned()


def _retrieved(done: asyncio.Future[Any]) -> None:
    """Take a look's failure, so one that fails after it was given up on
    isn't reported as never retrieved."""
    if not done.cancelled():
        done.exception()
