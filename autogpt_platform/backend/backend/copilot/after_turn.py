"""Work a turn leaves for after it has fully finished on its executor.

A turn that starts the chat's next turn from inside itself races its own
executor: the queue can hand the new turn to the pod still finishing this one,
which drops it as a duplicate. So the executor registers every turn it runs
here, work deferred while that turn runs is handed back to the executor's
done-callback (after the turn has left ``active_tasks`` and released its
lock), and work outside a registered turn, as in the API process, runs at once.

Process-local, like ``engine_switch``: a turn's end and its done-callback run
in the same executor process.
"""

import threading
from collections.abc import Awaitable, Callable

Work = Callable[[], Awaitable[None]]

_lock = threading.Lock()
_deferred: dict[str, list[Work]] = {}


async def run_after_turn(turn_id: str, work: Work) -> None:
    """Run *work* once turn *turn_id* has fully finished here, or now if it
    does not run in this process."""
    with _lock:
        deferred = _deferred.get(turn_id)
        if deferred is not None:
            deferred.append(work)
            return
    await work()


def turn_started(turn_id: str) -> None:
    with _lock:
        _deferred[turn_id] = []


def turn_finished(turn_id: str) -> list[Work]:
    """The work *turn_id* deferred, for its done-callback to run."""
    with _lock:
        return _deferred.pop(turn_id, [])
