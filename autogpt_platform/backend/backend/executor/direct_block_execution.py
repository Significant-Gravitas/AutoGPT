"""Direct block execution, run off the REST server's event loop.

The block-execute routes run one block for the caller and return its outputs.
Each run is a task on one of a small pool of worker threads, each with its own
event loop, so the server's loop only serves requests. A worker loop holds as
many runs as are waiting on I/O at once.

Each user may have ``DIRECT_BLOCK_EXECUTION_MAX_RUNS_PER_USER`` direct runs in
flight in this process, and a run has ``DIRECT_BLOCK_EXECUTION_TIMEOUT_SECONDS``
from the moment it starts on its worker loop.

Worker threads do not use the server's Prisma client (it reports itself
disconnected there, see ``backend.data.db``), so block code reaches the
database through the database manager, as it does in the executor.
"""

import asyncio
import concurrent.futures
import contextvars
import threading
from collections import defaultdict
from collections.abc import Callable
from functools import cache

from backend.data.block import BlockOutput, CompletedBlockOutput

DIRECT_BLOCK_EXECUTION_TIMEOUT_SECONDS = 300
DIRECT_BLOCK_EXECUTION_WORKERS = 8
DIRECT_BLOCK_EXECUTION_MAX_RUNS_PER_USER = 4


class DirectBlockExecutionTimeoutError(Exception):
    """The block did not finish within the direct execution time budget."""

    def __init__(self, timeout_seconds: float):
        super().__init__(
            f"Block execution did not finish within {timeout_seconds:g} seconds."
        )


class DirectBlockExecutionLimitError(Exception):
    """The user already has the most direct block runs allowed in flight."""

    def __init__(self, limit: int):
        super().__init__(
            f"You already have {limit} blocks running. "
            "Wait for one to finish, then try again."
        )


_runs_per_user: dict[str, int] = defaultdict(int)
_runs_lock = threading.Lock()
_workers_lock = threading.Lock()


class DirectBlockExecutionSlot:
    """One of a user's in-flight direct runs.

    Released when the ``with`` block exits, or, once handed to
    ``run_block_off_loop``, when the block itself stops running.
    """

    def __init__(self, user_id: str):
        self._user_id = user_id
        self._released = False
        self._handed_off = False

    def __enter__(self) -> "DirectBlockExecutionSlot":
        return self

    def __exit__(self, *exc_info: object) -> None:
        if not self._handed_off:
            self.release()

    def hand_off(self) -> Callable[[], None]:
        """Keep the slot past the ``with`` block; the caller releases it."""
        self._handed_off = True
        return self.release

    def release(self) -> None:
        with _runs_lock:
            if self._released:
                return
            self._released = True
            _runs_per_user[self._user_id] -= 1
            if _runs_per_user[self._user_id] <= 0:
                del _runs_per_user[self._user_id]


def claim_direct_block_execution_slot(user_id: str) -> DirectBlockExecutionSlot:
    """Claim one of *user_id*'s in-flight direct runs.

    Raises:
        DirectBlockExecutionLimitError: The user already has
            ``DIRECT_BLOCK_EXECUTION_MAX_RUNS_PER_USER`` runs in flight.
    """
    limit = DIRECT_BLOCK_EXECUTION_MAX_RUNS_PER_USER
    with _runs_lock:
        if _runs_per_user[user_id] >= limit:
            raise DirectBlockExecutionLimitError(limit)
        _runs_per_user[user_id] += 1
    return DirectBlockExecutionSlot(user_id)


async def run_block_off_loop(
    outputs: BlockOutput, slot: DirectBlockExecutionSlot | None = None
) -> CompletedBlockOutput:
    """Collect a block's ``execute()`` stream on a worker loop.

    The time budget starts when the run starts on its worker loop. A run that
    has not started within one budget is given up too.

    Raises:
        DirectBlockExecutionTimeoutError: The block ran past
            ``DIRECT_BLOCK_EXECUTION_TIMEOUT_SECONDS``. It is cancelled at its
            next ``await``.
    """
    timeout = DIRECT_BLOCK_EXECUTION_TIMEOUT_SECONDS
    server_loop = asyncio.get_running_loop()
    finished = False

    try:
        async with asyncio.timeout(timeout) as budget:

            def start_budget() -> None:
                if not finished and not budget.expired():
                    budget.reschedule(server_loop.time() + timeout)

            future = _least_busy_worker().submit(
                outputs,
                contextvars.copy_context(),
                on_start=lambda: server_loop.call_soon_threadsafe(start_budget),
                on_finish=slot.hand_off() if slot else _nothing,
            )
            try:
                return await asyncio.wrap_future(future)
            finally:
                finished = True
    except TimeoutError as e:
        raise DirectBlockExecutionTimeoutError(timeout) from e


class _WorkerLoop:
    """A thread running one event loop for its lifetime.

    Thread-cached async clients created by a block stay bound to a live loop.
    """

    def __init__(self, index: int):
        self.loop = asyncio.new_event_loop()
        self.active = 0
        self._thread = threading.Thread(
            target=self._serve,
            name=f"direct-block-execution-{index}",
            daemon=True,
        )
        self._thread.start()

    def _serve(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(
        self,
        outputs: BlockOutput,
        context: contextvars.Context,
        on_start: Callable[[], object],
        on_finish: Callable[[], object],
    ) -> concurrent.futures.Future[CompletedBlockOutput]:
        """Run *outputs* as a task on this loop, in *context*.

        Cancelling the returned future cancels the task. *on_finish* runs when
        the task has stopped, which can be after the future is cancelled.
        """
        future: concurrent.futures.Future[CompletedBlockOutput] = (
            concurrent.futures.Future()
        )
        with _workers_lock:
            self.active += 1

        def finish(task: asyncio.Task[CompletedBlockOutput] | None) -> None:
            with _workers_lock:
                self.active -= 1
            on_finish()
            if task is not None:
                _settle(future, task)

        def start() -> None:
            if future.cancelled():
                finish(None)
                return
            task = self.loop.create_task(_collect(outputs), context=context)
            task.add_done_callback(finish)

            def cancel_task(_: concurrent.futures.Future) -> None:
                if future.cancelled():
                    self.loop.call_soon_threadsafe(task.cancel)

            future.add_done_callback(cancel_task)
            # The task's first step runs after this callback returns.
            on_start()

        self.loop.call_soon_threadsafe(start)
        return future


@cache
def _worker_loops() -> tuple[_WorkerLoop, ...]:
    return tuple(_WorkerLoop(i) for i in range(DIRECT_BLOCK_EXECUTION_WORKERS))


def _least_busy_worker() -> _WorkerLoop:
    workers = _worker_loops()
    with _workers_lock:
        return min(workers, key=lambda worker: worker.active)


def _settle(
    future: concurrent.futures.Future[CompletedBlockOutput],
    task: asyncio.Task[CompletedBlockOutput],
) -> None:
    try:
        if task.cancelled():
            future.cancel()
        elif (error := task.exception()) is not None:
            future.set_exception(error)
        else:
            future.set_result(task.result())
    except concurrent.futures.InvalidStateError:
        pass  # The caller cancelled the future first.


def _nothing() -> None:
    pass


async def _collect(outputs: BlockOutput) -> CompletedBlockOutput:
    collected: CompletedBlockOutput = defaultdict(list)
    async for name, data in outputs:
        collected[name].append(data)
    return collected
