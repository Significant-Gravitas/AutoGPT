"""Direct block execution, run off the REST server's event loop.

The block-execute routes run one block for the caller and return its outputs.
Each run is a task on one of a small pool of worker threads, each with its own
event loop, so the server's loop only serves requests. A worker loop holds as
many runs as are waiting on I/O at once.

Each user may have ``DIRECT_BLOCK_EXECUTION_MAX_RUNS_PER_USER`` direct runs in
flight in this process, and a run has ``DIRECT_BLOCK_EXECUTION_TIMEOUT_SECONDS``
from the moment it starts on its worker loop. A run that is still running
``DIRECT_BLOCK_EXECUTION_CANCEL_GRACE_SECONDS`` after its budget ran out (stuck
in synchronous code, or ignoring cancellation) has its worker retired: a fresh
worker takes its place in the pool and its waiting runs, and the old thread
ends once its runs stop. While ``DIRECT_BLOCK_EXECUTION_MAX_RETIRED_WORKERS``
retired workers are still running, new runs are refused, since only a restart
may free them.

Worker threads do not use the server's Prisma client (it reports itself
disconnected there, see ``backend.data.db``), so block code reaches the
database through the database manager, as it does in the executor.
"""

import asyncio
import concurrent.futures
import contextvars
import itertools
import logging
import threading
from collections import defaultdict
from collections.abc import Callable

from backend.data.block import BlockOutput, CompletedBlockOutput

logger = logging.getLogger(__name__)

DIRECT_BLOCK_EXECUTION_TIMEOUT_SECONDS = 300
DIRECT_BLOCK_EXECUTION_CANCEL_GRACE_SECONDS = 5
DIRECT_BLOCK_EXECUTION_WORKERS = 8
DIRECT_BLOCK_EXECUTION_MAX_RUNS_PER_USER = 4
# A held run keeps its user's slot, so one user's runs fill at most
# DIRECT_BLOCK_EXECUTION_MAX_RUNS_PER_USER of these.
DIRECT_BLOCK_EXECUTION_MAX_RETIRED_WORKERS = 8


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


class DirectBlockExecutionBusyError(DirectBlockExecutionLimitError):
    """Too many retired workers are still running in this process.

    Unlike a user's own cap, it can last until the process restarts, so the
    routes answer it with a 503 rather than a 429.
    """

    def __init__(self) -> None:
        Exception.__init__(
            self,
            "Block execution is temporarily unavailable; please try again later.",
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
        DirectBlockExecutionBusyError: ``DIRECT_BLOCK_EXECUTION_MAX_RETIRED_WORKERS``
            retired workers are still running.
    """
    with _workers_lock:
        if len(_retired_workers) >= DIRECT_BLOCK_EXECUTION_MAX_RETIRED_WORKERS:
            raise DirectBlockExecutionBusyError()
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
            next ``await``, and its worker is retired if it has not stopped
            ``DIRECT_BLOCK_EXECUTION_CANCEL_GRACE_SECONDS`` later.
    """
    timeout = DIRECT_BLOCK_EXECUTION_TIMEOUT_SECONDS
    server_loop = asyncio.get_running_loop()
    # The worker is picked and the run submitted under two acquisitions of
    # _workers_lock. That is safe only because nothing yields in between:
    # retirement runs on this loop, so it cannot retire the worker meanwhile.
    worker = _least_busy_worker()
    finished = False
    stopped = threading.Event()
    budget = asyncio.timeout(timeout)

    try:
        async with budget:

            def start_budget() -> None:
                if not finished and not budget.expired():
                    budget.reschedule(server_loop.time() + timeout)

            release = slot.hand_off() if slot else _nothing

            def on_finish() -> None:
                stopped.set()
                release()

            try:
                future = worker.submit(
                    outputs,
                    contextvars.copy_context(),
                    on_start=lambda: server_loop.call_soon_threadsafe(start_budget),
                    on_finish=on_finish,
                )
            except BaseException:
                release()  # The run never reached the worker.
                raise
            try:
                return await asyncio.wrap_future(future)
            finally:
                finished = True
    except TimeoutError as e:
        if not budget.expired():
            raise  # The block raised it, within its budget.
        server_loop.call_later(
            DIRECT_BLOCK_EXECUTION_CANCEL_GRACE_SECONDS,
            _retire_unless_stopped,
            worker,
            stopped,
        )
        raise DirectBlockExecutionTimeoutError(timeout) from e


class _Run:
    """A block run handed to a worker, and the future its caller waits on."""

    def __init__(
        self,
        outputs: BlockOutput,
        context: contextvars.Context,
        on_start: Callable[[], object],
        on_finish: Callable[[], object],
    ):
        self.outputs = outputs
        self.context = context
        self.on_start = on_start
        self.on_finish = on_finish
        self.future: concurrent.futures.Future[CompletedBlockOutput] = (
            concurrent.futures.Future()
        )


class _WorkerLoop:
    """A thread running one event loop until the worker is retired.

    Thread-cached async clients created by a block stay bound to a live loop.
    """

    def __init__(self, index: int):
        self.loop = asyncio.new_event_loop()
        self.active = 0
        self.retired = False
        # Insertion-ordered, so runs moved to a replacement keep their order.
        self._waiting: dict[_Run, None] = {}
        self._thread = threading.Thread(
            target=self._serve,
            name=f"direct-block-execution-{index}",
            daemon=True,
        )
        self._thread.start()

    @property
    def name(self) -> str:
        return self._thread.name

    def _serve(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        # Only a retired worker stops, once its last run has stopped.
        with _workers_lock:
            _retired_workers.discard(self)
        self.loop.close()

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
        run = _Run(outputs, context, on_start, on_finish)
        self.queue(run)
        return run.future

    def queue(self, run: _Run) -> None:
        with _workers_lock:
            self.active += 1
            self._waiting[run] = None
        try:
            self.loop.call_soon_threadsafe(self._start, run)
        except BaseException:
            with _workers_lock:
                if run in self._waiting:
                    del self._waiting[run]
                    self.active -= 1
            raise

    def take_waiting_runs(self) -> list[_Run]:
        """Retire this worker and hand back the runs it has not started.

        The caller holds ``_workers_lock``.
        """
        self.retired = True
        runs = list(self._waiting)
        self._waiting.clear()
        self.active -= len(runs)
        if self.active == 0:
            self.loop.call_soon_threadsafe(self.loop.stop)
        return runs

    def _start(self, run: _Run) -> None:
        with _workers_lock:
            if run not in self._waiting:
                return  # Moved to the worker that replaced this one.
            self._waiting.pop(run, None)
        if run.future.cancelled():
            self._finish(run, None)
            return
        task = self.loop.create_task(_collect(run.outputs), context=run.context)
        task.add_done_callback(lambda task: self._finish(run, task))

        def cancel_task(_: concurrent.futures.Future) -> None:
            if run.future.cancelled():
                self.loop.call_soon_threadsafe(task.cancel)

        run.future.add_done_callback(cancel_task)
        # The task's first step runs after this callback returns.
        run.on_start()

    def _finish(
        self, run: _Run, task: asyncio.Task[CompletedBlockOutput] | None
    ) -> None:
        with _workers_lock:
            self.active -= 1
            drained = self.retired and self.active == 0
        run.on_finish()
        if task is not None:
            _settle(run.future, task)
        if drained:
            self.loop.stop()


_pool: list[_WorkerLoop] = []
_retired_workers: set[_WorkerLoop] = set()
_worker_numbers = itertools.count()


def _least_busy_worker() -> _WorkerLoop:
    with _workers_lock:
        if not _pool:
            _pool.extend(
                _WorkerLoop(next(_worker_numbers))
                for _ in range(DIRECT_BLOCK_EXECUTION_WORKERS)
            )
        return min(_pool, key=lambda worker: worker.active)


def _retire_unless_stopped(worker: _WorkerLoop, stopped: threading.Event) -> None:
    """Replace *worker* in the pool if its timed-out run is still running."""
    if stopped.is_set():
        return
    replacement: _WorkerLoop | None = None
    moved: list[_Run] = []
    with _workers_lock:
        if worker.retired or worker not in _pool:
            return
        retired = len(_retired_workers)
        if retired < DIRECT_BLOCK_EXECUTION_MAX_RETIRED_WORKERS:
            replacement = _WorkerLoop(next(_worker_numbers))
            _pool[_pool.index(worker)] = replacement
            _retired_workers.add(worker)
            moved = worker.take_waiting_runs()
    if replacement is None:
        logger.warning(
            "Direct block execution worker %s is still running a run past its "
            "time budget, and %s retired workers are still running; keeping it",
            worker.name,
            retired,
        )
        return
    logger.warning(
        "Direct block execution worker %s is still running a run %ss past its "
        "time budget; retired it and started %s",
        worker.name,
        DIRECT_BLOCK_EXECUTION_CANCEL_GRACE_SECONDS,
        replacement.name,
    )
    for run in moved:
        replacement.queue(run)


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
