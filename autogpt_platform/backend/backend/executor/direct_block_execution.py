"""Direct block execution, run off the REST server's event loop.

The block-execute routes run one block for the caller and return its outputs.
A block can do a lot of synchronous work, which on the server's event loop
would hold up every other request. Blocks run on a small pool of worker
threads instead, each with its own event loop, and the caller waits at most
``DIRECT_BLOCK_EXECUTION_TIMEOUT_SECONDS`` for the result.

Worker threads do not use the server's Prisma client (it reports itself
disconnected there, see ``backend.data.db``), so block code reaches the
database through the database manager, as it does in the executor.
"""

import asyncio
import contextvars
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import cache

from backend.copilot.constants import MAX_TOOL_WAIT_SECONDS
from backend.data.block import BlockOutput, CompletedBlockOutput

DIRECT_BLOCK_EXECUTION_TIMEOUT_SECONDS = MAX_TOOL_WAIT_SECONDS
DIRECT_BLOCK_EXECUTION_WORKERS = 8

_worker = threading.local()


class DirectBlockExecutionTimeoutError(Exception):
    """The block did not finish within the direct execution time budget."""

    def __init__(self, timeout_seconds: float):
        super().__init__(
            f"Block execution did not finish within {timeout_seconds:g} seconds."
        )


async def run_block_off_loop(outputs: BlockOutput) -> CompletedBlockOutput:
    """Collect a block's ``execute()`` stream on a worker thread.

    Raises:
        DirectBlockExecutionTimeoutError: The block ran past
            ``DIRECT_BLOCK_EXECUTION_TIMEOUT_SECONDS``. It is cancelled at its
            next ``await``.
    """
    timeout = DIRECT_BLOCK_EXECUTION_TIMEOUT_SECONDS
    job = _Job(outputs)
    future = _worker_pool().submit(contextvars.copy_context().run, job.run)
    try:
        return await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout)
    except TimeoutError as e:
        raise DirectBlockExecutionTimeoutError(timeout) from e
    finally:
        if not future.done():
            job.cancel()


class _Job:
    """One block run, cancellable from any thread before or while it runs."""

    def __init__(self, outputs: BlockOutput):
        self._outputs = outputs
        self._lock = threading.Lock()
        self._cancelled = False
        self._running: tuple[asyncio.AbstractEventLoop, asyncio.Task] | None = None

    def run(self) -> CompletedBlockOutput:
        loop: asyncio.AbstractEventLoop = _worker.loop
        with self._lock:
            if self._cancelled:
                raise asyncio.CancelledError()
            task = loop.create_task(_collect(self._outputs))
            self._running = (loop, task)
        return loop.run_until_complete(task)

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            if self._running is not None:
                loop, task = self._running
                loop.call_soon_threadsafe(task.cancel)


async def _collect(outputs: BlockOutput) -> CompletedBlockOutput:
    collected: CompletedBlockOutput = defaultdict(list)
    async for name, data in outputs:
        collected[name].append(data)
    return collected


@cache
def _worker_pool() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        max_workers=DIRECT_BLOCK_EXECUTION_WORKERS,
        thread_name_prefix="direct-block-execution",
        initializer=_start_worker_loop,
    )


def _start_worker_loop() -> None:
    # Each worker keeps one loop for its lifetime, so thread-cached async
    # clients created by a block stay bound to a live loop.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _worker.loop = loop
