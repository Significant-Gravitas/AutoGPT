"""Direct block execution: one block run for the caller, in a process of its own.

The block-execute routes run one block and return its outputs. Each run is a
child process forked from this server's run parent, which the server forks in
start-up before it starts any thread or opens any connection
(``start_run_parent``). The run parent kills a child still running
``DIRECT_BLOCK_EXECUTION_TIMEOUT_SECONDS`` after it started, so a run ends at its
deadline whatever its block is doing, and its memory is capped
(``direct_block_run_parent``).

Each user may have ``DIRECT_BLOCK_EXECUTION_MAX_RUNS_PER_USER`` direct runs in
flight on this server, and the server ``DIRECT_BLOCK_EXECUTION_MAX_RUNS``.

A run reaches the database through the database manager, as the executor does:
its process has no Prisma connection of its own.
"""

import asyncio
import json
import logging
import os
import pickle
import signal
import socket
import sys
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from backend.data.block import BlockInput, CompletedBlockOutput
from backend.executor import direct_block_run_parent
from backend.executor.direct_block_run_parent import (
    DIRECT_BLOCK_EXECUTION_MAX_OUTPUT_SIZE,
    FRAME_HEADER,
    OUTPUTS,
    REQUEST,
    RUN_ORDER,
)
from backend.util.exceptions import BlockExecutionError, BlockUnknownError

if TYPE_CHECKING:
    from backend.blocks._base import AnyBlockSchema

logger = logging.getLogger(__name__)

DIRECT_BLOCK_EXECUTION_TIMEOUT_SECONDS = 300
# How long the server waits for a timed-out run's process to be killed.
DIRECT_BLOCK_EXECUTION_KILL_WAIT_SECONDS = 5
# Documented in docs/platform/rate-limits.md: update it with any change here.
DIRECT_BLOCK_EXECUTION_MAX_RUNS_PER_USER = 4
DIRECT_BLOCK_EXECUTION_MAX_RUNS = 32


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
    """This server can't start a run: its run parent isn't running, or
    ``DIRECT_BLOCK_EXECUTION_MAX_RUNS`` are in flight.

    Not the user's own cap, so the routes answer it with a 503 rather than a 429.
    """

    def __init__(self) -> None:
        Exception.__init__(
            self,
            "Block execution is temporarily unavailable; please try again later.",
        )


_runs_per_user: dict[str, int] = defaultdict(int)
_runs_lock = threading.Lock()


class DirectBlockExecutionSlot:
    """One of a user's direct runs in flight, released when its ``with`` exits."""

    def __init__(self, user_id: str):
        self._user_id = user_id
        self._released = False

    def __enter__(self) -> "DirectBlockExecutionSlot":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

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
        DirectBlockExecutionBusyError: The run parent isn't running, or
            ``DIRECT_BLOCK_EXECUTION_MAX_RUNS`` runs are in flight.
    """
    with _runs_lock:
        if (
            _run_parent is None
            or not _run_parent.running()
            or sum(_runs_per_user.values()) >= DIRECT_BLOCK_EXECUTION_MAX_RUNS
        ):
            raise DirectBlockExecutionBusyError()
        limit = DIRECT_BLOCK_EXECUTION_MAX_RUNS_PER_USER
        if _runs_per_user[user_id] >= limit:
            raise DirectBlockExecutionLimitError(limit)
        _runs_per_user[user_id] += 1
    return DirectBlockExecutionSlot(user_id)


async def run_block_in_child(
    block: "AnyBlockSchema", input_data: BlockInput, **execute_kwargs: Any
) -> CompletedBlockOutput:
    """Run ``block.execute(input_data, **execute_kwargs)`` in a child process
    and collect its outputs.

    Raises:
        DirectBlockExecutionTimeoutError: The block ran past
            ``DIRECT_BLOCK_EXECUTION_TIMEOUT_SECONDS``; its process has been killed.
        DirectBlockExecutionBusyError: The run parent stopped.
        BlockExecutionError: The block failed on a ``ValueError``, ran out of
            memory, or returned too much output.
        BlockUnknownError: The block failed on any other error.
    """
    if _run_parent is None:
        raise DirectBlockExecutionBusyError()
    timeout = DIRECT_BLOCK_EXECUTION_TIMEOUT_SECONDS
    try:
        return await _run_parent.run(block, input_data, execute_kwargs, timeout)
    except TimeoutError as e:
        raise DirectBlockExecutionTimeoutError(timeout) from e


def start_run_parent() -> None:
    """Fork this server's run parent, which forks every direct run.

    Call before the process starts any thread or opens any connection: the run
    parent, and every run, inherit the process as it is at this call.
    """
    global _run_parent
    if sys.platform == "win32":
        logger.warning("Direct block runs need os.fork; this server refuses them")
        return
    direct_block_run_parent.load_blocks()
    ours, theirs = socket.socketpair()
    pid = os.fork()
    if pid == 0:
        ours.close()
        direct_block_run_parent.serve_runs(theirs)
    theirs.close()
    ours.setblocking(False)
    _run_parent = _RunParent(pid, ours)
    logger.info(f"Started the direct block run parent (pid {pid})")


def stop_run_parent() -> None:
    """Stop the run parent, killing any runs still in flight."""
    global _run_parent
    parent, _run_parent = _run_parent, None
    if parent is not None:
        parent.stop()


class _RunParent:
    """The server's end of its run parent."""

    def __init__(self, pid: int, control: socket.socket):
        self.pid = pid
        self._control = control
        self._exited = False

    def running(self) -> bool:
        if not self._exited and _has_exited(self.pid):
            self._exited = True
            logger.error(
                f"The direct block run parent (pid {self.pid}) exited; this "
                "server refuses direct block runs until it restarts"
            )
        return not self._exited

    async def run(
        self,
        block: "AnyBlockSchema",
        input_data: BlockInput,
        execute_kwargs: dict[str, Any],
        timeout: float,
    ) -> CompletedBlockOutput:
        request = pickle.dumps(
            (block.id, input_data, execute_kwargs), pickle.HIGHEST_PROTOCOL
        )
        deadline = time.monotonic() + timeout
        reader, writer = await self._start(timeout)
        try:
            try:
                async with asyncio.timeout(timeout):
                    writer.write(FRAME_HEADER.pack(REQUEST, len(request)) + request)
                    await writer.drain()
                    kind, body = await _read_answer(reader)
            except (ConnectionError, asyncio.IncompleteReadError) as e:
                if time.monotonic() < deadline:
                    raise RuntimeError("The block's process ended unexpectedly") from e
                raise TimeoutError() from e  # Killed at its deadline.
            except TimeoutError:
                await _wait_for_exit(reader)
                raise
        finally:
            writer.close()
        answer = json.loads(body)
        if kind == OUTPUTS:
            return answer
        error = BlockExecutionError if answer["value_error"] else BlockUnknownError
        raise error(message=answer["message"], block_name=block.name, block_id=block.id)

    async def _start(
        self, timeout: float
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Have the run parent fork a child, and connect to it."""
        ours, theirs = socket.socketpair()
        try:
            socket.send_fds(self._control, [RUN_ORDER.pack(timeout)], [theirs.fileno()])
        except OSError as e:
            ours.close()
            raise DirectBlockExecutionBusyError() from e
        finally:
            theirs.close()
        return await asyncio.open_connection(sock=ours)

    def stop(self) -> None:
        self._control.close()
        stop_by = time.monotonic() + DIRECT_BLOCK_EXECUTION_KILL_WAIT_SECONDS
        while not _has_exited(self.pid):
            if time.monotonic() > stop_by:
                os.kill(self.pid, signal.SIGKILL)
                os.waitpid(self.pid, 0)
                return
            time.sleep(0.01)


_run_parent: _RunParent | None = None


async def _read_answer(reader: asyncio.StreamReader) -> tuple[bytes, bytes]:
    kind, size = FRAME_HEADER.unpack(await reader.readexactly(FRAME_HEADER.size))
    # The run parent caps outputs at this; anything longer is not its answer.
    if size > DIRECT_BLOCK_EXECUTION_MAX_OUTPUT_SIZE + 2**16:
        raise RuntimeError(f"The block's process sent a {size}-byte answer")
    return kind, await reader.readexactly(size)


async def _wait_for_exit(reader: asyncio.StreamReader) -> None:
    # The run parent kills the run at its deadline; its socket ends with it.
    try:
        async with asyncio.timeout(DIRECT_BLOCK_EXECUTION_KILL_WAIT_SECONDS):
            while await reader.read(2**16):
                pass
    except (TimeoutError, ConnectionError):
        logger.warning("A direct block run's process outlived its deadline")


def _has_exited(pid: int) -> bool:
    try:
        return bool(os.waitpid(pid, os.WNOHANG)[0])
    except ChildProcessError:
        return True  # Reaped already.
