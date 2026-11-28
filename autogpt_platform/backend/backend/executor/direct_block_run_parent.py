"""The run parent: the process every direct block run is forked from.

The server forks it early in start-up (``direct_block_execution.start_run_parent``),
before it starts a thread or opens a connection, so each fork here copies a
process with one thread and the block registry already loaded. It forks one
child per run, cancels a child still running at its deadline, and kills it if
it outlasts the cancel. Nothing in this module runs in the server's own process.
"""

import asyncio
import gc
import json
import logging
import math
import os
import pickle
import selectors
import signal
import socket
import struct
import sys
import time
from collections import defaultdict
from typing import NoReturn

import pydantic
import sentry_sdk

from backend.blocks import get_blocks
from backend.data.block import BlockOutput, CompletedBlockOutput
from backend.data.llm_registry import load_catalog
from backend.integrations.managed_providers import register_all
from backend.sdk.registry import AutoRegistry

if sys.platform != "win32":
    import resource

logger = logging.getLogger(__name__)

# Bytes a run may allocate beyond what its process held when it was forked.
DIRECT_BLOCK_EXECUTION_MEMORY_LIMIT = 1024 * 2**20
DIRECT_BLOCK_EXECUTION_MAX_OUTPUT_SIZE = 32 * 2**20

# A run is cancelled at its deadline, so the block's own clean-up runs, and killed
# this long after.
_CANCEL_GRACE_SECONDS = 2
# The kernel ends a child this long after its deadline if the run parent is gone.
_ALARM_GRACE_SECONDS = 5

# A frame is a kind and a length, then that many bytes.
FRAME_HEADER = struct.Struct("!cQ")
REQUEST, OUTPUTS, FAILED = b"R", b"O", b"E"
# What the server sends the run parent, with the run's socket attached: its budget.
RUN_ORDER = struct.Struct("!d")

_OUTPUTS = pydantic.TypeAdapter(CompletedBlockOutput)

# Each live run: when the run parent next signals it, and with which signal.
_Runs = dict[int, tuple[float, signal.Signals]]


def load_blocks() -> None:
    """Load the registry and freeze the heap, so forks share it unchanged."""
    get_blocks()
    gc.collect()
    gc.freeze()


# Runs in a forked process, which coverage doesn't measure.
def serve_runs(control: socket.socket) -> NoReturn:  # pragma: no cover
    """The run parent's loop. It ends, killing its runs, once the server closes
    its end of *control*."""
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        _prepare_registry()
        runs: _Runs = {}
        with selectors.DefaultSelector() as selector:
            selector.register(control, selectors.EVENT_READ)
            while True:
                _reap(runs, os.WNOHANG)
                _end_overdue(runs)
                if selector.select(_next_wake(runs)) and not _start_run(control, runs):
                    break
        for pid in runs:
            os.kill(pid, signal.SIGKILL)
        _reap(runs, 0)
    except BaseException:
        logger.exception("The direct block run parent failed")
        os._exit(1)
    os._exit(0)


def _prepare_registry() -> None:
    # The set-up the server's lifespan does, done once here for every child.
    AutoRegistry.patch_integrations()
    register_all()
    load_catalog()
    load_blocks()


def _reap(runs: _Runs, options: int) -> None:
    for pid in list(runs):
        if os.waitpid(pid, options)[0]:
            del runs[pid]


def _end_overdue(runs: _Runs) -> None:
    # Only unreaped children are listed, so no pid here can have been reused.
    now = time.monotonic()
    for pid, (due, signum) in runs.items():
        if due > now:
            continue
        os.kill(pid, signum)
        if signum == signal.SIGTERM:
            runs[pid] = (now + _CANCEL_GRACE_SECONDS, signal.SIGKILL)
        else:
            runs[pid] = (math.inf, signal.SIGKILL)


def _next_wake(runs: _Runs) -> float | None:
    if not runs:
        return None
    # Wakes at least each second while runs are alive, to reap the finished.
    soonest = min(due for due, _ in runs.values())
    return max(0.0, min(1.0, soonest - time.monotonic()))


def _start_run(control: socket.socket, runs: _Runs) -> bool:
    """Fork a child for the next run the server sent; False once it is gone."""
    order, fds, _, _ = socket.recv_fds(control, RUN_ORDER.size, 1)
    if not order:
        return False
    conn = socket.socket(fileno=fds[0])
    (budget,) = RUN_ORDER.unpack(order)
    # Refreshes the registry's hourly cache here, not once in every child.
    get_blocks()
    try:
        pid = os.fork()
    except OSError:
        # Out of processes or memory for now: this run fails, the rest still run.
        logger.exception("Could not fork a direct block run")
        conn.close()
        return True
    if pid == 0:
        control.close()
        _run(conn, budget)
    conn.close()
    runs[pid] = (time.monotonic() + budget, signal.SIGTERM)
    return True


# Runs in a forked process, which coverage doesn't measure.
def _run(conn: socket.socket, budget: float) -> NoReturn:  # pragma: no cover
    """A child: read one run, answer it, exit."""
    try:
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.alarm(math.ceil(budget) + _ALARM_GRACE_SECONDS)
        _limit_memory()
        _prefer_for_oom_kill()
        with conn, conn.makefile("rb") as incoming:
            _, size = FRAME_HEADER.unpack(incoming.read(FRAME_HEADER.size))
            conn.sendall(asyncio.run(_answer_until_cancelled(incoming.read(size))))
    except BaseException:
        pass  # Cancelled at its deadline, or the server stopped waiting.
    finally:
        sentry_sdk.flush(timeout=2)
        os._exit(0)


# Runs in a forked process, which coverage doesn't measure.
async def _answer_until_cancelled(request: bytes) -> bytes:  # pragma: no cover
    # The run parent's SIGTERM at the deadline cancels the block where it awaits.
    task = asyncio.ensure_future(_answer(request))
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
    return await task


async def _answer(request: bytes) -> bytes:
    try:
        block_id, input_data, execute_kwargs = pickle.loads(request)
        block = get_blocks()[block_id]()
        outputs = await _collect(block.execute(input_data, **execute_kwargs))
        body = _OUTPUTS.dump_json(outputs, by_alias=True)
    except Exception as error:
        if _ran_out_of_memory(error):
            limit = DIRECT_BLOCK_EXECUTION_MEMORY_LIMIT // 2**20
            return _failure(
                f"The block used more than {limit} MB of memory.", value_error=True
            )
        return _failure(str(error), isinstance(error, ValueError))
    if len(body) > DIRECT_BLOCK_EXECUTION_MAX_OUTPUT_SIZE:
        limit = DIRECT_BLOCK_EXECUTION_MAX_OUTPUT_SIZE // 2**20
        return _failure(
            f"The block's output is larger than {limit} MB, "
            "the most a block run on its own can return.",
            value_error=True,
        )
    return FRAME_HEADER.pack(OUTPUTS, len(body)) + body


def _failure(message: str, value_error: bool) -> bytes:
    body = json.dumps({"message": message, "value_error": value_error}).encode()
    return FRAME_HEADER.pack(FAILED, len(body)) + body


def _ran_out_of_memory(error: Exception) -> bool:
    # Block.execute wraps the MemoryError, so look down the chain.
    link: BaseException | None = error
    seen: set[int] = set()
    while link is not None and id(link) not in seen:
        if isinstance(link, MemoryError):
            return True
        seen.add(id(link))
        link = link.__cause__ or link.__context__
    return False


async def _collect(outputs: BlockOutput) -> CompletedBlockOutput:
    collected: CompletedBlockOutput = defaultdict(list)
    async for name, data in outputs:
        collected[name].append(data)
    return collected


def _limit_memory() -> None:
    if sys.platform != "linux":
        return
    with open("/proc/self/status") as status:
        in_use = next(
            int(line.split()[1]) * 1024 for line in status if line.startswith("VmData:")
        )
    _, hard = resource.getrlimit(resource.RLIMIT_DATA)
    limit = in_use + DIRECT_BLOCK_EXECUTION_MEMORY_LIMIT
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_DATA, (limit, hard))


def _prefer_for_oom_kill() -> None:
    # Under the pod's memory limit the kernel kills a run before the server.
    try:
        with open("/proc/self/oom_score_adj", "w") as score:
            score.write("1000")
    except OSError:
        pass
