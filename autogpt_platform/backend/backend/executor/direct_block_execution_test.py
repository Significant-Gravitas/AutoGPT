"""Direct block runs happen in a child of the run parent, which kills one still
running at its deadline."""

import asyncio
import gc
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from backend.data.execution import ExecutionContext
from backend.executor import direct_block_execution, direct_block_run_parent
from backend.executor.direct_block_execution import (
    DirectBlockExecutionBusyError,
    DirectBlockExecutionLimitError,
    DirectBlockExecutionTimeoutError,
    claim_direct_block_execution_slot,
    run_block_in_child,
    start_run_parent,
    stop_run_parent,
)
from backend.util.exceptions import BlockExecutionError, BlockUnknownError


class _ReportsItsProcess:
    id = name = "reports-its-process"

    async def execute(self, input_data, *, execution_context, **kwargs):
        yield "pid", os.getpid()
        yield "user", execution_context.user_id
        yield "input", input_data


class _Computes:
    """Writes a beat every 20 ms until its process ends."""

    id = name = "computes"

    async def execute(self, input_data, **kwargs):
        while True:
            with open(input_data["beats"], "a") as beats:
                beats.write(".")
            time.sleep(0.02)
        yield "never", None


class _Fails:
    id = name = "fails"

    async def execute(self, input_data, **kwargs):
        raise {"value": ValueError, "runtime": RuntimeError}[input_data["error"]](
            "it went wrong"
        )
        yield "never", None


class _Allocates:
    id = name = "allocates"

    async def execute(self, input_data, **kwargs):
        yield "size", len(bytearray(input_data["bytes"]))


class _Outputs:
    id = name = "outputs"

    async def execute(self, input_data, **kwargs):
        yield "text", "x" * input_data["characters"]


_BLOCKS = {
    block.id: block
    for block in (_ReportsItsProcess, _Computes, _Fails, _Allocates, _Outputs)
}


@pytest.fixture
def run_parent(monkeypatch: pytest.MonkeyPatch):
    # Set before the fork: the run parent and its children see the process as it is now.
    monkeypatch.setattr(direct_block_run_parent, "get_blocks", lambda: _BLOCKS)
    monkeypatch.setattr(
        direct_block_run_parent, "DIRECT_BLOCK_EXECUTION_MEMORY_LIMIT", 64 * 2**20
    )
    monkeypatch.setattr(
        direct_block_run_parent, "DIRECT_BLOCK_EXECUTION_MAX_OUTPUT_SIZE", 2**20
    )
    start_run_parent()
    yield direct_block_execution._run_parent
    stop_run_parent()
    gc.unfreeze()


def _context() -> ExecutionContext:
    return ExecutionContext(user_id="user-a", user_timezone="UTC")


@pytest.mark.asyncio
async def test_a_run_happens_in_a_process_of_its_own(run_parent):
    outputs = await run_block_in_child(
        _ReportsItsProcess(), {"n": 1}, execution_context=_context()
    )

    (pid,) = outputs["pid"]
    assert pid not in (os.getpid(), run_parent.pid)
    assert outputs["user"] == ["user-a"]
    assert outputs["input"] == [{"n": 1}]


@pytest.mark.asyncio
async def test_a_block_computing_past_its_budget_is_killed_at_its_deadline(
    run_parent, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(
        direct_block_execution, "DIRECT_BLOCK_EXECUTION_TIMEOUT_SECONDS", 0.5
    )
    monkeypatch.setattr(
        direct_block_execution, "DIRECT_BLOCK_EXECUTION_KILL_WAIT_SECONDS", 1
    )
    monkeypatch.setattr(
        direct_block_execution, "DIRECT_BLOCK_EXECUTION_MAX_RUNS_PER_USER", 1
    )
    beats = tmp_path / "beats"

    with claim_direct_block_execution_slot("user-a"):
        with pytest.raises(DirectBlockExecutionTimeoutError, match="0.5 seconds"):
            await run_block_in_child(_Computes(), {"beats": str(beats)})

    assert beats.stat().st_size > 0
    stopped_at = beats.stat().st_size
    await asyncio.sleep(0.5)
    assert beats.stat().st_size == stopped_at
    claim_direct_block_execution_slot("user-a").release()


@pytest.mark.asyncio
async def test_stopping_the_run_parent_kills_its_runs(run_parent, tmp_path: Path):
    beats = tmp_path / "beats"
    run = asyncio.create_task(run_block_in_child(_Computes(), {"beats": str(beats)}))
    while not beats.exists():
        await asyncio.sleep(0.02)

    stop_run_parent()

    with pytest.raises(RuntimeError, match="ended unexpectedly"):
        await asyncio.wait_for(run, 10)
    stopped_at = beats.stat().st_size
    await asyncio.sleep(0.5)
    assert beats.stat().st_size == stopped_at


@pytest.mark.parametrize(
    "error, raised", [("value", BlockExecutionError), ("runtime", BlockUnknownError)]
)
@pytest.mark.asyncio
async def test_a_block_error_comes_back_as_the_error_the_routes_map(
    run_parent, error: str, raised: type[Exception]
):
    with pytest.raises(raised, match="it went wrong") as caught:
        await run_block_in_child(_Fails(), {"error": error})

    assert caught.value.block_id == "fails"


@pytest.mark.skipif(sys.platform != "linux", reason="the memory limit is Linux-only")
@pytest.mark.asyncio
async def test_a_run_past_its_memory_limit_is_refused(run_parent):
    with pytest.raises(BlockExecutionError, match="more than 64 MB of memory"):
        await run_block_in_child(_Allocates(), {"bytes": 256 * 2**20})

    assert await run_block_in_child(_Allocates(), {"bytes": 2**20}) == {"size": [2**20]}


@pytest.mark.asyncio
async def test_output_past_the_limit_is_refused(run_parent):
    with pytest.raises(BlockExecutionError, match="larger than 1 MB"):
        await run_block_in_child(_Outputs(), {"characters": 2 * 2**20})


def test_runs_are_refused_once_the_run_parent_has_exited(run_parent):
    claim_direct_block_execution_slot("user-a").release()

    os.kill(run_parent.pid, signal.SIGKILL)
    while run_parent.running():
        time.sleep(0.01)

    with pytest.raises(DirectBlockExecutionBusyError):
        claim_direct_block_execution_slot("user-a")


def test_runs_are_refused_without_a_run_parent():
    with pytest.raises(DirectBlockExecutionBusyError):
        claim_direct_block_execution_slot("user-a")


def test_a_user_gets_a_limited_number_of_runs(run_parent):
    limit = direct_block_execution.DIRECT_BLOCK_EXECUTION_MAX_RUNS_PER_USER
    slots = [claim_direct_block_execution_slot("user-a") for _ in range(limit)]
    try:
        with pytest.raises(DirectBlockExecutionLimitError, match=f"{limit} blocks"):
            claim_direct_block_execution_slot("user-a")
        claim_direct_block_execution_slot("user-b").release()
        slots.pop().release()
        slots.append(claim_direct_block_execution_slot("user-a"))
    finally:
        for slot in slots:
            slot.release()
    assert "user-a" not in direct_block_execution._runs_per_user


def test_the_server_runs_a_limited_number_of_runs(
    run_parent, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(direct_block_execution, "DIRECT_BLOCK_EXECUTION_MAX_RUNS", 2)
    slots = [claim_direct_block_execution_slot(user) for user in ("user-a", "user-b")]
    try:
        with pytest.raises(DirectBlockExecutionBusyError):
            claim_direct_block_execution_slot("user-c")
    finally:
        for slot in slots:
            slot.release()
    claim_direct_block_execution_slot("user-c").release()
