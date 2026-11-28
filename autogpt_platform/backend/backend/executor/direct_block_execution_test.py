"""Direct block runs happen in a child of the run parent, which cancels one still
running at its deadline and kills it if it outlasts the cancel."""

import asyncio
import gc
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from backend.blocks._base import Block
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


class _Block:
    """Has Block's default deadline, as a real block without its own does."""

    @property
    def direct_run_timeout_seconds(self) -> float:
        return Block.direct_run_timeout_seconds


class _ReportsItsProcess(_Block):
    id = name = "reports-its-process"

    async def execute(self, input_data, *, execution_context, **kwargs):
        yield "pid", os.getpid()
        yield "user", execution_context.user_id
        yield "input", input_data


class _Computes(_Block):
    """Writes a beat every 20 ms and takes no cancel, so it runs until it is killed."""

    id = name = "computes"

    async def execute(self, input_data, **kwargs):
        while True:
            with open(input_data["beats"], "a") as beats:
                beats.write(".")
            time.sleep(0.02)
        yield "never", None


class _Fails(_Block):
    id = name = "fails"

    async def execute(self, input_data, **kwargs):
        raise {"value": ValueError, "runtime": RuntimeError}[input_data["error"]](
            "it went wrong"
        )
        yield "never", None


class _Allocates(_Block):
    id = name = "allocates"

    async def execute(self, input_data, **kwargs):
        yield "size", len(bytearray(input_data["bytes"]))


class _Outputs(_Block):
    id = name = "outputs"

    async def execute(self, input_data, **kwargs):
        yield "text", "x" * input_data["characters"]


class _StartsAHelper(_Block):
    """Starts a helper in a session of its own and ends it when cancelled, as the
    video blocks do with their renders."""

    id = name = "starts-a-helper"

    async def execute(self, input_data, **kwargs):
        helper = await asyncio.create_subprocess_exec(
            "sleep", "60", start_new_session=True
        )
        Path(input_data["pid_file"]).write_text(str(helper.pid))
        try:
            await helper.wait()
        finally:
            os.killpg(helper.pid, signal.SIGKILL)
        yield "never", None


class _ReportsItsOOMScore(_Block):
    id = name = "reports-its-oom-score"

    async def execute(self, input_data, **kwargs):
        with open("/proc/self/oom_score_adj") as score:
            yield "oom_score_adj", int(score.read())


_BLOCKS = {
    block.id: block
    for block in (
        _ReportsItsProcess,
        _Computes,
        _Fails,
        _Allocates,
        _Outputs,
        _StartsAHelper,
        _ReportsItsOOMScore,
    )
}


@pytest.fixture
def run_parent(monkeypatch: pytest.MonkeyPatch):
    yield from _run_parent(monkeypatch)


@pytest.fixture
def run_parent_with_alarms_at_the_deadline(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(direct_block_run_parent, "_ALARM_GRACE_SECONDS", 0)
    yield from _run_parent(monkeypatch)


def _run_parent(monkeypatch: pytest.MonkeyPatch):
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
async def test_the_run_parent_outlives_a_run_it_has_reaped(run_parent):
    await run_block_in_child(_ReportsItsProcess(), {}, execution_context=_context())
    # Its loop wakes each second while it lists a run, so by now it has reaped this one.
    await asyncio.sleep(2.5)

    assert run_parent.running()


@pytest.mark.asyncio
async def test_a_block_computing_past_its_budget_is_killed(
    run_parent, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(Block, "direct_run_timeout_seconds", 0.5)
    # Past the cancel's grace, and short of the child's own alarm.
    monkeypatch.setattr(
        direct_block_execution, "DIRECT_BLOCK_EXECUTION_KILL_WAIT_SECONDS", 4
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


@pytest.mark.skipif(sys.platform != "linux", reason="reads the helper's /proc entry")
@pytest.mark.asyncio
async def test_a_run_past_its_deadline_ends_the_helpers_it_started(
    run_parent, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(Block, "direct_run_timeout_seconds", 0.5)
    pid_file = tmp_path / "helper.pid"

    with pytest.raises(DirectBlockExecutionTimeoutError):
        await run_block_in_child(_StartsAHelper(), {"pid_file": str(pid_file)})

    helper = int(pid_file.read_text())
    try:
        assert not _running(helper)
    finally:
        if _running(helper):
            os.kill(helper, signal.SIGKILL)


@pytest.mark.asyncio
async def test_a_run_outliving_its_run_parent_ends_at_its_own_alarm(
    run_parent_with_alarms_at_the_deadline,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(Block, "direct_run_timeout_seconds", 1)
    monkeypatch.setattr(
        direct_block_execution, "DIRECT_BLOCK_EXECUTION_KILL_WAIT_SECONDS", 3
    )
    beats = tmp_path / "beats"
    run = asyncio.create_task(run_block_in_child(_Computes(), {"beats": str(beats)}))
    while not beats.exists():
        await asyncio.sleep(0.02)

    os.kill(run_parent_with_alarms_at_the_deadline.pid, signal.SIGKILL)

    with pytest.raises(DirectBlockExecutionTimeoutError):
        await asyncio.wait_for(run, 10)
    stopped_at = beats.stat().st_size
    await asyncio.sleep(0.5)
    assert beats.stat().st_size == stopped_at


@pytest.mark.skipif(sys.platform != "linux", reason="oom_score_adj is Linux-only")
@pytest.mark.asyncio
async def test_under_memory_pressure_the_kernel_kills_a_run_first(run_parent):
    outputs = await run_block_in_child(_ReportsItsOOMScore(), {})

    assert outputs["oom_score_adj"] == [1000]


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


def _running(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False
