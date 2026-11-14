"""Tests for the SpecStream run status and wait blocks."""

import asyncio
from itertools import count
from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.specstream import run_status
from backend.blocks.specstream._models import RunStatus
from backend.blocks.specstream._test_helpers import run_block
from backend.blocks.specstream._testdata import (
    TEST_COMPLETED,
    TEST_FIND_ID,
    TEST_RUNNING,
)
from backend.blocks.specstream.run_status import SpecStreamWaitForRunBlock
from backend.data.model import APIKeyCredentials


@pytest.mark.parametrize(
    "intervals, expected",
    [
        ({"check_interval": 10}, [10.0, 15.0]),
        ({"check_interval": 60, "max_interval": 1}, [1.0, 1.0]),
    ],
)
@pytest.mark.asyncio
async def test_wait_polls_with_a_growing_gap_until_the_run_ends(
    intervals: dict[str, int], expected: list[float]
):
    block = SpecStreamWaitForRunBlock()
    block.get_run = AsyncMock(side_effect=[TEST_RUNNING, TEST_RUNNING, TEST_COMPLETED])
    waited: list[float] = []

    async def record(seconds: float) -> None:
        waited.append(seconds)

    with patch.object(run_status, "sleep", record):
        outputs = await run_block(block, run_id=TEST_FIND_ID, **intervals)

    assert waited == expected
    assert outputs["status"] == "completed"
    assert outputs["timed_out"] is False
    assert outputs["charged_cents"] == 375


def test_wait_block_limit_covers_the_longest_timeout():
    limit = SpecStreamWaitForRunBlock().execution_timeout_seconds
    assert limit is not None
    assert limit > run_status.MAX_WAIT_SECONDS


@pytest.mark.asyncio
async def test_wait_gives_up_at_the_timeout_and_leaves_the_run_going():
    block = SpecStreamWaitForRunBlock()
    block.get_run = AsyncMock(return_value=TEST_RUNNING)
    clock = count(step=100.0)

    async def no_wait(seconds: float) -> None:
        return None

    with (
        patch.object(run_status, "sleep", no_wait),
        patch.object(run_status, "monotonic", lambda: next(clock)),
    ):
        outputs = await run_block(block, run_id=TEST_FIND_ID, timeout=250)

    assert outputs["timed_out"] is True
    assert outputs["is_done"] is False
    assert outputs["status"] == "running"


@pytest.mark.asyncio
async def test_wait_bounds_each_poll_by_the_time_left_and_reports_the_last_status():
    block = SpecStreamWaitForRunBlock()
    block.get_run = AsyncMock(return_value=TEST_RUNNING)
    budgets: list[float] = []
    now = 0.0

    async def bounded(awaitable, timeout):
        nonlocal now
        budgets.append(timeout)
        if len(budgets) > 1:
            awaitable.close()
            now += timeout
            raise TimeoutError()
        return await awaitable

    async def no_wait(seconds: float) -> None:
        nonlocal now
        now += seconds

    with (
        patch.object(run_status, "sleep", no_wait),
        patch.object(run_status, "wait_for", bounded),
        patch.object(run_status, "monotonic", lambda: now),
    ):
        outputs = await run_block(block, run_id=TEST_FIND_ID, timeout=60)

    assert budgets == [60.0, 50.0]
    assert outputs["timed_out"] is True
    assert outputs["status"] == "running"


@pytest.mark.asyncio
async def test_wait_fails_when_a_poll_times_out_with_time_left():
    block = SpecStreamWaitForRunBlock()
    block.get_run = AsyncMock(side_effect=TimeoutError())

    with pytest.raises(TimeoutError):
        await run_block(block, run_id=TEST_FIND_ID, timeout=60)


@pytest.mark.asyncio
async def test_wait_reports_timed_out_when_the_first_poll_stalls():
    block = SpecStreamWaitForRunBlock()

    async def stall(credentials: APIKeyCredentials, run_id: str) -> RunStatus:
        await asyncio.sleep(3600)
        return TEST_RUNNING

    block.get_run = stall

    outputs = await run_block(block, run_id=TEST_FIND_ID, timeout=1)

    assert outputs["timed_out"] is True
    assert outputs["is_done"] is False
    assert outputs["status"] == "unknown"
    assert outputs["run_id"] == TEST_FIND_ID
