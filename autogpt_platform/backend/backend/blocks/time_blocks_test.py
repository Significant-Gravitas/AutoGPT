from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from backend.blocks.time_blocks import CountdownTimerBlock


async def _run(block: CountdownTimerBlock, **input_kwargs):
    outputs = []
    async for name, value in block.run(block.input_schema(**input_kwargs)):
        outputs.append((name, value))
    return outputs


@pytest.mark.asyncio
async def test_countdown_timer_rejects_excessive_duration():
    block = CountdownTimerBlock()
    with pytest.raises(ValueError, match="exceeds max"):
        await _run(block, days=365, repeat=1)


@pytest.mark.asyncio
async def test_countdown_timer_rejects_cumulative_duration_over_cap():
    block = CountdownTimerBlock()
    with pytest.raises(ValueError, match="exceeds max"):
        await _run(block, days=1, repeat=10)


@pytest.mark.asyncio
async def test_countdown_timer_rejects_negative_duration():
    block = CountdownTimerBlock()
    with pytest.raises(ValueError, match="non-negative"):
        await _run(block, seconds="-1")


def test_countdown_timer_rejects_repeat_zero_at_schema():
    block = CountdownTimerBlock()
    with pytest.raises(ValidationError):
        block.input_schema(seconds=1, repeat=0)


def test_countdown_timer_rejects_repeat_over_max_at_schema():
    block = CountdownTimerBlock()
    with pytest.raises(ValidationError):
        block.input_schema(seconds=1, repeat=1001)


@pytest.mark.asyncio
async def test_countdown_timer_run_rejects_repeat_zero_defense_in_depth():
    block = CountdownTimerBlock()
    bypassed = block.input_schema.model_construct(seconds=1, repeat=0)
    with pytest.raises(ValueError, match="Repeat must be between"):
        async for _ in block.run(bypassed):
            pass


@pytest.mark.asyncio
async def test_countdown_timer_run_rejects_repeat_over_max_defense_in_depth():
    block = CountdownTimerBlock()
    bypassed = block.input_schema.model_construct(seconds=1, repeat=1001)
    with pytest.raises(ValueError, match="Repeat must be between"):
        async for _ in block.run(bypassed):
            pass


@pytest.mark.asyncio
async def test_countdown_timer_allows_duration_at_cap(mocker):
    sleep_mock = mocker.patch(
        "backend.blocks.time_blocks.asyncio.sleep", new_callable=AsyncMock
    )
    block = CountdownTimerBlock()
    outputs = await _run(block, days=7, repeat=1)
    assert outputs == [("output_message", "timer finished")]
    sleep_mock.assert_awaited_once_with(7 * 86400)


def test_countdown_timer_execution_timeout_covers_max_duration():
    block = CountdownTimerBlock()
    assert block.execution_timeout_seconds is not None
    assert block.execution_timeout_seconds >= block.MAX_TOTAL_SECONDS
