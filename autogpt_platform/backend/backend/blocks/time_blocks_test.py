from unittest.mock import AsyncMock

import pytest

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
async def test_countdown_timer_rejects_negative_duration():
    block = CountdownTimerBlock()
    with pytest.raises(ValueError, match="non-negative"):
        await _run(block, seconds="-1")


@pytest.mark.asyncio
async def test_countdown_timer_allows_duration_at_cap(mocker):
    sleep_mock = mocker.patch(
        "backend.blocks.time_blocks.asyncio.sleep", new_callable=AsyncMock
    )
    block = CountdownTimerBlock()
    outputs = await _run(block, days=7, repeat=1)
    assert outputs == [("output_message", "timer finished")]
    sleep_mock.assert_awaited_once_with(7 * 86400)
