from datetime import datetime

import pytest

from backend.blocks.time_blocks import GetCurrentDateBlock


@pytest.mark.asyncio
async def test_get_current_date_without_execution_context():
    block = GetCurrentDateBlock()
    input_data = block.Input(trigger="Hello", offset="0")

    results = [output async for output in block.run(input_data=input_data)]

    assert len(results) == 1
    key, value = results[0]
    assert key == "date"
    datetime.strptime(value, "%Y-%m-%d")
