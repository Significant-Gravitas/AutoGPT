from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from backend.blocks.time_blocks import (
    DateStrftimeFormat,
    GetCurrentDateAndTimeBlock,
    GetCurrentDateBlock,
    GetCurrentTimeBlock,
    TimeStrftimeFormat,
)
from backend.data.execution import ExecutionContext


async def _collect_single_output(output_stream):
    results = [output async for output in output_stream]

    assert len(results) == 1
    return results[0]


def _assert_time_close_to_now(value: str, timezone: str, max_delta_seconds: int = 60):
    now = datetime.now(tz=ZoneInfo(timezone))
    parsed_time = datetime.strptime(value, "%H:%M:%S").time()
    parsed_datetime = datetime.combine(now.date(), parsed_time, tzinfo=now.tzinfo)
    candidates = (
        parsed_datetime - timedelta(days=1),
        parsed_datetime,
        parsed_datetime + timedelta(days=1),
    )

    assert min(abs(now - candidate) for candidate in candidates) <= timedelta(
        seconds=max_delta_seconds
    )


@pytest.mark.asyncio
async def test_get_current_date_without_execution_context_returns_current_date():
    block = GetCurrentDateBlock()
    input_data = block.Input(trigger="Hello", offset="0")

    key, value = await _collect_single_output(block.run(input_data=input_data))

    assert key == "date"
    parsed = datetime.strptime(value, "%Y-%m-%d").date()
    assert abs((datetime.now(tz=ZoneInfo("UTC")).date() - parsed).days) <= 1


@pytest.mark.asyncio
async def test_get_current_date_uses_user_timezone_from_execution_context():
    block = GetCurrentDateBlock()
    input_data = block.Input(
        trigger="Hello",
        offset="0",
        format_type=DateStrftimeFormat(
            discriminator="strftime",
            format="%Y-%m-%d",
            timezone="UTC",
            use_user_timezone=True,
        ),
    )
    execution_context = ExecutionContext(user_timezone="America/New_York")

    key, value = await _collect_single_output(
        block.run(input_data=input_data, execution_context=execution_context)
    )

    assert key == "date"
    assert value == datetime.now(tz=ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


@pytest.mark.asyncio
async def test_get_current_time_without_execution_context_returns_current_time():
    block = GetCurrentTimeBlock()
    input_data = block.Input(
        trigger="Hello",
        format_type=TimeStrftimeFormat(
            discriminator="strftime",
            format="%H:%M:%S",
            timezone="UTC",
        ),
    )

    key, value = await _collect_single_output(block.run(input_data=input_data))

    assert key == "time"
    _assert_time_close_to_now(value, "UTC")


@pytest.mark.asyncio
async def test_get_current_date_and_time_without_execution_context_returns_current_value():
    block = GetCurrentDateAndTimeBlock()
    input_data = block.Input(trigger="Hello")

    key, value = await _collect_single_output(block.run(input_data=input_data))

    assert key == "date_time"
    parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=ZoneInfo("UTC")
    )
    assert abs(datetime.now(tz=ZoneInfo("UTC")) - parsed) <= timedelta(seconds=10)
