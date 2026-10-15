from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data import expert_spend

# A Wednesday in ISO week 37; that week starts on Monday 2026-09-07.
WEDNESDAY = datetime(2026, 9, 9, 15, 30, tzinfo=timezone.utc)


def test_keys_bucket_by_window() -> None:
    assert expert_spend.spend_key("e", "week", WEDNESDAY) == "expert-spend:e:2026-W37"
    assert expert_spend.spend_key("e", "day", WEDNESDAY) == "expert-spend:e:2026-09-09"
    assert expert_spend.weekly_spend_key("e", WEDNESDAY) == "expert-spend:e:2026-W37"


def test_window_start_is_monday_or_midnight() -> None:
    assert expert_spend.window_start("week", WEDNESDAY) == datetime(
        2026, 9, 7, tzinfo=timezone.utc
    )
    assert expert_spend.window_start("day", WEDNESDAY) == datetime(
        2026, 9, 9, tzinfo=timezone.utc
    )


@pytest.mark.asyncio
async def test_add_spend_feeds_both_windows(mocker) -> None:
    redis = MagicMock(incrby=AsyncMock(), expire=AsyncMock())
    mocker.patch.object(expert_spend, "get_redis_async", AsyncMock(return_value=redis))

    await expert_spend.add_weekly_spend("e", 7)

    assert [c.args for c in redis.incrby.await_args_list] == [
        (expert_spend.spend_key("e", "week"), 7),
        (expert_spend.spend_key("e", "day"), 7),
    ]


@pytest.mark.asyncio
async def test_reset_clears_both_windows(mocker) -> None:
    redis = MagicMock(delete=AsyncMock())
    mocker.patch.object(expert_spend, "get_redis_async", AsyncMock(return_value=redis))

    await expert_spend.reset_weekly_spend("e")

    redis.delete.assert_awaited_once_with(
        expert_spend.spend_key("e", "week"), expert_spend.spend_key("e", "day")
    )


@pytest.mark.asyncio
async def test_get_spend_reads_the_requested_window_and_clamps(mocker) -> None:
    redis = MagicMock(get=AsyncMock(return_value=b"-3"))
    mocker.patch.object(expert_spend, "get_redis_async", AsyncMock(return_value=redis))

    assert await expert_spend.get_spend("e", "day") == 0

    redis.get.assert_awaited_once_with(expert_spend.spend_key("e", "day"))
