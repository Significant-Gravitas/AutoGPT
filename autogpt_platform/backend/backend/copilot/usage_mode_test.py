"""The launch switch: live from a named second, and steady through outages."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot import usage_mode
from backend.copilot.usage_mode import (
    LAST_ANSWER_KEY,
    LaunchAnswer,
    billing_clock_live,
    launch_second,
    parse_launch_answer,
)

NOW = 1_800_000_000
LAUNCH = NOW + 600


@pytest.fixture(autouse=True)
def _fresh_switch(monkeypatch: pytest.MonkeyPatch):
    usage_mode.launch_second.cache_clear()
    monkeypatch.setattr(usage_mode, "_last_answer", None)
    monkeypatch.setattr(usage_mode, "now_seconds", lambda: NOW)
    yield
    usage_mode.launch_second.cache_clear()


def _posthog(value: Any, answered: bool = True):
    return patch.object(
        usage_mode, "read_feature_flag", AsyncMock(return_value=(value, answered))
    )


def _redis(stored: str | None = None) -> MagicMock:
    redis = MagicMock()
    redis.get = AsyncMock(return_value=stored)
    redis.set = AsyncMock()
    redis.delete = AsyncMock()
    return redis


def _with_redis(redis: MagicMock):
    return patch.object(usage_mode, "get_redis_async", AsyncMock(return_value=redis))


def _posthog_configured(configured: bool = True):
    client = MagicMock() if configured else None
    return patch.object(usage_mode.posthog, "get_flag_client", return_value=client)


@pytest.mark.parametrize(
    "value,expected",
    [
        ({"live_from": LAUNCH}, LaunchAnswer(live_from=LAUNCH)),
        ({"live_from": None}, LaunchAnswer(live_from=None)),
        ({}, LaunchAnswer(live_from=None)),
        (False, LaunchAnswer(live_from=None)),
        # Malformed: no answer, so the last one given stands.
        ({"live_from": str(LAUNCH)}, None),
        ({"live_from": float(LAUNCH)}, None),
        ({"live_from": True}, None),
        ({"live_from": LAUNCH * 1000}, None),  # milliseconds
        ({"live_from": 1_500_000_000}, None),  # years ago
        ({"live_from": NOW + 400 * 86_400}, None),  # over a year ahead
        ({"launch": LAUNCH}, None),
        (True, None),
        ("soon", None),
    ],
)
def test_reading_the_flag_value(value: Any, expected: LaunchAnswer | None):
    assert parse_launch_answer(value, NOW) == expected


@pytest.mark.asyncio
async def test_it_is_live_from_the_launch_second_and_not_before():
    with _posthog({"live_from": LAUNCH}), _with_redis(_redis()):
        assert not await billing_clock_live(LAUNCH - 1)
        assert await billing_clock_live(LAUNCH)
        assert await billing_clock_live(LAUNCH + 86_400)


@pytest.mark.asyncio
async def test_with_no_posthog_and_nothing_configured_limits_stay_on_the_calendar():
    redis = _redis(str(LAUNCH))
    with _posthog(None, answered=False), _posthog_configured(False), _with_redis(redis):
        assert await launch_second() is None

    # Without PostHog nothing was ever stored: Redis isn't asked at all.
    redis.get.assert_not_called()


@pytest.mark.asyncio
async def test_an_answer_is_stored_for_the_servers_that_get_none():
    redis = _redis()
    with _posthog({"live_from": LAUNCH}), _with_redis(redis):
        assert await launch_second() == LAUNCH

    redis.set.assert_awaited_once_with(LAST_ANSWER_KEY, str(LAUNCH))


@pytest.mark.asyncio
async def test_a_posthog_outage_keeps_the_last_answer_given():
    redis = _redis()
    with _with_redis(redis), _posthog_configured():
        with _posthog({"live_from": NOW - 10}):
            assert await billing_clock_live()
        usage_mode.launch_second.cache_clear()
        with _posthog(None, answered=False):
            assert await billing_clock_live()

    # Remembered in the process: the stored copy isn't even needed.
    redis.get.assert_not_called()


@pytest.mark.asyncio
async def test_a_fresh_server_in_an_outage_takes_the_stored_answer():
    with (
        _posthog(None, answered=False),
        _posthog_configured(),
        _with_redis(_redis(str(NOW - 10))),
    ):
        assert await billing_clock_live()


@pytest.mark.asyncio
async def test_an_unreadable_store_in_an_outage_is_not_live_and_does_not_raise():
    with (
        _posthog(None, answered=False),
        _posthog_configured(),
        patch.object(
            usage_mode, "get_redis_async", AsyncMock(side_effect=ConnectionError())
        ),
    ):
        assert not await billing_clock_live()


@pytest.mark.asyncio
@pytest.mark.parametrize("off", [False, {}, {"live_from": None}])
async def test_a_conclusive_off_rolls_back_and_clears_the_stored_answer(off: Any):
    redis = _redis()
    with _with_redis(redis):
        with _posthog({"live_from": NOW - 10}):
            assert await billing_clock_live()
        usage_mode.launch_second.cache_clear()
        with _posthog(off):
            assert not await billing_clock_live()

    redis.delete.assert_awaited_once_with(LAST_ANSWER_KEY)


@pytest.mark.asyncio
async def test_a_malformed_value_does_not_roll_back_a_launch():
    with _with_redis(_redis()), _posthog_configured():
        with _posthog({"live_from": NOW - 10}):
            assert await billing_clock_live()
        usage_mode.launch_second.cache_clear()
        with _posthog({"live_from": "tomorrow"}):
            assert await billing_clock_live()


@pytest.mark.asyncio
async def test_a_failing_flag_read_is_no_answer():
    with (
        patch.object(
            usage_mode, "read_feature_flag", AsyncMock(side_effect=RuntimeError())
        ),
        _posthog_configured(False),
    ):
        assert await launch_second() is None


@pytest.mark.asyncio
async def test_the_config_default_applies_only_where_posthog_gave_no_answer(
    monkeypatch: pytest.MonkeyPatch,
):
    config = MagicMock(usage_billing_clock_live_from=NOW - 5)
    monkeypatch.setattr(usage_mode, "_config", config)
    with _posthog(None, answered=False), _posthog_configured(False):
        assert await billing_clock_live()
    usage_mode.launch_second.cache_clear()
    with _posthog(False), _with_redis(_redis()):
        assert not await billing_clock_live()
