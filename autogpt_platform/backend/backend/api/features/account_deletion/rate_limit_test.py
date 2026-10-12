"""The per-user cap on exit feedback: 10 an hour, failing open on Redis errors."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from pytest_mock import MockerFixture

from backend.api.features.account_deletion.rate_limit import (
    FEEDBACK_MAX_REQUESTS,
    FEEDBACK_WINDOW_SECONDS,
    enforce_feedback_rate_limit,
)

MODULE = "backend.api.features.account_deletion.rate_limit"
USER_ID = "user-1"


def _redis(count: int) -> MagicMock:
    redis = MagicMock()
    redis.set = AsyncMock()
    redis.incr = AsyncMock(return_value=count)
    return redis


@pytest.fixture
def hits(mocker: MockerFixture) -> MagicMock:
    return mocker.patch(f"{MODULE}.record_rate_limit_hit")


def test_the_limit_is_ten_an_hour():
    assert FEEDBACK_MAX_REQUESTS == 10
    assert FEEDBACK_WINDOW_SECONDS == 3600


@pytest.mark.asyncio
async def test_calls_up_to_the_limit_pass(mocker: MockerFixture, hits: MagicMock):
    redis = _redis(FEEDBACK_MAX_REQUESTS)
    mocker.patch(f"{MODULE}.get_redis_async", AsyncMock(return_value=redis))

    await enforce_feedback_rate_limit(USER_ID)

    hits.assert_not_called()


@pytest.mark.asyncio
async def test_the_window_is_per_user_and_expires(mocker: MockerFixture):
    redis = _redis(1)
    mocker.patch(f"{MODULE}.get_redis_async", AsyncMock(return_value=redis))

    await enforce_feedback_rate_limit(USER_ID)

    key = redis.set.await_args.args[0]
    assert USER_ID in key
    assert redis.set.await_args.kwargs == {"ex": FEEDBACK_WINDOW_SECONDS, "nx": True}
    redis.incr.assert_awaited_once_with(key)


@pytest.mark.asyncio
async def test_a_call_past_the_limit_gets_a_429(mocker: MockerFixture, hits: MagicMock):
    redis = _redis(FEEDBACK_MAX_REQUESTS + 1)
    mocker.patch(f"{MODULE}.get_redis_async", AsyncMock(return_value=redis))

    with pytest.raises(HTTPException) as raised:
        await enforce_feedback_rate_limit(USER_ID)

    assert raised.value.status_code == 429
    assert raised.value.detail == "Too many answers. Try again later."
    hits.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["connect", "incr"])
async def test_a_redis_error_fails_open(
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
    hits: MagicMock,
    failing: str,
):
    redis = _redis(1)
    if failing == "connect":
        get_redis = AsyncMock(side_effect=ConnectionError("redis down"))
    else:
        redis.incr.side_effect = ConnectionError("redis down")
        get_redis = AsyncMock(return_value=redis)
    mocker.patch(f"{MODULE}.get_redis_async", get_redis)

    with caplog.at_level(logging.WARNING):
        await enforce_feedback_rate_limit(USER_ID)

    hits.assert_not_called()
    assert "failed open" in caplog.text
