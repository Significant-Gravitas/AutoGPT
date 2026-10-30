"""The per-user cap on exit feedback: 20 an hour, failing open on Redis errors."""

import asyncio
import logging
import time
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


@pytest.fixture
def redis(mocker: MockerFixture) -> MagicMock:
    client = MagicMock()
    mocker.patch(f"{MODULE}.get_redis_async", AsyncMock(return_value=client))
    return client


@pytest.fixture
def incr(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch(f"{MODULE}.incr_with_ttl", AsyncMock(return_value=1))


@pytest.fixture
def hits(mocker: MockerFixture) -> MagicMock:
    return mocker.patch(f"{MODULE}.record_rate_limit_hit")


def test_the_limit_is_twenty_an_hour():
    assert FEEDBACK_MAX_REQUESTS == 20
    assert FEEDBACK_WINDOW_SECONDS == 3600


@pytest.mark.asyncio
async def test_calls_up_to_the_limit_pass(
    redis: MagicMock, incr: AsyncMock, hits: MagicMock
):
    incr.return_value = FEEDBACK_MAX_REQUESTS

    await enforce_feedback_rate_limit(USER_ID)

    hits.assert_not_called()


@pytest.mark.asyncio
async def test_the_window_is_per_user_and_expires(redis: MagicMock, incr: AsyncMock):
    await enforce_feedback_rate_limit(USER_ID)

    incr.assert_awaited_once()
    key = incr.await_args.args[1]
    assert USER_ID in key
    incr.assert_awaited_once_with(redis, key, FEEDBACK_WINDOW_SECONDS)


@pytest.mark.asyncio
async def test_a_call_past_the_limit_gets_a_429(
    redis: MagicMock, incr: AsyncMock, hits: MagicMock
):
    incr.return_value = FEEDBACK_MAX_REQUESTS + 1

    with pytest.raises(HTTPException) as raised:
        await enforce_feedback_rate_limit(USER_ID)

    assert raised.value.status_code == 429
    assert raised.value.detail == "Too many answers. Try again later."
    hits.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["connect", "incr"])
async def test_a_redis_error_fails_open(
    mocker: MockerFixture,
    redis: MagicMock,
    incr: AsyncMock,
    caplog: pytest.LogCaptureFixture,
    hits: MagicMock,
    failing: str,
):
    if failing == "connect":
        mocker.patch(
            f"{MODULE}.get_redis_async",
            AsyncMock(side_effect=ConnectionError("redis down")),
        )
    else:
        incr.side_effect = ConnectionError("redis down")

    with caplog.at_level(logging.WARNING):
        await enforce_feedback_rate_limit(USER_ID)

    hits.assert_not_called()
    assert "failed open" in caplog.text


@pytest.mark.asyncio
async def test_a_stalled_redis_fails_open_fast(
    mocker: MockerFixture,
    incr: AsyncMock,
    caplog: pytest.LogCaptureFixture,
    hits: MagicMock,
):
    async def stall():
        await asyncio.sleep(5)

    mocker.patch(f"{MODULE}.get_redis_async", side_effect=stall)

    started = time.monotonic()
    with caplog.at_level(logging.WARNING):
        await enforce_feedback_rate_limit(USER_ID)

    # Ten times the 0.25 s deadline, still well short of the 5 s stall.
    assert time.monotonic() - started < 2.5
    hits.assert_not_called()
    assert "failed open" in caplog.text
