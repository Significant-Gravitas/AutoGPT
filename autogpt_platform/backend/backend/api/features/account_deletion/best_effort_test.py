"""attempt_step: one bounded call that logs instead of raising."""

import asyncio
import logging

import pytest
from pytest_mock import MockerFixture

from backend.api.features.account_deletion.best_effort import attempt_step
from backend.util.exceptions import NotFoundError

MODULE = "backend.api.features.account_deletion.best_effort"
USER_ID = "user-1"


async def _value() -> int:
    return 7


async def _raise(error: Exception) -> None:
    raise error


@pytest.mark.asyncio
async def test_returns_the_value(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger=MODULE)

    assert await attempt_step("count", USER_ID, _value()) == 7
    assert not caplog.records


@pytest.mark.asyncio
async def test_a_stalled_call_is_given_up_on_and_logged(
    mocker: MockerFixture, caplog: pytest.LogCaptureFixture
):
    mocker.patch(f"{MODULE}.STEP_TIMEOUT_SECONDS", 0.05)

    result = await attempt_step("wait", USER_ID, asyncio.Event().wait())

    assert result is None
    [record] = caplog.records
    assert record.levelno == logging.ERROR
    assert "TimeoutError" in record.getMessage()
    assert USER_ID in record.getMessage()


@pytest.mark.asyncio
async def test_already_gone_is_logged_as_info(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger=MODULE)

    result = await attempt_step("delete", USER_ID, _raise(NotFoundError("gone")))

    assert result is None
    [record] = caplog.records
    assert record.levelno == logging.INFO
    assert "already gone" in record.getMessage()


@pytest.mark.asyncio
async def test_any_other_error_is_logged_and_not_raised(
    caplog: pytest.LogCaptureFixture,
):
    result = await attempt_step("stop", USER_ID, _raise(RuntimeError("db down")))

    assert result is None
    [record] = caplog.records
    assert record.levelno == logging.ERROR
    assert "stop" in record.getMessage() and "db down" in record.getMessage()
