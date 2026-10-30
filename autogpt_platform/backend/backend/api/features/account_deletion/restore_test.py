"""Keeping a closed account: the status the sign-in screen reads, and the
reopen. routes_test.py runs both against the real database."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from backend.api.features.account_deletion.restore import (
    TooLateToRestoreError,
    get_deletion_status,
    restore_account,
)
from backend.data.account_deletion import ReopenOutcome

MODULE = "backend.api.features.account_deletion.restore"
USER_ID = "user-1"
ERASE_AFTER = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_an_open_account_has_nothing_scheduled(mocker: MockerFixture):
    """And this worker stops answering it as closed from its cache."""
    mocker.patch(f"{MODULE}.get_deletion_request", AsyncMock(return_value=None))
    forget = mocker.patch(f"{MODULE}.forget_account_state")

    status = await get_deletion_status(USER_ID)

    assert (status.status, status.erase_after) == ("open", None)
    forget.assert_called_once_with(USER_ID)


@pytest.mark.asyncio
async def test_a_closed_account_shows_its_erasure_date(mocker: MockerFixture):
    request = MagicMock(eraseAfter=ERASE_AFTER)
    mocker.patch(f"{MODULE}.get_deletion_request", AsyncMock(return_value=request))

    status = await get_deletion_status(USER_ID)

    assert (status.status, status.erase_after) == ("scheduled", ERASE_AFTER)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [ReopenOutcome.REOPENED, ReopenOutcome.NOT_CLOSED])
async def test_keeping_the_account_lets_the_owner_in_at_once(
    mocker: MockerFixture, outcome: ReopenOutcome
):
    reopen = mocker.patch(f"{MODULE}.reopen_account", AsyncMock(return_value=outcome))
    forget = mocker.patch(f"{MODULE}.forget_account_state")

    await restore_account(USER_ID)

    reopen.assert_awaited_once_with(USER_ID)
    forget.assert_called_once_with(USER_ID)


@pytest.mark.asyncio
async def test_too_late_changes_nothing(mocker: MockerFixture):
    mocker.patch(
        f"{MODULE}.reopen_account",
        AsyncMock(return_value=ReopenOutcome.TOO_LATE),
    )
    forget = mocker.patch(f"{MODULE}.forget_account_state")

    with pytest.raises(TooLateToRestoreError):
        await restore_account(USER_ID)

    forget.assert_not_called()
