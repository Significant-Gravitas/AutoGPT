"""What closing and reopening an account write, without a database.

account_deletion_test.py runs the same functions against the real database;
these pin down every statement of the close's transaction, including the
ones that need too many rows to set up there (runs, reviews).
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import AccountDeletionSource
from prisma.errors import UniqueViolationError

import backend.data.account_deletion as account_deletion
from backend.data.account_deletion import (
    GRACE_PERIOD,
    ReopenOutcome,
    close_account,
    reopen_account,
)

USER_ID = "user-1"


@pytest.fixture
def tx(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    tx = MagicMock()
    for table in (
        "accountdeletionrequest",
        "authsession",
        "storelisting",
        "skilllisting",
        "storelistingreview",
        "sharedchatfile",
        "sharedexecutionfile",
        "chatlinkedshare",
        "chatsession",
        "agentgraphexecution",
        "platformlink",
        "platformuserlink",
        "user",
    ):
        actions = getattr(tx, table)
        for action in ("create", "update_many", "delete_many"):
            setattr(actions, action, AsyncMock())

    @asynccontextmanager
    async def transaction():
        yield tx

    monkeypatch.setattr(account_deletion, "transaction", transaction)
    return tx


@pytest.mark.asyncio
async def test_the_request_leaves_fourteen_days_to_change_your_mind(tx: MagicMock):
    request = await close_account(USER_ID, AccountDeletionSource.SETTINGS)

    data = tx.accountdeletionrequest.create.call_args.kwargs["data"]
    assert data["userId"] == USER_ID
    assert data["source"] == AccountDeletionSource.SETTINGS
    assert data["eraseAfter"] - data["requestedAt"] == timedelta(days=14)
    assert data["requestedAt"].tzinfo is not None
    assert request is tx.accountdeletionrequest.create.return_value


@pytest.mark.asyncio
async def test_the_close_bans_nothing(tx: MagicMock):
    """Signing in is how the owner keeps the account; a ban for another
    reason is left as it was."""
    await close_account(USER_ID, AccountDeletionSource.SETTINGS)

    assert not tx.authuser.mock_calls


@pytest.mark.asyncio
async def test_the_close_signs_out_and_takes_the_marketplace_presence_down(
    tx: MagicMock,
):
    await close_account(USER_ID, AccountDeletionSource.SETTINGS)

    tx.authsession.delete_many.assert_awaited_once_with(where={"userId": USER_ID})
    for listings in (tx.storelisting, tx.skilllisting):
        listings.update_many.assert_awaited_once_with(
            where={"owningUserId": USER_ID, "isDeleted": False},
            data={"isDeleted": True},
        )
    tx.storelistingreview.delete_many.assert_awaited_once_with(
        where={"reviewByUserId": USER_ID}
    )


@pytest.mark.asyncio
async def test_the_close_turns_off_every_share_link(tx: MagicMock):
    await close_account(USER_ID, AccountDeletionSource.SETTINGS)

    tx.chatsession.update_many.assert_awaited_once_with(
        where={"userId": USER_ID, "isShared": True},
        data={
            "isShared": False,
            "shareToken": None,
            "sharedAt": None,
            "autoShareExecutions": False,
        },
    )
    tx.agentgraphexecution.update_many.assert_awaited_once_with(
        where={"userId": USER_ID, "isShared": True},
        data={
            "isShared": False,
            "shareToken": None,
            "sharedAt": None,
            "sharedVia": None,
        },
    )
    tx.sharedchatfile.delete_many.assert_awaited_once_with(
        where={"Session": {"is": {"userId": USER_ID}}}
    )
    tx.sharedexecutionfile.delete_many.assert_awaited_once_with(
        where={"Execution": {"is": {"userId": USER_ID}}}
    )
    tx.chatlinkedshare.delete_many.assert_awaited_once_with(
        where={
            "OR": [
                {"Session": {"is": {"userId": USER_ID}}},
                {"Execution": {"is": {"userId": USER_ID}}},
            ]
        }
    )


@pytest.mark.asyncio
async def test_the_close_unlinks_every_chat_platform(tx: MagicMock):
    await close_account(USER_ID, AccountDeletionSource.PAYWALL)

    tx.platformlink.delete_many.assert_awaited_once_with(where={"userId": USER_ID})
    tx.platformuserlink.delete_many.assert_awaited_once_with(where={"userId": USER_ID})


@pytest.mark.asyncio
async def test_a_closed_account_is_not_closed_again(tx: MagicMock):
    tx.accountdeletionrequest.create.side_effect = UniqueViolationError(
        {"user_facing_error": {"message": "Unique constraint failed on (`userId`)"}}
    )

    assert await close_account(USER_ID, AccountDeletionSource.SETTINGS) is None
    tx.authsession.delete_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_reopening_deletes_the_request_only_before_its_erasure_date(
    tx: MagicMock,
):
    now = datetime(2026, 11, 1, tzinfo=timezone.utc)
    tx.accountdeletionrequest.delete_many.return_value = 1

    assert await reopen_account(USER_ID, now=now) is ReopenOutcome.REOPENED

    tx.accountdeletionrequest.delete_many.assert_awaited_once_with(
        where={"userId": USER_ID, "completedAt": None, "eraseAfter": {"gt": now}}
    )
    tx.user.update_many.assert_awaited_once_with(
        where={"id": USER_ID}, data={"maxEmailsPerDay": 3}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "still_closed, outcome",
    [(True, ReopenOutcome.TOO_LATE), (False, ReopenOutcome.NOT_CLOSED)],
)
async def test_nothing_to_reopen(
    tx: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    still_closed: bool,
    outcome: ReopenOutcome,
):
    tx.accountdeletionrequest.delete_many.return_value = 0
    monkeypatch.setattr(
        account_deletion, "is_account_closed", AsyncMock(return_value=still_closed)
    )

    assert await reopen_account(USER_ID) is outcome
    tx.user.update_many.assert_not_awaited()


def test_the_grace_period_is_fourteen_days():
    assert GRACE_PERIOD == timedelta(days=14)
