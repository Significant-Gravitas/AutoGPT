"""stop_account_emails against the real database.

A closed account's Briefing, Alerts and Verdicts are switched off, it is opted
out of marketing, and MailerLite is told to stop mailing it. Nobody else's
settings move.
"""

from datetime import datetime, timezone
from typing import AsyncIterator
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma.enums import BriefingFrequency
from prisma.models import User
from pytest_mock import MockerFixture

from backend.api.features.account_deletion.email_opt_out import (
    MARKETING_OPT_OUT_SOURCE_ACCOUNT_DELETED,
    stop_account_emails,
)
from backend.data import db
from backend.data.notifications import AudienceAction, NotificationResult
from backend.util.exceptions import DatabaseError, NotFoundError

MODULE = "backend.api.features.account_deletion.email_opt_out"
# The MailerLite unsubscribe is queued by the opt-out helper in data/user.py.
QUEUE = "backend.data.user.queue_audience_change"


@pytest_asyncio.fixture(loop_scope="session")
async def accounts() -> AsyncIterator[tuple[str, str]]:
    """The account being closed, and a bystander that must stay untouched."""
    owned_connection = not db.is_connected()
    if owned_connection:
        await db.connect()
    user_id, bystander_id = str(uuid4()), str(uuid4())
    try:
        for account_id in (user_id, bystander_id):
            await User.prisma().create(
                data={"id": account_id, "email": f"opt-out-{account_id}@example.com"}
            )
        yield user_id, bystander_id
    finally:
        await User.prisma().delete_many(where={"id": {"in": [user_id, bystander_id]}})
        if owned_connection:
            await db.disconnect()


@pytest.fixture
def queued(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch(QUEUE, return_value=NotificationResult(success=True))


async def _row(user_id: str) -> User:
    return await User.prisma().find_unique_or_raise(where={"id": user_id})


@pytest.mark.asyncio(loop_scope="session")
async def test_turns_off_the_briefing_alerts_and_verdicts(
    accounts: tuple[str, str], queued: AsyncMock
):
    user_id, _ = accounts

    await stop_account_emails(user_id)

    row = await _row(user_id)
    assert row.briefingFrequency == BriefingFrequency.OFF
    assert row.alertsEnabled is False
    assert row.notifyOnStoreVerdict is False
    # The one-click unsubscribe's "send nothing": no product email of any kind.
    assert row.maxEmailsPerDay == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_records_the_opt_out_and_queues_the_mailerlite_unsubscribe(
    accounts: tuple[str, str], queued: AsyncMock
):
    user_id, _ = accounts

    await stop_account_emails(user_id)

    row = await _row(user_id)
    assert row.marketingOptOutAt is not None
    assert row.marketingOptOutSource == MARKETING_OPT_OUT_SOURCE_ACCOUNT_DELETED
    assert MARKETING_OPT_OUT_SOURCE_ACCOUNT_DELETED == "account_deleted"
    queued.assert_awaited_once()
    event = queued.await_args.args[0]
    assert event.action is AudienceAction.UNSUBSCRIBE
    assert event.user_id == user_id
    assert event.email == row.email


@pytest.mark.asyncio(loop_scope="session")
async def test_a_second_call_changes_nothing(
    accounts: tuple[str, str], queued: AsyncMock
):
    user_id, _ = accounts

    await stop_account_emails(user_id)
    first = await _row(user_id)
    await stop_account_emails(user_id)
    second = await _row(user_id)

    assert second.marketingOptOutAt == first.marketingOptOutAt
    assert second.marketingOptOutSource == MARKETING_OPT_OUT_SOURCE_ACCOUNT_DELETED
    assert second.briefingFrequency == BriefingFrequency.OFF
    assert second.alertsEnabled is False
    assert second.notifyOnStoreVerdict is False
    assert second.maxEmailsPerDay == 0
    assert second.email == first.email


@pytest.mark.asyncio(loop_scope="session")
async def test_an_earlier_opt_out_keeps_its_date_and_source(
    accounts: tuple[str, str], queued: AsyncMock
):
    """First refusal wins, as everywhere else. MailerLite is still told: the
    unsubscribe is a no-op for someone it already does not mail, and it is the
    last chance to catch one whose earlier unsubscribe never got queued."""
    user_id, _ = accounts
    refused_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    await User.prisma().update(
        where={"id": user_id},
        data={"marketingOptOutAt": refused_at, "marketingOptOutSource": "signup"},
    )

    await stop_account_emails(user_id)

    row = await _row(user_id)
    assert row.marketingOptOutAt == refused_at
    assert row.marketingOptOutSource == "signup"
    queued.assert_awaited_once()


@pytest.mark.asyncio(loop_scope="session")
async def test_other_users_are_untouched(accounts: tuple[str, str], queued: AsyncMock):
    user_id, bystander_id = accounts
    before = await _row(bystander_id)

    await stop_account_emails(user_id)

    after = await _row(bystander_id)
    assert after.briefingFrequency == before.briefingFrequency
    assert after.briefingFrequency == BriefingFrequency.WEEKLY
    assert after.alertsEnabled is True
    assert after.notifyOnStoreVerdict is True
    assert after.maxEmailsPerDay == before.maxEmailsPerDay
    assert after.marketingOptOutAt is None
    assert after.marketingOptOutSource is None
    assert all(call.args[0].user_id == user_id for call in queued.await_args_list)


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "failure",
    [
        {"return_value": NotificationResult(success=False, message="broker down")},
        {"side_effect": RuntimeError("broker down")},
    ],
    ids=["not_queued", "raised"],
)
async def test_a_mailerlite_queueing_failure_is_logged_not_raised(
    accounts: tuple[str, str],
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
    failure: dict,
):
    user_id, _ = accounts
    mocker.patch(QUEUE, **failure)

    await stop_account_emails(user_id)

    row = await _row(user_id)
    assert row.marketingOptOutSource == MARKETING_OPT_OUT_SOURCE_ACCOUNT_DELETED
    assert row.briefingFrequency == BriefingFrequency.OFF
    assert "Could not queue the MailerLite unsubscribe" in caplog.text


@pytest.mark.asyncio(loop_scope="session")
async def test_an_unknown_account_raises(accounts: tuple[str, str], queued: AsyncMock):
    with pytest.raises(NotFoundError):
        await stop_account_emails(str(uuid4()))

    queued.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "failing",
    ["update_user_notification_preference", "record_marketing_opt_out_by_email"],
)
async def test_a_database_error_propagates(
    accounts: tuple[str, str],
    mocker: MockerFixture,
    queued: AsyncMock,
    failing: str,
):
    user_id, _ = accounts
    mocker.patch(f"{MODULE}.{failing}", side_effect=DatabaseError("database down"))

    with pytest.raises(DatabaseError, match="database down"):
        await stop_account_emails(user_id)

    queued.assert_not_awaited()
