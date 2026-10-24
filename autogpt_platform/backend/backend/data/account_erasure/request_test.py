"""The hand-off from closing an account: which requests are pending, the
confirmation the close could not send goes out before the address does, the
request is marked done once, and a request made by email is closed exactly
the way Settings closes one."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import AccountDeletionSource

from backend.data.account_erasure import request
from backend.data.account_erasure.models import StepStatus, Subject

USER_ID = "3f0b8c1e-4c2a-4f7e-9a51-0c6f2d8e1a77"
EMAIL = "sam@example.com"
NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
PENDING = Subject(
    user_id=USER_ID,
    email=EMAIL,
    has_user_row=True,
    request_source=AccountDeletionSource.SETTINGS,
)


@pytest.fixture
def requests_table(monkeypatch):
    table = MagicMock()
    table.prisma.return_value.find_many = AsyncMock(
        return_value=[SimpleNamespace(userId="a"), SimpleNamespace(userId="b")]
    )
    table.prisma.return_value.update_many = AsyncMock(return_value=1)
    monkeypatch.setattr(request, "AccountDeletionRequest", table)
    return table.prisma.return_value


@pytest.mark.asyncio
async def test_pending_requests_are_the_unfinished_ones_past_the_grace(
    requests_table,
):
    """A request closed moments ago is left alone: the close sends its
    confirmation just after writing the row."""
    ids = await request.pending_user_ids(grace=timedelta(minutes=15), now=NOW)

    assert ids == ["a", "b"]
    kwargs = requests_table.find_many.await_args.kwargs
    assert kwargs["where"] == {
        "completedAt": None,
        "requestedAt": {"lte": NOW - timedelta(minutes=15)},
    }
    assert kwargs["order"] == {"requestedAt": "asc"}


@pytest.fixture
def email(monkeypatch):
    calls = SimpleNamespace(send=AsyncMock(), mark=AsyncMock())
    monkeypatch.setattr(request, "send_account_deleted_email", calls.send)
    monkeypatch.setattr(request, "mark_confirmation_sent", calls.mark)
    return calls


@pytest.mark.asyncio
async def test_a_confirmation_the_close_could_not_send_goes_out_first(email):
    report = await request.send_confirmation(PENDING, dry_run=False)

    email.send.assert_awaited_once_with(EMAIL, AccountDeletionSource.SETTINGS)
    email.mark.assert_awaited_once_with(USER_ID)
    assert report.counts == {"emails": 1}
    assert report.status == StepStatus.DONE


@pytest.mark.asyncio
async def test_the_confirmation_is_worded_for_the_request_source(email):
    """A request made by email gets the email's own reason line, not the one
    for deleting the account in AutoGPT."""
    by_email = PENDING.model_copy(
        update={"request_source": AccountDeletionSource.EMAIL}
    )

    await request.send_confirmation(by_email, dry_run=False)

    email.send.assert_awaited_once_with(EMAIL, AccountDeletionSource.EMAIL)


@pytest.mark.asyncio
async def test_a_confirmation_dry_run_sends_nothing(email):
    report = await request.send_confirmation(PENDING, dry_run=True)

    email.send.assert_not_awaited()
    email.mark.assert_not_awaited()
    assert report.counts == {"emails": 1}


@pytest.mark.asyncio
async def test_a_confirmation_already_sent_is_not_sent_again(email):
    sent = PENDING.model_copy(update={"confirmation_sent": True})

    report = await request.send_confirmation(sent, dry_run=False)

    email.send.assert_not_awaited()
    assert report.counts == {"emails": 0}


@pytest.mark.asyncio
async def test_a_failed_send_is_not_recorded_and_fails_the_step(email):
    """The routine reports the step failed and stops before the rows (and
    the address) go, so the next run sends it."""
    email.send.side_effect = RuntimeError("smtp down")

    with pytest.raises(RuntimeError):
        await request.send_confirmation(PENDING, dry_run=False)

    email.mark.assert_not_awaited()


@pytest.mark.asyncio
async def test_without_an_address_the_confirmation_is_skipped(email):
    no_address = PENDING.model_copy(update={"email": None})

    report = await request.send_confirmation(no_address, dry_run=False)

    assert report.status == StepStatus.SKIPPED
    email.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_request_is_marked_done_once(requests_table):
    report = await request.complete_request(PENDING, dry_run=False)

    where = requests_table.update_many.await_args.kwargs["where"]
    # A rerun keeps the first date.
    assert where == {"userId": USER_ID, "completedAt": None}
    assert report.counts == {"completed": 1}


@pytest.mark.asyncio
async def test_a_completion_dry_run_writes_nothing(requests_table):
    done = PENDING.model_copy(update={"request_completed": True})

    assert (await request.complete_request(PENDING, dry_run=True)).counts == {
        "completed": 1
    }
    assert (await request.complete_request(done, dry_run=True)).counts == {
        "completed": 0
    }
    requests_table.update_many.assert_not_awaited()


@pytest.fixture
def closing(monkeypatch):
    """`delete_account` faked, and an identity banned the way `ban` says."""
    monkeypatch.setattr(
        request, "get_user_by_id", AsyncMock(return_value=SimpleNamespace(email=EMAIL))
    )
    close = AsyncMock()
    monkeypatch.setattr(request, "delete_account", close)
    identity = SimpleNamespace(banned=None, banReason=None)
    identities = MagicMock()
    identities.prisma.return_value.find_unique = AsyncMock(return_value=identity)
    monkeypatch.setattr(request, "AuthUser", identities)
    return SimpleNamespace(close=close, identity=identity)


@pytest.mark.asyncio
async def test_an_email_request_is_closed_the_way_settings_closes_one(closing):
    await request.close_for_email_request(USER_ID)

    closing.close.assert_awaited_once_with(USER_ID, EMAIL, AccountDeletionSource.EMAIL)


@pytest.mark.asyncio
async def test_an_account_banned_for_another_reason_is_not_closed(closing):
    """Refused before Stripe, revokes or the email, as the Settings route
    refuses it."""
    closing.identity.banned, closing.identity.banReason = True, "spam"

    with pytest.raises(request.AccountBannedError):
        await request.close_for_email_request(USER_ID)

    closing.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_operator_can_close_a_banned_account_anyway(closing):
    closing.identity.banned, closing.identity.banReason = True, "spam"

    await request.close_for_email_request(USER_ID, allow_banned=True)

    closing.close.assert_awaited_once_with(USER_ID, EMAIL, AccountDeletionSource.EMAIL)
