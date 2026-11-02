"""The hand-off from closing an account: which requests are pending, the
claim that keeps Keep from racing the erasure, the confirmation the close
could not send goes out before the address does, the claimed request is
marked done once, and a request made by email is closed exactly the way
Settings closes one."""

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
ERASE_AFTER = NOW - timedelta(hours=1)
REQUEST_ID = "9d4f6a2b-1c3e-4b5a-8f7d-2e6c0a9b1d34"
PENDING = Subject(
    user_id=USER_ID,
    email=EMAIL,
    has_user_row=True,
    request_source=AccountDeletionSource.SETTINGS,
    erase_after=ERASE_AFTER,
    request_id=REQUEST_ID,
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
async def test_pending_requests_are_the_unfinished_ones_past_their_erase_date(
    requests_table,
):
    """Not filtered on startedAt: an erasure that stopped halfway is
    resumed."""
    ids = await request.pending_user_ids(now=NOW)

    assert ids == ["a", "b"]
    kwargs = requests_table.find_many.await_args.kwargs
    assert kwargs["where"] == {"completedAt": None, "eraseAfter": {"lte": NOW}}
    assert kwargs["order"] == {"eraseAfter": "asc"}


@pytest.fixture
def claim_query(monkeypatch):
    query = AsyncMock(return_value=[{"id": REQUEST_ID}])
    monkeypatch.setattr(request, "query_raw_with_schema", query)
    return query


@pytest.mark.asyncio
async def test_the_claim_is_one_conditional_statement(claim_query):
    """Only an unfinished request past its erase date, on the database's
    clock; an earlier claim keeps its date and nothing ever clears it."""
    assert await request.claim_request(USER_ID) == REQUEST_ID

    sql, user_id = claim_query.await_args.args
    assert user_id == USER_ID
    assert sql.startswith('UPDATE {schema_prefix}"AccountDeletionRequest"')
    # The columns hold UTC without a zone: now() is read in UTC, so a
    # database in another time zone doesn't cut the 14 days short.
    utc_now = "now() AT TIME ZONE 'UTC'"
    assert f'SET "startedAt" = COALESCE("startedAt", {utc_now})' in sql
    assert '"completedAt" IS NULL' in sql
    assert f'"eraseAfter" <= {utc_now}' in sql
    assert 'RETURNING "id"' in sql
    assert "NULL," not in sql and '"startedAt" = NULL' not in sql


@pytest.mark.asyncio
async def test_a_kept_or_early_request_is_not_claimed(claim_query):
    claim_query.return_value = []

    assert await request.claim_request(USER_ID) is None


@pytest.fixture
def email(monkeypatch):
    calls = SimpleNamespace(send=AsyncMock(), mark=AsyncMock())
    monkeypatch.setattr(request, "send_account_deleted_email", calls.send)
    monkeypatch.setattr(request, "mark_confirmation_sent", calls.mark)
    return calls


@pytest.mark.asyncio
async def test_a_confirmation_the_close_could_not_send_goes_out_first(email):
    report = await request.send_confirmation(PENDING, dry_run=False)

    email.send.assert_awaited_once()
    assert email.send.await_args.args == (
        EMAIL,
        AccountDeletionSource.SETTINGS,
        ERASE_AFTER,
    )
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

    assert email.send.await_args.args == (
        EMAIL,
        AccountDeletionSource.EMAIL,
        ERASE_AFTER,
    )


@pytest.mark.asyncio
async def test_a_late_confirmation_follows_the_claim_not_this_pods_clock(email):
    """Claimed, so the 14 days are over: even a pod whose clock is behind the
    erase date sends the "being erased" wording, never a Keep it would
    refuse."""
    ahead = PENDING.model_copy(
        update={"erase_after": datetime.now(UTC) + timedelta(minutes=5)}
    )

    await request.send_confirmation(ahead, dry_run=False)

    assert email.send.await_args.kwargs["now"] >= ahead.erase_after


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
async def test_the_claimed_request_is_marked_done_once(requests_table):
    """The row this erasure claimed, not any row of the user ID; a rerun
    keeps the first date."""
    report = await request.complete_request(PENDING, dry_run=False)

    where = requests_table.update_many.await_args.kwargs["where"]
    assert where == {
        "id": REQUEST_ID,
        "startedAt": {"not": None},
        "completedAt": None,
    }
    assert report.counts == {"completed": 1}


@pytest.mark.asyncio
async def test_an_unclaimed_request_is_never_marked_done(requests_table):
    unclaimed = PENDING.model_copy(update={"request_id": None})

    with pytest.raises(RuntimeError):
        await request.complete_request(unclaimed, dry_run=False)

    requests_table.update_many.assert_not_awaited()


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


@pytest.mark.asyncio
async def test_an_email_request_is_closed_the_way_settings_closes_one(monkeypatch):
    monkeypatch.setattr(
        request, "get_user_by_id", AsyncMock(return_value=SimpleNamespace(email=EMAIL))
    )
    close = AsyncMock(return_value=NOW + timedelta(days=14))
    monkeypatch.setattr(request, "delete_account", close)

    erase_after = await request.close_for_email_request(USER_ID)

    close.assert_awaited_once_with(USER_ID, EMAIL, AccountDeletionSource.EMAIL)
    assert erase_after == NOW + timedelta(days=14)
