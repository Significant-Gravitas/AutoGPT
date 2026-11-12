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
from backend.util.exceptions import EmailUndeliverableError

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
    calls = SimpleNamespace(
        send=AsyncMock(), mark=AsyncMock(), skipped=AsyncMock(return_value=False)
    )
    monkeypatch.setattr(request, "send_account_deleted_email", calls.send)
    monkeypatch.setattr(request, "mark_confirmation_sent", calls.mark)
    monkeypatch.setattr(request, "_skipped_since_loaded", calls.skipped)
    return calls


@pytest.mark.asyncio
async def test_a_skip_marked_after_the_run_loaded_the_account_still_holds(email):
    """The job loaded the account, then an operator's --no-email marked the
    request: the row is read again just before sending, and nothing goes."""
    email.skipped.return_value = True

    report = await request.send_confirmation(PENDING, dry_run=False)

    email.skipped.assert_awaited_once_with(USER_ID)
    email.send.assert_not_awaited()
    email.mark.assert_not_awaited()
    assert report.counts == {"emails": 0}


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
async def test_an_address_the_mail_provider_refuses_does_not_stop_the_erasure(email):
    """A bounced or suppressed address never becomes deliverable, so
    retrying would hold the erasure forever: it goes on without the email,
    which is not recorded as sent."""
    email.send.side_effect = EmailUndeliverableError("refused (error 406)")

    report = await request.send_confirmation(PENDING, dry_run=False)

    assert report.status == StepStatus.DONE
    assert report.counts == {"emails": 0}
    assert "won't deliver" in report.note
    email.mark.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("dry_run", [True, False])
async def test_a_stack_with_no_mail_provider_sends_nothing(email, monkeypatch, dry_run):
    monkeypatch.setattr(
        request,
        "Settings",
        lambda: SimpleNamespace(
            config=SimpleNamespace(account_erasure_unused_services=["postmark"])
        ),
    )

    report = await request.send_confirmation(PENDING, dry_run=dry_run)

    assert report.counts == {"emails": 0}
    email.send.assert_not_awaited()
    email.mark.assert_not_awaited()


@pytest.mark.asyncio
async def test_without_an_address_there_is_no_one_to_send_it_to(email):
    """Not a skip: a skip means a service this stack does not run."""
    no_address = PENDING.model_copy(update={"email": None})

    report = await request.send_confirmation(no_address, dry_run=False)

    assert report.status == StepStatus.DONE
    assert report.counts == {"emails": 0}
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

    close.assert_awaited_once_with(
        USER_ID, EMAIL, AccountDeletionSource.EMAIL, send_confirmation=True
    )
    assert erase_after == NOW + timedelta(days=14)


@pytest.mark.asyncio
async def test_an_email_request_to_be_erased_at_once_has_no_14_days(monkeypatch):
    monkeypatch.setattr(
        request, "get_user_by_id", AsyncMock(return_value=SimpleNamespace(email=EMAIL))
    )
    close = AsyncMock(return_value=NOW)
    monkeypatch.setattr(request, "delete_account", close)

    await request.close_for_email_request(USER_ID, erase_now=True)

    close.assert_awaited_once_with(
        USER_ID,
        EMAIL,
        AccountDeletionSource.EMAIL,
        grace=timedelta(0),
        send_confirmation=True,
    )


@pytest.mark.asyncio
async def test_an_email_request_closed_without_the_email_sends_none(monkeypatch):
    """--no-email: closing sends no confirmation and records that on the
    request, so the erasure sends none either."""
    monkeypatch.setattr(
        request, "get_user_by_id", AsyncMock(return_value=SimpleNamespace(email=EMAIL))
    )
    close = AsyncMock(return_value=NOW)
    monkeypatch.setattr(request, "delete_account", close)

    await request.close_for_email_request(USER_ID, erase_now=True, send_email=False)

    assert close.await_args.kwargs["send_confirmation"] is False


@pytest.mark.asyncio
async def test_closing_without_the_email_goes_only_with_erase_now(monkeypatch):
    close = AsyncMock()
    monkeypatch.setattr(request, "delete_account", close)

    with pytest.raises(ValueError):
        await request.close_for_email_request(USER_ID, send_email=False)

    close.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_account_closed_already_can_skip_the_late_confirmation(
    requests_table,
):
    """Only a request whose email did not go out and that is not erased."""
    assert await request.skip_confirmation(USER_ID) is True

    kwargs = requests_table.update_many.await_args.kwargs
    assert kwargs["where"] == {
        "userId": USER_ID,
        "confirmationSentAt": None,
        "completedAt": None,
    }
    assert kwargs["data"] == {"confirmationSkipped": True}

    requests_table.update_many.return_value = 0
    assert await request.skip_confirmation(USER_ID) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("dry_run", [True, False])
async def test_a_request_closed_without_the_email_gets_no_late_one(
    monkeypatch, dry_run
):
    sent = AsyncMock()
    monkeypatch.setattr(request, "send_account_deleted_email", sent)
    marked = AsyncMock()
    monkeypatch.setattr(request, "mark_confirmation_sent", marked)
    subject = PENDING.model_copy(update={"confirmation_skipped": True})

    report = await request.send_confirmation(subject, dry_run=dry_run)

    assert report.counts == {"emails": 0}
    assert "--no-email" in report.note
    sent.assert_not_awaited()
    marked.assert_not_awaited()


@pytest.mark.asyncio
async def test_ending_the_14_days_dates_them_a_minute_back(claim_query):
    """On the database's clock, a minute back, so the CLI's dry run finds the
    account due even on a host whose clock is a little behind; never later
    than the date the request had, and never for an erasure under way."""
    assert await request.end_grace_now(USER_ID) is True

    sql, user_id = claim_query.await_args.args
    assert user_id == USER_ID
    assert (
        'SET "eraseAfter" = LEAST("eraseAfter", '
        "(now() - interval '1 minute') AT TIME ZONE 'UTC')" in sql
    )
    assert '"startedAt" IS NULL AND "completedAt" IS NULL' in sql
