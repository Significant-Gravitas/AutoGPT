"""The erase-account command: a dry run unless --apply, a dry run first even
with it, blockers stop it before anything goes, a partial erasure fails the
command, and nothing it prints carries the email. --close only closes an
account the way Settings does (its owner then has 14 days to keep it), and
--pending erases every closed account whose 14 days are over."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import click
import pytest
from click.testing import CliRunner

from backend.api.features.account_deletion.service import PlanCancellationError
from backend.cli import erase_account as cli
from backend.data import account_deletion, db
from backend.data.account_erasure import request, routine, sweep
from backend.data.account_erasure.deleted_user import DELETED_USER_ID
from backend.data.account_erasure.models import (
    AccountErasureBlocked,
    Blocker,
    ErasureReport,
    StepReport,
    StepStatus,
)

USER_ID = "3f0b8c1e-4c2a-4f7e-9a51-0c6f2d8e1a77"
EMAIL = "sam@example.com"
ERASE_AFTER = datetime(2026, 11, 16, 9, 30, tzinfo=UTC)


def _report(
    *,
    dry_run: bool,
    failed: bool = False,
    blocked: bool = False,
    user_id: str = USER_ID,
):
    status = StepStatus.WOULD_DO if dry_run else StepStatus.DONE
    return ErasureReport(
        user_id=user_id,
        dry_run=dry_run,
        blockers=(
            [Blocker(name="shared_orgs_without_another_owner", count=1, resolution="r")]
            if blocked
            else []
        ),
        steps=[
            StepReport(name="files", status=status, counts={"objects": 3}),
            StepReport(
                name="mailerlite",
                status=StepStatus.FAILED if failed else status,
                note="MailerLiteError" if failed else "",
            ),
        ],
    )


def _run_on_a_private_loop(coro):
    """What the command's `asyncio.run` does, without leaving the thread with
    no event loop for the async tests after these."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def run(monkeypatch):
    """The command against a fake routine and no database."""
    monkeypatch.setattr(cli.asyncio, "run", _run_on_a_private_loop)
    monkeypatch.setattr(db, "connect", AsyncMock())
    monkeypatch.setattr(db, "disconnect", AsyncMock())
    monkeypatch.setattr(cli, "_resolve", AsyncMock(return_value=(USER_ID, EMAIL)))
    closed = AsyncMock(return_value=SimpleNamespace(eraseAfter=ERASE_AFTER))
    monkeypatch.setattr(account_deletion, "get_deletion_request", closed)
    erasing = AsyncMock(
        side_effect=lambda user_id, *, dry_run, **kw: _report(dry_run=dry_run)
    )
    monkeypatch.setattr(routine, "erase_account", erasing)
    closing = AsyncMock(return_value=ERASE_AFTER)
    monkeypatch.setattr(request, "close_for_email_request", closing)
    ending = AsyncMock(return_value=True)
    monkeypatch.setattr(request, "end_grace_now", ending)

    def invoke(*args: str, input: str | None = None):
        return CliRunner().invoke(cli.erase_account_command, [*args], input=input)

    return SimpleNamespace(
        invoke=invoke, erasing=erasing, closed=closed, closing=closing, ending=ending
    )


def _dry_runs(erasing: AsyncMock) -> list[bool]:
    return [call.kwargs["dry_run"] for call in erasing.await_args_list]


def test_without_apply_it_is_a_dry_run(run):
    result = run.invoke(EMAIL)

    assert result.exit_code == 0
    assert _dry_runs(run.erasing) == [True]
    assert "Dry run: nothing was changed" in result.output
    assert "objects=3" in result.output
    assert EMAIL not in result.output


def test_apply_runs_the_dry_run_first_then_erases(run):
    result = run.invoke(EMAIL, "--apply", "--yes")

    assert result.exit_code == 0, result.output
    assert _dry_runs(run.erasing) == [True, False]
    run.closing.assert_not_awaited()
    assert "Done." in result.output


def test_apply_asks_first_and_stops_on_no(run):
    result = run.invoke(EMAIL, "--apply", input="n\n")

    assert result.exit_code == 1
    assert _dry_runs(run.erasing) == [True]


def test_blockers_stop_an_apply_before_anything_goes(run):
    run.erasing.side_effect = lambda user_id, *, dry_run, **kw: _report(
        dry_run=dry_run, blocked=True
    )

    result = run.invoke(EMAIL, "--apply", "--yes")

    assert result.exit_code == 1
    assert _dry_runs(run.erasing) == [True]
    assert "BLOCKED shared_orgs_without_another_owner: 1" in result.output


def test_a_blocker_that_appears_after_the_dry_run_stops_too(run):
    def erasing(user_id, *, dry_run, **kw):
        if dry_run:
            return _report(dry_run=True)
        raise AccountErasureBlocked(_report(dry_run=False, blocked=True))

    run.erasing.side_effect = erasing

    result = run.invoke(EMAIL, "--apply", "--yes")

    assert result.exit_code == 1
    assert "nothing erased" in result.output


def test_a_partial_erasure_fails_the_command(run):
    run.erasing.side_effect = lambda user_id, *, dry_run, **kw: _report(
        dry_run=dry_run, failed=not dry_run
    )

    result = run.invoke(EMAIL, "--apply", "--yes")

    assert result.exit_code == 1
    assert "mailerlite" in result.output and "failed" in result.output
    assert "run it again to resume" in result.output


def test_close_only_closes_an_open_account(run):
    """Its owner then has 14 days to keep it, as from Settings; the job
    erases it after that."""
    run.closed.return_value = None

    result = run.invoke(EMAIL, "--close", "--apply", "--yes")

    assert result.exit_code == 0, result.output
    run.closing.assert_awaited_once_with(USER_ID)
    run.erasing.assert_not_awaited()
    assert "can keep it until 2026-11-16 09:30 UTC" in result.output
    assert EMAIL not in result.output


def test_close_without_apply_closes_nothing(run):
    run.closed.return_value = None

    result = run.invoke(EMAIL, "--close")

    assert result.exit_code == 0, result.output
    run.closing.assert_not_awaited()
    assert "Dry run" in result.output


def test_close_on_a_closed_account_changes_nothing(run):
    result = run.invoke(EMAIL, "--close", "--apply", "--yes")

    assert result.exit_code == 0, result.output
    run.closing.assert_not_awaited()
    run.erasing.assert_not_awaited()
    assert "closed already" in result.output
    assert "2026-11-16 09:30 UTC" in result.output


def test_close_now_closes_with_no_14_days_and_erases_in_the_same_run(run):
    run.closed.return_value = None

    result = run.invoke(EMAIL, "--close", "--now", "--apply", "--yes")

    assert result.exit_code == 0, result.output
    run.closing.assert_awaited_once_with(USER_ID, erase_now=True)
    run.ending.assert_awaited_once_with(USER_ID)
    # A dry run for the blockers before closing, then the usual dry run and
    # the erasure.
    assert _dry_runs(run.erasing) == [True, True, False]
    assert EMAIL not in result.output


def test_close_now_on_a_closed_account_ends_its_14_days_and_erases(run):
    result = run.invoke(EMAIL, "--close", "--now", "--apply", "--yes")

    assert result.exit_code == 0, result.output
    run.closing.assert_not_awaited()
    run.ending.assert_awaited_once_with(USER_ID)
    assert _dry_runs(run.erasing) == [True, True, False]


def test_close_now_without_apply_changes_nothing(run):
    run.closed.return_value = None

    result = run.invoke(EMAIL, "--close", "--now")

    assert result.exit_code == 0, result.output
    run.closing.assert_not_awaited()
    run.ending.assert_not_awaited()
    assert _dry_runs(run.erasing) == [True]
    assert "Dry run" in result.output


@pytest.mark.parametrize("apply", [[], ["--apply", "--yes"]])
def test_close_now_on_a_blocked_account_closes_nothing(run, apply):
    """Closing first would leave the account closed, its owner unable to
    keep it, and not erased."""
    run.closed.return_value = None
    run.erasing.side_effect = lambda user_id, *, dry_run, **kw: _report(
        dry_run=dry_run, blocked=True
    )

    result = run.invoke(EMAIL, "--close", "--now", *apply)

    assert result.exit_code == 1
    assert "shared_orgs_without_another_owner" in result.output
    run.closing.assert_not_awaited()
    run.ending.assert_not_awaited()
    assert _dry_runs(run.erasing) == [True]


@pytest.mark.parametrize("apply", [[], ["--apply", "--yes"]])
def test_close_now_closes_nothing_when_the_erasure_would_stop(run, apply):
    """A step the dry run shows failing (no ban key, FalkorDB down) or a
    service not set up: closing first would leave the account closed and
    half erased until it is fixed."""
    run.closed.return_value = None
    run.erasing.side_effect = lambda user_id, *, dry_run, **kw: _report(
        dry_run=dry_run, failed=True
    )

    result = run.invoke(EMAIL, "--close", "--now", *apply)

    assert result.exit_code == 1
    assert "would stop at: mailerlite" in result.output
    run.closing.assert_not_awaited()
    run.ending.assert_not_awaited()


def test_close_now_goes_ahead_while_only_closing_stands_in_the_way(run):
    """An open account is "not closed" and a closed one "not due": those are
    what --close --now clears."""
    run.closed.return_value = None
    not_closed = Blocker(name="not_closed", count=0, resolution="r")
    run.erasing.side_effect = lambda user_id, *, dry_run, **kw: (
        _report(dry_run=True).model_copy(update={"blockers": [not_closed]})
        if run.closing.await_count == 0
        else _report(dry_run=dry_run)
    )

    result = run.invoke(EMAIL, "--close", "--now", "--apply", "--yes")

    assert result.exit_code == 0, result.output
    run.closing.assert_awaited_once_with(USER_ID, erase_now=True)
    assert _dry_runs(run.erasing) == [True, True, False]


@pytest.mark.parametrize("now", [[], ["--now"]])
@pytest.mark.parametrize("user_id", [DELETED_USER_ID, "../everyone"])
def test_close_refuses_the_placeholder_and_malformed_ids(
    run, monkeypatch, now, user_id
):
    monkeypatch.setattr(cli, "_resolve", AsyncMock(return_value=(user_id, None)))
    run.closed.return_value = None

    result = run.invoke(user_id, "--close", *now, "--apply", "--yes")

    assert result.exit_code == 1
    run.closing.assert_not_awaited()
    run.ending.assert_not_awaited()
    run.erasing.assert_not_awaited()


def test_a_stripe_refusal_on_close_leaves_the_account_open(run):
    run.closed.return_value = None
    run.closing.side_effect = PlanCancellationError()

    result = run.invoke(EMAIL, "--close", "--apply", "--yes")

    assert result.exit_code == 1
    run.erasing.assert_not_awaited()
    assert "still open" in result.output


def test_redis_down_is_a_clear_error_not_a_traceback(run):
    from backend.data.account_erasure.lease import ErasureLeaseUnavailable

    def erasing(user_id, *, dry_run, **kw):
        if dry_run:
            return _report(dry_run=True)
        raise ErasureLeaseUnavailable("account erasure lease unavailable")

    run.erasing.side_effect = erasing

    result = run.invoke(EMAIL, "--apply", "--yes")

    assert result.exit_code == 1
    assert "Redis can't be reached" in result.output
    assert "Traceback" not in result.output


def test_an_account_erased_already_is_a_success(run):
    """Rerunning a finished erasure (to confirm it, say) is not a failure."""

    def erasing(user_id, *, dry_run, **kw):
        if dry_run:
            return _report(dry_run=True)
        return ErasureReport(
            user_id=user_id, dry_run=False, not_claimed=routine.ERASED_ALREADY
        )

    run.erasing.side_effect = erasing

    result = run.invoke(EMAIL, "--apply", "--yes")

    assert result.exit_code == 0, result.output
    assert "nothing ran: erased already" in result.output


@pytest.mark.parametrize(
    "status, exit_code", [(StepStatus.DONE, 0), (StepStatus.FAILED, 1)]
)
def test_a_rerun_of_a_finished_erasure_reports_its_repeated_deletes(
    run, status, exit_code
):
    """A rerun repeats the PostHog delete; it fails the command only when
    that delete failed, so a leftover person is never read as gone."""

    def erasing(user_id, *, dry_run, **kw):
        if dry_run:
            return _report(dry_run=True)
        return ErasureReport(
            user_id=user_id,
            dry_run=False,
            not_claimed=routine.ERASED_ALREADY,
            steps=[StepReport(name="posthog", status=status, counts={"persons": 1})],
        )

    run.erasing.side_effect = erasing

    result = run.invoke(EMAIL, "--apply", "--yes")

    assert result.exit_code == exit_code, result.output
    assert "erased already: repeats only the PostHog delete" in result.output
    assert f"posthog           {status.value}" in result.output


def test_an_erasure_the_owner_kept_meanwhile_fails_the_command(run):
    def erasing(user_id, *, dry_run, **kw):
        if dry_run:
            return _report(dry_run=True)
        return ErasureReport(user_id=user_id, dry_run=False, not_claimed="kept")

    run.erasing.side_effect = erasing

    result = run.invoke(EMAIL, "--apply", "--yes")

    assert result.exit_code == 1
    assert "nothing ran: kept" in result.output


@pytest.mark.parametrize(
    "args", [(), (EMAIL, "--pending"), ("--pending", "--close"), (EMAIL, "--now")]
)
def test_conflicting_arguments_are_refused(run, args):
    result = run.invoke(*args)

    assert result.exit_code != 0
    run.erasing.assert_not_awaited()


@pytest.fixture
def pending(monkeypatch):
    plans = AsyncMock(
        return_value=[
            _report(dry_run=True, user_id="a"),
            _report(dry_run=True, user_id="b", blocked=True),
        ]
    )
    monkeypatch.setattr(sweep, "erase_pending", plans)
    each = AsyncMock(
        side_effect=lambda ids, *, dry_run: [
            _report(dry_run=dry_run, user_id=i) for i in ids
        ]
    )
    monkeypatch.setattr(sweep, "erase_each", each)
    purge = AsyncMock(return_value=2)
    monkeypatch.setattr(sweep, "purge_ban_records", purge)
    return SimpleNamespace(plans=plans, each=each, purge=purge)


def test_pending_without_apply_lists_what_would_go(run, pending):
    result = run.invoke("--pending")

    assert result.exit_code == 0, result.output
    assert "1 of 2 closed accounts can be erased" in result.output
    pending.each.assert_not_awaited()


def test_pending_apply_erases_the_unblocked_ones_from_the_dry_run(run, pending):
    result = run.invoke("--pending", "--apply", "--yes")

    assert result.exit_code == 0, result.output
    pending.each.assert_awaited_once_with(["a"], dry_run=False)
    pending.purge.assert_awaited_once_with()
    assert "Purged 2 expired ban record(s)" in result.output


def test_pending_purges_even_with_nothing_to_erase(run, pending):
    pending.plans.return_value = []

    assert run.invoke("--pending").exit_code == 0
    pending.purge.assert_not_awaited()
    result = run.invoke("--pending", "--apply", "--yes")

    assert result.exit_code == 0, result.output
    pending.purge.assert_awaited_once_with()


def test_pending_purges_even_when_every_account_is_blocked(run, pending):
    pending.plans.return_value = [_report(dry_run=True, user_id="b", blocked=True)]

    result = run.invoke("--pending", "--apply", "--yes")

    assert result.exit_code == 1
    assert "Every one is blocked" in result.output
    pending.purge.assert_awaited_once_with()
    pending.each.assert_not_awaited()


@pytest.fixture
def accounts(monkeypatch):
    users = AsyncMock(return_value=None)
    identities = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "prisma.models.User.prisma",
        lambda: SimpleNamespace(find_first=users),
    )
    monkeypatch.setattr(
        "prisma.models.AuthUser.prisma",
        lambda: SimpleNamespace(find_first=identities),
    )
    return SimpleNamespace(users=users, identities=identities)


@pytest.mark.asyncio
async def test_a_user_id_is_taken_as_it_is(accounts):
    assert await cli._resolve(USER_ID) == (USER_ID, None)
    accounts.users.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_email_finds_the_account_or_its_leftover_identity(accounts):
    accounts.identities.return_value = SimpleNamespace(id=USER_ID)

    assert await cli._resolve(EMAIL) == (USER_ID, EMAIL)
    where = accounts.users.await_args.kwargs["where"]
    assert where["email"]["mode"] == "insensitive"


@pytest.mark.asyncio
async def test_an_unknown_email_is_refused_without_echoing_it(accounts):
    with pytest.raises(click.ClickException) as raised:
        await cli._resolve(EMAIL)

    assert EMAIL not in raised.value.message
