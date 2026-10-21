"""The erase-account command: a dry run unless --apply, a dry run first even
with it, blockers stop it before anything goes, a partial erasure fails the
command, and nothing it prints carries the email. --close closes an account
the way Settings does before erasing it, and --pending erases every closed
account the same way."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import click
import pytest
from click.testing import CliRunner

from backend.api.features.account_deletion.service import PlanCancellationError
from backend.cli import erase_account as cli
from backend.data import account_deletion, db
from backend.data.account_erasure import request, routine, sweep
from backend.data.account_erasure.models import (
    AccountErasureBlocked,
    Blocker,
    ErasureReport,
    StepReport,
    StepStatus,
)

USER_ID = "3f0b8c1e-4c2a-4f7e-9a51-0c6f2d8e1a77"
EMAIL = "sam@example.com"


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


@pytest.fixture
def run(monkeypatch):
    """The command against a fake routine and no database."""
    monkeypatch.setattr(db, "connect", AsyncMock())
    monkeypatch.setattr(db, "disconnect", AsyncMock())
    monkeypatch.setattr(cli, "_resolve", AsyncMock(return_value=(USER_ID, EMAIL)))
    closed = AsyncMock(return_value=True)
    monkeypatch.setattr(account_deletion, "is_account_closed", closed)
    erasing = AsyncMock(
        side_effect=lambda user_id, *, dry_run, **kw: _report(dry_run=dry_run)
    )
    monkeypatch.setattr(routine, "erase_account", erasing)
    closing = AsyncMock()
    monkeypatch.setattr(request, "close_for_email_request", closing)

    def invoke(*args: str, input: str | None = None):
        return CliRunner().invoke(cli.erase_account_command, [*args], input=input)

    return SimpleNamespace(
        invoke=invoke, erasing=erasing, closed=closed, closing=closing
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


def test_close_closes_an_open_account_before_erasing_it(run):
    run.closed.return_value = False

    result = run.invoke(EMAIL, "--close", "--apply", "--yes")

    assert result.exit_code == 0, result.output
    run.closing.assert_awaited_once_with(USER_ID, allow_banned=False)
    # The dry run is not refused for the request the close will write.
    assert run.erasing.await_args_list[0].kwargs["closing"] is True
    assert run.erasing.await_args_list[0].kwargs["allow_banned"] is False
    assert "closes it as a request made by email" in result.output


def test_allow_banned_reaches_the_dry_run_and_the_close(run):
    run.closed.return_value = False

    result = run.invoke(EMAIL, "--close", "--allow-banned", "--apply", "--yes")

    assert result.exit_code == 0, result.output
    assert run.erasing.await_args_list[0].kwargs["allow_banned"] is True
    run.closing.assert_awaited_once_with(USER_ID, allow_banned=True)


def test_a_ban_that_appears_after_the_dry_run_erases_nothing(run):
    run.closed.return_value = False
    run.closing.side_effect = request.AccountBannedError()

    result = run.invoke(EMAIL, "--close", "--apply", "--yes")

    assert result.exit_code == 1
    assert _dry_runs(run.erasing) == [True]
    assert "nothing was closed or erased" in result.output


def test_close_without_apply_closes_nothing(run):
    run.closed.return_value = False

    result = run.invoke(EMAIL, "--close")

    assert result.exit_code == 0, result.output
    run.closing.assert_not_awaited()


def test_close_on_a_closed_account_only_erases(run):
    result = run.invoke(EMAIL, "--close", "--apply", "--yes")

    assert result.exit_code == 0, result.output
    run.closing.assert_not_awaited()
    assert run.erasing.await_args_list[0].kwargs["closing"] is False
    assert "closed already" in result.output


def test_a_stripe_refusal_on_close_erases_nothing(run):
    run.closed.return_value = False
    run.closing.side_effect = PlanCancellationError()

    result = run.invoke(EMAIL, "--close", "--apply", "--yes")

    assert result.exit_code == 1
    assert _dry_runs(run.erasing) == [True]
    assert "still open" in result.output


@pytest.mark.parametrize(
    "args",
    [(), (EMAIL, "--pending"), ("--pending", "--close"), (EMAIL, "--allow-banned")],
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
    return SimpleNamespace(plans=plans, each=each)


def test_pending_without_apply_lists_what_would_go(run, pending):
    result = run.invoke("--pending")

    assert result.exit_code == 0, result.output
    assert "1 of 2 closed accounts can be erased" in result.output
    pending.each.assert_not_awaited()


def test_pending_apply_erases_the_unblocked_ones_from_the_dry_run(run, pending):
    result = run.invoke("--pending", "--apply", "--yes")

    assert result.exit_code == 0, result.output
    pending.each.assert_awaited_once_with(["a"], dry_run=False)


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
