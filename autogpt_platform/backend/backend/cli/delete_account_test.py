"""The delete-account command: a dry run unless --apply, a dry run first even
with it, blockers stop it before anything goes, a partial deletion fails the
command, and nothing it prints carries the email."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import click
import pytest
from click.testing import CliRunner

from backend.cli import delete_account as cli
from backend.data import db
from backend.data.account_deletion import routine
from backend.data.account_deletion.models import (
    AccountDeletionBlocked,
    Blocker,
    DeletionReport,
    StepReport,
    StepStatus,
)

USER_ID = "3f0b8c1e-4c2a-4f7e-9a51-0c6f2d8e1a77"
EMAIL = "sam@example.com"


def _report(*, dry_run: bool, failed: bool = False, blocked: bool = False):
    status = StepStatus.WOULD_DO if dry_run else StepStatus.DONE
    return DeletionReport(
        user_id=USER_ID,
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
    deleting = AsyncMock(
        side_effect=lambda user_id, *, dry_run, **kw: _report(dry_run=dry_run)
    )
    monkeypatch.setattr(routine, "delete_account", deleting)

    def invoke(*args: str):
        return CliRunner().invoke(cli.delete_account_command, [EMAIL, *args])

    return SimpleNamespace(invoke=invoke, deleting=deleting)


def _dry_runs(deleting: AsyncMock) -> list[bool]:
    return [call.kwargs["dry_run"] for call in deleting.await_args_list]


def test_without_apply_it_is_a_dry_run(run):
    result = run.invoke()

    assert result.exit_code == 0
    assert _dry_runs(run.deleting) == [True]
    assert "Dry run: nothing was deleted" in result.output
    assert "objects=3" in result.output
    assert EMAIL not in result.output


def test_apply_runs_the_dry_run_first_then_deletes(run):
    result = run.invoke("--apply", "--yes")

    assert result.exit_code == 0, result.output
    assert _dry_runs(run.deleting) == [True, False]
    assert "Done." in result.output


def test_apply_asks_first_and_stops_on_no(run):
    result = CliRunner().invoke(
        cli.delete_account_command, [EMAIL, "--apply"], input="n\n"
    )

    assert result.exit_code == 1
    assert _dry_runs(run.deleting) == [True]


def test_blockers_stop_an_apply_before_anything_goes(run):
    run.deleting.side_effect = lambda user_id, *, dry_run, **kw: _report(
        dry_run=dry_run, blocked=True
    )

    result = run.invoke("--apply", "--yes")

    assert result.exit_code == 1
    assert _dry_runs(run.deleting) == [True]
    assert "BLOCKED shared_orgs_without_another_owner: 1" in result.output


def test_a_blocker_that_appears_after_the_dry_run_stops_too(run):
    def deleting(user_id, *, dry_run, **kw):
        if dry_run:
            return _report(dry_run=True)
        raise AccountDeletionBlocked(_report(dry_run=False, blocked=True))

    run.deleting.side_effect = deleting

    result = run.invoke("--apply", "--yes")

    assert result.exit_code == 1
    assert "nothing deleted" in result.output


def test_a_partial_deletion_fails_the_command(run):
    run.deleting.side_effect = lambda user_id, *, dry_run, **kw: _report(
        dry_run=dry_run, failed=not dry_run
    )

    result = run.invoke("--apply", "--yes")

    assert result.exit_code == 1
    assert "mailerlite" in result.output and "failed" in result.output
    assert "run it again to resume" in result.output


def test_shared_agent_removal_is_passed_through(run):
    run.invoke("--apply", "--yes", "--remove-shared-agents")

    for call in run.deleting.await_args_list:
        assert call.kwargs["remove_shared_agents"] is True


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
