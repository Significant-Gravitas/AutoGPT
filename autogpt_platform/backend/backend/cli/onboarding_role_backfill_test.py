"""The role backfill's dry run reports reliable against skipped accounts and
where each one would go, by count only, and a run that could not write
everyone fails as a whole after reporting."""

import click
import pytest
from click.testing import CliRunner

from backend.cli import onboarding_role_backfill as cli
from backend.notifications import role_backfill
from backend.notifications.role_backfill import RoleRecord


# The root conftest spins a full test server for every test via an autouse
# session fixture; these never touch it.
@pytest.fixture(scope="session")
def server():
    yield None


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    yield


def test_the_dry_run_report_counts_reliable_and_skipped(capsys):
    records = [
        RoleRecord(user_id="1", email="kept@x.io", choice="Marketing"),
        RoleRecord(user_id="2", email="exact@x.io", understanding_role="Marketing"),
        RoleRecord(user_id="3", email="typed@x.io", understanding_role="Dentist"),
        RoleRecord(
            user_id="4",
            email="sam@firma.ru",
            understanding_role="Engineering",
        ),
    ]
    cli._report(
        role_backfill.plan(records, {"kept@x.io": {}, "exact@x.io": {"role": "x"}})
    )
    out = capsys.readouterr().out
    assert "Accounts with a role on record: 4" in out
    assert "reliable, kept by the wizard: 1" in out
    assert "reliable, an exact option in the understanding: 2" in out
    assert "skipped, Other's text or an AutoPilot rewrite: 1" in out
    assert "Roles: Marketing=2, Engineering=1" in out
    assert "PostHog: 3 persons" in out
    assert "MailerLite: 1 subscribers to fill in" in out
    assert "already have a role (kept): 1" in out
    assert "placed in Iran or Russia (skipped): 1" in out
    assert "@x.io" not in out and "firma.ru" not in out


def test_a_complete_run_reports_and_succeeds(capsys):
    cli._finish(5, 0, 2)
    out = capsys.readouterr().out
    assert "5 ok, 0 failed, 2 no longer needed" in out
    assert "Run the dry run again" in out


def test_the_command_exits_non_zero_when_anyone_failed(monkeypatch):
    async def run(*, apply: bool, yes: bool) -> None:
        cli._finish(4, 1, 0)

    monkeypatch.setattr(cli, "_run", run)
    result = CliRunner().invoke(
        cli.onboarding_role_backfill_command, ["--apply", "--yes"]
    )
    assert result.exit_code == 1
    assert "4 ok, 1 failed" in result.output
    assert "1 MailerLite subscribers were not written" in result.output


def test_a_run_with_failures_reports_then_fails():
    with pytest.raises(click.ClickException, match="2 MailerLite subscribers"):
        cli._finish(1, 2, 0)
