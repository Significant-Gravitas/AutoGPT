"""A checkout backfill that could not write everyone fails as a whole, after
reporting, so a Job or script never reads a partial run as done."""

import click
import pytest
from click.testing import CliRunner

from backend.cli import mailerlite_backfill as cli


# The root conftest spins a full test server for every test via an autouse
# session fixture; these never touch it.
@pytest.fixture(scope="session")
def server():
    yield None


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    yield


def test_a_complete_run_reports_and_succeeds(capsys):
    cli._finish_checkout(5, 0, 2)
    out = capsys.readouterr().out
    assert "5 ok, 0 failed, 2 already up to date" in out
    assert "Run the dry run again" in out


def test_a_run_with_failures_reports_then_fails():
    with pytest.raises(
        click.ClickException, match="3 checkout openers were not written"
    ):
        cli._finish_checkout(5, 3, 0)


def test_the_command_exits_non_zero_when_anyone_failed(monkeypatch):
    async def run(*, apply: bool, yes: bool) -> None:
        cli._finish_checkout(4, 1, 0)

    monkeypatch.setattr(cli, "_run_checkout", run)
    result = CliRunner().invoke(
        cli.mailerlite_checkout_backfill_command, ["--apply", "--yes"]
    )
    assert result.exit_code == 1
    assert "4 ok, 1 failed" in result.output
    assert "1 checkout openers were not written" in result.output
