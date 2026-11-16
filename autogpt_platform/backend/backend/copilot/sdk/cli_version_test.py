"""``require_verbatim_prompts_cli`` refuses a CLI older than the floor and
returns the binary it checked."""

import logging
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from claude_agent_sdk import CLINotFoundError
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

from backend.copilot.sdk.cli_version import (
    parse_cli_version,
    require_verbatim_prompts_cli,
)

_RUN = "backend.copilot.sdk.cli_version.subprocess.run"


@pytest.fixture
def cli(tmp_path: Path) -> str:
    """A file standing in for the CLI binary; ``subprocess.run`` is patched."""
    path = tmp_path / "claude"
    path.write_text("binary")
    return str(path)


def _probe(stdout: str, returncode: int = 0) -> MagicMock:
    return MagicMock(return_value=MagicMock(stdout=stdout, returncode=returncode))


@pytest.mark.parametrize(
    "stdout",
    [
        "2.1.248",
        "2.1.284 (Claude Code)",
        "2.1.284 (Claude Code)\n",
        "2.1.284 (Claude Code)\nsecond line",
        "3.0.0",
    ],
)
def test_supported_cli_is_accepted_and_its_path_returned(cli: str, stdout: str):
    with patch(_RUN, _probe(stdout)) as run:
        checked = require_verbatim_prompts_cli(cli)

    assert checked == os.path.realpath(cli)
    assert run.call_args.args[0] == [checked, "-v"]
    assert run.call_args.kwargs["stdin"] is subprocess.DEVNULL
    assert run.call_args.kwargs["encoding"] == "utf-8"
    assert run.call_args.kwargs["errors"] == "replace"
    assert run.call_args.kwargs["timeout"] == 10


@pytest.mark.parametrize("stdout", ["2.1.247 (Claude Code)", "2.0.999", "1.9.300"])
def test_older_cli_is_refused(cli: str, stdout: str):
    with (
        patch(_RUN, _probe(stdout)),
        pytest.raises(RuntimeError, match="older than 2.1.248"),
    ):
        require_verbatim_prompts_cli(cli)


@pytest.mark.parametrize(
    ("stdout", "returncode"),
    [
        ("", 0),
        ("Claude Code", 0),
        ("2.1.284", 1),
        ("2.1.300-beta.1 (Claude Code)", 0),
        ("2.1.300-dev", 0),
        ("2.1.248evil", 0),
        ("2.1.248.0", 0),
        ("v2.1.284", 0),
        ("2.1.284 (Other)", 0),
        ("2.1.284 (Claude Code) extra", 0),
        ("warning: something\n2.1.284 (Claude Code)", 0),
        ("2.1.284�", 0),
    ],
)
def test_unknown_version_output_is_refused(cli: str, stdout: str, returncode: int):
    with (
        patch(_RUN, _probe(stdout, returncode)),
        pytest.raises(RuntimeError, match="Could not read the version"),
    ):
        require_verbatim_prompts_cli(cli)


@pytest.mark.parametrize(
    "error", [OSError("no such file"), subprocess.TimeoutExpired(["claude"], 10)]
)
def test_cli_that_will_not_run_is_refused_without_its_directory(
    cli: str, error: Exception
):
    with (
        patch(_RUN, MagicMock(side_effect=error)),
        pytest.raises(RuntimeError, match="Could not run") as raised,
    ):
        require_verbatim_prompts_cli(cli)

    assert os.path.dirname(cli) not in str(raised.value)


def test_missing_cli_is_refused(tmp_path: Path):
    with (
        patch(_RUN) as run,
        pytest.raises(RuntimeError, match="Could not run"),
    ):
        require_verbatim_prompts_cli(str(tmp_path / "claude"))

    run.assert_not_called()


@pytest.mark.parametrize(
    ("stdout", "returncode", "error"),
    [
        ("2.1.247", 0, None),
        ("garbage", 0, None),
        ("", 0, subprocess.TimeoutExpired(["claude"], 10)),
    ],
)
def test_each_refusal_is_logged_once_with_the_path(
    cli: str,
    caplog: pytest.LogCaptureFixture,
    stdout: str,
    returncode: int,
    error: Exception | None,
):
    run = MagicMock(
        side_effect=error, return_value=MagicMock(stdout=stdout, returncode=returncode)
    )
    with (
        patch(_RUN, run),
        caplog.at_level(logging.ERROR, logger="backend.copilot.sdk.cli_version"),
        pytest.raises(RuntimeError),
    ):
        require_verbatim_prompts_cli(cli)

    assert len(caplog.records) == 1
    assert os.path.realpath(cli) in caplog.records[0].getMessage()


def test_without_a_path_the_cli_the_sdk_finds_is_checked(cli: str):
    with (
        patch.object(SubprocessCLITransport, "_find_cli", return_value=cli),
        patch(_RUN, _probe("2.1.284")) as run,
    ):
        checked = require_verbatim_prompts_cli(None)

    assert checked == os.path.realpath(cli)
    assert run.call_args.args[0] == [checked, "-v"]


def test_without_a_path_and_no_cli_to_find_the_sdk_error_is_raised():
    with (
        patch.object(
            SubprocessCLITransport,
            "_find_cli",
            side_effect=CLINotFoundError("Claude Code not found"),
        ),
        patch(_RUN) as run,
        pytest.raises(CLINotFoundError),
    ):
        require_verbatim_prompts_cli(None)

    run.assert_not_called()


def test_the_cli_found_first_stays_the_one_returned(cli: str, tmp_path: Path):
    """A ``PATH`` change after the first check does not swap the binary."""
    later = tmp_path / "later-claude"
    later.write_text("another binary")
    find = MagicMock(side_effect=[cli, str(later)])
    with (
        patch.object(SubprocessCLITransport, "_find_cli", find),
        patch(_RUN, _probe("2.1.284")) as run,
    ):
        first = require_verbatim_prompts_cli(None)
        second = require_verbatim_prompts_cli(None)

    assert first == second == os.path.realpath(cli)
    assert find.call_count == 1
    assert run.call_count == 1


def test_a_swapped_symlink_returns_and_probes_its_new_target(tmp_path: Path):
    first, second = tmp_path / "claude-a", tmp_path / "claude-b"
    first.write_text("binary a")
    second.write_text("binary b, longer")
    link = tmp_path / "claude"
    try:
        link.symlink_to(first)
    except OSError:
        pytest.skip("this account cannot create symlinks")
    run = MagicMock(
        side_effect=[
            MagicMock(stdout="2.1.284", returncode=0),
            MagicMock(stdout="2.1.247", returncode=0),
        ]
    )
    with patch(_RUN, run):
        assert require_verbatim_prompts_cli(link) == os.path.realpath(first)
        link.unlink()
        link.symlink_to(second)
        with pytest.raises(RuntimeError, match="older than"):
            require_verbatim_prompts_cli(link)

    assert [c.args[0][0] for c in run.call_args_list] == [
        os.path.realpath(first),
        os.path.realpath(second),
    ]


def test_each_binary_is_probed_once(cli: str, tmp_path: Path):
    other = tmp_path / "other-claude"
    other.write_text("binary")
    with patch(_RUN, _probe("2.1.284")) as run:
        require_verbatim_prompts_cli(cli)
        require_verbatim_prompts_cli(cli)
        require_verbatim_prompts_cli(str(other))

    assert run.call_count == 2


def test_a_failed_probe_is_tried_again(cli: str):
    run = MagicMock(
        side_effect=[
            subprocess.TimeoutExpired(["claude"], 10),
            MagicMock(stdout="2.1.284", returncode=0),
        ]
    )
    with patch(_RUN, run):
        with pytest.raises(RuntimeError):
            require_verbatim_prompts_cli(cli)
        require_verbatim_prompts_cli(cli)

    assert run.call_count == 2


def test_a_binary_replaced_in_place_is_probed_again(cli: str):
    run = MagicMock(
        side_effect=[
            MagicMock(stdout="2.1.284", returncode=0),
            MagicMock(stdout="2.1.247", returncode=0),
        ]
    )
    with patch(_RUN, run):
        require_verbatim_prompts_cli(cli)
        Path(cli).write_text("a different binary")
        with pytest.raises(RuntimeError, match="older than"):
            require_verbatim_prompts_cli(cli)

    assert run.call_count == 2


def test_a_binary_that_changes_while_it_is_probed_is_refused_and_not_kept(cli: str):
    def replace_during_probe(*_args, **_kwargs):
        Path(cli).write_text("replaced while running -v")
        return MagicMock(stdout="2.1.284", returncode=0)

    with patch(_RUN, MagicMock(side_effect=replace_during_probe)):
        with pytest.raises(RuntimeError, match="changed while"):
            require_verbatim_prompts_cli(cli)
    with patch(_RUN, _probe("2.1.284")) as run:
        require_verbatim_prompts_cli(cli)

    run.assert_called_once()


@pytest.mark.parametrize(
    ("output", "version"),
    [
        ("2.1.284", (2, 1, 284)),
        ("2.1.284 (Claude Code)", (2, 1, 284)),
        ("2.1.284 (Claude Code)\r\nnext line", (2, 1, 284)),
        ("\n2.1.284\n", (2, 1, 284)),
        ("2.1.300-beta.1 (Claude Code)", None),
        ("Update available\n2.1.284 (Claude Code)", None),
        ("v2.1.284", None),
        ("2.1.284 (Claude Code) extra", None),
        ("2.1.284\ufffd", None),
        ("", None),
    ],
)
def test_the_version_is_read_from_an_exact_first_line(output: str, version):
    assert parse_cli_version(output) == version
