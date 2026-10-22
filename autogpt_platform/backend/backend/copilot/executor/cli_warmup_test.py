"""The CLI warm-up reads the whole binary, runs it, and never fails the executor."""

import logging
import os
import threading
from unittest.mock import MagicMock, patch

import pytest

from backend.copilot.executor import cli_warmup


def test_reads_every_byte_of_the_binary(tmp_path):
    binary = tmp_path / "claude"
    binary.write_bytes(b"x" * (cli_warmup._CHUNK_BYTES * 2 + 5))

    assert cli_warmup._read_through(str(binary)) == binary.stat().st_size


def test_reads_the_binary_then_runs_it(tmp_path, monkeypatch):
    binary = tmp_path / "claude"
    binary.write_bytes(b"cli")
    steps: list[tuple[str, object]] = []
    read_through = cli_warmup._read_through

    def read(path: str) -> int:
        steps.append(("read", path))
        return read_through(path)

    def run(args: list[str], **_kwargs: object) -> MagicMock:
        steps.append(("run", args))
        return MagicMock(returncode=0)

    monkeypatch.setattr(cli_warmup, "_read_through", read)
    monkeypatch.setattr(cli_warmup.subprocess, "run", run)

    cli_warmup.warm_cli(str(binary))

    assert steps == [("read", str(binary)), ("run", [str(binary), "-v"])]


def test_runs_the_binary_with_only_what_a_launch_keeps(tmp_path, monkeypatch):
    binary = tmp_path / "claude"
    binary.write_bytes(b"cli")
    monkeypatch.setenv("SOME_PLATFORM_SECRET", "sk-FAKE-platform-secret")
    run = MagicMock(return_value=MagicMock(returncode=0))

    with patch("backend.copilot.executor.cli_warmup.subprocess.run", run):
        cli_warmup.warm_cli(str(binary))

    env = run.call_args.kwargs["env"]
    assert "SOME_PLATFORM_SECRET" not in env
    assert env["PATH"] == os.environ["PATH"]


def test_a_failing_version_run_is_a_warning(tmp_path, caplog):
    binary = tmp_path / "claude"
    binary.write_bytes(b"cli")
    run = MagicMock(return_value=MagicMock(returncode=1))
    caplog.set_level(logging.INFO, logger=cli_warmup.__name__)

    with patch("backend.copilot.executor.cli_warmup.subprocess.run", run):
        cli_warmup.warm_cli(str(binary))

    [failed] = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "rc=1" in failed.getMessage()


@pytest.mark.parametrize("returncode", [0, 1])
def test_the_logs_name_the_binary_but_not_where_it_is(tmp_path, caplog, returncode):
    binary = tmp_path / "claude"
    binary.write_bytes(b"cli")
    run = MagicMock(return_value=MagicMock(returncode=returncode))
    caplog.set_level(logging.INFO, logger=cli_warmup.__name__)

    with patch("backend.copilot.executor.cli_warmup.subprocess.run", run):
        cli_warmup.warm_cli(str(binary))

    messages = [r.getMessage() for r in caplog.records]
    assert messages[-1].endswith(": claude")
    assert not any(str(tmp_path) in message for message in messages)


def test_a_missing_binary_is_skipped_without_raising(tmp_path):
    run = MagicMock()

    with patch("backend.copilot.executor.cli_warmup.subprocess.run", run):
        cli_warmup.warm_cli(str(tmp_path / "missing"))

    run.assert_not_called()


def test_the_configured_cli_path_wins(monkeypatch):
    monkeypatch.setenv("CHAT_CLAUDE_AGENT_CLI_PATH", "/opt/claude")
    monkeypatch.setenv("CLAUDE_AGENT_CLI_PATH", "/usr/bin/claude")

    assert cli_warmup._resolve_cli_path() == "/opt/claude"


def test_the_unprefixed_cli_path_is_used_on_its_own(monkeypatch):
    monkeypatch.delenv("CHAT_CLAUDE_AGENT_CLI_PATH", raising=False)
    monkeypatch.setenv("CLAUDE_AGENT_CLI_PATH", "/usr/bin/claude")

    assert cli_warmup._resolve_cli_path() == "/usr/bin/claude"


def test_startup_does_not_wait_for_the_warm_up():
    release = threading.Event()

    with patch(
        "backend.copilot.executor.cli_warmup.warm_cli",
        side_effect=lambda: release.wait(5),
    ):
        thread = cli_warmup.start_cli_warmup()
        assert thread.daemon
        assert thread.is_alive()
        release.set()
        thread.join(5)

    assert not thread.is_alive()
