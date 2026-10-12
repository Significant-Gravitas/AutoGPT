"""The CLI warm-up reads the whole binary, runs it, and never fails the executor."""

import threading
from unittest.mock import MagicMock, patch

from backend.copilot.executor import cli_warmup


def test_reads_every_byte_of_the_binary(tmp_path):
    binary = tmp_path / "claude"
    binary.write_bytes(b"x" * (cli_warmup._CHUNK_BYTES * 2 + 5))

    assert cli_warmup._read_through(str(binary)) == binary.stat().st_size


def test_reads_the_binary_then_runs_it(tmp_path):
    binary = tmp_path / "claude"
    binary.write_bytes(b"cli")
    run = MagicMock(return_value=MagicMock(returncode=0))

    with patch("backend.copilot.executor.cli_warmup.subprocess.run", run):
        cli_warmup.warm_cli(str(binary))

    run.assert_called_once()
    assert run.call_args.args[0] == [str(binary), "-v"]


def test_a_missing_binary_is_skipped_without_raising(tmp_path):
    run = MagicMock()

    with patch("backend.copilot.executor.cli_warmup.subprocess.run", run):
        cli_warmup.warm_cli(str(tmp_path / "missing"))

    run.assert_not_called()


def test_the_configured_cli_path_wins(monkeypatch):
    monkeypatch.setenv("CHAT_CLAUDE_AGENT_CLI_PATH", "/opt/claude")
    monkeypatch.setenv("CLAUDE_AGENT_CLI_PATH", "/usr/bin/claude")

    assert cli_warmup._resolve_cli_path() == "/opt/claude"


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
