"""A command that can't start says what is missing: its folder or its program.

The jail lets a command name a `cwd` that isn't there yet. Starting in one
fails, and the handler took every "not found" from the start for a missing
program: the platform was told SHELL_NOT_AVAILABLE, and told the model that
the shell isn't installed, when the folder was what was missing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from autogpt_local_executor import platform_info
from autogpt_local_executor.audit import AuditWriter
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.handlers import CommandHandler
from autogpt_local_executor.protocol import (
    CommandResultMessage,
    ErrorCode,
    ErrorMessage,
    ExecuteCommandMessage,
    ExecuteCommandPayload,
    new_id,
    now_ts,
)

# A command that runs anywhere, as argv and as a shell command.
ARGV = [sys.executable, "-c", "print('ran')"]
SHELL_COMMAND = "echo ran"


def _config(tmp_path: Path) -> ShimConfig:
    return ShimConfig(
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "_audit.log",
        platform_url="http://localhost:9999",
        machine_id="test-machine",
        enable_shell=True,
    )


def _command(how: str, **payload: Any) -> ExecuteCommandMessage:
    if how == "argv":
        payload = {"argv": ARGV, **payload}
    else:
        payload = {"command": SHELL_COMMAND, **payload}
    return ExecuteCommandMessage(id=new_id(), ts=now_ts(), payload=ExecuteCommandPayload(**payload))


def _audited(tmp_path: Path) -> list[dict[str, Any]]:
    log = tmp_path / "_audit.log"
    return [
        record
        for record in map(json.loads, log.read_text(encoding="utf-8").splitlines())
        if record["op"] == "EXECUTE_COMMAND"
    ]


@pytest.mark.parametrize("how", ["argv", "shell"])
async def test_a_command_in_a_folder_that_is_not_there_is_told_the_folder_is_missing(
    tmp_path: Path, how: str
) -> None:
    config = _config(tmp_path)
    handler = CommandHandler(config, audit=AuditWriter(config.audit_log_path, b"a" * 32))
    cwd = tmp_path / "not" / "made" / "yet"

    reply = await handler.handle(_command(how, cwd=str(cwd)))

    assert isinstance(reply, ErrorMessage)
    assert reply.payload.code == ErrorCode.PATH_NOT_FOUND
    # The folder as it was asked for, for the platform's own text.
    assert reply.payload.details == {"path": str(cwd), "reason": "cwd_not_found"}
    assert repr(str(cwd)) in reply.payload.message
    [record] = _audited(tmp_path)
    assert record["result"]["error_code"] == "PATH_NOT_FOUND"


@pytest.mark.parametrize("how", ["argv", "shell"])
async def test_a_command_whose_cwd_is_a_file_is_told_so(tmp_path: Path, how: str) -> None:
    """POSIX gives NotADirectoryError for this, which was answered as an
    internal error with the system's text."""
    a_file = tmp_path / "notes.txt"
    a_file.write_text("x", encoding="utf-8")

    reply = await CommandHandler(_config(tmp_path)).handle(_command(how, cwd=str(a_file)))

    assert isinstance(reply, ErrorMessage)
    assert reply.payload.code == ErrorCode.PATH_NOT_FOUND
    assert reply.payload.details == {"path": str(a_file), "reason": "cwd_not_a_directory"}
    assert repr(str(a_file)) in reply.payload.message
    assert "not a folder" in reply.payload.message


async def test_a_relative_cwd_is_named_as_it_was_asked(tmp_path: Path) -> None:
    """A relative `cwd` is taken from the shared folder. The reply names it
    as the request did, not where that comes to."""
    reply = await CommandHandler(_config(tmp_path)).handle(_command("argv", cwd="not/there"))

    assert isinstance(reply, ErrorMessage)
    assert reply.payload.code == ErrorCode.PATH_NOT_FOUND
    assert reply.payload.details == {"path": "not/there", "reason": "cwd_not_found"}
    assert str(tmp_path) not in reply.payload.message


async def test_a_program_that_is_not_there_is_still_a_missing_program(tmp_path: Path) -> None:
    """With the folder there, "not found" is the program, as before. It is
    named as the request named it."""
    reply = await CommandHandler(_config(tmp_path)).handle(
        ExecuteCommandMessage(
            id=new_id(),
            ts=now_ts(),
            payload=ExecuteCommandPayload(argv=["no-such-program-9f3a", "--version"]),
        )
    )

    assert isinstance(reply, ErrorMessage)
    assert reply.payload.code == ErrorCode.SHELL_NOT_AVAILABLE
    assert "no-such-program-9f3a" in reply.payload.message


async def test_a_shell_whose_program_is_gone_is_still_a_missing_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shell the shim knows of, whose program was removed since: the
    reply says which shell was asked for, for the platform's own text, and
    not where the shim looked for it."""
    gone = tmp_path / "removed" / "shell-program"
    monkeypatch.setattr(platform_info, "resolve_shell", lambda _name: (str(gone), ["-c"]))

    reply = await CommandHandler(_config(tmp_path)).handle(_command("shell", shell="bash"))

    assert isinstance(reply, ErrorMessage)
    assert reply.payload.code == ErrorCode.SHELL_NOT_AVAILABLE
    assert reply.payload.details == {"requested_shell": "bash"}
    assert "bash" in reply.payload.message
    assert str(gone) not in reply.payload.message
    assert str(tmp_path) not in reply.payload.message


async def test_a_shell_this_os_does_not_have_names_the_shell_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(platform_info, "resolve_shell", lambda _name: None)

    reply = await CommandHandler(_config(tmp_path)).handle(_command("shell", shell="zsh"))

    assert isinstance(reply, ErrorMessage)
    assert reply.payload.code == ErrorCode.SHELL_NOT_AVAILABLE
    assert reply.payload.details == {"requested_shell": "zsh"}


async def test_a_command_the_shim_would_not_run_without_a_job_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Windows a command that can't be put in a job of its own isn't run.
    That refusal is the shim's own, with no path in it, and reaches the
    platform as it is; any other failure to start is answered by its kind."""
    from autogpt_local_executor import windows_jobs

    async def no_job(*_args: Any, **_kwargs: Any) -> Any:
        raise windows_jobs.JobError(
            "the command was not run: it could not be put in a job of its own"
        )

    monkeypatch.setattr(CommandHandler, "_spawn_argv", staticmethod(no_job))

    reply = await CommandHandler(_config(tmp_path)).handle(_command("argv"))

    assert isinstance(reply, ErrorMessage)
    assert reply.payload.code == ErrorCode.INTERNAL_ERROR
    assert "job of its own" in reply.payload.message

    async def some_other_failure(*_args: Any, **_kwargs: Any) -> Any:
        raise PermissionError(13, "Permission denied", str(tmp_path / "somewhere"))

    monkeypatch.setattr(CommandHandler, "_spawn_argv", staticmethod(some_other_failure))

    reply = await CommandHandler(_config(tmp_path)).handle(_command("argv"))

    assert isinstance(reply, ErrorMessage)
    assert reply.payload.code == ErrorCode.INTERNAL_ERROR
    assert str(tmp_path) not in reply.payload.message
    assert "PermissionError" in reply.payload.message


@pytest.mark.parametrize("how", ["argv", "shell"])
async def test_a_command_in_a_folder_that_is_there_still_runs(tmp_path: Path, how: str) -> None:
    (tmp_path / "work").mkdir()

    reply = await CommandHandler(_config(tmp_path)).handle(
        _command(how, cwd=str(tmp_path / "work"))
    )

    assert isinstance(reply, CommandResultMessage)
    assert reply.payload.exit_code == 0
    assert "ran" in reply.payload.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="what Windows itself reports")
@pytest.mark.parametrize("what", ["missing", "a-file"])
def test_windows_reports_a_cwd_it_cannot_use_as_an_invalid_directory_name(
    tmp_path: Path, what: str
) -> None:
    """Why the handler looks at the folder itself, not at the error: on
    Windows a start in a folder that isn't there, or in a file, fails with
    error 267 (the directory name is invalid), a NotADirectoryError that
    names no file, where POSIX gives FileNotFoundError or NotADirectoryError
    with the folder as its file name."""
    cwd = tmp_path / "not-there"
    if what == "a-file":
        cwd.write_text("x", encoding="utf-8")

    with pytest.raises(OSError) as failed:
        subprocess.Popen(ARGV, cwd=str(cwd), stdin=subprocess.DEVNULL)

    assert (type(failed.value).__name__, failed.value.winerror, failed.value.filename) == (
        "NotADirectoryError",
        267,
        None,
    )
