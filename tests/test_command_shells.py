"""What each shell is handed for EXECUTE_COMMAND's `command`.

On Windows cmd.exe is the default shell, and it got its command as one argv
item. subprocess joins an argv into a Windows command line by the Microsoft
C runtime's rules (list2cmdline), which write every `"` as `\\"`, and cmd.exe
reads its command line itself and takes `\\` for no escape: `dir "C:\\Program
Files"` ran as `dir \\"C:\\Program Files\\"`. It now gets one command line,
built here, with the command in it as the platform wrote it.

The tests of the real Windows shells run only on Windows, so in CI.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import autogpt_local_executor.handlers as handlers_module
from autogpt_local_executor import platform_info
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.handlers import CommandHandler
from autogpt_local_executor.protocol import (
    CommandResultMessage,
    CommandResultPayload,
    ExecuteCommandMessage,
    ExecuteCommandPayload,
    Shell,
    new_id,
    now_ts,
)

CMD_EXE = r"C:\Windows\System32\cmd.exe"
# What a command has to come through as written: quotes, a quoted path with a
# space, cmd.exe's own escape and its special characters, a `\"`, backslashes
# at the end and before a quote, and nothing special at all.
COMMANDS = [
    'dir "C:\\Program Files"',
    'echo "a b"',
    'type "C:\\a folder\\a b.txt"',
    "echo one&& echo two",
    "echo hello| findstr hello",
    "echo a^&b",
    "echo %SHIM_TEST_VALUE%",
    "echo plain",
    'echo \\"',
    "dir C:\\",
    'copy "C:\\a b\\" "D:\\c d\\"',
    '"C:\\Program Files\\tool.exe" "an argument"',
    "",
]


def _config(tmp_path: Path) -> ShimConfig:
    return ShimConfig(
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "_audit.log",
        platform_url="http://localhost:9999",
        machine_id="test-machine",
        enable_shell=True,
        # A shell's first start on a busy CI machine can take seconds.
        command_timeout_seconds=300,
    )


def _message(command: str, tmp_path: Path, shell: Shell, **payload: Any) -> ExecuteCommandMessage:
    return ExecuteCommandMessage(
        id=new_id(),
        ts=now_ts(),
        payload=ExecuteCommandPayload(command=command, cwd=str(tmp_path), shell=shell, **payload),
    )


async def _run(command: str, tmp_path: Path, shell: Shell, **payload: Any) -> CommandResultPayload:
    """Run `command` in `shell` through the handler; its result."""
    response = await CommandHandler(_config(tmp_path)).handle(
        _message(command, tmp_path, shell, **payload)
    )
    assert isinstance(response, CommandResultMessage), response
    assert response.payload.timed_out is False
    return response.payload


def _after_slash_s(command_line: str) -> str:
    """What cmd.exe runs for a command line with /s, by `cmd /?`: it takes
    off the first quote after /c and the last quote on the line, and keeps
    the rest as it is."""
    switches, _, rest = command_line.partition(" /c ")
    assert switches.endswith(" /d /s"), command_line
    assert rest.startswith('"'), command_line
    last = rest.rindex('"')
    return rest[1:last] + rest[last + 1 :]


# ── The command line that is built ───────────────────────────────────────────


def test_the_cmd_exe_command_line_holds_the_command_as_written() -> None:
    line = handlers_module._cmd_exe_command_line(CMD_EXE, 'dir "C:\\Program Files"')

    assert line == CMD_EXE + ' /d /s /c "chcp 65001 >NUL & dir "C:\\Program Files""'


@pytest.mark.parametrize("command", COMMANDS)
def test_cmd_exe_runs_exactly_the_command_it_was_sent(command: str) -> None:
    line = handlers_module._cmd_exe_command_line(CMD_EXE, command)

    assert _after_slash_s(line) == "chcp 65001 >NUL & " + command


@pytest.mark.parametrize("command", [c for c in COMMANDS if '"' in c or c.endswith("\\")])
def test_an_argv_item_did_not_carry_the_command_as_written(command: str) -> None:
    """What went wrong: as an argv item, a command's quotes came out as
    `\\"`, and a backslash at its end was doubled. cmd.exe took the outer
    quotes off and ran the rest."""
    sent = "chcp 65001 >NUL & " + command
    as_argv = subprocess.list2cmdline([CMD_EXE, "/C", sent])
    _, _, rest = as_argv.partition(" /C ")

    assert rest.startswith('"') and rest.endswith('"')
    assert rest[1:-1] != sent


def test_a_cmd_exe_path_with_a_space_is_quoted_on_its_command_line() -> None:
    line = handlers_module._cmd_exe_command_line(r"C:\Other Tools\cmd.exe", "echo hi")

    assert line == '"C:\\Other Tools\\cmd.exe" /d /s /c "chcp 65001 >NUL & echo hi"'


# ── What the handler starts, on any OS ───────────────────────────────────────


class _FakeProc:
    returncode = 0
    pid = 12345

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"", b""


def _as_on_windows(
    monkeypatch: pytest.MonkeyPatch, shell: tuple[str, list[str]]
) -> tuple[list[dict[str, Any]], list[tuple[Any, ...]]]:
    """Have the handler act as on Windows with `shell` resolved, and record
    what it starts: command lines, and argv lists."""
    command_lines: list[dict[str, Any]] = []
    argvs: list[tuple[Any, ...]] = []

    async def start_with_command_line(command_line: str, **kwargs: Any) -> _FakeProc:
        command_lines.append({"command_line": command_line, **kwargs})
        return _FakeProc()

    async def create_subprocess_exec(*argv: Any, **kwargs: Any) -> _FakeProc:
        argvs.append(argv)
        return _FakeProc()

    monkeypatch.setattr(
        handlers_module, "_start_with_command_line", start_with_command_line, raising=False
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    monkeypatch.setattr(handlers_module.platform_info, "detect_platform", lambda: "windows")
    monkeypatch.setattr(handlers_module.platform_info, "resolve_shell", lambda _shell: shell)
    return command_lines, argvs


@pytest.mark.parametrize("shell", [Shell.AUTO, Shell.CMD])
@pytest.mark.parametrize("command", COMMANDS)
async def test_cmd_exe_is_started_with_one_command_line_not_an_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shell: Shell, command: str
) -> None:
    command_lines, argvs = _as_on_windows(monkeypatch, (CMD_EXE, ["/C"]))

    await _run(command, tmp_path, shell)

    assert argvs == [], "cmd.exe got its command as an argv item, for subprocess to quote"
    [started] = command_lines
    assert started["command_line"] == f'{CMD_EXE} /d /s /c "chcp 65001 >NUL & {command}"'
    assert started["executable"] == CMD_EXE
    assert Path(started["cwd"]) == tmp_path.resolve()
    assert started["env"]["PYTHONIOENCODING"] == "utf-8"
    assert started["stdin"] == asyncio.subprocess.DEVNULL
    assert started["stdout"] == started["stderr"] == asyncio.subprocess.PIPE


@pytest.mark.parametrize(
    ("shell", "resolved"),
    [
        (Shell.POWERSHELL, ("powershell.exe", ["-NoLogo", "-NoProfile", "-Command"])),
        (Shell.PWSH, ("pwsh", ["-NoLogo", "-NoProfile", "-Command"])),
        (Shell.BASH, (r"C:\Program Files\Git\bin\bash.exe", ["-c"])),
    ],
)
@pytest.mark.parametrize("command", COMMANDS)
async def test_the_other_windows_shells_get_the_command_as_one_argument(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shell: Shell,
    resolved: tuple[str, list[str]],
    command: str,
) -> None:
    """PowerShell and bash read their command line as the C runtime does, so
    what subprocess quotes for them they read back whole: one argument, the
    command as written."""
    command_lines, argvs = _as_on_windows(monkeypatch, resolved)

    await _run(command, tmp_path, shell)

    assert command_lines == []
    assert argvs == [(resolved[0], *resolved[1], command)]


def test_the_event_loop_still_starts_a_process_from_a_command_line() -> None:
    """The cmd.exe command line goes to the event loop's own way of starting
    a process, which asyncio's public calls wrap: this is the shape it has
    had from Python 3.4 on. A Python that changes it fails here first."""
    from asyncio import base_events

    start = base_events.BaseEventLoop._make_subprocess_transport
    parameters = list(inspect.signature(start).parameters)

    assert inspect.iscoroutinefunction(start)
    assert parameters[:8] == [
        "self",
        "protocol",
        "args",
        "shell",
        "stdin",
        "stdout",
        "stderr",
        "bufsize",
    ]
    assert inspect.signature(start).parameters["kwargs"].kind is inspect.Parameter.VAR_KEYWORD


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX takes the whole string as argv[0]")
async def test_a_process_is_started_from_one_string_and_read_like_any_other(
    tmp_path: Path,
) -> None:
    """The start cmd.exe gets on Windows, run for real where there is no
    cmd.exe: the string is all of the command line there is, the program is
    the one named beside it, and the process is read and waited for as one
    from create_subprocess_exec is."""
    shell = shutil.which("sh")
    assert shell is not None

    proc = await handlers_module._start_with_command_line(
        "the whole string",
        executable=shell,
        cwd=str(tmp_path),
        env={"PATH": "/usr/bin:/bin"},
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    # sh, started as "the whole string" with no arguments, reads its script
    # from stdin, which is empty.
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

    assert isinstance(proc, asyncio.subprocess.Process)
    assert proc.returncode == 0
    assert (stdout, stderr) == (b"", b"")


# ── The real shells, on Windows ──────────────────────────────────────────────

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="the Windows shells")


@pytest.fixture
def spaced(tmp_path: Path) -> Path:
    """A file whose folder and name both hold a space, holding `inside`."""
    path = tmp_path / "a folder" / "a b.txt"
    path.parent.mkdir()
    path.write_bytes(b"inside")
    return path


@windows_only
@pytest.mark.parametrize("shell", [Shell.AUTO, Shell.CMD])
async def test_cmd_exe_reads_a_file_by_a_quoted_path_with_spaces(
    tmp_path: Path, spaced: Path, shell: Shell
) -> None:
    result = await _run(f'type "{spaced}"', tmp_path, shell)

    assert (result.stdout, result.exit_code) == ("inside", 0), result.stderr


@windows_only
async def test_cmd_exe_echoes_a_quoted_string_as_cmd_exe_does(tmp_path: Path) -> None:
    result = await _run('echo "a b"', tmp_path, Shell.CMD)

    # cmd.exe's echo writes its argument as it is, quotes and all.
    assert (result.stdout, result.exit_code) == ('"a b"\r\n', 0)


@windows_only
async def test_cmd_exe_does_not_read_an_argv_item_back_as_sent(tmp_path: Path) -> None:
    """What went wrong, on the real cmd.exe: the same command as an argv
    item, which is how `command` used to reach it and how an `argv` that
    names cmd.exe still does. subprocess writes its quotes as `\\"`, and
    cmd.exe runs that as it stands."""
    resolved = platform_info.resolve_shell("cmd")
    assert resolved is not None
    response = await CommandHandler(_config(tmp_path)).handle(
        ExecuteCommandMessage(
            id=new_id(),
            ts=now_ts(),
            payload=ExecuteCommandPayload(
                argv=[resolved[0], "/d", "/c", 'echo "a b"'], cwd=str(tmp_path)
            ),
        )
    )

    assert isinstance(response, CommandResultMessage), response
    assert response.payload.stdout == '\\"a b\\"\r\n'


@windows_only
async def test_cmd_exe_was_given_exactly_the_command_line_that_was_built(tmp_path: Path) -> None:
    """%CMDCMDLINE% is the command line cmd.exe was started with."""
    command = "echo %CMDCMDLINE%"
    resolved = platform_info.resolve_shell("cmd")
    assert resolved is not None

    result = await _run(command, tmp_path, Shell.CMD)

    assert result.exit_code == 0
    assert result.stdout == handlers_module._cmd_exe_command_line(resolved[0], command) + "\r\n"
    assert ' /d /s /c "chcp 65001 >NUL & echo %CMDCMDLINE%"' in result.stdout


@windows_only
@pytest.mark.parametrize(
    ("command", "stdout", "exit_code"),
    [
        ("echo one&& echo two", "one\r\ntwo\r\n", 0),
        # The second runs only when the first succeeds.
        ('dir /b "no such file here" >NUL 2>&1 && echo two', "", 1),
        ("echo hello| findstr hello", "hello\r\n", 0),
        ("echo a^&b", "a&b\r\n", 0),
        ("echo %SHIM_TEST_VALUE%", "from the request\r\n", 0),
        ("echo plain", "plain\r\n", 0),
        # A quoted program, and a quoted argument after it.
        (f'"{sys.executable}" -c "print(40 + 2)"', "42\r\n", 0),
    ],
    ids=["and-then", "and-then-after-a-failure", "pipe", "caret", "variable", "plain", "program"],
)
async def test_cmd_exe_runs_the_command_as_cmd_exe_reads_it(
    tmp_path: Path, command: str, stdout: str, exit_code: int
) -> None:
    result = await _run(command, tmp_path, Shell.CMD, env={"SHIM_TEST_VALUE": "from the request"})

    assert (result.stdout, result.exit_code) == (stdout, exit_code)


@windows_only
async def test_a_cmd_exe_command_runs_in_its_cwd_with_its_environment(tmp_path: Path) -> None:
    folder = tmp_path / "in here"
    folder.mkdir()

    result = await _run(
        "cd & echo %SHIM_TEST_VALUE%",
        folder,
        Shell.CMD,
        env={"SHIM_TEST_VALUE": "from the request"},
    )

    where, value = result.stdout.splitlines()
    assert Path(where).resolve() == folder.resolve()
    assert value == "from the request"


@windows_only
async def test_a_cmd_exe_command_is_in_a_job_and_its_stop_takes_what_it_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The job a Windows command runs in, and the stop that ends the job,
    hold for a command cmd.exe runs as they do for an argv."""
    monkeypatch.setattr(handlers_module, "_WINDOWS_BREAK_GRACE_SECONDS", 0.5)
    held, ready = tmp_path / "held", tmp_path / "ready"
    script = tmp_path / "hold it.py"
    script.write_text(
        "import os, signal, time\n"
        "signal.signal(signal.SIGBREAK, signal.SIG_IGN)\n"
        f"held = open({str(held)!r}, 'w')\n"
        f"open({str(ready)!r} + '.tmp', 'w').close()\n"
        f"os.replace({str(ready)!r} + '.tmp', {str(ready)!r})\n"
        "time.sleep(600)\n",
        encoding="utf-8",
    )
    handler = CommandHandler(_config(tmp_path))
    task = asyncio.create_task(
        handler.handle(_message(f'"{sys.executable}" "{script}"', tmp_path, Shell.CMD))
    )
    try:
        await _until(ready.exists)
        assert not _released(held)
        assert len(handlers_module._command_jobs) == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        await _until(lambda: _released(held))
        assert not handlers_module._command_jobs
    finally:
        if not task.done():
            task.cancel()


@windows_only
@pytest.mark.parametrize("shell", [Shell.POWERSHELL, Shell.PWSH])
@pytest.mark.parametrize(
    ("command", "stdout"),
    [
        ('Write-Output "a b"', "a b\r\n"),
        ("Write-Output 'say \"hi\"'", 'say "hi"\r\n'),
        ("Write-Output one; Write-Output two", "one\r\ntwo\r\n"),
        ("Write-Output hello | ForEach-Object { $_.ToUpper() }", "HELLO\r\n"),
        ("Write-Output $env:SHIM_TEST_VALUE", "from the request\r\n"),
        ("Write-Output plain", "plain\r\n"),
        ('Write-Output "C:\\a b\\"', "C:\\a b\\\r\n"),
    ],
    ids=["quoted", "quote-inside", "two", "pipe", "variable", "plain", "backslash-then-quote"],
)
async def test_powershell_runs_the_command_as_it_was_sent(
    tmp_path: Path, shell: Shell, command: str, stdout: str
) -> None:
    if platform_info.resolve_shell(shell.value) is None:
        pytest.skip(f"{shell.value} isn't installed")

    result = await _run(command, tmp_path, shell, env={"SHIM_TEST_VALUE": "from the request"})

    assert (result.stdout, result.exit_code) == (stdout, 0), result.stderr


@windows_only
@pytest.mark.parametrize("shell", [Shell.POWERSHELL, Shell.PWSH])
async def test_powershell_reads_a_file_by_a_quoted_path_with_spaces(
    tmp_path: Path, spaced: Path, shell: Shell
) -> None:
    if platform_info.resolve_shell(shell.value) is None:
        pytest.skip(f"{shell.value} isn't installed")

    result = await _run(f'Get-Content "{spaced}"', tmp_path, shell)

    assert (result.stdout.strip(), result.exit_code) == ("inside", 0), result.stderr


@functools.lru_cache(maxsize=1)
def _git_bash() -> str | None:
    """A bash this Windows machine runs commands with, of the kind Git for
    Windows installs (MSYS): the one the shim would find, else Git's own.
    The first `bash` on PATH is often WSL's launcher, which runs nothing
    until a distribution is installed, and reads paths another way."""
    found = platform_info.resolve_shell("bash")
    candidates = [found[0]] if found else []
    candidates += [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
    ]
    for bash in candidates:
        try:
            ran = subprocess.run(
                [bash, "-c", "echo $OSTYPE"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if ran.returncode == 0 and ran.stdout.strip() in (b"msys", b"cygwin"):
            return bash
    return None


@windows_only
@pytest.mark.parametrize(
    ("command", "stdout"),
    [
        ('echo "a b"', "a b\n"),
        ("echo 'say \"hi\"'", 'say "hi"\n'),
        ("echo one && echo two", "one\ntwo\n"),
        ('echo hello | while read -r line; do echo "${line^^}"; done', "HELLO\n"),
        ('echo "$SHIM_TEST_VALUE"', "from the request\n"),
        ("echo plain", "plain\n"),
    ],
    ids=["quoted", "quote-inside", "and-then", "pipe", "variable", "plain"],
)
async def test_bash_on_windows_runs_the_command_as_it_was_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, stdout: str
) -> None:
    bash = _git_bash()
    if bash is None:
        pytest.skip("no Git bash here")
    monkeypatch.setattr(
        handlers_module.platform_info, "resolve_shell", lambda _shell: (bash, ["-c"])
    )

    result = await _run(command, tmp_path, Shell.BASH, env={"SHIM_TEST_VALUE": "from the request"})

    assert (result.stdout.replace("\r\n", "\n"), result.exit_code) == (stdout, 0), result.stderr


@windows_only
async def test_bash_on_windows_reads_a_file_by_a_quoted_path_with_spaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spaced: Path
) -> None:
    bash = _git_bash()
    if bash is None:
        pytest.skip("no Git bash here")
    monkeypatch.setattr(
        handlers_module.platform_info, "resolve_shell", lambda _shell: (bash, ["-c"])
    )

    # The file's text, read by bash itself.
    result = await _run(f'echo "$(< "{spaced.as_posix()}")"', tmp_path, Shell.BASH)

    assert (result.stdout.replace("\r\n", "\n"), result.exit_code) == ("inside\n", 0), result.stderr


async def _until(condition: Any, seconds: float = 120) -> None:
    deadline = asyncio.get_running_loop().time() + seconds
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "timed out waiting"
        await asyncio.sleep(0.05)


def _released(held: Path) -> bool:
    """Whether no process holds `held` open any more: Windows refuses to
    delete a file another process has open."""
    try:
        held.unlink()
    except FileNotFoundError:
        return True
    except PermissionError:
        return False
    return True
