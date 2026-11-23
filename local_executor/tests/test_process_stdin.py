"""Nothing the shim starts reads the shim's own standard input.

A command inherited it. Started in a terminal, as the README shows, the shim
handed every command that terminal: one that prompts (`pip uninstall`, a bare
`python`) waited on a prompt nobody saw until its timeout, and whatever the
person typed in that window went to the command. Under launchd or systemd the
same command ran straight through. A command's stdin is now the null device,
however the shim was started.

Every other process the package starts names its stdin too: a pipe it uses,
or the null device. The check below reads the source for one that doesn't.
(The sign-in's `webbrowser.open` is the standard library's own start, in a
command the person runs at their terminal.)
"""

from __future__ import annotations

import ast
import contextlib
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import autogpt_local_executor.handlers as handlers_module
from autogpt_local_executor import platform_info
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.handlers import CommandHandler
from autogpt_local_executor.protocol import (
    CommandResultMessage,
    ExecuteCommandMessage,
    ExecuteCommandPayload,
    Shell,
    new_id,
    now_ts,
)

ROOT = Path(__file__).resolve().parent.parent
TYPED = b"typed in the shim's terminal\n"
# A program that reads one line of its stdin and shows what it got.
READ_A_LINE = "import sys; print(repr(sys.stdin.readline()))"


@contextlib.contextmanager
def _own_stdin_is_a_pipe_with_data_waiting() -> Iterator[None]:
    """Make this process's standard input a pipe with a line in it and its
    other end still open, as a terminal someone has typed a line into is: a
    process that inherits it reads the line, and would wait for the next."""
    read_end, write_end = os.pipe()
    saved = os.dup(0)
    try:
        os.write(write_end, TYPED)
        os.dup2(read_end, 0)
        yield
    finally:
        os.dup2(saved, 0)
        for descriptor in (saved, read_end, write_end):
            os.close(descriptor)


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


async def _stdout_of(tmp_path: Path, **payload: Any) -> str:
    """What a command wrote, run while this process's stdin is a pipe with
    a line waiting in it."""
    with _own_stdin_is_a_pipe_with_data_waiting():
        response = await CommandHandler(_config(tmp_path)).handle(
            ExecuteCommandMessage(
                id=new_id(),
                ts=now_ts(),
                payload=ExecuteCommandPayload(cwd=str(tmp_path), **payload),
            )
        )
    assert isinstance(response, CommandResultMessage), response
    assert response.payload.timed_out is False
    assert response.payload.exit_code == 0, response.payload.stderr
    return response.payload.stdout.replace("\r\n", "\n")


# ── A command's stdin ────────────────────────────────────────────────────────


async def test_an_argv_command_reads_the_end_of_input_at_once(tmp_path: Path) -> None:
    stdout = await _stdout_of(tmp_path, argv=[sys.executable, "-c", READ_A_LINE])

    assert stdout == "''\n"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shells")
@pytest.mark.parametrize("shell", [Shell.AUTO, Shell.BASH, Shell.SH, Shell.ZSH])
async def test_a_posix_shell_command_reads_the_end_of_input_at_once(
    tmp_path: Path, shell: Shell
) -> None:
    if platform_info.resolve_shell(shell.value) is None:
        pytest.skip(f"{shell.value} isn't installed")

    stdout = await _stdout_of(
        tmp_path, command='IFS= read -r line; echo "read=$? line=[$line]"', shell=shell
    )

    # `read` fails at the end of input, having read nothing.
    assert stdout == "read=1 line=[]\n"


windows_only = pytest.mark.skipif(sys.platform != "win32", reason="the Windows shells")


@windows_only
@pytest.mark.parametrize("shell", [Shell.AUTO, Shell.CMD])
async def test_a_cmd_exe_command_reads_the_end_of_input_at_once(
    tmp_path: Path, shell: Shell
) -> None:
    # What cmd.exe starts has cmd.exe's own stdin.
    stdout = await _stdout_of(
        tmp_path, command=f'"{sys.executable}" -c "{READ_A_LINE}"', shell=shell
    )

    assert stdout == "''\n"


@windows_only
@pytest.mark.parametrize("shell", [Shell.POWERSHELL, Shell.PWSH])
async def test_a_powershell_command_reads_the_end_of_input_at_once(
    tmp_path: Path, shell: Shell
) -> None:
    if platform_info.resolve_shell(shell.value) is None:
        pytest.skip(f"{shell.value} isn't installed")

    stdout = await _stdout_of(
        tmp_path,
        command=(
            "$line = [Console]::In.ReadLine(); "
            "if ($null -eq $line) { 'end of input' } else { \"line=[$line]\" }"
        ),
        shell=shell,
    )

    assert stdout == "end of input\n"


@windows_only
async def test_a_bash_command_on_windows_reads_the_end_of_input_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_command_shells import _git_bash

    bash = _git_bash()
    if bash is None:
        pytest.skip("no Git bash here")
    monkeypatch.setattr(
        handlers_module.platform_info, "resolve_shell", lambda _shell: (bash, ["-c"])
    )

    stdout = await _stdout_of(
        tmp_path, command='IFS= read -r line; echo "read=$? line=[$line]"', shell=Shell.BASH
    )

    assert stdout == "read=1 line=[]\n"


# ── Every process start names its stdin ──────────────────────────────────────

# The calls that start a process and take `stdin=`, by module.
TAKE_STDIN = {
    "subprocess": {"run", "Popen", "call", "check_call", "check_output"},
    "asyncio": {"create_subprocess_exec", "create_subprocess_shell"},
}
# The ones that can't be told what stdin to give, so can't be used here.
INHERIT_STDIN = {
    "subprocess": {"getoutput", "getstatusoutput"},
    "os": {
        "system",
        "popen",
        "startfile",
        "posix_spawn",
        "posix_spawnp",
        *(f"spawn{suffix}" for suffix in ("l", "le", "lp", "lpe", "v", "ve", "vp", "vpe")),
        *(f"exec{suffix}" for suffix in ("l", "le", "lp", "lpe", "v", "ve", "vp", "vpe")),
    },
}
# An event loop's own starts, by the method's name, and the package's own
# (handlers._start_with_command_line). `_make_subprocess_transport` takes its
# stdin fourth, after the protocol, the arguments and `shell`.
LOOP_STARTS = {"subprocess_exec", "subprocess_shell"}
OWN_STARTS = {"_start_with_command_line"}
BY_POSITION = {"_make_subprocess_transport": 3}


def _starts_without_stdin(tree: ast.Module) -> Iterator[tuple[int, str]]:
    """Each call in `tree` that starts a process without saying what its
    stdin is, as its line and the call: through the module or by a name
    imported from it, under whatever name either was imported as."""
    modules: dict[str, str] = {}
    names: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("subprocess", "asyncio", "os", "asyncio.subprocess"):
                    module = "asyncio" if alias.name == "asyncio.subprocess" else alias.name
                    modules[alias.asname or alias.name] = module
        elif isinstance(node, ast.ImportFrom) and node.module in (
            "subprocess",
            "asyncio",
            "asyncio.subprocess",
            "os",
        ):
            module = "asyncio" if node.module == "asyncio.subprocess" else node.module
            for alias in node.names:
                if alias.name == "subprocess" and module == "asyncio":
                    modules[alias.asname or alias.name] = "asyncio"
                else:
                    names[alias.asname or alias.name] = (module, alias.name)
    # asyncio's starts are also reached as asyncio.subprocess.<name>.
    modules.update(
        {f"{name}.subprocess": "asyncio" for name, module in modules.items() if module == "asyncio"}
    )

    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        module: str | None = None
        if isinstance(func, ast.Attribute):
            name = func.attr
            module = modules.get(ast.unparse(func.value))
        elif isinstance(func, ast.Name):
            module, name = names.get(func.id, (None, func.id))
        else:
            continue
        shown = f"{ast.unparse(func)}(...)"
        if module is not None and name in INHERIT_STDIN.get(module, ()):
            yield call.lineno, shown
            continue
        takes_stdin = (
            (module is not None and name in TAKE_STDIN.get(module, ()))
            or (isinstance(func, ast.Attribute) and name in LOOP_STARTS)
            or name in OWN_STARTS
        )
        keywords = {keyword.arg for keyword in call.keywords}
        if takes_stdin:
            # `input=` makes stdin a pipe the call itself writes and closes.
            if not keywords & {"stdin", "input"}:
                yield call.lineno, shown
        elif name in BY_POSITION:
            if "stdin" not in keywords and len(call.args) <= BY_POSITION[name]:
                yield call.lineno, shown


def test_every_process_the_package_starts_is_told_its_stdin() -> None:
    unnamed = [
        f"{path.relative_to(ROOT).as_posix()}:{line}: {call}"
        for path in sorted((ROOT / "autogpt_local_executor").rglob("*.py"))
        for line, call in _starts_without_stdin(ast.parse(path.read_bytes(), str(path)))
    ]
    assert not unnamed, (
        "Say what the process's stdin is (`stdin=subprocess.DEVNULL` unless it is written "
        "to): left out, it is the shim's own, a terminal when the shim was started in one:\n"
        + "\n".join(unnamed)
    )


def test_the_check_reaches_the_starts_the_package_has() -> None:
    """The starts the check has to see: were it to miss one of these it
    would miss its like."""
    started = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "autogpt_local_executor").rglob("*.py")
        for node in ast.walk(ast.parse(path.read_bytes(), str(path)))
        if isinstance(node, ast.Call) and any(keyword.arg == "stdin" for keyword in node.keywords)
    }

    assert {
        "autogpt_local_executor/handlers.py",
        "autogpt_local_executor/install.py",
        "autogpt_local_executor/doctor.py",
        "autogpt_local_executor/platform_info.py",
        "autogpt_local_executor/audit.py",
        "autogpt_local_executor/computer_use/mcp_client.py",
        "autogpt_local_executor/computer_use/concealed.py",
        "autogpt_local_executor/recording/consent.py",
    } <= started


def test_the_check_finds_a_start_without_stdin_under_any_name() -> None:
    source = """\
import asyncio
import os
import subprocess
import subprocess as sp
from asyncio import create_subprocess_shell as shell
from asyncio import subprocess as aio_subprocess
from subprocess import Popen, run as execute
subprocess.run(argv)
sp.check_output(argv, stderr=sp.DEVNULL)
Popen(argv, stdout=sp.PIPE)
execute(argv, **options)
os.system("ls")
subprocess.getoutput("ls")
async def start(loop, protocol):
    await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE)
    await asyncio.subprocess.create_subprocess_exec(*argv)
    await aio_subprocess.create_subprocess_exec(*argv)
    await shell("ls")
    await loop.subprocess_exec(factory, *argv)
    await loop._make_subprocess_transport(protocol, argv, False)
    await _start_with_command_line(line, executable=program)
    # Each of these says what stdin is.
    subprocess.run(argv, stdin=subprocess.DEVNULL)
    subprocess.run(argv, input=b"")
    await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.PIPE, **options)
    await loop._make_subprocess_transport(protocol, argv, False, stdin, stdout, stderr, 0)
    await _start_with_command_line(line, executable=program, stdin=stdin)
    # And these start nothing.
    os.path.join("a", "b")
    asyncio.run(main())
    runner.run(argv)
"""
    found = sorted(line for line, _call in _starts_without_stdin(ast.parse(source)))

    assert found == [8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21]
