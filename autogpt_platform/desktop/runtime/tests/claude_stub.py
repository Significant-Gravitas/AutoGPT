"""A stand-in for the Claude Code CLI, for tests.

    make(directory, logged_in=True)  ->  the path of a `claude` to run

It answers `-v` and `auth status` the way the real one does, and can
be told to answer wrongly: signed out, a failing exit code, or not at all.
Like the real one, `auth status` leaves a `.claude.json` in a home directory
that has none.

It has to be a real program. On Windows the runtime and the SDK both refuse
a `.cmd` or `.bat` (CreateProcess does not run them), so the stub there is a
copy of this interpreter under the name claude.exe: Python reads
`sitecustomize` on every start, and the one written beside the copy answers
in the interpreter's place. A `claude._pth` beside it says where the standard
library is, so the stub runs in any environment, as the real CLI does.
`--version` is the one thing the interpreter answers itself, before it reads
anything, so that build-only probe reports Python's version on Windows.
`-v` reaches STUB on every platform. Elsewhere the stub is a shell script
that runs STUB below.
"""

from __future__ import annotations

import atexit
import json
import os
import shlex
import shutil
import sys
import tempfile
from pathlib import Path

WINDOWS = sys.platform == "win32"
NAME = "claude.exe" if WINDOWS else "claude"
POSIX_VERSION = "2.1.284"

# Runs in the stub's process. `behaviour.json` beside the program says how
# to answer; the answers have the real CLI's shape, with an account that
# must never show up in a log or an event.
STUB = r"""
import json, os, sys, time

def answer(program, arguments):
    with open(os.path.join(os.path.dirname(program), "behaviour.json"), encoding="utf-8") as file:
        behaviour = json.load(file)
    with open(os.path.join(os.path.dirname(program), "calls.log"), "a", encoding="utf-8") as log:
        log.write(json.dumps({"arguments": arguments, "home": os.path.expanduser("~"),
                              "config": os.environ.get("CLAUDE_CONFIG_DIR"),
                              "api_key": os.environ.get("ANTHROPIC_API_KEY"),
                              "updater": os.environ.get("DISABLE_AUTOUPDATER")}) + "\n")
    time.sleep(behaviour.get("sleep", 0))
    if arguments in (["-v"], ["--version"]):
        print(behaviour.get("version_output", behaviour["version"] + " (Claude Code)"))
        sys.stdout.flush()
        os._exit(behaviour.get("version_exit", 0))
    elif arguments == ["auth", "status"]:
        open(os.path.join(os.path.expanduser("~"), ".claude.json"), "a").close()
        time.sleep(behaviour.get("auth_sleep", 0))
        with open(os.path.join(os.path.dirname(program), "calls.log"), encoding="utf-8") as log:
            asked = sum(json.loads(line)["arguments"] == arguments for line in log)
        if asked <= behaviour.get("silent_times", 0):
            os._exit(1)
        if behaviour.get("garbage"):
            print("not json")
        else:
            print(json.dumps({"loggedIn": behaviour["logged_in"], "authMethod": "claude.ai",
                              "email": "secret.person@example.com"}))
    sys.stdout.flush()
    os._exit(behaviour.get("exit", 0))
"""

SITECUSTOMIZE = (
    STUB
    + r"""
if os.path.exists(os.path.join(os.path.dirname(sys.executable), "behaviour.json")):
    answer(sys.executable, sys.orig_argv[1:])
"""
)


def make(directory: Path, **behaviour: object) -> Path:
    """A stub CLI in `directory`. Keywords: logged_in, exit, sleep (before any
    answer), auth_sleep (before that one), silent_times (how many times
    `auth status` exits without a word before it answers), garbage, version,
    version_output, version_exit. Windows ignores these for --version."""
    directory.mkdir(parents=True, exist_ok=True)
    cli = directory / NAME
    if WINDOWS:
        _copy_interpreter(cli)
        (directory / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
        # Named after the program: the interpreter then takes its path from
        # this file and from nothing in the environment.
        home = Path(sys.base_prefix)
        # The bundle's interpreter has its standard library in an archive
        # beside it (build/site_zip.py); any other has it in Lib.
        archive = home / f"python{sys.version_info.major}{sys.version_info.minor}.zip"
        lines = [
            str(archive),
            str(home / "Lib"),
            str(home / "DLLs"),
            ".",
            "import site",
        ]
        (directory / "claude._pth").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    else:
        (directory / "stub.py").write_text(
            STUB + "\nanswer(sys.argv[1], sys.argv[2:])\n", encoding="utf-8"
        )
        run = f"exec {shlex.quote(sys.executable)} -B {shlex.quote(str(directory / 'stub.py'))}"
        cli.write_text(f'#!/bin/sh\n{run} "$0" "$@"\n', encoding="utf-8")
        cli.chmod(0o755)
    behave(cli, **behaviour)
    return cli


def behave(cli: Path, **behaviour: object) -> None:
    settings = {"logged_in": True, "version": POSIX_VERSION, **behaviour}
    (cli.parent / "behaviour.json").write_text(json.dumps(settings), encoding="utf-8")


def version_of(cli: Path, *, build_probe: bool = False) -> str:
    """What the stub answers to -v, or the build's --version probe."""
    if WINDOWS and build_probe:
        return ".".join(str(part) for part in sys.version_info[:3])
    return json.loads((cli.parent / "behaviour.json").read_text(encoding="utf-8"))[
        "version"
    ]


def calls(cli: Path) -> list[dict]:
    """Every time the stub was run: its arguments and what it was given."""
    log = cli.parent / "calls.log"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def _copy_interpreter(cli: Path) -> None:
    """The interpreter and the libraries it loads from its own directory.
    Copied out of the installation once per run; each stub links to that."""
    for source in _interpreter_files():
        target = cli.parent / (cli.name if source.suffix == ".exe" else source.name)
        try:
            os.link(source, target)
        except OSError:  # another volume
            shutil.copy2(source, target)


def _interpreter_files() -> list[Path]:
    global _copied
    if _copied is None:
        _copied = Path(tempfile.mkdtemp(prefix="claude-stub-"))
        atexit.register(shutil.rmtree, _copied, ignore_errors=True)
        home = Path(sys.base_prefix)
        for source in (home / "python.exe", *home.glob("*.dll")):
            shutil.copy2(source, _copied / source.name)
    return sorted(_copied.iterdir())


_copied: Path | None = None
