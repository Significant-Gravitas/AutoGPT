"""Send AutoPilot one real turn through an assembled runtime, and show what
it ran on and where it wrote.

    <runtime>/python/python -B autopilot_turn.py <runtime> [--message TEXT] [--turns N]
                                   [--setting NAME=VALUE ...] [--data-dir DIR]

Not part of the smoke test: a turn needs a model, so it costs somebody
something. With Claude Code signed in on this machine and nothing else
configured, it is the user's own Claude plan (autogpt_desktop/claude_code.py);
keep the message short. `--setting` lines go into the data directory's
settings.env first: an API key, `AUTOGPT_CLAUDE_CODE=off`, or
`AUTOGPT_CLAUDE_CLI=<path>` to try another CLI.

It starts the runtime the way the shell does, signs up the owner, creates a
chat session and posts the message to the API the window uses, then prints:

  * what the runtime reported about Claude Code, and for each service host
    its services, its memory, its home directory and the settings the
    sign-in changes;
  * the reply, and when each piece of it arrived (it should stream);
  * the workspace directories seen while the turns ran (`/tmp/copilot-*`, or
    `tmp/copilot-*` in the data directory on Windows) and which are left,
    the transcript directories, that the app's own `home/.claude` stayed
    empty, and that no credentials file is anywhere in the data directory;
  * what is new at the top of the user's home directory (names only).

The runtime's log goes to stderr as it is written. It has a line for each
question the runtime asked the CLI, with its exit code and how long it took
(`claude auth status: exit code 0 after 0.4s`, or `no answer after 10.0s`):
that tells a CLI that is signed out from one that sat in a system dialog.

Exit code 0 when every turn got a reply, the runtime stopped cleanly, only
the copilot executor ran in the user's home (in a process of its own), and
nothing new is at the top of that home. Without `--data-dir` the data
directory is temporary and removed afterwards. What the CLI wrote in the
user's own Claude directory (the transcript of the session) is left, as the
app leaves it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil

import smoke_test

READY_SECONDS = 600
SHOWN = (
    "HOME",
    "USERPROFILE",
    "CLAUDE_CONFIG_DIR",
    "CHAT_USE_CLAUDE_CODE_SUBSCRIPTION",
    "CHAT_CLAUDE_AGENT_CLI_PATH",
    "MEM0_DIR",
    "COPILOT_WORKSPACE_PREFIX",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--message", default="Reply with the single word ready")
    parser.add_argument("--turns", type=int, default=1, help="in one chat session")
    parser.add_argument("--setting", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--data-dir", type=Path, help="use this (empty) data directory")
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    sys.path.append(str(runtime))  # for autogpt_desktop, as the bundle has it
    data = (args.data_dir or Path(tempfile.mkdtemp(prefix="autogpt-turn-"))).resolve()
    aliases = smoke_test.Aliases(runtime, data)
    try:
        return run(runtime, data, args)
    finally:
        aliases.remove_new()
        if not args.data_dir:
            smoke_test.remove_data(data)


def run(runtime: Path, data: Path, args: argparse.Namespace) -> int:
    (data / "config").mkdir(parents=True, exist_ok=True)
    if args.setting:
        settings = "\n".join(args.setting) + "\n"
        (data / "config" / "settings.env").write_text(settings, encoding="utf-8")
    process = subprocess.Popen(
        smoke_test.shell_command(runtime),
        cwd=runtime,
        env={**os.environ, "AUTOGPT_DESKTOP_DATA_DIR": str(data)},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    replies = 0
    problems: list[str] = []
    home_before = top_of_home()
    workspaces = Workspaces(data)
    try:
        url, state = wait_for_ready(process)
        if url:
            problems += show_hosts(process.pid, state)
            workspaces.watch()
            replies = converse(url, args.message, args.turns)
            time.sleep(5)  # the turn's own clearing up
            workspaces.show()
            show_what_landed(data)
    finally:
        workspaces.stop()
        assert process.stdin
        process.stdin.close()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(120)
        if process.poll() is None:
            process.kill()
    new = sorted(top_of_home() - home_before)
    print(f"new at the top of {Path.home()}: {new}")
    if new:
        problems.append(f"the run left {new} in the user's home directory")
    for problem in problems:
        print(f"PROBLEM: {problem}")
    print(
        f"runtime exit code: {process.returncode}; turns answered: {replies} of {args.turns}"
    )
    return (
        0 if process.returncode == 0 and replies == args.turns and not problems else 1
    )


def top_of_home() -> set[str]:
    """Names only. With CLAUDE_CONFIG_DIR unset the CLI keeps its state in
    ~/.claude and ~/.claude.json, which a home it is started in already has
    (claude_code.used_in); anything else that appears is the app's doing."""
    return {path.name for path in Path.home().iterdir()}


class Workspaces:
    """The per-session working directories, looked for while the turns run:
    the backend removes one when its turn ends."""

    def __init__(self, data: Path) -> None:
        self.directory = data / "tmp" if sys.platform == "win32" else Path("/tmp")
        self.before = self.present()
        self.seen: set[str] = set()
        self.done = threading.Event()

    def present(self) -> set[str]:
        with contextlib.suppress(OSError):
            return {path.name for path in self.directory.glob("copilot-*")}
        return set()

    def watch(self) -> None:
        def look() -> None:
            while not self.done.wait(0.2):
                self.seen |= self.present() - self.before

        threading.Thread(target=look, daemon=True).start()

    def stop(self) -> None:
        self.done.set()

    def show(self) -> None:
        left = sorted(self.present() & self.seen)
        print(
            f"workspaces seen in {self.directory} during the turns: {sorted(self.seen)}"
        )
        print(f"    still there afterwards: {left}")


def wait_for_ready(process: subprocess.Popen[str]) -> tuple[str | None, str | None]:
    """The app's address once it is ready, and the state of the Claude Code
    sign-in the runtime reported."""
    result: list[str | None] = [None, None]

    def read() -> None:
        assert process.stdout
        for line in process.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("event") == "claude_code":
                print(f"runtime: {json.dumps(event)}")
                result[1] = event.get("state")
            if event.get("event") == "ready":
                result[0] = event["url"]
            if event.get("event") == "error":
                print(f"runtime: {event.get('message')}")
            if event.get("event") == "ready" or event.get("fatal"):
                return

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    reader.join(READY_SECONDS)
    return result[0], result[1]


def show_hosts(runtime_pid: int, state: str | None) -> list[str]:
    """Each service host, with its memory and the variables the sign-in
    changes, as that process really got them. Returns what is wrong: with
    the sign-in in use the copilot executor, alone in its process, is the
    one service in the user's home; without it no service is."""
    home_variable = "USERPROFILE" if sys.platform == "win32" else "HOME"
    in_the_users_home: list[list[str]] = []
    total = 0.0
    for child in psutil.Process(runtime_pid).children():
        with contextlib.suppress(psutil.Error):
            command, env = child.cmdline(), child.environ()
            if "--name" not in command:
                continue
            services = [
                argument.split("=")[0]
                for argument in command
                if "=backend." in argument
            ]
            own = child.memory_full_info().uss / 1024**2
            total += own
            host = command[command.index("--name") + 1]
            print(f"host {host}: {', '.join(services)} ({own:.0f} MB of its own)")
            for name in SHOWN:
                print(f"    {name}={env.get(name, '<unset>')}")
            if Path(env.get(home_variable, "")) == Path.home():
                in_the_users_home.append(services)
    print(f"service hosts together: {total:.0f} MB of their own")
    expected = [["copilot-executor"]] if state == "in_use" else []
    if in_the_users_home == expected:
        return []
    return [
        f"the services in the user's home are {in_the_users_home}, not {expected} "
        f"(the runtime reported {state})"
    ]


def converse(url: str, message: str, turns: int) -> int:
    """The number of turns that got a reply; stops at the first that did not."""
    owner = smoke_test.Browser(url)
    status, _ = owner.sign_up(smoke_test.OWNER_EMAIL, smoke_test.OWNER_PASSWORD)
    print(f"owner sign-up -> {status}")
    status, session = owner.api("/chat/sessions", {})
    print(f"new chat session -> {status}")
    if status != 200:
        print(f"    {json.dumps(session)[:400]}")
        return 0
    for number in range(turns):
        if not turn(owner, url, str(session.get("id")), message, number + 1):
            return number
        time.sleep(3)
    return turns


def turn(
    owner: smoke_test.Browser, url: str, session: str, message: str, number: int
) -> bool:
    _, minted = owner.get("/api/auth/token")
    request = urllib.request.Request(
        f"{url}/_agpt/api/chat/sessions/{session}/stream",
        data=json.dumps({"message": message, "is_user_message": True}).encode(),
        headers={
            "Origin": url,
            "Content-Type": "application/json",
            "Authorization": f"Bearer {minted.get('token')}",
        },
    )
    began = time.monotonic()
    text: list[str] = []
    arrived: list[float] = []
    try:
        response = owner.opener.open(request, timeout=READY_SECONDS)
    except urllib.error.HTTPError as refused:
        print(f"turn {number}: {refused.code} {refused.read()[:400]!r}")
        return False
    with response:
        print(
            f"turn {number}: {response.status} {response.headers.get('content-type')}"
        )
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: {"):
                continue
            try:
                chunk = json.loads(line[5:])
            except ValueError:
                continue
            if chunk.get("type") == "text-delta":
                text.append(str(chunk.get("delta", "")))
                arrived.append(round(time.monotonic() - began, 1))
            if chunk.get("type") == "error":
                print(f"    error: {chunk.get('errorText')}")
    reply = "".join(text).strip()
    print(f"    reply: {reply!r}")
    print(f"    {len(arrived)} pieces, at seconds {arrived[:12]}")
    return bool(reply)


def show_what_landed(data: Path) -> None:
    for directory in (data / "tmp", data / "home", data / "home" / ".claude"):
        names = sorted(
            str(path.relative_to(directory)) for path in directory.rglob("*")
        )
        state = (
            f"{len(names)} entries {names[:12]}"
            if directory.is_dir()
            else "does not exist"
        )
        print(f"{directory}: {state}")
    print(
        f"credentials files in the data directory: {[str(p) for p in data.rglob('*credentials*')]}"
    )
    for projects in {
        Path.home() / ".claude" / "projects",
        data / "home" / ".claude" / "projects",
    }:
        if projects.is_dir():
            ours = [path for path in projects.iterdir() if "copilot-" in path.name]
            newest = sorted(ours, key=lambda path: path.stat().st_mtime)[-3:]
            print(f"newest AutoPilot transcript directories in {projects}:")
            for path in newest:
                print(f"    {time.ctime(path.stat().st_mtime)}  {path.name}")


if __name__ == "__main__":
    sys.exit(main())
