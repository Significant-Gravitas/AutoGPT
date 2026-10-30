"""AutoPilot on the user's own Claude Code sign-in.

When the Claude Code CLI on this machine is signed in, AutoPilot runs on that
sign-in: its turns count against the user's Claude plan, and no API key is
needed. Nothing is configured for it. `AUTOGPT_CLAUDE_CODE=off` turns it off,
in settings.env or in the environment the app is started with, as do the
backend's own switches in settings.env
(`CHAT_USE_CLAUDE_CODE_SUBSCRIPTION=false`, `CHAT_USE_LOCAL=true`).

The sign-in itself is never touched. The runtime only ever starts the stock
CLI (`-v`, `auth status`) and reads the two answers: no credentials
file, keychain entry or token is opened, copied or logged, and neither is
the account's email, which `auth status` prints. The backend finds the
sign-in the way the user's own terminal does, because the copilot executor
then runs in a process of its own (apps.copilot_alone) that is given the
user's real home directory (settings.claude_code_host_environment). Every
other process keeps the app's private one.

The CLI is started only in a home where Claude Code has been used before
(`used_in`): `auth status` sets up `~/.claude.json` and `~/.claude` where
there are none, and a machine without Claude Code must stay without them.

Which CLI: the one the Claude Agent SDK bundles, when the bundle has it. It
is the version the SDK was built against, the binary the Docker image runs,
and the one the backend's own check asks whether it is signed in.
The user's own install is used only where none is bundled.
`AUTOGPT_CLAUDE_CLI=<path>` in settings.env names another one; its resolved
path is passed to both the SDK and the backend's subscription check.

It is read once, while the app starts: signing in or out of Claude Code takes
effect the next time the app starts.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from autogpt_desktop.layout import EXE, WINDOWS, Bundle, DataDir
from autogpt_desktop.process import CREATE_NO_WINDOW, base_env, run_tool

logger = logging.getLogger("autogpt_desktop")

OFF_SETTING = "AUTOGPT_CLAUDE_CODE"
CLI_SETTING = "AUTOGPT_CLAUDE_CLI"
SUBSCRIPTION = "CHAT_USE_CLAUDE_CODE_SUBSCRIPTION"
LOCAL = "CHAT_USE_LOCAL"

# What the backend allows the same two commands (sdk/subscription.py).
PROBE_TIMEOUT_SECONDS = 10
# Importing the backend's settings takes five seconds on a warm machine.
CONFIG_TIMEOUT_SECONDS = 120
# Longer than everything `detect` can wait for, one after the other.
DETECTION_TIMEOUT_SECONDS = 240
# The backend's own answer to "is this configuration valid with the sign-in
# on": ChatConfig refuses some (a title model no key can reach), and every
# service that builds one would then fail to start.
CONFIG_CHECK = "from backend.copilot.config import ChatConfig; ChatConfig()"

# The states the shell words its status line from (src/claude-code.js).
IN_USE = "in_use"
SIGNED_OUT = "signed_out"
# The CLI is there and did not say whether it is signed in: it took too long
# (a virus scanner, a system dialog nobody answered) or said something else.
NO_ANSWER = "no_answer"
NOT_FOUND = "not_found"
OFF = "off"
REFUSED = "refused"

_FALSE = {"0", "false", "f", "no", "n", "off"}
_TRUE = {"1", "true", "t", "yes", "y", "on"}
_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)")
_VERSION_LINE = re.compile(r"(\d+)\.(\d+)\.(\d+)(?: \(Claude Code\))?")
# backend/copilot/sdk/cli_version.py: the first release that honors client_composed.
MIN_CLI_VERSION = (2, 1, 248)
_NATIVE_ON_WINDOWS = (".exe", ".com")

Environment = Callable[[Path | None], dict[str, str]]


@dataclass(frozen=True)
class ClaudeCode:
    """What was found, and whether AutoPilot uses it."""

    state: str = NOT_FOUND
    cli: Path | None = None
    version: str | None = None
    bundled: bool = False
    signed_in: bool = False
    # For the log: why it is not in use.
    detail: str = ""

    @property
    def in_use(self) -> bool:
        return self.state == IN_USE

    @property
    def host_cli(self) -> Path | None:
        """The CLI the hosting process has to be pointed at: none for the
        bundled one, which the SDK finds by itself."""
        return None if self.bundled else self.cli

    def describe(self) -> str:
        if self.in_use:
            return (
                "AutoPilot uses the Claude Code sign-in on this machine "
                "(turns count against that Claude plan)"
            )
        return f"AutoPilot does not use a Claude Code sign-in: {self.detail}"


class Detection:
    """`detect`, started while the databases start and asked for when the
    services are about to. It never fails a start: whatever goes wrong, the
    answer is that there is no sign-in to use."""

    def __init__(self, find: Callable[[], ClaudeCode]) -> None:
        self._found = ClaudeCode(detail="the search for Claude Code did not finish")
        self._thread = threading.Thread(
            target=self._run, args=(find,), name="claude-code"
        )
        self._thread.daemon = True
        self._thread.start()

    def _run(self, find: Callable[[], ClaudeCode]) -> None:
        try:
            self._found = find()
        except Exception as exc:
            logger.exception("looking for Claude Code failed")
            self._found = ClaudeCode(
                detail=f"the search for Claude Code failed ({exc})"
            )

    def result(self) -> ClaudeCode:
        self._thread.join(DETECTION_TIMEOUT_SECONDS)
        return self._found


def detect(
    bundle: Bundle,
    data: DataDir,
    user: dict[str, str],
    environment: Environment,
    home: Path,
) -> ClaudeCode:
    """`environment(cli)` is everything the process hosting the copilot
    executor would get with the sign-in in use (`cli` when it has to be
    pointed at one that is not bundled). The CLI is asked with exactly that:
    what the user's terminal would answer is not what counts, since a service
    inherits neither a proxy nor a CLAUDE_CONFIG_DIR exported in a shell."""
    off = turned_off(user)
    if off:
        return ClaudeCode(state=OFF, detail=off)
    probe = Probe(data, sdk_cli_versions(bundle), home)
    found = probe.first_signed_in(
        candidates(bundle, user, environment(None), home), environment
    )
    if not found.signed_in:
        return found
    refusal = backend_refuses(bundle, environment(found.host_cli))
    if refusal:
        return _replace(
            found, REFUSED, f"the backend refuses this configuration: {refusal}"
        )
    return found


def turned_off(
    user: dict[str, str], started_with: Mapping[str, str] | None = None
) -> str | None:
    """Why the sign-in is ruled out, if it is: by settings.env, or by
    AUTOGPT_CLAUDE_CODE in the environment the app was started with
    (`started_with`). The second is for a test run of the whole stack, which
    must not depend on whether the machine it runs on is signed in."""
    started_with = os.environ if started_with is None else started_with
    if started_with.get(OFF_SETTING, "").strip().lower() in _FALSE:
        return f"{OFF_SETTING}={started_with[OFF_SETTING]} in the environment"
    if user.get(OFF_SETTING, "").strip().lower() in _FALSE:
        return f"{OFF_SETTING}={user[OFF_SETTING]} in settings.env"
    if user.get(SUBSCRIPTION, "").strip().lower() in _FALSE:
        return f"{SUBSCRIPTION}={user[SUBSCRIPTION]} in settings.env"
    if user.get(LOCAL, "").strip().lower() in _TRUE:
        return f"{LOCAL}={user[LOCAL]} in settings.env (a local model runs AutoPilot)"
    return None


def candidates(
    bundle: Bundle, user: dict[str, str], env: dict[str, str], home: Path
) -> list[tuple[Path, bool]]:
    """(CLI, whether it is the bundled one), in the order to try them.
    Each launcher is resolved by the probe, so the checked executable is
    the one passed on to the backend."""
    configured = user.get(CLI_SETTING)
    if configured:
        return [(Path(configured), False)]
    bundled = bundled_cli(bundle)
    if bundled:
        return [(bundled, True)]
    found: list[Path] = []
    for cli in (_on_path(env.get("PATH")), home / ".local" / "bin" / f"claude{EXE}"):
        if cli and cli.is_file() and cli not in found:
            found.append(cli)
    return [(cli, False) for cli in found]


def bundled_cli(bundle: Bundle) -> Path | None:
    directory = sdk_directory(bundle)
    cli = directory / "_bundled" / f"claude{EXE}" if directory else None
    return cli if cli and cli.is_file() else None


def sdk_directory(bundle: Bundle) -> Path | None:
    """The installed claude_agent_sdk package: in the bundle's site/, or
    wherever this interpreter has it (a bundle that was not relocated)."""
    relocated = bundle.root / "site" / "claude_agent_sdk"
    if relocated.is_dir():
        return relocated
    spec = importlib.util.find_spec("claude_agent_sdk")
    return Path(spec.origin).parent if spec and spec.origin else None


def native(cli: Path) -> bool:
    """Whether the SDK can start it. On Windows that is a program, not the
    `claude.cmd` shim npm installs, which the SDK refuses to run."""
    return not WINDOWS or cli.name.lower().endswith(_NATIVE_ON_WINDOWS)


def used_in(home: Path, env: dict[str, str]) -> bool:
    """Whether Claude Code has been used in this home directory, going by
    names alone: its configuration directory (the one `env` names, else
    ~/.claude) or its state file is there. Nothing is opened."""
    configured = env.get("CLAUDE_CONFIG_DIR")
    if configured:
        return Path(configured).is_dir()
    return (home / ".claude").is_dir() or (home / ".claude.json").is_file()


def backend_refuses(bundle: Bundle, env: dict[str, str]) -> str | None:
    """What the backend says against starting with this environment, if
    anything. Asked of the backend itself, in an interpreter that is gone
    again before the services start: which combinations of settings it
    accepts is upstream's to change."""
    try:
        result = run_tool(
            [str(bundle.python), "-B", "-c", CONFIG_CHECK],
            env=env,
            cwd=bundle.backend_dir,
            capture_output=True,
            timeout=CONFIG_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"its settings could not be checked ({type(exc).__name__})"
    if result.returncode == 0:
        return None
    lines = result.stderr.decode("utf-8", errors="replace").strip().splitlines()
    said = next((line.strip() for line in lines if "Value error" in line), None)
    return (said or (lines[-1].strip() if lines else f"exit code {result.returncode}"))[
        :400
    ]


@dataclass(frozen=True)
class Probe:
    """Asks a CLI its version and whether it is signed in."""

    data: DataDir
    # sdk_cli_versions: the oldest CLI the SDK accepts, the one it was built with.
    sdk: tuple[str | None, str | None]
    # The home directory the CLI would be started in: the user's own.
    home: Path

    def first_signed_in(
        self, found: list[tuple[Path, bool]], environment: Environment
    ) -> ClaudeCode:
        """The first that is signed in; failing that, the first there is."""
        first: ClaudeCode | None = None
        for cli, bundled in found:
            with contextlib.suppress(OSError, RuntimeError):
                cli = cli.resolve(strict=True)
            probed = self.ask(cli, bundled, environment(None if bundled else cli))
            if probed.signed_in:
                return probed
            first = first or probed
        return first or ClaudeCode(detail="no Claude Code CLI was found")

    def ask(self, cli: Path, bundled: bool, env: dict[str, str]) -> ClaudeCode:
        def no(state: str, detail: str, version: str | None = None) -> ClaudeCode:
            return ClaudeCode(state, cli, version, bundled, False, detail)

        if not cli.is_file():
            return no(NOT_FOUND, f"{cli} does not exist")
        try:
            cli = cli.resolve(strict=True)
            identity = _identity(cli)
        except (OSError, RuntimeError):
            return no(NOT_FOUND, f"{cli} could not be resolved or read")
        if not native(cli):
            return no(
                NOT_FOUND, f"{cli} is a script, and AutoPilot can only start a program"
            )
        if not used_in(self.home, env):
            # Before anything is started there: see the module docstring.
            return no(SIGNED_OUT, f"Claude Code has not been used in {self.home}")
        # The backend asks with these three blanked, so that a key in
        # settings.env is not taken for a sign-in.
        env = {
            **env,
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_AUTH_TOKEN": "",
            "ANTHROPIC_BASE_URL": "",
        }
        # Twice: the first start of a 250 MB program on a machine that scans
        # what it starts can take longer than the answer is waited for.
        version = _version(self.run(cli, ["-v"], env) or self.run(cli, ["-v"], env))
        if version is None:
            return no(NOT_FOUND, f"{cli} -v did not answer with a version")
        try:
            unchanged = _identity(cli) == identity
        except OSError:
            unchanged = False
        if not unchanged:
            return no(NOT_FOUND, f"{cli} changed while its version was read", version)
        too_old = self.older_than_supported(cli, version)
        if too_old:
            return no(NOT_FOUND, too_old, version)
        signed_in = self.signed_in(cli, env)
        if signed_in is None:
            return no(NO_ANSWER, f"{cli} did not say whether it is signed in", version)
        if not signed_in:
            return no(SIGNED_OUT, f"{cli} is not signed in", version)
        return ClaudeCode(IN_USE, cli, version, bundled, True, "")

    def signed_in(self, cli: Path, env: dict[str, str]) -> bool | None:
        """None when it did not say. It is then asked once more, as for its
        version."""
        for _ in range(2):
            answer = _logged_in(self.run(cli, ["auth", "status"], env))
            if answer is not None:
                return answer
        return None

    def run(
        self, cli: Path, arguments: list[str], env: dict[str, str]
    ) -> subprocess.CompletedProcess[bytes] | None:
        """None when it could not be run or did not finish. Its output is for
        `ask` to read two facts from and is never logged: `auth status` names
        the account."""
        asked, began = f"{cli.name} {' '.join(arguments)}", time.monotonic()
        try:
            answer = _run(
                [str(cli), *arguments], env, self.data.home, PROBE_TIMEOUT_SECONDS
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            took = time.monotonic() - began
            logger.info(f"{asked}: no answer after {took:.1f}s ({type(exc).__name__})")
            return None
        took = time.monotonic() - began
        logger.info(f"{asked}: exit code {answer.returncode} after {took:.1f}s")
        return answer

    def older_than_supported(self, cli: Path, version: str) -> str | None:
        """Every CLI must meet both the backend and SDK floors. Being older
        than the SDK's bundled release alone only warrants a warning."""
        minimum, built_with = self.sdk
        if _numbers(version) < MIN_CLI_VERSION:
            backend_minimum = ".".join(str(part) for part in MIN_CLI_VERSION)
            return f"{cli} is version {version}; the backend needs {backend_minimum} or newer"
        if minimum and _numbers(version) < _numbers(minimum):
            return f"{cli} is version {version}; the Claude Agent SDK needs {minimum} or newer"
        if built_with and _numbers(version) < _numbers(built_with):
            logger.warning(
                f"{cli} is version {version}, older than the {built_with} the Claude Agent "
                "SDK was built with; AutoPilot may misbehave until Claude Code is updated"
            )
        return None


def _version(answer: subprocess.CompletedProcess[bytes] | None) -> str | None:
    if answer is None or answer.returncode != 0:
        return None
    first_line = (
        answer.stdout.decode("utf-8", errors="replace")
        .strip()
        .split("\n", 1)[0]
        .rstrip()
    )
    match = _VERSION_LINE.fullmatch(first_line)
    return ".".join(str(int(part)) for part in match.groups()) if match else None


def _identity(cli: Path) -> tuple[int, int, int, int]:
    stat = cli.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _run(
    argv: list[str], env: dict[str, str], cwd: Path, timeout: float
) -> subprocess.CompletedProcess[bytes]:
    """process.run_tool, except that when the time is up, what the program
    started goes with it: on macOS the CLI has a system tool fetch its
    sign-in, and that can sit in a dialog nobody sees."""
    with subprocess.Popen(
        argv,
        env=env,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=CREATE_NO_WINDOW,
        start_new_session=sys.platform != "win32",
    ) as process:
        try:
            out, err = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_with_what_it_started(process)
            raise
    return subprocess.CompletedProcess(argv, process.returncode, out, err)


def _kill_with_what_it_started(process: subprocess.Popen[bytes]) -> None:
    if sys.platform != "win32":
        with contextlib.suppress(OSError):
            os.killpg(process.pid, signal.SIGKILL)  # its session, made for this
        return
    import psutil

    started: list[psutil.Process] = []
    with contextlib.suppress(psutil.Error):
        started = psutil.Process(process.pid).children(recursive=True)
    process.kill()
    for child in started:
        with contextlib.suppress(psutil.Error):
            child.kill()
    # What the reader threads are waiting for: the pipes are closed now.
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.communicate(timeout=5)


def _logged_in(answer: subprocess.CompletedProcess[bytes] | None) -> bool | None:
    """None when the CLI did not say: it gave no answer, or one that is not
    its status. A signed-out CLI says `loggedIn: false`, with a failing exit
    code; the backend takes either for signed out (sdk/subscription.py)."""
    if answer is None:
        return None
    try:
        status = json.loads(answer.stdout.decode("utf-8", errors="replace"))
    except ValueError:
        return None
    if not isinstance(status, dict) or "loggedIn" not in status:
        return None
    return status["loggedIn"] is True and answer.returncode == 0


def sdk_cli_versions(bundle: Bundle) -> tuple[str | None, str | None]:
    """(the oldest CLI the installed SDK accepts, the one it was built with),
    read from the SDK's source. None for one that is no longer where it was:
    the check it feeds is then skipped."""
    directory = sdk_directory(bundle)
    if directory is None:
        return None, None
    minimum = _constant(
        directory / "_internal" / "transport" / "subprocess_cli.py",
        "MINIMUM_CLAUDE_CODE_VERSION",
    )
    return minimum, _constant(directory / "_cli_version.py", "__cli_version__")


def _constant(path: Path, name: str) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(
        rf'^{re.escape(name)}\s*=\s*"(\d+\.\d+\.\d+)"', text, re.MULTILINE
    )
    return match.group(1) if match else None


def _numbers(version: str) -> tuple[int, ...]:
    match = _VERSION.fullmatch(version)
    return tuple(int(part) for part in match.groups()) if match else ()


def _on_path(path: str | None) -> Path | None:
    """`claude` as the service would find it by name. On Windows a hit that
    is not a program (npm's shim, a script from Git Bash) can shadow a real
    claude.exe further along PATH."""
    for name in _names():
        hit = shutil.which(name, path=path)
        if hit and native(Path(hit)):
            return Path(hit)
    return None


def _names() -> Iterator[str]:
    yield "claude"
    if WINDOWS:
        yield "claude.exe"


def _replace(found: ClaudeCode, state: str, detail: str) -> ClaudeCode:
    return ClaudeCode(
        state, found.cli, found.version, found.bundled, found.signed_in, detail
    )


def real_home() -> Path:
    """The user's own home directory: the runtime is started with the
    user's environment, and only hands its services a different one."""
    return Path(os.path.expanduser("~"))


def service_environment(env: dict[str, str]) -> dict[str, str]:
    """`env` as a process gets it (apps.host_process)."""
    return {**base_env(), **env}
