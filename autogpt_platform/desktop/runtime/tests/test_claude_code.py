"""AutoPilot on the user's own Claude Code sign-in (claude_code.py).

Against a stand-in CLI (claude_stub.py): a real program that answers
`-v` and `auth status` as the test tells it to. The real one is never
started here, whatever the machine has installed: PATH is emptied and the
home directory is a temporary one.

The last section is what all of it rests on in the backend and the SDK, read
from their source, each failing with what to look at when upstream moves.
"""

import ast
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import claude_stub
import pytest

from autogpt_desktop import apps, claude_code, events, ports, runs, settings, supervisor
from autogpt_desktop.claude_code import (
    IN_USE,
    NO_ANSWER,
    NOT_FOUND,
    OFF,
    REFUSED,
    SIGNED_OUT,
)
from autogpt_desktop.layout import EXE, Bundle, DataDir
from autogpt_desktop.process import ManagedProcess, base_env

DESKTOP = Path(__file__).resolve().parents[2]
PLATFORM = DESKTOP.parent
BACKEND = PLATFORM / "backend" / "backend"
REAL_BUNDLE = DESKTOP / "build" / "runtime"
ACCOUNT = "secret.person"

# Stands in for backend/copilot/config.py: refuses what the real one refuses.
CHAT_CONFIG = """
import os

class ChatConfig:
    def __init__(self):
        assert os.environ.get("CHAT_USE_CLAUDE_CODE_SUBSCRIPTION") == "true"
        keys = ("CHAT_AUX_API_KEY", "CHAT_DIRECT_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")
        if os.environ.get("CHAT_USE_OPENROUTER") == "false" and not any(map(os.environ.get, keys)):
            raise ValueError("Value error, Subscription mode with no CHAT_AUX_API_KEY")
"""


@pytest.fixture(autouse=True)
def no_claude_code_of_the_machines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    empty = tmp_path / "nothing-on-path"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    # A run of these tests is not to depend on how it was started either.
    monkeypatch.delenv(claude_code.OFF_SETTING, raising=False)


@pytest.fixture
def data(tmp_path: Path) -> DataDir:
    data = DataDir(tmp_path / "data")
    data.prepare()
    return data


@pytest.fixture
def new_home(tmp_path: Path) -> Path:
    """A home directory Claude Code has never been used in."""
    home = tmp_path / "the user's home"
    home.mkdir()
    return home


@pytest.fixture
def home(new_home: Path) -> Path:
    """One it has been used in: only there is the CLI started at all."""
    (new_home / ".claude").mkdir()
    return new_home


@pytest.fixture
def bundle(tmp_path: Path) -> Bundle:
    """A bundle with the SDK's files and a backend that can be asked, and no
    CLI yet (`bundled_stub` adds one)."""
    root = tmp_path / "runtime"
    sdk = root / "site" / "claude_agent_sdk"
    (sdk / "_internal" / "transport").mkdir(parents=True)
    write_sdk_versions(Bundle(root), minimum="2.0.0", built_with="2.1.284")
    config = root / "backend" / "backend" / "copilot" / "config.py"
    config.parent.mkdir(parents=True)
    for package in (config.parent, config.parent.parent):
        (package / "__init__.py").write_text("", encoding="utf-8")
    config.write_text(CHAT_CONFIG, encoding="utf-8")
    return Bundle(root)


def write_sdk_versions(bundle: Bundle, minimum: str, built_with: str) -> None:
    sdk = bundle.root / "site" / "claude_agent_sdk"
    (sdk / "_cli_version.py").write_text(
        f'__cli_version__ = "{built_with}"\n', encoding="utf-8"
    )
    (sdk / "_internal" / "transport" / "subprocess_cli.py").write_text(
        f'MINIMUM_CLAUDE_CODE_VERSION = "{minimum}"\n', encoding="utf-8"
    )


def bundled_stub(bundle: Bundle, **behaviour: object) -> Path:
    return claude_stub.make(
        bundle.root / "site" / "claude_agent_sdk" / "_bundled", **behaviour
    )


def users_stub(home: Path, **behaviour: object) -> Path:
    """Where the native installer puts it."""
    return claude_stub.make(home / ".local" / "bin", **behaviour)


def backend_env(bundle: Bundle, data: DataDir, user: dict[str, str]) -> dict[str, str]:
    secret = settings.ensure_secrets(bundle, data)
    return settings.backend_environment(
        bundle, data, ports.allocate(data.ports_file), secret, user
    )


def detect(
    bundle: Bundle,
    data: DataDir,
    home: Path,
    user: dict[str, str] | None = None,
) -> claude_code.ClaudeCode:
    """As the supervisor calls it (Stack.look_for_claude_code)."""
    user = user or {}
    env = backend_env(bundle, data, user)

    def as_the_host_gets_it(cli: Path | None) -> dict[str, str]:
        host = settings.claude_code_host_environment(env, data, user, home, cli)
        return claude_code.service_environment(host)

    return claude_code.detect(bundle, data, user, as_the_host_gets_it, home)


# --- which CLI, and whether it is signed in --------------------------------------


def test_the_bundled_cli_signed_in_is_used(bundle: Bundle, data: DataDir, home: Path):
    cli = bundled_stub(bundle)

    found = detect(bundle, data, home)

    assert (found.state, found.cli, found.bundled, found.signed_in) == (
        IN_USE,
        cli,
        True,
        True,
    )
    assert found.in_use and found.host_cli is None
    assert found.version == claude_stub.version_of(cli)
    asked = [call["arguments"] for call in claude_stub.calls(cli)]
    assert asked == [["-v"], ["auth", "status"]]


def test_the_cli_is_asked_with_what_the_hosting_process_gets(
    bundle: Bundle, data: DataDir, home: Path
):
    """The user's real home and no CLAUDE_CONFIG_DIR of the app's; a key in
    settings.env blanked, as the backend blanks it for the same question."""
    cli = bundled_stub(bundle)

    detect(bundle, data, home, {"ANTHROPIC_API_KEY": "sk-from-settings"})

    for call in claude_stub.calls(cli):
        assert Path(call["home"]) == home
        assert call["config"] is None
        assert call["api_key"] == ""
        assert call["updater"] == "1"


def test_a_signed_out_bundled_cli_is_not_replaced_by_the_users(
    bundle: Bundle, data: DataDir, home: Path
):
    """The backend's own check asks the bundled CLI whenever there is one
    (sdk/subscription.py find_bundled_cli), so a signed-in CLI of the user's
    would only make every turn fail there."""
    bundled = bundled_stub(bundle, logged_in=False)
    users = users_stub(home)

    found = detect(bundle, data, home)

    assert (found.state, found.cli, found.signed_in) == (SIGNED_OUT, bundled, False)
    assert claude_stub.calls(users) == []


def test_without_a_bundled_cli_the_users_own_is_used(
    bundle: Bundle, data: DataDir, home: Path
):
    users = users_stub(home)

    found = detect(bundle, data, home)

    assert (found.state, found.cli, found.bundled) == (IN_USE, users, False)
    assert found.host_cli == users


def test_a_cli_on_path_comes_before_the_one_in_the_home_directory(
    bundle: Bundle,
    data: DataDir,
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    on_path = claude_stub.make(tmp_path / "on-path")
    in_home = users_stub(home, logged_in=False)
    monkeypatch.setenv("PATH", str(on_path.parent))

    found = detect(bundle, data, home)

    assert (found.state, found.cli) == (IN_USE, on_path)
    assert claude_stub.calls(in_home) == []


def test_a_signed_out_cli_on_path_gives_way_to_a_signed_in_one(
    bundle: Bundle,
    data: DataDir,
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    on_path = claude_stub.make(tmp_path / "on-path", logged_in=False)
    in_home = users_stub(home)
    monkeypatch.setenv("PATH", str(on_path.parent))

    found = detect(bundle, data, home)

    assert (found.state, found.cli) == (IN_USE, in_home)


def test_no_cli_at_all(bundle: Bundle, data: DataDir, home: Path):
    found = detect(bundle, data, home)
    assert (found.state, found.cli, found.in_use) == (NOT_FOUND, None, False)
    assert "no Claude Code CLI was found" in found.describe()


@pytest.mark.parametrize(
    "behaviour",
    [{"logged_in": False}, {"exit": 1}, {"logged_in": False, "exit": 1}],
    ids=[
        "loggedIn false",
        "a failing exit code",
        "both, as the real CLI answers signed out",
    ],
)
def test_a_cli_that_says_it_is_signed_out_is_not_used(
    bundle: Bundle, data: DataDir, home: Path, behaviour: dict
):
    cli = bundled_stub(bundle, **behaviour)
    found = detect(bundle, data, home)
    assert (found.state, found.signed_in, found.in_use) == (SIGNED_OUT, False, False)
    assert auth_status_calls(cli) == 1, "it answered: it is not asked again"


def auth_status_calls(cli: Path) -> int:
    return [call["arguments"] for call in claude_stub.calls(cli)].count(
        ["auth", "status"]
    )


def test_an_answer_that_is_not_the_clis_status_is_no_answer_and_not_signed_out(
    bundle: Bundle, data: DataDir, home: Path
):
    """The shell then asks for a restart, not for a sign-in the user has."""
    cli = bundled_stub(bundle, garbage=True)
    found = detect(bundle, data, home)
    assert (found.state, found.signed_in, found.in_use) == (NO_ANSWER, False, False)
    assert "did not say whether it is signed in" in found.describe()
    assert auth_status_calls(cli) == 2, "asked once more before giving up"


def test_a_cli_that_does_not_answer_in_time_is_not_used(
    bundle: Bundle, data: DataDir, home: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(claude_code, "PROBE_TIMEOUT_SECONDS", 2)
    bundled_stub(bundle, sleep=30)
    found = detect(bundle, data, home)
    assert not found.signed_in and not found.in_use
    assert found.state == NOT_FOUND


def test_auth_status_alone_running_out_of_time_is_no_answer_and_leaves_nothing_running(
    bundle: Bundle,
    data: DataDir,
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """A Keychain dialog nobody answers, or a virus scanner on a first start."""
    import psutil

    monkeypatch.setattr(claude_code, "PROBE_TIMEOUT_SECONDS", 3)
    cli = bundled_stub(bundle, auth_sleep=60)

    with caplog.at_level(logging.INFO, logger="autogpt_desktop"):
        found = detect(bundle, data, home)

    assert (found.state, found.version, found.in_use) == (
        NO_ANSWER,
        claude_stub.version_of(cli),
        False,
    )
    assert auth_status_calls(cli) == 2
    assert caplog.text.count("auth status: no answer after") == 2
    left = [
        process.pid
        for process in psutil.process_iter(["exe"])
        if process.info["exe"] and Path(process.info["exe"]) == cli
    ]
    assert left == [], "a CLI that ran out of time is killed, not left waiting"


def test_a_cli_that_answers_the_second_time_is_used(
    bundle: Bundle, data: DataDir, home: Path
):
    cli = bundled_stub(bundle, silent_times=1)
    found = detect(bundle, data, home)
    assert (found.state, found.in_use) == (IN_USE, True)
    assert auth_status_calls(cli) == 2


def test_every_question_is_logged_with_how_it_ended_and_how_long_it_took(
    bundle: Bundle, data: DataDir, home: Path, caplog: pytest.LogCaptureFixture
):
    """What tells a slow machine or a dialog from a signed-out CLI afterwards."""
    cli = bundled_stub(bundle, logged_in=False, exit=1)
    with caplog.at_level(logging.INFO, logger="autogpt_desktop"):
        detect(bundle, data, home)
    assert re.search(
        rf"{re.escape(cli.name)} -v: exit code 0 after \d+\.\ds", caplog.text
    )
    assert re.search(
        rf"{re.escape(cli.name)} auth status: exit code 1 after \d+\.\ds", caplog.text
    )
    assert ACCOUNT not in caplog.text


# --- a machine without Claude Code stays without it ----------------------------------


def test_nothing_is_started_in_a_home_claude_code_was_never_used_in(
    bundle: Bundle, data: DataDir, new_home: Path
):
    """The real `auth status` sets up ~/.claude.json and ~/.claude where
    there are none (the stand-in leaves the first, as proof it would)."""
    cli = bundled_stub(bundle)

    found = detect(bundle, data, new_home)

    assert (found.state, found.cli, found.in_use) == (SIGNED_OUT, cli, False)
    assert "has not been used in" in found.detail
    assert claude_stub.calls(cli) == []
    assert (
        list(new_home.iterdir()) == []
    ), "the app left something in a home without Claude Code"

    (new_home / ".claude").mkdir()
    assert detect(bundle, data, new_home).state == IN_USE
    assert (
        new_home / ".claude.json"
    ).exists(), "the stand-in no longer writes what the CLI does"


def test_the_state_file_alone_is_claude_code_having_been_used(
    bundle: Bundle, data: DataDir, new_home: Path
):
    bundled_stub(bundle)
    (new_home / ".claude.json").write_text("{}", encoding="utf-8")
    assert detect(bundle, data, new_home).state == IN_USE


def test_a_config_directory_named_in_settings_is_where_it_is_looked_for(
    bundle: Bundle, data: DataDir, home: Path
):
    """`home` has a ~/.claude, which is then not the one that counts."""
    cli = bundled_stub(bundle)
    elsewhere = home / "claude-elsewhere"

    found = detect(bundle, data, home, {"CLAUDE_CONFIG_DIR": str(elsewhere)})
    assert found.state == SIGNED_OUT and claude_stub.calls(cli) == []

    elsewhere.mkdir()
    found = detect(bundle, data, home, {"CLAUDE_CONFIG_DIR": str(elsewhere)})
    assert found.state == IN_USE
    assert {call["config"] for call in claude_stub.calls(cli)} == {str(elsewhere)}


def test_a_script_is_not_a_cli_on_windows(monkeypatch: pytest.MonkeyPatch):
    """npm installs `claude.cmd`; the SDK refuses to start a batch file."""
    monkeypatch.setattr(claude_code, "WINDOWS", True)
    assert not claude_code.native(Path("C:/Users/x/AppData/Roaming/npm/claude.cmd"))
    assert not claude_code.native(Path("C:/somewhere/claude.bat"))
    assert not claude_code.native(Path("C:/Program Files/Git/usr/bin/claude"))
    assert claude_code.native(Path("C:/Users/x/.local/bin/claude.exe"))
    assert claude_code.native(Path("C:/Users/x/.local/bin/CLAUDE.EXE"))
    monkeypatch.setattr(claude_code, "WINDOWS", False)
    assert claude_code.native(Path("/usr/local/bin/claude"))


@pytest.mark.skipif(sys.platform != "win32", reason="only Windows has .cmd shims")
def test_a_cmd_shim_on_path_is_passed_over_for_a_program_further_along(
    bundle: Bundle,
    data: DataDir,
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    shim = tmp_path / "npm" / "claude.cmd"
    shim.parent.mkdir()
    shim.write_text('@echo {"loggedIn": true}\r\n', encoding="ascii")
    monkeypatch.setenv("PATH", str(shim.parent))
    assert detect(bundle, data, home).state == NOT_FOUND

    program = claude_stub.make(tmp_path / "native")
    monkeypatch.setenv("PATH", os.pathsep.join([str(shim.parent), str(program.parent)]))
    found = detect(bundle, data, home)
    assert (found.state, found.cli) == (IN_USE, program)

    named = detect(bundle, data, home, {claude_code.CLI_SETTING: str(shim)})
    assert named.state == NOT_FOUND and "is a script" in named.detail


@pytest.mark.parametrize("bundled", [False, True])
def test_a_cli_older_than_the_sdk_accepts_is_refused(
    bundle: Bundle, data: DataDir, home: Path, bundled: bool
):
    cli = bundled_stub(bundle) if bundled else users_stub(home)
    write_sdk_versions(bundle, minimum="99.0.0", built_with="99.0.0")

    found = detect(bundle, data, home)

    assert found.state == NOT_FOUND and not found.in_use
    assert "needs 99.0.0 or newer" in found.detail
    assert auth_status_calls(cli) == 0


@pytest.mark.parametrize("bundled", [False, True])
@pytest.mark.parametrize("sdk", [("2.0.0", "2.1.284"), (None, None)])
def test_a_cli_below_the_backend_floor_is_refused_before_auth(
    data: DataDir, home: Path, monkeypatch: pytest.MonkeyPatch, bundled: bool, sdk
):
    cli = users_stub(home)
    asked = []

    def run(self, path, arguments, env):
        asked.append(arguments)
        output = (
            b"2.1.247 (Claude Code)"
            if arguments != ["auth", "status"]
            else b'{"loggedIn": true}'
        )
        return subprocess.CompletedProcess(arguments, 0, output, b"")

    monkeypatch.setattr(claude_code.Probe, "run", run)
    found = claude_code.Probe(data, sdk, home).ask(cli, bundled, {})

    assert found.state == NOT_FOUND and not found.signed_in
    assert "2.1.248" in found.detail
    assert asked == [["-v"]]


@pytest.mark.parametrize("bundled", [False, True])
@pytest.mark.parametrize(
    ("output", "version"),
    [
        ("2.1.248", "2.1.248"),
        (" 2.1.284 (Claude Code)\r\n", "2.1.284"),
        ("2.1.284 (Claude Code)\nsecond line", "2.1.284"),
        ("3.0.0", "3.0.0"),
    ],
)
def test_supported_first_line_versions_are_accepted(
    bundle: Bundle, data: DataDir, home: Path, bundled: bool, output: str, version: str
):
    cli = (
        bundled_stub(bundle, version_output=output)
        if bundled
        else users_stub(home, version_output=output)
    )

    found = detect(bundle, data, home)

    assert found.in_use and found.version == version
    assert [call["arguments"] for call in claude_stub.calls(cli)] == [
        ["-v"],
        ["auth", "status"],
    ]


@pytest.mark.parametrize(
    "behaviour",
    [{"version": "2.1.247"}, {"version_output": "Python 3.11.9"}, {"version_exit": 1}],
)
def test_the_version_probe_refuses_a_native_cli_without_asking_auth(
    bundle: Bundle, data: DataDir, home: Path, behaviour: dict
):
    cli = bundled_stub(bundle, **behaviour)

    found = detect(bundle, data, home)

    assert found.state == NOT_FOUND and not found.signed_in
    assert [call["arguments"] for call in claude_stub.calls(cli)] == [["-v"]]


@pytest.mark.parametrize(
    ("stdout", "stderr", "returncode"),
    [(b"2.1.284 (Claude Code)", b"", 1), (b"", b"2.1.284 (Claude Code)", 0)],
)
def test_a_version_requires_successful_stdout(
    data: DataDir,
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdout: bytes,
    stderr: bytes,
    returncode: int,
):
    cli = users_stub(home)
    asked = []

    def run(self, path, arguments, env):
        asked.append(arguments)
        return subprocess.CompletedProcess(arguments, returncode, stdout, stderr)

    monkeypatch.setattr(claude_code.Probe, "run", run)
    found = claude_code.Probe(data, (None, None), home).ask(cli, True, {})

    assert found.state == NOT_FOUND and not found.signed_in
    assert asked == [["-v"]]


@pytest.mark.parametrize(
    "output",
    [
        b"Python 3.11.9",
        b"wrapper 2.1.284 (Claude Code)",
        b"2.1.284-beta",
        b"2.1.284.1",
        b"2.1.284 trailing text",
        b"warning\n2.1.284 (Claude Code)",
        b"\xff2.1.284 (Claude Code)",
    ],
)
def test_unrecognized_version_output_never_reaches_auth(
    data: DataDir, home: Path, monkeypatch: pytest.MonkeyPatch, output: bytes
):
    cli = users_stub(home)
    asked = []

    def run(self, path, arguments, env):
        asked.append(arguments)
        stdout = b'{"loggedIn": true}' if arguments == ["auth", "status"] else output
        return subprocess.CompletedProcess(arguments, 0, stdout, b"")

    monkeypatch.setattr(claude_code.Probe, "run", run)
    found = claude_code.Probe(data, (None, None), home).ask(cli, True, {})

    assert found.state == NOT_FOUND and not found.signed_in
    assert asked == [["-v"]]


def test_a_cli_changed_during_its_version_probe_is_refused_before_auth(
    data: DataDir, home: Path, monkeypatch: pytest.MonkeyPatch
):
    cli = home / f"claude{EXE}"
    cli.write_bytes(b"original executable")
    asked = []

    def run(self, path, arguments, env):
        asked.append(arguments)
        if arguments == ["auth", "status"]:
            output = b'{"loggedIn": true}'
        else:
            cli.write_bytes(b"replacement executable with a different size")
            output = b"2.1.284 (Claude Code)"
        return subprocess.CompletedProcess(arguments, 0, output, b"")

    monkeypatch.setattr(claude_code.Probe, "run", run)
    found = claude_code.Probe(data, (None, None), home).ask(cli, True, {})

    assert found.state == NOT_FOUND and not found.signed_in
    assert "changed" in found.detail
    assert asked == [["-v"]]


def test_the_checked_executable_is_resolved_and_kept_when_the_launcher_changes(
    data: DataDir, home: Path, monkeypatch: pytest.MonkeyPatch
):
    original = home / "original" / f"claude{EXE}"
    replacement = home / "replacement" / f"claude{EXE}"
    for path in (original, replacement):
        path.parent.mkdir()
        path.write_bytes(b"stand-in executable")
    launcher = home / f"claude{EXE}"
    try:
        launcher.symlink_to(original)
    except OSError:
        pytest.skip("creating a symlink requires permission on this Windows host")
    asked = []

    def run(self, path, arguments, env):
        asked.append((path, arguments))
        if arguments == ["auth", "status"]:
            output = b'{"loggedIn": true}'
        else:
            launcher.unlink()
            launcher.symlink_to(replacement)
            output = b"2.1.284 (Claude Code)"
        return subprocess.CompletedProcess(arguments, 0, output, b"")

    monkeypatch.setattr(claude_code.Probe, "run", run)
    environments = []

    def environment(cli):
        environments.append(cli)
        return {}

    found = claude_code.Probe(data, (None, None), home).first_signed_in(
        [(launcher, False)], environment
    )

    assert found.in_use and found.cli == original and found.host_cli == original
    assert asked == [(original, ["-v"]), (original, ["auth", "status"])]
    assert environments == [original]


def test_fallback_launcher_keeps_subscription_auth_on_the_checked_version(
    bundle: Bundle, data: DataDir, home: Path, monkeypatch: pytest.MonkeyPatch
):
    target = home / ".local" / "share" / "claude" / "versions" / f"2.1.284{EXE}"
    replacement = target.with_name(f"2.1.247{EXE}")
    target.parent.mkdir(parents=True)
    for path in (target, replacement):
        path.write_bytes(b"stand-in executable that must never be started")
        path.chmod(0o755)
    launcher = home / ".local" / "bin" / f"claude{EXE}"
    launcher.parent.mkdir(parents=True)
    try:
        launcher.symlink_to(target)
    except OSError:
        pytest.skip("creating a symlink requires permission on this Windows host")
    probed = []

    def run(self, cli, arguments, env):
        probed.append((cli, arguments))
        stdout = (
            b'{"loggedIn": true}' if arguments == ["auth", "status"] else b"2.1.284"
        )
        return subprocess.CompletedProcess(arguments, 0, stdout, b"")

    monkeypatch.setattr(claude_code.Probe, "run", run)
    monkeypatch.setattr(claude_code, "backend_refuses", lambda *args: None)
    found = detect(bundle, data, home)
    assert found.in_use and found.host_cli == target
    assert probed == [(target, ["-v"]), (target, ["auth", "status"])]

    launcher.unlink()
    launcher.symlink_to(replacement)
    monkeypatch.setattr(claude_code, "real_home", lambda: home)
    stack = stack_that_found(bundle, data, found)
    stack.use_claude_code()
    host = stack.copilot_env
    assert host is not None
    assert host["CHAT_CLAUDE_AGENT_CLI_PATH"] == str(target)
    assert shutil.which("claude", path=host["PATH"]) is None

    inherited = ModuleType("backend.copilot.sdk.launch_env")
    inherited.inherited_env = dict
    monkeypatch.setitem(sys.modules, inherited.__name__, inherited)
    transport = ModuleType("claude_agent_sdk._internal.transport.subprocess_cli")
    transport.SubprocessCLITransport = SimpleNamespace(_find_bundled_cli=lambda _: None)
    monkeypatch.setitem(sys.modules, transport.__name__, transport)
    spec = importlib.util.spec_from_file_location(
        "desktop_subscription_contract", BACKEND / "copilot" / "sdk" / "subscription.py"
    )
    assert spec is not None and spec.loader is not None
    subscription = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(subscription)
    monkeypatch.setenv("PATH", host["PATH"])
    monkeypatch.setenv("CHAT_CLAUDE_AGENT_CLI_PATH", host["CHAT_CLAUDE_AGENT_CLI_PATH"])
    monkeypatch.delenv("CLAUDE_AGENT_CLI_PATH", raising=False)
    monkeypatch.setattr(subscription, "provision_credentials_file", lambda: None)
    monkeypatch.setattr(subscription, "host_login_env", dict)
    checked = []

    def run_subscription(argv, **kwargs):
        checked.append(argv)
        stdout = '{"loggedIn": true}' if argv[1:] == ["auth", "status"] else "2.1.284"
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(subscription.subprocess, "run", run_subscription)
    subscription.validate_subscription()
    assert checked == [[str(target), "--version"], [str(target), "auth", "status"]]


def test_a_cli_older_than_the_sdk_was_built_with_is_used_with_a_warning(
    bundle: Bundle, data: DataDir, home: Path, caplog: pytest.LogCaptureFixture
):
    """The stable channels run about a week behind the SDK's CLI."""
    users_stub(home)
    write_sdk_versions(bundle, minimum="0.0.1", built_with="99.0.0")

    with caplog.at_level(logging.WARNING, logger="autogpt_desktop"):
        found = detect(bundle, data, home)

    assert found.state == IN_USE
    assert "older than the 99.0.0" in caplog.text


def test_sdk_files_that_moved_skip_only_the_sdk_version_floor(
    bundle: Bundle, data: DataDir, home: Path
):
    users_stub(home)
    (bundle.root / "site" / "claude_agent_sdk" / "_cli_version.py").unlink()
    (
        bundle.root
        / "site"
        / "claude_agent_sdk"
        / "_internal"
        / "transport"
        / "subprocess_cli.py"
    ).unlink()
    assert claude_code.sdk_cli_versions(bundle) == (None, None)
    assert detect(bundle, data, home).state == IN_USE


# --- settings.env ------------------------------------------------------------------


@pytest.mark.parametrize(
    "user",
    [
        {"AUTOGPT_CLAUDE_CODE": "off"},
        {"AUTOGPT_CLAUDE_CODE": "FALSE"},
        {"CHAT_USE_CLAUDE_CODE_SUBSCRIPTION": "false"},
        {"CHAT_USE_LOCAL": "true"},
    ],
    ids=lambda user: "=".join(next(iter(user.items()))),
)
def test_turned_off_in_settings_nothing_is_even_started(
    bundle: Bundle, data: DataDir, home: Path, user: dict[str, str]
):
    cli = bundled_stub(bundle)
    found = detect(bundle, data, home, user)
    assert (found.state, found.in_use, found.cli) == (OFF, False, None)
    assert next(iter(user)) in found.detail
    assert claude_stub.calls(cli) == []


def test_the_environment_the_app_is_started_with_can_turn_it_off_too(
    bundle: Bundle, data: DataDir, home: Path, monkeypatch: pytest.MonkeyPatch
):
    """For a test run of the whole stack (build/smoke_test.py), which must
    come out the same on a machine that is signed in to Claude Code."""
    cli = bundled_stub(bundle)
    monkeypatch.setenv("AUTOGPT_CLAUDE_CODE", "off")

    found = detect(bundle, data, home)

    assert (found.state, found.in_use) == (OFF, False)
    assert "in the environment" in found.detail
    assert claude_stub.calls(cli) == []
    assert claude_code.turned_off({}, {"AUTOGPT_CLAUDE_CODE": "on"}) is None
    # Only that one name: the backend's switches are settings, not environment.
    assert claude_code.turned_off({}, {"CHAT_USE_LOCAL": "true"}) is None


def test_other_values_of_the_switches_leave_it_on():
    assert claude_code.turned_off({}) is None
    assert claude_code.turned_off({"AUTOGPT_CLAUDE_CODE": "on"}) is None
    assert claude_code.turned_off({"CHAT_USE_CLAUDE_CODE_SUBSCRIPTION": "true"}) is None
    assert claude_code.turned_off({"CHAT_USE_LOCAL": "false"}) is None


def test_a_configuration_the_backend_refuses_with_the_sign_in_on_keeps_it_off(
    bundle: Bundle, data: DataDir, home: Path
):
    """CHAT_USE_OPENROUTER=false with no key for the title model: every
    service that builds a ChatConfig would fail to start."""
    bundled_stub(bundle)

    refused = detect(bundle, data, home, {"CHAT_USE_OPENROUTER": "false"})
    assert (refused.state, refused.signed_in, refused.in_use) == (REFUSED, True, False)
    assert "no CHAT_AUX_API_KEY" in refused.detail

    user = {"CHAT_USE_OPENROUTER": "false", "ANTHROPIC_API_KEY": "sk-ant"}
    assert detect(bundle, data, home, user).state == IN_USE


def test_a_backend_that_cannot_be_asked_keeps_it_off(
    bundle: Bundle, data: DataDir, home: Path
):
    bundled_stub(bundle)
    (bundle.root / "backend" / "backend" / "copilot" / "config.py").unlink()
    found = detect(bundle, data, home)
    assert found.state == REFUSED and not found.in_use


def test_the_cli_named_in_settings_is_the_one_asked(
    bundle: Bundle, data: DataDir, home: Path
):
    """How a machine with no sign-in is simulated on one that has one."""
    bundled = bundled_stub(bundle)
    named = claude_stub.make(data.root.parent / "named", logged_in=False)

    found = detect(bundle, data, home, {claude_code.CLI_SETTING: str(named)})

    assert (found.state, found.cli, found.bundled) == (SIGNED_OUT, named, False)
    assert claude_stub.calls(bundled) == []


def test_a_named_cli_is_used_even_when_the_bundled_one_is_signed_out(
    bundle: Bundle, data: DataDir, home: Path
):
    bundled = bundled_stub(bundle, logged_in=False)
    named = claude_stub.make(data.root.parent / "named")

    found = detect(bundle, data, home, {claude_code.CLI_SETTING: str(named)})

    assert found.in_use and found.host_cli == named.resolve()
    assert claude_stub.calls(bundled) == []


def test_a_named_cli_that_is_not_there(bundle: Bundle, data: DataDir, home: Path):
    found = detect(
        bundle, data, home, {claude_code.CLI_SETTING: str(home / "nowhere" / "claude")}
    )
    assert found.state == NOT_FOUND and "does not exist" in found.detail


# --- what the services are given ---------------------------------------------------


def test_claude_tokens_in_settings_reach_no_process(
    bundle: Bundle, data: DataDir, home: Path
):
    """With one of them the backend writes a credentials file into the home
    directory (sdk/subscription.py provision_credentials_file)."""
    user = {
        "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat-secret",
        "CLAUDE_CODE_REFRESH_TOKEN": "sk-ant-ort-secret",
        # Windows has one variable for every spelling of a name.
        "claude_code_oauth_token": "sk-ant-oat-secret-lower",
        "Claude_Code_Refresh_Token": "sk-ant-ort-secret-mixed",
        "OPEN_ROUTER_API_KEY": "sk-or",
    }
    secret = settings.ensure_secrets(bundle, data)
    port = ports.allocate(data.ports_file)
    backend = settings.backend_environment(bundle, data, port, secret, user)
    host = settings.claude_code_host_environment(backend, data, user, home)
    frontend = settings.frontend_environment(backend, port, secret, data)

    for env in (backend, host, frontend, claude_code.service_environment(host)):
        assert not {"CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_REFRESH_TOKEN"} & {
            name.upper() for name in env
        }
        assert "secret" not in json.dumps(env)
    assert backend["OPEN_ROUTER_API_KEY"] == "sk-or"


def test_the_hosting_process_gets_the_real_home_and_nothing_of_the_apps_claude_setup(
    bundle: Bundle, data: DataDir, home: Path
):
    backend = backend_env(bundle, data, {})
    host = settings.claude_code_host_environment(backend, data, {}, home)

    assert host["HOME"] == str(home)
    assert "CLAUDE_CONFIG_DIR" not in host
    assert host["CHAT_USE_CLAUDE_CODE_SUBSCRIPTION"] == "true"
    assert host["DISABLE_AUTOUPDATER"] == "1"
    # What would otherwise be written into the user's home goes to the app's.
    assert host["MEM0_DIR"] == str(data.home / ".mem0")
    assert host["XDG_CACHE_HOME"] == str(data.backend_cache)
    assert host["CODEX_HOME"] == str(data.home / ".codex")
    assert "CHAT_CLAUDE_AGENT_CLI_PATH" not in host
    if sys.platform == "win32":  # where HOME is ignored
        assert host["USERPROFILE"] == str(home)
    changed = {name for name in host if host[name] != backend.get(name)} | (
        backend.keys() - host.keys()
    )
    assert changed <= {
        "HOME",
        "USERPROFILE",
        "CLAUDE_CONFIG_DIR",
        "CHAT_USE_CLAUDE_CODE_SUBSCRIPTION",
        "DISABLE_AUTOUPDATER",
        "MEM0_DIR",
    }


def test_a_config_directory_the_user_chose_is_kept(
    bundle: Bundle, data: DataDir, home: Path
):
    user = {"CLAUDE_CONFIG_DIR": str(home / "claude-elsewhere")}
    backend = backend_env(bundle, data, user)
    assert backend["CLAUDE_CONFIG_DIR"] == str(data.home / ".claude")
    host = settings.claude_code_host_environment(backend, data, user, home)
    assert host["CLAUDE_CONFIG_DIR"] == str(home / "claude-elsewhere")


def test_a_cli_that_is_not_bundled_is_named_and_put_first_on_path(
    bundle: Bundle, data: DataDir, home: Path, monkeypatch: pytest.MonkeyPatch
):
    """Execution and subscription validation share the configured CLI;
    its directory is also available for command discovery."""
    monkeypatch.setenv("PATH", "the-systems-path")
    cli = home / ".local" / "bin" / f"claude{EXE}"
    backend = backend_env(bundle, data, {})
    host = settings.claude_code_host_environment(backend, data, {}, home, cli)
    assert host["CHAT_CLAUDE_AGENT_CLI_PATH"] == str(cli)
    assert host["PATH"].split(os.pathsep)[0] == str(cli.parent)
    assert host["PATH"].split(os.pathsep)[-1] == "the-systems-path"


def test_workspaces_stay_in_the_data_directory_on_windows(
    bundle: Bundle, data: DataDir
):
    backend = backend_env(bundle, data, {"COPILOT_WORKSPACE_PREFIX": "C:/tmp/copilot-"})
    if sys.platform != "win32":
        assert "COPILOT_WORKSPACE_PREFIX" not in settings._copilot_workspaces(data)
        return
    prefix = backend["COPILOT_WORKSPACE_PREFIX"]
    assert Path(prefix).parent == data.root / "tmp"
    # The backend sweeps transcript directories by this part of their name.
    assert re.sub(r"[^a-zA-Z0-9]", "-", prefix + "0123").endswith("-tmp-copilot-0123")


def test_workspaces_left_by_an_earlier_run_are_cleared_at_start(
    data: DataDir, monkeypatch: pytest.MonkeyPatch
):
    left = data.root / "tmp" / "copilot-0123" / "claude"
    left.mkdir(parents=True)
    (left / "probe").write_text("x", encoding="utf-8")
    kept = data.workspaces / "file"
    kept.write_text("the user's", encoding="utf-8")

    settings.clear_copilot_workspaces(data)
    assert left.exists() == (
        sys.platform != "win32"
    ), "elsewhere they are in /tmp, the system's"

    monkeypatch.setattr(sys, "platform", "win32")
    settings.clear_copilot_workspaces(data)
    assert not (data.root / "tmp").exists()
    assert kept.read_text(encoding="utf-8") == "the user's"
    settings.clear_copilot_workspaces(data)  # nothing there: nothing to do


@pytest.mark.parametrize(
    "merged", [True, False], ids=["grouped", "one process per service"]
)
def test_the_copilot_executor_alone_runs_in_the_real_home(
    bundle: Bundle, data: DataDir, home: Path, merged: bool
):
    """No other service shares its process: the graph executor starts the
    Claude Code CLI too (the orchestrator block), with nothing to keep it
    from loading the user's own settings, hooks and MCP servers."""
    backend = backend_env(bundle, data, {})
    host = settings.claude_code_host_environment(backend, data, {}, home)
    groups = apps.copilot_alone(apps.layout(merged))

    processes = apps.backend_processes(
        bundle, data, backend, groups, runs.Cache(0, ""), host
    )

    in_the_real_home = [
        group.services
        for group, process in zip(groups, processes, strict=True)
        if process.env["HOME"] == str(home)
    ]
    assert in_the_real_home == [(apps.COPILOT_EXECUTOR,)]
    hosting = [apps.COPILOT_EXECUTOR]
    for process in processes:
        if process.name in hosting:
            assert "CLAUDE_CONFIG_DIR" not in process.env
            continue
        assert process.env["HOME"] == str(data.home)
        assert process.env["CLAUDE_CONFIG_DIR"] == str(data.home / ".claude")
        assert "MEM0_DIR" not in process.env
        if sys.platform == "win32":
            assert process.env["USERPROFILE"] == str(data.home)


def test_taking_the_copilot_executor_out_keeps_every_other_service_where_it_was():
    merged = apps.layout(merged=True)
    apart = apps.copilot_alone(merged)

    assert [group.name for group in apart] == [
        "database-manager",
        "swap-credentials",
        "workers",
        apps.COPILOT_EXECUTOR,
        "api",
    ]
    for before in merged:
        after = next(group for group in apart if group.name == before.name)
        assert after.services == tuple(
            name for name in before.services if name != apps.COPILOT_EXECUTOR
        )
    hosted = sorted(name for group in apart for name in group.services)
    assert hosted == sorted(
        service.name for service in apps.SERVICES
    ), "each once, none lost"
    # One process per service already has it alone; done twice changes nothing.
    alone = apps.layout(merged=False)
    assert apps.copilot_alone(alone) == alone
    assert apps.copilot_alone(apart) == apart


def test_a_process_the_copilot_executor_shares_never_gets_the_real_home(
    bundle: Bundle, data: DataDir, home: Path
):
    """Whatever layout it is called with: the real home is for a process
    that hosts the copilot executor and nothing else."""
    backend = backend_env(bundle, data, {})
    host = settings.claude_code_host_environment(backend, data, {}, home)
    shared = apps.layout(merged=True)

    processes = apps.backend_processes(
        bundle, data, backend, shared, runs.Cache(0, ""), host
    )

    assert {process.env["HOME"] for process in processes} == {str(data.home)}


def test_without_a_sign_in_every_process_keeps_the_apps_home(
    bundle: Bundle, data: DataDir
):
    backend = backend_env(bundle, data, {})
    groups = apps.layout(merged=True)
    processes = apps.backend_processes(bundle, data, backend, groups, runs.Cache(0, ""))
    assert {process.env["HOME"] for process in processes} == {str(data.home)}
    assert all(
        "CHAT_USE_CLAUDE_CODE_SUBSCRIPTION" not in process.env for process in processes
    )


# --- the supervisor, and what the shell is told --------------------------------------


def stack_that_found(
    bundle: Bundle, data: DataDir, found: claude_code.ClaudeCode
) -> supervisor.Stack:
    stack = supervisor.Stack(bundle, data)
    stack.env = backend_env(bundle, data, {})
    stack.claude_code = claude_code.Detection(lambda: found)
    return stack


def last_event(capsys: pytest.CaptureFixture[str]) -> dict:
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_a_sign_in_turns_subscription_on_for_every_service_and_tells_the_shell(
    bundle: Bundle, data: DataDir, capsys: pytest.CaptureFixture[str]
):
    cli = bundle.root / "site" / "claude_agent_sdk" / "_bundled" / f"claude{EXE}"
    stack = stack_that_found(
        bundle, data, claude_code.ClaudeCode(IN_USE, cli, "2.1.284", True, True)
    )

    stack.use_claude_code()

    assert stack.env["CHAT_USE_CLAUDE_CODE_SUBSCRIPTION"] == "true"
    assert stack.env["HOME"] == str(data.home)
    assert stack.copilot_env and stack.copilot_env["HOME"] == str(
        claude_code.real_home()
    )
    assert last_event(capsys) == {
        "event": "claude_code",
        "state": "in_use",
        "message": "AutoPilot uses the Claude Code sign-in on this machine "
        "(turns count against that Claude plan)",
        "cli": str(cli),
        "version": "2.1.284",
        "bundled": True,
    }


def test_without_a_sign_in_nothing_changes_and_the_shell_is_told_why(
    bundle: Bundle, data: DataDir, capsys: pytest.CaptureFixture[str]
):
    cli = bundle.root / "claude"
    found = claude_code.ClaudeCode(
        SIGNED_OUT, cli, "2.1.284", True, False, f"{cli} is not signed in"
    )
    stack = stack_that_found(bundle, data, found)
    before = dict(stack.env)

    stack.use_claude_code()

    assert stack.env == before and stack.copilot_env is None
    event = last_event(capsys)
    assert (event["event"], event["state"]) == ("claude_code", "signed_out")
    assert "is not signed in" in event["message"]


class NoProxy:
    def __init__(self, *args: object) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def hosts_started(
    bundle: Bundle,
    data: DataDir,
    found: claude_code.ClaudeCode,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[tuple[str, ...], dict[str, str]]:
    """Stack.start_apps with stand-in processes: the services of each host
    it starts, and the environment that host is given."""
    started: dict[tuple[str, ...], dict[str, str]] = {}

    def stand_in(name: str) -> ManagedProcess:
        argv = [sys.executable, "-c", "import time; time.sleep(60)"]
        return ManagedProcess(name, argv, {}, data.root, data.logs, stop_timeout=2)

    def host_process(bundle, data, env, group, cache) -> ManagedProcess:
        started[group.services] = env
        return stand_in(group.name)

    monkeypatch.setattr(apps, "host_process", host_process)
    monkeypatch.setattr(apps, "frontend_process", lambda *args: stand_in(apps.FRONTEND))
    monkeypatch.setattr(supervisor, "ProxyThread", NoProxy)
    monkeypatch.setattr(supervisor.Stack, "wait_for_apps", lambda self, port: None)
    stack = stack_that_found(bundle, data, found)
    try:
        stack.start_apps(
            ports.allocate(data.ports_file), settings.ensure_secrets(bundle, data)
        )
    finally:
        stack.stop()
    return started


def test_with_a_sign_in_the_supervisor_starts_the_copilot_executor_in_a_process_of_its_own(
    bundle: Bundle,
    data: DataDir,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    cli = bundle.root / "site" / "claude_agent_sdk" / "_bundled" / f"claude{EXE}"
    found = claude_code.ClaudeCode(IN_USE, cli, "2.1.284", True, True)

    started = hosts_started(bundle, data, found, monkeypatch)

    real_home = str(claude_code.real_home())
    assert [
        services for services, env in started.items() if env["HOME"] == real_home
    ] == [(apps.COPILOT_EXECUTOR,)]
    assert len(started) == 5
    for services, env in started.items():
        assert env["CHAT_USE_CLAUDE_CODE_SUBSCRIPTION"] == "true"
        if services != (apps.COPILOT_EXECUTOR,):
            assert apps.COPILOT_EXECUTOR not in services
            assert env["HOME"] == str(data.home)
            assert env["CLAUDE_CONFIG_DIR"] == str(data.home / ".claude")


def test_without_a_sign_in_the_supervisor_starts_the_usual_four(
    bundle: Bundle,
    data: DataDir,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    found = claude_code.ClaudeCode(
        SIGNED_OUT, None, None, False, False, "not signed in"
    )

    started = hosts_started(bundle, data, found, monkeypatch)

    assert sorted(started) == sorted(
        group.services for group in apps.layout(merged=True)
    )
    assert {env["HOME"] for env in started.values()} == {str(data.home)}


def test_a_search_that_fails_is_no_sign_in_and_not_a_failed_start():
    def broken() -> claude_code.ClaudeCode:
        raise OSError("the disk went away")

    found = claude_code.Detection(broken).result()
    assert (found.state, found.in_use) == (NOT_FOUND, False)
    assert "the disk went away" in found.detail


def test_the_account_is_named_nowhere(
    bundle: Bundle,
    data: DataDir,
    home: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
):
    """`auth status` prints the account's email address."""
    bundled_stub(bundle)
    with caplog.at_level(logging.DEBUG, logger="autogpt_desktop"):
        found = detect(bundle, data, home)
        events.claude_code(
            found.state, found.describe(), str(found.cli), found.version, found.bundled
        )
        signed_out = detect(bundle, data, home, {"CHAT_USE_OPENROUTER": "false"})
    assert found.in_use and signed_out.state == REFUSED
    said = caplog.text + capsys.readouterr().out + repr(found) + repr(signed_out)
    assert ACCOUNT not in said


def test_the_runtime_never_opens_a_credential_store():
    """It starts the CLI and reads two answers; that is all."""
    source = (DESKTOP / "runtime" / "autogpt_desktop" / "claude_code.py").read_text(
        encoding="utf-8"
    )
    code = source.split('"""', 2)[2]
    for forbidden in (
        ".credentials",
        "keychain",
        "Keychain",
        "security find",
        "accessToken",
    ):
        assert forbidden not in code


# --- what this rests on upstream ------------------------------------------------------


def backend_source(*parts: str) -> str:
    return BACKEND.joinpath(*parts).read_text(encoding="utf-8")


def test_the_backend_version_contract_is_still_the_one_the_runtime_mirrors():
    source = ast.parse(backend_source("copilot", "sdk", "cli_version.py"))
    definitions = [
        node
        for node in source.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id in {"MIN_CLI_VERSION", "_VERSION_LINE_RE"}
                for target in node.targets
            )
        )
        or (isinstance(node, ast.FunctionDef) and node.name == "parse_cli_version")
    ]
    contract = {"re": re}
    exec(  # noqa: S102 - Only the pure version contract from trusted backend source.
        compile(
            ast.Module(body=definitions, type_ignores=[]), "cli_version.py", "exec"
        ),
        contract,
    )
    assert contract["MIN_CLI_VERSION"] == claude_code.MIN_CLI_VERSION
    for output in (
        "2.1.248",
        "2.1.284 (Claude Code)",
        " 2.1.284 (Claude Code)\r\nsecond line",
        "02.01.284",
        "Python 3.11.9",
        "2.1.284-beta",
        "2.1.284+build",
        "2.1.284.1",
        "warning\n2.1.284 (Claude Code)",
        "2.1.284 (Claude Code) extra",
        "",
    ):
        expected = contract["parse_cli_version"](output)
        answer = subprocess.CompletedProcess(["-v"], 0, output.encode(), b"")
        version = claude_code._version(answer)
        assert version == (
            ".".join(map(str, expected)) if expected is not None else None
        )


def test_the_backend_still_takes_its_sign_in_from_these_settings():
    config = backend_source("copilot", "config.py")
    update = "Update desktop/runtime/autogpt_desktop/claude_code.py and settings.py."
    assert re.search(
        r"^class ChatConfig\(", config, re.MULTILINE
    ), f"backend.copilot.config.ChatConfig is gone: claude_code.CONFIG_CHECK builds one. {update}"
    for field in ("use_claude_code_subscription", "use_local", "claude_agent_cli_path"):
        assert re.search(
            rf"^    {field}: ", config, re.MULTILINE
        ), f"ChatConfig.{field} is gone or renamed (CHAT_{field.upper()}). {update}"
    assert 'os.getenv("CHAT_CLAUDE_AGENT_CLI_PATH")' in config, update


def test_the_backends_sign_in_check_is_still_the_one_the_runtime_mirrors():
    subscription = backend_source("copilot", "sdk", "subscription.py")
    update = (
        "Read backend/copilot/sdk/subscription.py again and update "
        "desktop/runtime/autogpt_desktop/claude_code.py to ask what it asks."
    )
    for needed in (
        '[cli, "auth", "status"]',
        '[cli, "--version"]',
        'status.get("loggedIn")',
    ):
        assert (
            needed in subscription
        ), f"validate_subscription no longer has {needed}. {update}"
    assert (
        "_find_bundled_cli" in subscription and 'shutil.which("claude")' in subscription
    ), f"find_bundled_cli no longer looks for the bundled CLI and then `claude` on PATH. {update}"
    for token in settings.CLAUDE_TOKEN_SETTINGS:
        assert f'os.environ.get("{token}"' in subscription, (
            f"the backend no longer reads {token}; settings.CLAUDE_TOKEN_SETTINGS may be "
            "dropping a name that means nothing now, or missing a new one."
        )


def test_the_real_bundle_has_the_sdk_files_the_runtime_reads():
    if not (REAL_BUNDLE / "manifest.json").is_file():
        pytest.skip("no assembled bundle (build/runtime)")
    bundle = Bundle(REAL_BUNDLE)
    minimum, built_with = claude_code.sdk_cli_versions(bundle)
    assert minimum and built_with, (
        "claude_agent_sdk no longer has MINIMUM_CLAUDE_CODE_VERSION in "
        "_internal/transport/subprocess_cli.py, or __cli_version__ in _cli_version.py. "
        "Update sdk_cli_versions in desktop/runtime/autogpt_desktop/claude_code.py."
    )
    assert claude_code.bundled_cli(bundle), (
        "the bundle has no claude_agent_sdk/_bundled/claude: run the deps step "
        "(desktop/build/claude_cli.py)"
    )


def test_the_real_bundle_still_reads_what_keeps_files_out_of_the_users_home():
    """The copilot executor's process runs in the user's real home. Two
    names of other people's keep it from leaving things there, and both are
    only set by settings.claude_code_host_environment, never checked."""
    if not (REAL_BUNDLE / "manifest.json").is_file():
        pytest.skip("no assembled bundle (build/runtime)")
    update = "Update claude_code_host_environment in desktop/runtime/autogpt_desktop/settings.py."
    mem0 = bundled_sources("mem0")
    for module in ("configs/base.py", "memory/setup.py"):
        assert 'os.environ.get("MEM0_DIR")' in mem0[module], (
            f"mem0/{module} no longer takes its directory from MEM0_DIR: "
            f"mem0 would write ~/.mem0 into the user's home. {update}"
        )
    homes = sorted(module for module, source in mem0.items() if "expanduser(" in source)
    assert homes == ["configs/base.py", "memory/setup.py"], (
        f"mem0 looks for the home directory in {homes} now; see that each is sent to the "
        f"data directory. {update}"
    )
    cli = claude_code.bundled_cli(Bundle(REAL_BUNDLE))
    assert cli, "the bundle has no Claude Code CLI: run the deps step"
    with open(cli, "rb") as program:
        found = any(
            b"DISABLE_AUTOUPDATER" in chunk
            for chunk in overlapping_chunks(program, 1 << 24, 32)
        )
    assert found, (
        "the bundled Claude Code CLI no longer knows DISABLE_AUTOUPDATER: started in the "
        f"user's home, it could update the user's own install. {update}"
    )


def bundled_sources(package: str) -> dict[str, str]:
    """The Python sources of a third-party package in the real bundle, by
    their path inside the package: from site/, or from the archive in
    site-zip/ where the build puts a package that is nothing but source
    (build/site_zip.py)."""
    directory = REAL_BUNDLE / "site" / package
    if directory.is_dir():
        return {
            path.relative_to(directory).as_posix(): path.read_text(
                encoding="utf-8", errors="replace"
            )
            for path in directory.rglob("*.py")
        }
    prefix = f"{package}/"
    found: dict[str, str] = {}
    for zipped in sorted((REAL_BUNDLE / "site-zip").glob("*.zip")):
        with zipfile.ZipFile(zipped) as archive:
            found.update(
                {
                    name.removeprefix(prefix): archive.read(name).decode(
                        "utf-8", errors="replace"
                    )
                    for name in archive.namelist()
                    if name.startswith(prefix) and name.endswith(".py")
                }
            )
    return found


def overlapping_chunks(file, size: int, overlap: int):
    """`file` in pieces that share `overlap` bytes, so that a name lying
    across two pieces is whole in one of them."""
    previous = b""
    while piece := file.read(size):
        yield previous[-overlap:] + piece
        previous = piece


def test_base_env_hands_the_services_no_proxy_and_no_claude_setting(
    monkeypatch: pytest.MonkeyPatch,
):
    """Why the CLI is asked with the service's environment and not the
    runtime's: none of these reaches a service, whatever the user's shell
    exports."""
    names = (
        "HTTPS_PROXY",
        "CLAUDE_CONFIG_DIR",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "ANTHROPIC_API_KEY",
    )
    for name in names:
        monkeypatch.setenv(name, "from-the-users-shell")
    assert not set(names) & base_env().keys()
