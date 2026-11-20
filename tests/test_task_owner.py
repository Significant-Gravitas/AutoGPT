"""The Windows task belongs to the account that installs it.

The task's logon trigger and its principal named no account, which Task
Scheduler reads as "at any account's logon". Only an administrator may
register that, so `install` told people to use an elevated prompt. A person
without administrator rights who opened one with an administrator's
credentials registered the task for the administrator: the shim then ran at
the administrator's logon, with the administrator's Credential Manager, and
never for the person who installed it.

`install` now writes the account it runs as into both places. A task for one
account's own logon is that account's to register, from its own prompt.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from autogpt_local_executor import cli
from autogpt_local_executor import install as install_mod
from tests.test_uninstall import real_service_manager_asked_for

# The real command runner, kept before conftest.py stands a refusing one in
# for it: only the test at the end, on a task of its own, may use it.
_REAL_RUN_COMMAND = install_mod.run_command

ROOT = Path(__file__).resolve().parent.parent
NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
TASK = r"\AutoGPT\Shim"
SID = "S-1-5-21-3623811015-3361044348-30300820-1013"
WHOAMI = ["whoami", "/user", "/fo", "csv", "/nh"]


def _result(returncode: int | None, output: str = "", failure: str | None = None) -> Any:
    return install_mod.CommandResult(returncode, output, failure)


def _context(user_id: str | None) -> install_mod.AutostartContext:
    return install_mod.AutostartContext(
        shim_command=(r"C:\Tools\autogpt-shim.exe",),
        config_path=r"C:\Users\alice\config.toml",
        stdout_log=r"C:\logs\out.log",
        stderr_log=r"C:\logs\err.log",
        user_id=user_id,
    )


def _user_ids(xml: bytes | str) -> tuple[str | None, str | None]:
    """The account in the task's logon trigger, and the one in its principal."""
    task = ET.fromstring(xml)
    return (
        task.findtext("t:Triggers/t:LogonTrigger/t:UserId", namespaces=NS),
        task.findtext("t:Principals/t:Principal/t:UserId", namespaces=NS),
    )


# ── What the task says ───────────────────────────────────────────────────────


def test_the_task_names_the_account_in_its_trigger_and_in_its_principal() -> None:
    rendered = install_mod.render_template("taskscheduler.xml.template", _context(SID))

    assert _user_ids(rendered) == (SID, SID)
    # It still runs with the account's own rights, in its desktop session.
    task = ET.fromstring(rendered)
    principal = task.find("t:Principals/t:Principal", NS)
    assert principal is not None
    assert principal.findtext("t:LogonType", namespaces=NS) == "InteractiveToken"
    assert principal.findtext("t:RunLevel", namespaces=NS) == "LeastPrivilege"


@pytest.mark.parametrize("account", ["R&D\\<ops>", 'CORP\\o\'neil "jr"', "a]]>b\\c"])
def test_an_account_is_escaped_for_xml(account: str) -> None:
    """A Windows account name may hold `&`. Whatever the context holds comes
    back from the file as it went in, and can't add markup of its own."""
    rendered = install_mod.render_template("taskscheduler.xml.template", _context(account))

    assert _user_ids(rendered) == (account, account)
    if "&" in account:
        assert "R&amp;D\\&lt;ops&gt;" in rendered
    assert len(ET.fromstring(rendered).findall(".//t:UserId", NS)) == 2


def test_a_task_that_names_no_account_is_not_written(tmp_path: Path) -> None:
    """One would be a task for every account's logon again."""
    with pytest.raises(ValueError, match="account"):
        install_mod.render_template("taskscheduler.xml.template", _context(None))
    with pytest.raises(ValueError, match="account"):
        install_mod.write_windows_task_xml(_context(None), tmp_path / "task.xml")
    assert not (tmp_path / "task.xml").exists()


def test_an_account_with_a_control_character_in_it_is_refused() -> None:
    with pytest.raises(ValueError, match="control character"):
        install_mod.render_template("taskscheduler.xml.template", _context("CORP\\al\nice"))


@pytest.mark.parametrize("template", ["launchd.plist.template", "systemd.service.template"])
def test_the_other_service_files_need_no_account(template: str) -> None:
    """launchd and systemd keep each account's services apart themselves."""
    assert install_mod.render_template(template, _context(None))


# ── Whose account ────────────────────────────────────────────────────────────


class _Whoami:
    """`whoami /user /fo csv /nh`, which prints the account's name and its
    SID as one CSV line."""

    def __init__(self, output: str = f'"desktop-7f3k\\alice","{SID}"\r\n', code: int = 0) -> None:
        self.output = output
        self.code = code
        self.ran: list[list[str]] = []

    def __call__(self, argv: Sequence[str], timeout: float) -> Any:
        assert timeout > 0
        self.ran.append(list(argv))
        assert list(argv) == WHOAMI, argv
        return _result(self.code, self.output)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A home folder of the test's own for the autostart entry."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    monkeypatch.setattr(install_mod, "default_stdout_log", lambda: tmp_path / "logs" / "out.log")
    monkeypatch.setattr(install_mod, "default_stderr_log", lambda: tmp_path / "logs" / "err.log")
    monkeypatch.setattr(install_mod, "default_config_path", lambda: tmp_path / "config.toml")
    monkeypatch.setattr(install_mod, "resolve_shim_command", lambda: ("/opt/bin/autogpt-shim",))
    monkeypatch.setattr(install_mod, "_user_id", lambda: 501)
    for field in ("platform_url", "platform_ws_url", "platform_oauth_url", "oauth_client_id"):
        monkeypatch.delenv(f"AUTOGPT_SHIM_{field.upper()}", raising=False)
    return tmp_path


def test_install_writes_the_account_this_process_runs_as(home: Path) -> None:
    whoami = _Whoami()

    status = install_mod.install_autostart(home=home, platform_name="windows", run=whoami)

    assert whoami.ran == [WHOAMI]
    assert _user_ids(status.target_path.read_bytes()) == (SID, SID)


@pytest.mark.parametrize(
    "output",
    [
        f'"S-1-5-18\\S-1-5-19","{SID}"\r\n',
        f'"corp\\smith, john","{SID}"\r\n',
        f'"corp\\al ""the"" ice","{SID}"\n',
        f'\r\n"desktop\\alice","{SID}"\r\n\r\n',
    ],
    ids=["a-name-that-looks-like-a-sid", "a-comma-in-the-name", "a-quote-in-the-name", "blanks"],
)
def test_the_sid_is_read_from_its_own_column(home: Path, output: str) -> None:
    status = install_mod.install_autostart(home=home, platform_name="windows", run=_Whoami(output))

    assert _user_ids(status.target_path.read_bytes()) == (SID, SID)


@pytest.mark.parametrize(
    "whoami",
    [
        _Whoami("ERROR: Invalid argument/option - '/user'.\n", code=1),
        _Whoami("alice\n"),
        _Whoami(""),
        _Whoami(f'"desktop\\alice","{SID}"\r\n', code=1),
        lambda argv, timeout: _result(None, failure="`whoami` was not found on PATH"),
    ],
    ids=["refused", "no-sid", "nothing", "failed-with-a-sid", "not-found"],
)
def test_install_writes_no_task_when_the_account_cannot_be_read(home: Path, whoami: Any) -> None:
    """A task with no account in it would be the any-account one again, so
    nothing is written, and an entry already there is left as it was."""
    target = install_mod.target_path_for("windows", home)
    target.parent.mkdir(parents=True)
    target.write_text("the entry that was there", encoding="utf-8")

    with pytest.raises(install_mod.AutostartError, match="whoami"):
        install_mod.install_autostart(home=home, platform_name="windows", run=whoami)

    assert target.read_text(encoding="utf-8") == "the entry that was there"


@pytest.mark.parametrize("platform_name", ["darwin", "linux", "wsl2"])
def test_only_the_windows_task_asks_for_the_account(home: Path, platform_name: str) -> None:
    def no_question(argv: Sequence[str], timeout: float) -> Any:
        # macOS asks launchd one thing, of its own: whether a LaunchAgent is
        # left under the earlier label (tests/test_launchd_label.py).
        assert argv[0] == "launchctl" and platform_name == "darwin", f"{platform_name}: {argv}"
        return _result(113, "Could not find service\n")

    status = install_mod.install_autostart(home=home, platform_name=platform_name, run=no_question)

    assert status.target_path.is_file()


# ── What a person is told ────────────────────────────────────────────────────


def _run_cli(monkeypatch: pytest.MonkeyPatch, run: Any, *arguments: str) -> int:
    monkeypatch.setattr(install_mod, "_current_platform_for_install", lambda: "windows")
    monkeypatch.setattr(install_mod, "run_command", run)
    monkeypatch.setattr(sys, "argv", ["autogpt-shim", *arguments])
    try:
        cli.main()
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return 0


def _mentions_elevation(text: str) -> bool:
    return bool(re.search(r"elevat|administrator|run as admin", text, re.I))


def test_install_no_longer_asks_for_an_elevated_prompt(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = home / "config.toml"
    config.write_text("", encoding="utf-8")

    assert _run_cli(monkeypatch, _Whoami(), "--config", str(config), "install") == 0

    out, err = capsys.readouterr()
    target = install_mod.target_path_for("windows", home)
    assert f'schtasks /Create /TN "{TASK}" /XML "{target}"' in out
    said = " ".join(out.split())
    assert "(elevated" not in said and "elevated cmd" not in said
    # It says whose task it is, and that this account registers it itself.
    assert "this account" in said
    assert "no elevated prompt" in said
    assert err == ""


def test_install_says_why_when_the_account_cannot_be_read(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = home / "config.toml"
    config.write_text("", encoding="utf-8")
    refused = _Whoami("ERROR: Access is denied.\n", code=1)

    assert _run_cli(monkeypatch, refused, "--config", str(config), "install") == 1

    out, err = capsys.readouterr()
    assert "Could not write the autostart entry" in err
    assert "whoami" in err and "Access is denied" in err
    assert "Wrote autostart entry" not in out
    assert not install_mod.target_path_for("windows", home).exists()


def test_status_of_the_task_no_longer_asks_for_an_elevated_prompt(home: Path) -> None:
    status = install_mod.status_for("windows", home)

    assert not _mentions_elevation(" ".join(status.notes).replace("no elevated prompt", ""))
    assert "no elevated prompt" in " ".join(status.notes)


def test_the_rendered_task_file_no_longer_asks_for_an_elevated_prompt() -> None:
    rendered = install_mod.render_template("taskscheduler.xml.template", _context(SID))

    assert "from an elevated prompt" not in rendered
    assert "autogpt-shim uninstall" in rendered


def test_the_scoop_notes_no_longer_ask_for_an_elevated_prompt() -> None:
    manifest = json.loads(
        (ROOT / "packaging" / "scoop" / "autogpt-local-executor.json").read_text(encoding="utf-8")
    )
    notes = "\n".join(manifest["notes"])

    assert not _mentions_elevation(notes.replace("no elevated prompt", ""))
    assert "autogpt-shim install" in notes
    assert "autogpt-shim uninstall" in notes
    # The command to register the task is the one `install` prints, with the
    # file's real path: `%LOCALAPPDATA%` is cmd.exe's spelling, and Scoop runs
    # in PowerShell, which doesn't expand it.
    assert "%LOCALAPPDATA%" not in notes
    assert "the command it prints" in notes


def test_cross_platform_md_says_whose_the_task_is_and_how_many_a_computer_has() -> None:
    document = " ".join((ROOT / "docs" / "CROSS_PLATFORM.md").read_text(encoding="utf-8").split())
    start = document.index("#### The Windows task's account")
    section = document[start : document.index("####", start + 10)]

    assert "`whoami`" in section and "SID" in section
    assert "no elevated prompt" in section
    # Task Scheduler's names are the whole computer's: say so where a
    # second account on the computer will look.
    assert "one task of a name for the whole computer" in section
    assert "elevated cmd" not in document


# ── Uninstalling a task that is the account's own ────────────────────────────


class _TaskScheduler:
    """Task Scheduler, as schtasks shows it to a prompt that isn't
    elevated: the account's own task, which it may end and delete, or
    nothing."""

    def __init__(self, *, registered: bool) -> None:
        self.registered = registered
        self.ran: list[str] = []

    def __call__(self, argv: Sequence[str], timeout: float) -> Any:
        self.ran.append(" ".join(argv))
        tool, verb, flag, task, *_rest = argv
        assert (tool, flag, task) == ("schtasks", "/TN", TASK)
        if not self.registered:
            return _result(1, "ERROR: The system cannot find the file specified.\n")
        if verb == "/Delete":
            self.registered = False
        return _result(0, "SUCCESS\n")


def _installed(home: Path) -> Path:
    status = install_mod.install_autostart(home=home, platform_name="windows", run=_Whoami())
    return status.target_path


def test_the_accounts_own_task_is_ended_and_deleted_without_elevation(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(install_mod, "_is_elevated", lambda: False)
    target = _installed(home)
    scheduler = _TaskScheduler(registered=True)

    assert _run_cli(monkeypatch, scheduler, "uninstall") == 0

    out, err = capsys.readouterr()
    assert scheduler.ran == [
        f"schtasks /Query /TN {TASK}",
        f"schtasks /End /TN {TASK}",
        f"schtasks /Delete /TN {TASK} /F",
    ]
    assert f"Task Scheduler ended and deleted the task {TASK}" in out
    assert err == ""
    assert not target.exists()


@pytest.mark.parametrize("installed", [True, False])
def test_an_entry_whose_task_was_never_registered_is_removed_without_a_failure(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    installed: bool,
) -> None:
    """From a prompt that isn't elevated, as `install` leaves it when the
    schtasks command was never run. While registering took an elevated
    prompt, a task this prompt couldn't see might still have been this
    install's own, so `uninstall` failed and sent the person to an elevated
    prompt. The account's own task is one its own prompt sees."""
    monkeypatch.setattr(install_mod, "_is_elevated", lambda: False)
    target = _installed(home) if installed else install_mod.target_path_for("windows", home)
    scheduler = _TaskScheduler(registered=False)

    assert _run_cli(monkeypatch, scheduler, "uninstall") == 0

    out, err = capsys.readouterr()
    assert scheduler.ran == [f"schtasks /Query /TN {TASK}"]
    assert err == ""
    assert f"this account has no task {TASK} registered" in " ".join(out.split())
    assert "may still be running" not in out
    assert not target.exists()


def test_a_plain_prompt_still_says_what_it_cannot_see(home: Path) -> None:
    """A task registered from an elevated prompt can be hidden from a plain
    one. That is no longer how `install` says to register it, so it is a
    remark and not a failure."""
    done = install_mod._stop_scheduled_task(
        _TaskScheduler(registered=False), task=TASK, elevated=False
    )

    assert (done.found, done.problem) == (False, None)
    assert "registered from an elevated prompt" in done.said
    assert "`autogpt-shim uninstall` from one" in done.said


def test_an_elevated_prompt_says_plainly_that_no_task_is_registered(home: Path) -> None:
    done = install_mod._stop_scheduled_task(
        _TaskScheduler(registered=False), task=TASK, elevated=True
    )

    assert (done.found, done.problem) == (False, None)
    assert done.said == f"The service wasn't set to start: no task {TASK} is registered."


# ── The real Task Scheduler, on a task of the test's own ─────────────────────


def _real(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=120,
        check=False,
    )


@pytest.mark.skipif(
    not real_service_manager_asked_for(),
    reason="registers a throwaway task with the OS: runs in CI, or with "
    "AUTOGPT_SHIM_TEST_SERVICE_MANAGER=1",
)
@pytest.mark.skipif(sys.platform != "win32", reason="Task Scheduler is Windows'")
def test_real_task_scheduler_registers_the_task_for_the_account_that_installed_it(
    home: Path,
) -> None:
    """The file `install` writes, with the account the real `whoami` gives,
    registered by the real schtasks under a name of the test's own, read
    back, and then ended and deleted as `uninstall` does it."""
    status = install_mod.install_autostart(
        home=home, platform_name="windows", run=_REAL_RUN_COMMAND
    )
    written = _user_ids(status.target_path.read_bytes())
    asked = _real(WHOAMI)
    assert asked.returncode == 0, asked.stdout + asked.stderr
    name, sid = next(row for row in csv.reader(asked.stdout.splitlines()) if row)
    assert written == (sid, sid)
    assert re.fullmatch(r"S-1-\d+(-\d+)+", sid)

    task = rf"\AutoGPT-test\Owner-{uuid.uuid4().hex[:12]}"
    made = _real(["schtasks", "/Create", "/TN", task, "/XML", str(status.target_path)])
    assert made.returncode == 0, made.stdout + made.stderr
    try:
        shown = _real(["schtasks", "/Query", "/TN", task, "/XML"])
        assert shown.returncode == 0, shown.stdout + shown.stderr
        # Task Scheduler shows an account by its SID or by its name.
        accounts = re.findall(r"<UserId>([^<]*)</UserId>", shown.stdout)
        assert len(accounts) == 2, shown.stdout
        for account in accounts:
            assert account.lower() in (sid.lower(), name.lower()), shown.stdout
        assert "<LogonTrigger>" in shown.stdout
        assert "<LogonType>InteractiveToken</LogonType>" in shown.stdout
        # Task Scheduler leaves out what is at its default, the run level
        # among them: only a task that runs elevated has one written.
        assert "HighestAvailable" not in shown.stdout
        # The supervising `start` is what it runs.
        assert "--supervise</Arguments>" in shown.stdout

        # The task was registered and has never run. `schtasks /End` succeeds
        # for it, with no instance to end, so the delete after it is reached.
        stopped = install_mod._stop_scheduled_task(
            _REAL_RUN_COMMAND, task=task, elevated=install_mod._is_elevated()
        )

        assert stopped.problem is None, stopped.problem
        assert stopped.found is True
        assert _real(["schtasks", "/Query", "/TN", task]).returncode != 0
    finally:
        _real(["schtasks", "/End", "/TN", task])
        _real(["schtasks", "/Delete", "/TN", task, "/F"])
