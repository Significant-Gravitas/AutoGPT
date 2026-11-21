"""The macOS service is named for a domain the project owns.

A launchd label is the reversed name of a domain its author owns. The
LaunchAgent's was built on a domain that isn't the project's, the one the
default platform address and the docs' endpoints had named before they were
moved to agpt.co. The label is now `co.agpt.shim`, in the file `install`
writes, in every command the shim prints, and in what `uninstall` asks
launchd.

A Mac that installed the service under the earlier label still has it. Left
alone it would start a second shim for the same computer at every login,
beside the one under the new label, and the platform lets one connection per
computer stand. So `install` and `uninstall` stop it, unload it and remove
its file, and say nothing when there is none.
"""

from __future__ import annotations

import plistlib
import re
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from autogpt_local_executor import cli
from autogpt_local_executor import install as install_mod

ROOT = Path(__file__).resolve().parent.parent
LABEL = "co.agpt.shim"
# The earlier label, which only install.py spells (tests/test_packaging.py).
FORMER = getattr(install_mod, "FORMER_LAUNCHD_LABEL", "the-earlier-label")


def _result(returncode: int | None, output: str = "", failure: str | None = None) -> Any:
    return install_mod.CommandResult(returncode, output, failure)


class _Launchd:
    """launchd, as `launchctl print` and `launchctl bootout` show it:
    `loaded` holds each service as `<domain>/<label>`. A service in `stuck`
    stays loaded whatever it is told."""

    def __init__(self, *loaded: str, stuck: Sequence[str] = ()) -> None:
        self.loaded = set(loaded)
        self.stuck = set(stuck)
        self.ran: list[str] = []

    def __call__(self, argv: Sequence[str], timeout: float) -> Any:
        assert timeout > 0
        self.ran.append(" ".join(argv))
        tool, verb, target = argv
        assert tool == "launchctl"
        if verb == "print":
            if target in self.loaded:
                return _result(0, f"{target} = {{\n\tstate = running\n}}\n")
            return _result(113, f'Could not find service "{target}"\n')
        assert verb == "bootout"
        if target not in self.loaded:
            return _result(3, "Boot-out failed: 3: No such process\n")
        if target in self.stuck:
            return _result(5, "Boot-out failed: 5: Input/output error\n")
        self.loaded.discard(target)
        return _result(0)

    def asked_about(self, label: str) -> list[str]:
        return [line for line in self.ran if line.endswith(f"/{label}")]


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A home folder of the test's own for the autostart entry."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(install_mod, "default_stdout_log", lambda: tmp_path / "logs" / "out.log")
    monkeypatch.setattr(install_mod, "default_stderr_log", lambda: tmp_path / "logs" / "err.log")
    monkeypatch.setattr(install_mod, "default_config_path", lambda: tmp_path / "config.toml")
    monkeypatch.setattr(install_mod, "resolve_shim_command", lambda: ("/opt/bin/autogpt-shim",))
    monkeypatch.setattr(install_mod, "_user_id", lambda: 501)
    monkeypatch.setattr(install_mod, "_current_platform_for_install", lambda: "darwin")
    # No wall-clock waits for a service that never unloads.
    now = [1000.0]
    monkeypatch.setattr(install_mod, "_monotonic", lambda: now[0])
    monkeypatch.setattr(install_mod, "_sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    for field in ("platform_url", "platform_ws_url", "platform_oauth_url", "oauth_client_id"):
        monkeypatch.delenv(f"AUTOGPT_SHIM_{field.upper()}", raising=False)
    return tmp_path


def _agents(home: Path) -> Path:
    return home / "Library" / "LaunchAgents"


def _former_plist(home: Path) -> Path:
    """The file an earlier `install` wrote, as it left it."""
    plist = _agents(home) / f"{FORMER}.plist"
    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_bytes(plistlib.dumps({"Label": FORMER, "ProgramArguments": ["autogpt-shim"]}))
    return plist


def _run_cli(monkeypatch: pytest.MonkeyPatch, run: Any, *arguments: str) -> int:
    monkeypatch.setattr(install_mod, "run_command", run)
    monkeypatch.setattr(sys, "argv", ["autogpt-shim", *arguments])
    try:
        cli.main()
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return 0


# ── The label ────────────────────────────────────────────────────────────────


def test_the_label_is_in_a_domain_the_project_owns() -> None:
    assert install_mod.LAUNCHD_LABEL == LABEL
    # The other two services keep their names.
    assert install_mod.SYSTEMD_UNIT_NAME == "autogpt-shim.service"
    assert install_mod.TASKSCHED_TASK_NAME == r"\AutoGPT\Shim"


def test_the_rendered_file_has_the_label_and_is_named_for_it(home: Path) -> None:
    status = install_mod.install_autostart(home=home, platform_name="darwin", run=_Launchd())

    assert status.target_path == _agents(home) / "co.agpt.shim.plist"
    rendered = status.target_path.read_text(encoding="utf-8")
    assert plistlib.loads(rendered.encode("utf-8"))["Label"] == LABEL
    assert "~/Library/LaunchAgents/co.agpt.shim.plist" in rendered
    assert FORMER not in rendered


def test_install_and_status_print_the_file_under_the_label(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = home / "config.toml"
    config.write_text("", encoding="utf-8")

    assert _run_cli(monkeypatch, _Launchd(), "--config", str(config), "install") == 0

    out = capsys.readouterr().out
    plist = _agents(home) / "co.agpt.shim.plist"
    assert f"Wrote autostart entry: {plist}" in out
    assert f"  launchctl load {shlex.quote(str(plist))}" in out

    cli._print_autostart_status()

    assert f"  enable: launchctl load {shlex.quote(str(plist))}" in capsys.readouterr().out


def test_the_command_that_loads_the_service_keeps_a_path_with_a_space_whole(
    tmp_path: Path,
) -> None:
    """A home folder like `/Volumes/User Homes/alex`: printed bare, its path
    was two arguments to launchctl, and the command `install` and `status`
    give to copy loaded nothing."""
    home = tmp_path / "User Homes" / "alex"
    plist = _agents(home) / "co.agpt.shim.plist"

    command = install_mod.status_for("darwin", home).enable_command

    assert command is not None
    assert shlex.split(command) == ["launchctl", "load", str(plist)]
    if sys.platform != "win32":
        # A path with nothing a shell reads is printed as it is. (On a
        # Windows host these tests' paths have backslashes, which one does.)
        plain = install_mod.status_for("darwin", Path("/Users/alex")).enable_command
        assert plain == "launchctl load /Users/alex/Library/LaunchAgents/co.agpt.shim.plist"


def test_a_stopped_launchagent_is_restarted_by_its_label(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """What the shim prints when it stops under launchd for something only
    the person can fix: a configuration error, and a sign-in that failed."""
    monkeypatch.setenv("AUTOGPT_SHIM_SUPERVISOR", "launchd")

    assert cli._config_error_exit_code() == 0

    assert "`launchctl kickstart -k gui/$(id -u)/co.agpt.shim`" in capsys.readouterr().err

    cli._print_launchagent_stopped("Run `autogpt-shim auth`")

    said = capsys.readouterr().err
    assert (
        "Run `autogpt-shim auth`, then run `launchctl kickstart -k gui/$(id -u)/co.agpt.shim`"
        in said
    )


def test_every_command_the_cli_prints_for_launchd_names_the_label() -> None:
    """Each `launchctl` command in what the CLI prints, read from its
    source: none spells a label of its own."""
    source = (ROOT / "autogpt_local_executor" / "cli.py").read_text(encoding="utf-8")
    commands = re.findall(r"launchctl [a-z]+ [^`\"\n]*", source)

    assert commands == ["launchctl kickstart -k gui/$(id -u)/{LAUNCHD_LABEL}"]
    assert FORMER not in source and "net." + "autogpt" not in source


def test_uninstall_asks_launchd_by_the_label(home: Path) -> None:
    target = install_mod.install_autostart(
        home=home, platform_name="darwin", run=_Launchd()
    ).target_path
    launchd = _Launchd(f"gui/501/{LABEL}")

    done = install_mod.uninstall_autostart(home=home, platform_name="darwin", run=launchd)

    assert launchd.asked_about(LABEL) == [
        f"launchctl print gui/501/{LABEL}",
        f"launchctl bootout gui/501/{LABEL}",
        f"launchctl print gui/501/{LABEL}",
        f"launchctl print user/501/{LABEL}",
    ]
    assert done.service.stop_command == f"launchctl bootout gui/$(id -u)/{LABEL}"
    assert done.service.found is True and done.service.problem is None
    assert not target.exists()


# ── A service left under the earlier label ───────────────────────────────────


def test_the_earlier_label_is_another_one() -> None:
    assert FORMER == install_mod.FORMER_LAUNCHD_LABEL
    assert FORMER != LABEL and FORMER.endswith(".shim")


def test_install_stops_and_removes_a_service_under_the_earlier_label(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = home / "config.toml"
    config.write_text("", encoding="utf-8")
    former = _former_plist(home)
    launchd = _Launchd(f"gui/501/{FORMER}")

    assert _run_cli(monkeypatch, launchd, "--config", str(config), "install") == 0

    out, err = capsys.readouterr()
    # Stopped and unloaded, by its label, and its file removed.
    assert f"launchctl bootout gui/501/{FORMER}" in launchd.ran
    assert launchd.loaded == set()
    assert not former.exists()
    assert (_agents(home) / "co.agpt.shim.plist").is_file()
    # The new service isn't loaded by `install`: that stays the person's step.
    assert launchd.asked_about(LABEL) == []
    assert f"launchd unloaded {FORMER} from gui/501" in out
    assert f"Removed the earlier autostart entry: {former}" in out
    assert err == ""


def test_install_says_nothing_of_it_when_there_is_none(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = home / "config.toml"
    config.write_text("", encoding="utf-8")
    launchd = _Launchd()

    assert _run_cli(monkeypatch, launchd, "--config", str(config), "install") == 0

    out, err = capsys.readouterr()
    # launchd was asked, in both of the person's sessions, and had none.
    assert launchd.ran == [
        f"launchctl print gui/501/{FORMER}",
        f"launchctl print user/501/{FORMER}",
    ]
    assert FORMER not in out + err
    assert "earlier" not in out + err


def test_install_removes_an_earlier_entry_whose_service_is_not_loaded(home: Path) -> None:
    former = _former_plist(home)
    launchd = _Launchd()

    status = install_mod.install_autostart(home=home, platform_name="darwin", run=launchd)

    assert not former.exists()
    assert not any(line.startswith("launchctl bootout") for line in launchd.ran)
    assert f"Removed the earlier autostart entry: {former}" in status.notes


def test_install_warns_when_the_earlier_service_cannot_be_stopped(home: Path) -> None:
    """Two services would start two shims for this computer, so the person
    is told before they load the new one."""
    former = _former_plist(home)
    launchd = _Launchd(f"gui/501/{FORMER}", stuck=[f"gui/501/{FORMER}"])

    status = install_mod.install_autostart(home=home, platform_name="darwin", run=launchd)

    notes = " ".join(status.notes)
    assert status.target_path.is_file()
    assert not former.exists()
    assert "two shims" in notes
    assert f"launchctl bootout gui/$(id -u)/{FORMER}" in notes


@pytest.mark.parametrize("current", [True, False], ids=["both-installed", "only-the-earlier"])
def test_uninstall_stops_and_removes_a_service_under_the_earlier_label(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    current: bool,
) -> None:
    loaded = [f"user/501/{FORMER}"]
    if current:
        install_mod.install_autostart(home=home, platform_name="darwin", run=_Launchd())
        loaded.append(f"gui/501/{LABEL}")
    former = _former_plist(home)
    launchd = _Launchd(*loaded)

    assert _run_cli(monkeypatch, launchd, "uninstall") == 0

    out, err = capsys.readouterr()
    assert launchd.loaded == set()
    assert f"launchctl bootout user/501/{FORMER}" in launchd.ran
    assert not former.exists()
    assert not (_agents(home) / "co.agpt.shim.plist").exists()
    assert f"launchd unloaded {FORMER} from user/501" in out
    assert f"Removed the earlier autostart entry: {former}" in out
    assert err == ""


def test_uninstall_says_nothing_of_it_when_there_is_none(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_mod.install_autostart(home=home, platform_name="darwin", run=_Launchd())
    launchd = _Launchd(f"gui/501/{LABEL}")

    assert _run_cli(monkeypatch, launchd, "uninstall") == 0

    out, err = capsys.readouterr()
    assert launchd.asked_about(FORMER) == [
        f"launchctl print gui/501/{FORMER}",
        f"launchctl print user/501/{FORMER}",
    ]
    assert FORMER not in out + err
    assert "earlier" not in out + err


def test_uninstall_fails_when_the_earlier_service_cannot_be_stopped(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """It may still be running, which is what the exit status 1 says."""
    install_mod.install_autostart(home=home, platform_name="darwin", run=_Launchd())
    former = _former_plist(home)
    launchd = _Launchd(f"gui/501/{LABEL}", f"gui/501/{FORMER}", stuck=[f"gui/501/{FORMER}"])

    assert _run_cli(monkeypatch, launchd, "uninstall") == 1

    out, err = capsys.readouterr()
    assert f"gui/501/{LABEL}" not in launchd.loaded
    assert f"gui/501/{FORMER}" in launchd.loaded
    assert not former.exists()
    assert "The service may still be running" in err
    assert f"  launchctl bootout gui/$(id -u)/{FORMER}" in err
    # What did go through is still said.
    assert f"launchd unloaded {LABEL} from gui/501" in out


def test_no_other_os_looks_for_it(home: Path) -> None:
    def systemd(argv: Sequence[str], timeout: float) -> Any:
        assert argv[0] == "systemctl", argv
        return _result(0, "ActiveState=inactive\n")

    _former_plist(home)

    install_mod.install_autostart(home=home, platform_name="linux", run=systemd)
    install_mod.uninstall_autostart(home=home, platform_name="linux", run=systemd)

    assert (_agents(home) / f"{FORMER}.plist").is_file()


# ── The notes and the docs ───────────────────────────────────────────────────


def test_the_homebrew_notes_say_what_install_and_uninstall_do() -> None:
    formula = (ROOT / "packaging" / "homebrew" / "autogpt-local-executor.rb").read_text(
        encoding="utf-8"
    )
    caveats = formula[formula.index("def caveats") : formula.rindex("test do")]

    # `install` writes the file and prints the command that loads it: the
    # notes had it registering the service, and spelled a command of their
    # own with a file name in it.
    assert "launchctl load" not in caveats
    assert "LaunchAgents/" not in caveats
    assert "register the launchd LaunchAgent" not in caveats
    said = " ".join(caveats.split())
    assert "writes the LaunchAgent" in said and "the command it prints" in said
    assert "stops the service, unloads it and removes the LaunchAgent" in said


@pytest.mark.parametrize(
    "document", ["docs/CROSS_PLATFORM.md", "docs/CUA_DRIVER.md", "docs/COMPUTER_USE.md"]
)
def test_the_docs_name_the_label(document: str) -> None:
    text = (ROOT / document).read_text(encoding="utf-8")

    assert "co.agpt.shim" in text
    assert FORMER not in text


def test_cross_platform_md_says_what_happens_to_the_earlier_service() -> None:
    text = " ".join((ROOT / "docs" / "CROSS_PLATFORM.md").read_text(encoding="utf-8").split())

    assert "earlier label" in text
    assert "`FORMER_LAUNCHD_LABEL`" in text


# ── Review: what can go wrong around the earlier service ─────────────────────


def test_install_leaves_the_earlier_service_alone_when_the_new_entry_cannot_be_written(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The earlier service was unloaded and its file removed before the new
    entry was written. With the disk full at that write, the Mac was left
    with no service at all for the next login."""
    former = _former_plist(home)
    launchd = _Launchd(f"gui/501/{FORMER}")

    def disk_full(*_args: Any, **_kwargs: Any) -> Path:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(install_mod, "_write_atomic", disk_full)

    with pytest.raises(OSError, match="No space left"):
        install_mod.install_autostart(home=home, platform_name="darwin", run=launchd)

    assert launchd.loaded == {f"gui/501/{FORMER}"}
    assert not any(line.startswith("launchctl bootout") for line in launchd.ran)
    assert former.is_file()


def test_uninstall_fails_when_launchd_cannot_be_asked_about_the_earlier_service(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only the earlier service is installed, and launchd doesn't answer.
    Its file is removed, but nothing says its service has stopped: the
    exit status was 0 all the same, since the entry under the new label
    wasn't there."""
    former = _former_plist(home)

    def no_answer(argv: Sequence[str], timeout: float) -> Any:
        return _result(None, failure=f"`{' '.join(argv)}` did not finish within {timeout:g} s")

    assert _run_cli(monkeypatch, no_answer, "uninstall") == 1

    out, err = capsys.readouterr()
    assert not former.exists()
    assert "may still be running" in err
    assert f"launchctl bootout gui/$(id -u)/{FORMER}" in err


def test_uninstall_fails_when_the_earlier_entry_cannot_be_removed(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Left there, it loads the earlier service again at the next login."""
    former = _former_plist(home)
    launchd = _Launchd(f"gui/501/{FORMER}")
    unlink = Path.unlink

    def not_this_one(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == former:
            raise PermissionError(13, "Permission denied")
        unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", not_this_one)

    assert _run_cli(monkeypatch, launchd, "uninstall") == 1

    out, err = capsys.readouterr()
    assert launchd.loaded == set()
    assert former.is_file()
    assert f"Could not remove the earlier autostart entry {former}" in err
    assert "Permission denied" in err


def test_install_says_so_when_the_earlier_entry_cannot_be_removed(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    former = _former_plist(home)
    unlink = Path.unlink

    def not_this_one(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == former:
            raise PermissionError(13, "Permission denied")
        unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", not_this_one)

    status = install_mod.install_autostart(home=home, platform_name="darwin", run=_Launchd())

    notes = " ".join(status.notes)
    assert status.target_path.is_file()
    assert f"Could not remove the earlier autostart entry {former}" in notes
    assert "two shims" in notes
