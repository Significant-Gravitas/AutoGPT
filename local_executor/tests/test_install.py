"""Tests for `autogpt-shim install / uninstall / status`.

Covers:
- Per-OS template rendering shape (no env-leak, all placeholders bound).
- install_autostart writes to the right per-OS target (under tmp_path).
- uninstall_autostart removes the rendered file (idempotent).
- status_for returns the right shape when present vs. absent.
- Each per-OS branch is exercised by mocking platform.system().
"""

from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import pytest

from autogpt_local_executor import install as install_mod

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def ctx() -> install_mod.AutostartContext:
    return install_mod.AutostartContext(
        shim_command=("/opt/bin/autogpt-shim",),
        config_path="/opt/etc/autogpt-local-executor/config.toml",
        stdout_log="/opt/var/log/autogpt-local-executor/shim.out.log",
        stderr_log="/opt/var/log/autogpt-local-executor/shim.err.log",
    )


# ── Template rendering ───────────────────────────────────────────────────────


def test_render_macos_plist_has_all_placeholders_filled(
    ctx: install_mod.AutostartContext,
) -> None:
    out = install_mod.render_template("launchd.plist.template", ctx)
    # Every placeholder is gone.
    assert "{" not in out
    assert "{config_path}" not in out
    assert "{stdout_log}" not in out
    assert "{stderr_log}" not in out
    # Spec keys per CROSS_PLATFORM.md "Autostart" row.
    assert "<key>RunAtLoad</key>" in out
    assert "<key>KeepAlive</key>" in out
    assert "net.autogpt.shim" in out
    assert "/opt/bin/autogpt-shim" in out
    assert "/opt/etc/autogpt-local-executor/config.toml" in out
    # KeepAlive restarts any non-zero exit; the marker lets `start` exit 0
    # after an authentication failure so launchd stops retrying it.
    environment = plistlib.loads(out.encode("utf-8"))["EnvironmentVariables"]
    assert environment["AUTOGPT_SHIM_SUPERVISOR"] == "launchd"


def test_render_systemd_unit_has_all_placeholders_filled(
    ctx: install_mod.AutostartContext,
) -> None:
    out = install_mod.render_template("systemd.service.template", ctx)
    assert "{" not in out
    assert "{config_path}" not in out
    assert (
        "ExecStart=/opt/bin/autogpt-shim --config /opt/etc/autogpt-local-executor/config.toml start"
        in out
    )
    assert "Restart=on-failure" in out
    # Neither an authentication failure (77) nor a configuration or preflight
    # failure (78) fixes itself on a restart; Linux has no TCC relaunch chain.
    assert "RestartPreventExitStatus=77 78" in out
    assert "WantedBy=default.target" in out


def test_render_task_xml_has_all_placeholders_filled(
    ctx: install_mod.AutostartContext,
) -> None:
    out = install_mod.render_template("taskscheduler.xml.template", ctx)
    assert "{" not in out
    assert "{config_path}" not in out
    assert "<LogonTrigger>" in out
    assert "<Command>/opt/bin/autogpt-shim</Command>" in out
    assert (
        "<Arguments>--config /opt/etc/autogpt-local-executor/config.toml start</Arguments>" in out
    )


def test_render_no_user_specific_paths_leak(
    ctx: install_mod.AutostartContext,
) -> None:
    """Rendered output should only contain values from the context — never the
    test runner's $HOME or similar."""
    for tpl in (
        "launchd.plist.template",
        "systemd.service.template",
        "taskscheduler.xml.template",
    ):
        out = install_mod.render_template(tpl, ctx)
        home_str = str(Path.home())
        assert home_str not in out, f"{tpl} leaked the runtime $HOME"


# Valid paths that each format must quote: XML markup, spaces, and systemd's
# specifier (%) and variable ($) characters.
_AWKWARD = install_mod.AutostartContext(
    shim_command=("/opt/My Tools/autogpt-shim",),
    config_path="/home/alice/R&D <work>/50% $HOME/config.toml",
    stdout_log="/home/alice/logs & more/100%/out.log",
    stderr_log="/home/alice/logs & more/100%/err.log",
)


def _systemd_argv(exec_start: str) -> list[str]:
    """Undo systemd's specifier, quoting and variable escapes, in its order."""
    return [word.replace("$$", "$") for word in shlex.split(exec_start.replace("%%", "%"))]


def test_plist_keeps_each_argument_and_path_whole() -> None:
    plist = plistlib.loads(install_mod.render_template("launchd.plist.template", _AWKWARD).encode())

    assert plist["ProgramArguments"] == [
        "/opt/My Tools/autogpt-shim",
        "--config",
        _AWKWARD.config_path,
        "start",
    ]
    assert plist["StandardOutPath"] == _AWKWARD.stdout_log
    assert plist["StandardErrorPath"] == _AWKWARD.stderr_log


def test_systemd_unit_marks_the_shim_as_its_service() -> None:
    """The shim takes the desktop's display from the user manager only as
    this service (platform_info.SessionDisplay)."""
    lines = install_mod.render_template("systemd.service.template", _AWKWARD).splitlines()

    assert "Environment=AUTOGPT_SHIM_SUPERVISOR=systemd" in lines


def test_systemd_unit_keeps_each_argument_whole() -> None:
    lines = install_mod.render_template("systemd.service.template", _AWKWARD).splitlines()
    values = dict(line.split("=", 1) for line in lines if "=" in line and not line.startswith("#"))

    assert _systemd_argv(values["ExecStart"]) == [
        "/opt/My Tools/autogpt-shim",
        "--config",
        _AWKWARD.config_path,
        "start",
    ]
    assert values["StandardOutput"].replace("%%", "%") == f"append:{_AWKWARD.stdout_log}"
    assert values["StandardError"].replace("%%", "%") == f"append:{_AWKWARD.stderr_log}"


def test_task_xml_is_utf16_as_declared_and_quotes_the_command_line(tmp_path: Path) -> None:
    ctx = install_mod.AutostartContext(
        shim_command=(r"C:\Program Files\Python\python.exe", "-m", "autogpt_local_executor"),
        config_path=r"C:\Users\alice\R&D Files\config.toml",
        stdout_log=r"C:\logs\out.log",
        stderr_log=r"C:\logs\err.log",
    )
    target = install_mod.write_windows_task_xml(ctx, tmp_path / "task.xml")

    raw = target.read_bytes()
    assert raw.startswith(b"\xff\xfe")
    ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
    exec_action = ET.fromstring(raw).find("t:Actions/t:Exec", ns)
    assert exec_action is not None
    assert exec_action.findtext("t:Command", namespaces=ns) == (
        r'"C:\Program Files\Python\python.exe"'
    )
    assert exec_action.findtext("t:Arguments", namespaces=ns) == (
        r'-m autogpt_local_executor --config "C:\Users\alice\R&D Files\config.toml" start'
    )


def test_an_allowed_root_given_to_install_stays_one_argument_in_each_template(
    tmp_path: Path,
) -> None:
    """`install` passes the service the global options `start` needs too
    (_service_global_args), `--allowed-root` among them, and each template
    quotes them its own way: a folder with a space or a `%` in its name must
    come back whole from each."""
    root = "/home/alice/My Docs/100% done"
    ctx = install_mod.AutostartContext(
        shim_command=("/opt/bin/autogpt-shim",),
        config_path=None,
        stdout_log="/tmp/out.log",
        stderr_log="/tmp/err.log",
        global_args=("--allowed-root", root),
    )
    argv = ["/opt/bin/autogpt-shim", "--allowed-root", root, "start"]

    plist = plistlib.loads(install_mod.render_template("launchd.plist.template", ctx).encode())
    assert plist["ProgramArguments"] == argv

    lines = install_mod.render_template("systemd.service.template", ctx).splitlines()
    [exec_start] = [
        line.removeprefix("ExecStart=") for line in lines if line.startswith("ExecStart=")
    ]
    assert _systemd_argv(exec_start) == argv

    raw = install_mod.write_windows_task_xml(ctx, tmp_path / "task.xml").read_bytes()
    ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
    exec_action = ET.fromstring(raw).find("t:Actions/t:Exec", ns)
    assert exec_action is not None
    # As Windows splits a command line: the quoted folder is one argument.
    assert exec_action.findtext("t:Arguments", namespaces=ns) == (
        '--allowed-root "/home/alice/My Docs/100% done" start'
    )


def test_a_path_with_a_control_character_is_refused() -> None:
    ctx = install_mod.AutostartContext(
        shim_command=("/opt/bin/autogpt-shim",),
        config_path="/home/alice/evil\nExecStartPre=/bin/true",
        stdout_log="/tmp/out.log",
        stderr_log="/tmp/err.log",
    )

    with pytest.raises(ValueError, match="control character"):
        install_mod.render_template("systemd.service.template", ctx)


# ── Locating the shim ────────────────────────────────────────────────────────


def _which_without_path(found_on_path: str | None):
    real_which = shutil.which

    def which(name, mode=None, path=None):
        if path is None:
            return found_on_path
        return real_which(name, path=path)

    return which


def test_the_console_script_on_path_is_preferred(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Absolute on every OS: "/usr/bin/..." gains a drive letter on Windows.
    script = str(tmp_path / "bin" / "autogpt-shim")
    monkeypatch.setattr(install_mod.shutil, "which", _which_without_path(script))

    assert install_mod.resolve_shim_command() == (script,)


def test_the_console_script_beside_the_interpreter_is_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pipx or venv install run by absolute path has its script beside python."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / ("autogpt-shim.exe" if sys.platform == "win32" else "autogpt-shim")
    script.write_text("", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setattr(install_mod.sys, "executable", str(bin_dir / "python"))
    monkeypatch.setattr(install_mod.shutil, "which", _which_without_path(None))

    # Windows' which() spells the extension from PATHEXT (".EXE").
    [found] = install_mod.resolve_shim_command()
    assert os.path.normcase(found) == os.path.normcase(str(script))


def test_the_python_fallback_starts_the_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(install_mod.shutil, "which", lambda *a, **k: None)

    command = install_mod.resolve_shim_command()
    result = subprocess.run(
        [*command, "--help"], capture_output=True, text=True, timeout=120, check=False
    )

    assert command == (sys.executable, "-m", "autogpt_local_executor")
    assert result.returncode == 0
    assert "usage:" in result.stdout


# ── Per-OS target path resolution ────────────────────────────────────────────


def test_target_path_macos(tmp_path: Path) -> None:
    p = install_mod.target_path_for("darwin", home=tmp_path)
    assert p == tmp_path / "Library" / "LaunchAgents" / "net.autogpt.shim.plist"


def test_target_path_linux(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    p = install_mod.target_path_for("linux", home=tmp_path)
    assert p == tmp_path / ".config" / "systemd" / "user" / "autogpt-shim.service"


def test_target_path_linux_respects_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    xdg = tmp_path / "xdg-config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    p = install_mod.target_path_for("linux", home=tmp_path)
    assert p == xdg / "systemd" / "user" / "autogpt-shim.service"


def test_target_path_wsl2_matches_linux(tmp_path: Path) -> None:
    assert install_mod.target_path_for("wsl2", home=tmp_path) == (
        install_mod.target_path_for("linux", home=tmp_path)
    )


def test_target_path_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    localappdata = tmp_path / "LocalAppData"
    monkeypatch.setenv("LOCALAPPDATA", str(localappdata))
    p = install_mod.target_path_for("windows", home=tmp_path)
    assert p == localappdata / "autogpt-local-executor" / "autogpt-shim.xml"


def test_target_path_unsupported() -> None:
    with pytest.raises(ValueError):
        install_mod.target_path_for("freebsd")


# ── install_autostart writes the right file per OS ───────────────────────────


def _patch_platform(name: str):
    """Drive `_current_platform_for_install()` via platform.system()."""
    mapping = {"darwin": "Darwin", "linux": "Linux", "windows": "Windows", "wsl2": "Linux"}
    return patch("platform.system", return_value=mapping[name])


def test_install_macos_writes_plist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(install_mod, "default_stdout_log", lambda: tmp_path / "out.log")
    monkeypatch.setattr(install_mod, "default_stderr_log", lambda: tmp_path / "err.log")
    monkeypatch.setattr(
        install_mod, "resolve_shim_command", lambda: ("/usr/local/bin/autogpt-shim",)
    )
    with _patch_platform("darwin"):
        status = install_mod.install_autostart(
            config_path=tmp_path / "cfg.toml",
            home=tmp_path,
        )
    expected = tmp_path / "Library" / "LaunchAgents" / "net.autogpt.shim.plist"
    assert status.target_path == expected
    assert expected.is_file()
    content = expected.read_text(encoding="utf-8")
    assert "net.autogpt.shim" in content
    assert "<key>RunAtLoad</key>" in content
    assert "<key>KeepAlive</key>" in content
    assert str(tmp_path / "cfg.toml") in content


def test_install_linux_writes_systemd_unit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(install_mod, "default_stdout_log", lambda: tmp_path / "out.log")
    monkeypatch.setattr(install_mod, "default_stderr_log", lambda: tmp_path / "err.log")
    monkeypatch.setattr(install_mod, "resolve_shim_command", lambda: ("/usr/bin/autogpt-shim",))
    # Force the linux branch (not wsl2) by ensuring /proc/version doesn't say
    # microsoft. We do this by mocking open in the install module.
    with patch(
        "builtins.open",
        side_effect=_fake_open({"/proc/version": "Linux 6.1 generic"}),
    ):
        with _patch_platform("linux"):
            status = install_mod.install_autostart(
                config_path=tmp_path / "cfg.toml",
                home=tmp_path,
            )
    expected = tmp_path / ".config" / "systemd" / "user" / "autogpt-shim.service"
    assert status.target_path == expected
    assert expected.is_file()
    content = expected.read_text(encoding="utf-8")
    assert "ExecStart=/usr/bin/autogpt-shim --config" in content
    assert "Restart=on-failure" in content


def test_install_windows_writes_task_xml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    localappdata = tmp_path / "LocalAppData"
    monkeypatch.setenv("LOCALAPPDATA", str(localappdata))
    monkeypatch.setattr(install_mod, "default_stdout_log", lambda: tmp_path / "out.log")
    monkeypatch.setattr(install_mod, "default_stderr_log", lambda: tmp_path / "err.log")
    monkeypatch.setattr(
        install_mod, "resolve_shim_command", lambda: (r"C:\Program Files\autogpt-shim.exe",)
    )
    with _patch_platform("windows"):
        status = install_mod.install_autostart(
            config_path=tmp_path / "cfg.toml",
            home=tmp_path,
        )
    expected = localappdata / "autogpt-local-executor" / "autogpt-shim.xml"
    assert status.target_path == expected
    assert expected.is_file()
    content = expected.read_text(encoding="utf-16")
    assert "<LogonTrigger>" in content
    assert '<Command>"C:\\Program Files\\autogpt-shim.exe"</Command>' in content


def test_install_wsl2_uses_linux_path_and_surfaces_linger_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(install_mod, "default_stdout_log", lambda: tmp_path / "out.log")
    monkeypatch.setattr(install_mod, "default_stderr_log", lambda: tmp_path / "err.log")
    monkeypatch.setattr(install_mod, "resolve_shim_command", lambda: ("/usr/bin/autogpt-shim",))
    with patch(
        "builtins.open",
        side_effect=_fake_open({"/proc/version": "Linux 5.15 Microsoft@WSL2"}),
    ):
        with _patch_platform("wsl2"):
            status = install_mod.install_autostart(
                config_path=tmp_path / "cfg.toml",
                home=tmp_path,
            )
    assert status.platform == "wsl2"
    expected = tmp_path / ".config" / "systemd" / "user" / "autogpt-shim.service"
    assert status.target_path == expected
    assert expected.is_file()
    joined_notes = " ".join(status.notes)
    assert "enable-linger" in joined_notes


def test_install_rejects_non_at_login() -> None:
    with pytest.raises(NotImplementedError):
        install_mod.install_autostart(at_login=False)


# ── uninstall_autostart ──────────────────────────────────────────────────────


def test_uninstall_removes_rendered_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(install_mod, "default_stdout_log", lambda: tmp_path / "out.log")
    monkeypatch.setattr(install_mod, "default_stderr_log", lambda: tmp_path / "err.log")
    monkeypatch.setattr(
        install_mod, "resolve_shim_command", lambda: ("/usr/local/bin/autogpt-shim",)
    )
    with _patch_platform("darwin"):
        install_mod.install_autostart(home=tmp_path)
        target = tmp_path / "Library" / "LaunchAgents" / "net.autogpt.shim.plist"
        assert target.is_file()
        status = install_mod.uninstall_autostart(home=tmp_path)
    assert not target.exists()
    assert status.installed is False


def test_uninstall_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _patch_platform("darwin"):
        status1 = install_mod.uninstall_autostart(home=tmp_path)
        status2 = install_mod.uninstall_autostart(home=tmp_path)
    assert status1.installed is False
    assert status2.installed is False


# ── status_for ───────────────────────────────────────────────────────────────


def test_status_when_absent(tmp_path: Path) -> None:
    status = install_mod.status_for("darwin", home=tmp_path)
    assert status.installed is False
    assert status.target_path == (tmp_path / "Library" / "LaunchAgents" / "net.autogpt.shim.plist")
    assert status.enable_command and "launchctl load" in status.enable_command


def test_status_when_present_macos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(install_mod, "default_stdout_log", lambda: tmp_path / "out.log")
    monkeypatch.setattr(install_mod, "default_stderr_log", lambda: tmp_path / "err.log")
    monkeypatch.setattr(
        install_mod, "resolve_shim_command", lambda: ("/usr/local/bin/autogpt-shim",)
    )
    with _patch_platform("darwin"):
        install_mod.install_autostart(home=tmp_path)
    status = install_mod.status_for("darwin", home=tmp_path)
    assert status.installed is True


def test_status_linux_includes_systemctl_commands(tmp_path: Path) -> None:
    status = install_mod.status_for("linux", home=tmp_path)
    assert status.enable_command is not None
    assert "systemctl --user enable" in status.enable_command
    assert status.disable_command is not None
    assert "systemctl --user disable" in status.disable_command


def test_status_windows_includes_schtasks_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))
    status = install_mod.status_for("windows", home=tmp_path)
    assert status.enable_command is not None
    assert "schtasks /Create" in status.enable_command
    assert status.disable_command is not None
    assert "schtasks /Delete" in status.disable_command


def test_status_wsl2_mentions_enable_linger(tmp_path: Path) -> None:
    status = install_mod.status_for("wsl2", home=tmp_path)
    joined = " ".join(status.notes)
    assert "enable-linger" in joined


# ── Helpers ──────────────────────────────────────────────────────────────────


def _fake_open(file_contents: dict[str, str]):
    """Return a side_effect for `patch('builtins.open')` that returns the
    matching string when the requested path is in file_contents, else
    delegates to the real builtins.open.
    """
    import builtins
    import io

    real_open = builtins.open

    def _opener(path, *args, **kwargs):
        path_str = str(path)
        if path_str in file_contents:
            return io.StringIO(file_contents[path_str])
        return real_open(path, *args, **kwargs)

    return _opener


def test_a_relative_config_path_is_made_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The service manager runs the shim from its own working directory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(install_mod, "resolve_shim_command", lambda: ("/usr/bin/autogpt-shim",))

    ctx = install_mod.build_context(Path("conf") / "shim.toml")

    assert ctx.config_path == str(tmp_path.resolve() / "conf" / "shim.toml")


def test_a_relative_path_entry_still_gives_an_absolute_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relative PATH entry made `which` return a relative path, which a
    service started from another directory can't run."""
    monkeypatch.setattr(install_mod.shutil, "which", _which_without_path("bin/autogpt-shim"))

    [command] = install_mod.resolve_shim_command()

    assert Path(command).is_absolute()
    assert command.endswith(str(Path("bin") / "autogpt-shim"))


def _link_or_skip(link: Path, folder: Path) -> None:
    """Make `link` a symlink to `folder`, or skip where symlinks can't be
    made (Windows without the privilege)."""
    try:
        link.symlink_to(folder, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks can't be made here: {exc}")


def test_a_config_path_through_a_link_and_dotdot_names_the_same_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """abspath() drops `link/..` without looking at the disk, so where
    `link` is a symlink, `link/../config.toml` was stored as another file
    than the one it names, and the service ran on that one, or failed to
    start. The OS reads the path given: POSIX follows `link` before `..`,
    and Windows drops `link\\..` itself, as abspath does."""
    (tmp_path / "project" / "sub").mkdir(parents=True)
    (tmp_path / "project" / "config.toml").write_text("# the one named\n", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    (work / "config.toml").write_text("# another\n", encoding="utf-8")
    _link_or_skip(work / "link", tmp_path / "project" / "sub")
    monkeypatch.chdir(work)
    monkeypatch.setattr(install_mod, "resolve_shim_command", lambda: ("/usr/bin/autogpt-shim",))
    given = Path("link") / ".." / "config.toml"

    ctx = install_mod.build_context(given)

    assert ctx.config_path is not None and Path(ctx.config_path).is_absolute()
    assert os.path.samefile(ctx.config_path, given)


def test_a_path_entry_through_a_link_and_dotdot_gives_the_script_it_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """abspath() dropped `link/..` from the script `which` found without
    looking at the disk, so a PATH entry through a symlinked folder gave the
    service another program, or none."""
    name = "autogpt-shim.exe" if sys.platform == "win32" else "autogpt-shim"
    (tmp_path / "project" / "sub").mkdir(parents=True)
    for folder in (tmp_path / "project" / "bin", tmp_path / "bin"):  # the named one, another
        folder.mkdir()
        (folder / name).write_text("", encoding="utf-8")
        (folder / name).chmod(0o755)
    _link_or_skip(tmp_path / "link", tmp_path / "project" / "sub")
    entry = tmp_path / "link" / ".." / "bin"
    monkeypatch.setenv("PATH", str(entry))

    [command] = install_mod.resolve_shim_command()

    assert Path(command).is_absolute()
    assert os.path.samefile(command, entry / name)


def test_a_failed_autostart_write_leaves_the_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_write_atomic truncated the file in place, so a failed write left it
    half written."""
    target = tmp_path / "autogpt-shim.service"
    target.write_text("the working unit", encoding="utf-8")

    def refuse(*_args) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(install_mod.os, "replace", refuse)
    with pytest.raises(OSError, match="disk full"):
        install_mod._write_atomic(target, "a new unit")
    monkeypatch.undo()

    assert target.read_text(encoding="utf-8") == "the working unit"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["autogpt-shim.service"]


def test_two_writes_to_one_target_that_overlap_both_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The test below only finds the old fixed temporary name in place; this
    one runs a second write while the first waits to rename, so a shared
    name of any spelling would make one overwrite or remove the other's
    file."""
    import os

    target = tmp_path / "autogpt-shim.service"
    real_replace = os.replace
    renamed: list[str] = []

    def replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        renamed.append(Path(src).name)
        if len(renamed) == 1:
            install_mod._write_atomic(target, "second")  # while the first waits
        real_replace(src, dst)

    monkeypatch.setattr(install_mod.os, "replace", replace)
    install_mod._write_atomic(target, "first")
    monkeypatch.undo()

    assert len(set(renamed)) == 2
    assert target.read_text(encoding="utf-8") == "first"
    assert sorted(p.name for p in tmp_path.iterdir()) == [target.name]


def test_a_write_leaves_another_installs_temporary_file_alone(tmp_path: Path) -> None:
    """Two installs at once shared one temporary name, so one could write into
    the other's file, or rename it away before the other had finished."""
    target = tmp_path / "autogpt-shim.service"
    theirs = tmp_path / f".{target.name}.tmp"
    theirs.write_text("another install, part way through", encoding="utf-8")

    install_mod._write_atomic(target, "mine")

    assert target.read_text(encoding="utf-8") == "mine"
    assert theirs.read_text(encoding="utf-8") == "another install, part way through"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([theirs.name, target.name])


@pytest.mark.parametrize("exists", [True, False])
def test_the_default_config_is_pinned_only_when_it_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exists: bool
) -> None:
    """A missing --config file is an error now, so a service pinned to a
    default config that doesn't exist would never start; without the flag
    the shim finds the default itself."""
    default = tmp_path / "config.toml"
    if exists:
        default.write_text("", encoding="utf-8")
    monkeypatch.setattr(install_mod, "default_config_path", lambda: default)
    monkeypatch.setattr(install_mod, "resolve_shim_command", lambda: ("/usr/bin/autogpt-shim",))

    ctx = install_mod.build_context(None)

    if exists:
        assert ctx.start_argv == ("/usr/bin/autogpt-shim", "--config", str(default), "start")
    else:
        assert ctx.start_argv == ("/usr/bin/autogpt-shim", "start")


def test_the_service_starts_with_the_options_install_was_given(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(install_mod, "resolve_shim_command", lambda: ("/usr/bin/autogpt-shim",))
    monkeypatch.setattr(install_mod, "default_config_path", lambda: tmp_path / "none.toml")

    ctx = install_mod.build_context(
        None, global_args=("--platform-url", "https://platform.example.com")
    )

    assert ctx.start_argv == (
        "/usr/bin/autogpt-shim",
        "--platform-url",
        "https://platform.example.com",
        "start",
    )
