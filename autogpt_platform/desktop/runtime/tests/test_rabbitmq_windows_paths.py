"""Check Erlang's flag parser without starting an emulator or node."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from autogpt_desktop import rabbitmq
from autogpt_desktop.layout import Bundle

DESKTOP = Path(__file__).resolve().parents[2]
PROFILES = ["Nick", "John Smith", "O'Brien", "Jürgen", "O''Brien", "O'Brien's"]


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows Erlang parser")
@pytest.mark.parametrize("profile", PROFILES)
def test_erlang_patch_path_survives_native_flag_parsing(tmp_path, monkeypatch, profile):
    configured = os.environ.get("AUTOGPT_DESKTOP_RUNTIME")
    installed = Bundle(Path(configured) if configured else DESKTOP / "build/runtime")
    erl = installed.erlang_home / "bin" / "erl.exe"
    if not erl.is_file():
        pytest.skip("no assembled Windows Erlang bundle")

    bundle = Bundle(tmp_path / profile / "runtime")
    bundle.erlang_patches.mkdir(parents=True)
    (bundle.erlang_patches / "inet_udp.beam").touch()
    monkeypatch.setattr(rabbitmq, "_short", str)
    flags = rabbitmq._erlang_flags(bundle)
    assert flags.startswith(rabbitmq.LOOPBACK_DISTRIBUTION)
    path_flags = flags.removeprefix(rabbitmq.LOOPBACK_DISTRIBUTION).strip()
    expected = str(bundle.erlang_patches).replace(os.sep, "/")

    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper()
        in {"SYSTEMROOT", "WINDIR", "SYSTEMDRIVE", "TEMP", "TMP", "COMSPEC"}
    }
    env.update(
        PATH=str(Path(env["SYSTEMROOT"]) / "System32"),
        HOME=str(tmp_path),
        USERPROFILE=str(tmp_path),
        ERL_AFLAGS=path_flags,
    )
    result = subprocess.run(
        [str(erl), "-emu_args_exit"],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    assert result.returncode == 0, result.stderr
    arguments = result.stdout.splitlines()
    assert arguments.count("-pa") == 1, arguments
    assert arguments[arguments.index("-pa") + 1] == expected


def test_posix_erlang_patch_flags_keep_existing_path_spelling(tmp_path, monkeypatch):
    bundle = Bundle(tmp_path / "runtime")
    bundle.erlang_patches.mkdir(parents=True)
    (bundle.erlang_patches / "inet_udp.beam").touch()
    monkeypatch.setattr(rabbitmq, "WINDOWS", False)
    monkeypatch.setattr(rabbitmq, "_short", lambda _: "/tmp/autogpt-patches")
    assert rabbitmq._erlang_flags(bundle) == (
        f"{rabbitmq.LOOPBACK_DISTRIBUTION} -pa /tmp/autogpt-patches"
    )
