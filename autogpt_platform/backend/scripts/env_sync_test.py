import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import dotenv_values

from scripts import env_sync as sync

BACKEND_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def state_home(tmp_path, monkeypatch):
    state = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    return state


def test_sync_preserves_other_values_and_literal_secrets(tmp_path, state_home):
    target = tmp_path / ".env"
    original = "PORT=8123\n# keep this comment\nOTHER='stay'\n"
    target.write_text(original)
    value = "literal$VALUE#with'quotes"
    sync.apply_updates(target, {"MICROSOFT_CLIENT_SECRET": value})
    assert target.read_text().startswith(original)
    assert dotenv_values(target, interpolate=False)["MICROSOFT_CLIENT_SECRET"] == value
    assert target.stat().st_mode & 0o777 == 0o600
    backups = (state_home / "autogpt-env-sync").iterdir()
    assert [backup.read_text() for backup in backups] == [original]


def test_backup_stays_out_of_the_repository(tmp_path):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text(".env\n")
    target = repo / ".env"
    target.write_text("OTHER='stay'\n")
    sync.apply_updates(target, {"MICROSOFT_CLIENT_SECRET": "new-value"})
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert status.stdout.splitlines() == ["?? .gitignore"]


def test_backup_is_private_and_reported_without_values(tmp_path, state_home, capsys):
    target = tmp_path / ".env"
    target.write_text("OTHER='old-value'\n")
    sync.apply_updates(target, {"MICROSOFT_CLIENT_SECRET": "new-value"})
    directory = state_home / "autogpt-env-sync"
    [backup] = directory.iterdir()
    assert backup.read_text() == "OTHER='old-value'\n"
    assert directory.stat().st_mode & 0o777 == 0o700
    assert backup.stat().st_mode & 0o777 == 0o600
    output = capsys.readouterr().out
    assert str(backup) in output
    assert "old-value" not in output and "new-value" not in output


@pytest.mark.parametrize("state", [None, "relative/state"])
def test_backups_default_to_the_home_state_directory(tmp_path, monkeypatch, state):
    monkeypatch.setenv("HOME", str(tmp_path))
    if state is None:
        monkeypatch.delenv("XDG_STATE_HOME")
    else:
        monkeypatch.setenv("XDG_STATE_HOME", state)
    expected = tmp_path / ".local" / "state" / "autogpt-env-sync"
    assert sync.backup_directory() == expected


def test_value_written_after_the_report_still_needs_replace(
    tmp_path, monkeypatch, capsys
):
    source = tmp_path / "source.env"
    source.write_text("MICROSOFT_CLIENT_SECRET='mine'\n")
    target = tmp_path / ".env"
    target.write_text("OTHER='stay'\n")
    apply_updates = sync.apply_updates

    def another_run_writes_first(*args, **kwargs):
        target.write_text("OTHER='stay'\nMICROSOFT_CLIENT_SECRET='theirs'\n")
        return apply_updates(*args, **kwargs)

    monkeypatch.setattr(sync, "apply_updates", another_run_writes_first)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "env_sync",
            *("--source", f"local:{source}", "--target", str(target)),
            *("--key", "MICROSOFT_CLIENT_SECRET", "--apply"),
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        sync.main()
    assert stopped.value.code == 2
    assert "--replace" in capsys.readouterr().err
    secret = dotenv_values(target, interpolate=False)["MICROSOFT_CLIENT_SECRET"]
    assert secret == "theirs"


def test_non_ascii_values_survive_a_non_utf8_locale(tmp_path):
    source = tmp_path / "source.env"
    source.write_bytes("MICROSOFT_TENANT_ID='café'\n".encode())
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text(f"#!/bin/sh\ncat {shlex.quote(str(source))}\n")
    ssh.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "LC_ALL": "C",
        "PYTHONUTF8": "0",
    }
    read = (
        "import sys; from scripts.env_sync import read_source; "
        "print(ascii([read_source(source, []) for source in sys.argv[1:]]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", read, f"local:{source}", "laptop:/backend/.env"],
        cwd=BACKEND_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ascii([{"MICROSOFT_TENANT_ID": "café"}] * 2)


def test_conflicting_sources_do_not_expose_values():
    with pytest.raises(ValueError) as error:
        sync.select_values(
            [
                {"MICROSOFT_CLIENT_SECRET": "secret-one"},
                {"MICROSOFT_CLIENT_SECRET": "secret-two"},
            ],
            ["MICROSOFT_CLIENT_SECRET"],
        )
    assert "MICROSOFT_CLIENT_SECRET" in str(error.value)
    assert "secret-one" not in str(error.value)
    assert "secret-two" not in str(error.value)


def test_empty_source_values_are_not_updates():
    assert (
        sync.select_values(
            [{"MICROSOFT_CLIENT_SECRET": ""}], ["MICROSOFT_CLIENT_SECRET"]
        )
        == {}
    )
