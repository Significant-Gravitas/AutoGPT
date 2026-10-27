import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import dotenv_values

from scripts import env_sync as sync

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parents[1]

# Runs a sync that dies like a SIGKILL at its Nth file replacement.
KILLED_SYNC = """
import os, sys
from pathlib import Path
from scripts import env_sync

replace, calls = os.replace, []

def replace_or_die(*args):
    calls.append(args)
    if len(calls) == int(sys.argv[2]):
        os._exit(9)
    replace(*args)

os.replace = replace_or_die
env_sync.apply_updates(Path(sys.argv[1]), {"MICROSOFT_CLIENT_SECRET": "new-value"})
"""


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


def test_replace_overwrites_a_different_value_and_keeps_a_backup(tmp_path, state_home):
    target = tmp_path / ".env"
    target.write_text("MICROSOFT_CLIENT_SECRET='old-value'\n")
    updates = {"MICROSOFT_CLIENT_SECRET": "new-value"}
    assert sync.apply_updates(target, updates, replace=True) == 1
    secret = dotenv_values(target, interpolate=False)["MICROSOFT_CLIENT_SECRET"]
    assert secret == "new-value"
    [backup] = (state_home / "autogpt-env-sync").iterdir()
    assert backup.read_text() == "MICROSOFT_CLIENT_SECRET='old-value'\n"


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


@pytest.mark.parametrize("kill_at", [1, 2])
def test_a_killed_sync_leaves_nothing_to_stage(tmp_path, kill_at):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text((REPO_ROOT / ".gitignore").read_text())
    target = repo / ".env"
    target.write_text("OTHER='stay'\n")
    killed = subprocess.run(
        [sys.executable, "-c", KILLED_SYNC, str(target), str(kill_at)],
        cwd=BACKEND_DIR,
        check=False,
    )
    assert killed.returncode == 9
    assert target.read_text() == "OTHER='stay'\n"
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert status.stdout.splitlines() == ["?? .gitignore"]


def test_a_leftover_lock_is_named_in_the_error(tmp_path):
    target = tmp_path / ".env"
    target.write_text("OTHER='stay'\n")
    lock = tmp_path / ".env.sync-lock"
    lock.touch()
    with pytest.raises(ValueError, match=r"\.env\.sync-lock"):
        sync.apply_updates(target, {"MICROSOFT_CLIENT_SECRET": "new-value"})
    assert lock.exists()
    assert target.read_text() == "OTHER='stay'\n"


@pytest.mark.parametrize("name", [".env.default", ".env.example", ".env.template"])
def test_template_targets_are_refused(tmp_path, name):
    target = tmp_path / name
    target.write_text("OTHER='stay'\n")
    with pytest.raises(ValueError, match="private, regular env file"):
        sync.apply_updates(target, {"MICROSOFT_CLIENT_SECRET": "new-value"})
    assert target.read_text() == "OTHER='stay'\n"


def test_symlinked_targets_are_refused(tmp_path):
    real = tmp_path / "real.env"
    real.write_text("OTHER='stay'\n")
    target = tmp_path / ".env"
    target.symlink_to(real)
    with pytest.raises(ValueError, match="private, regular env file"):
        sync.apply_updates(target, {"MICROSOFT_CLIENT_SECRET": "new-value"})
    assert real.read_text() == "OTHER='stay'\n"


def test_tracked_targets_are_refused(tmp_path):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    target = repo / ".env"
    target.write_text("OTHER='stay'\n")
    subprocess.run(["git", "-C", str(repo), "add", "--force", ".env"], check=True)
    with pytest.raises(ValueError, match="tracked file"):
        sync.apply_updates(target, {"MICROSOFT_CLIENT_SECRET": "new-value"})
    assert target.read_text() == "OTHER='stay'\n"
