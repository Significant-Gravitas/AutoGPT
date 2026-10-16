"""The headless Claude credentials file: owner-only, never through a link.

Every credential here is fake.
"""

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from . import subscription

_ACCESS = "sk-ant-oat01-FAKE-access"
_REFRESH = "sk-ant-ort01-FAKE-refresh"
posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _ACCESS)
    monkeypatch.setenv("CLAUDE_CODE_REFRESH_TOKEN", _REFRESH)
    return tmp_path


@pytest.fixture
def loose_umask():
    previous = os.umask(0o022)
    yield
    os.umask(previous)


def _creds(home: Path) -> Path:
    return home / ".claude" / ".credentials.json"


def test_no_token_writes_nothing(home: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN")
    subscription.provision_credentials_file()
    assert not (home / ".claude").exists()


def test_writes_the_tokens(home: Path):
    subscription.provision_credentials_file()
    oauth = json.loads(_creds(home).read_text())["claudeAiOauth"]
    assert (oauth["accessToken"], oauth["refreshToken"]) == (_ACCESS, _REFRESH)


@posix_only
def test_the_file_and_directory_are_owner_only(home: Path, loose_umask: None):
    subscription.provision_credentials_file()
    assert stat.S_IMODE(os.stat(home / ".claude").st_mode) == 0o700
    assert stat.S_IMODE(os.stat(_creds(home)).st_mode) == 0o600


@posix_only
def test_an_open_directory_is_closed_to_others(home: Path, loose_umask: None):
    (home / ".claude").mkdir(mode=0o755)
    os.chmod(home / ".claude", 0o755)
    subscription.provision_credentials_file()
    assert stat.S_IMODE(os.stat(home / ".claude").st_mode) == 0o700


@posix_only
def test_a_private_existing_file_is_kept(home: Path):
    (home / ".claude").mkdir(mode=0o700)
    _creds(home).write_text("mounted")
    os.chmod(_creds(home), 0o600)
    subscription.provision_credentials_file()
    assert _creds(home).read_text() == "mounted"


@posix_only
def test_an_existing_file_others_can_read_is_refused(home: Path):
    (home / ".claude").mkdir(mode=0o700)
    _creds(home).write_text("mounted")
    os.chmod(_creds(home), 0o644)
    with pytest.raises(RuntimeError, match="chmod 600"):
        subscription.provision_credentials_file()
    assert _creds(home).read_text() == "mounted"


def test_a_link_in_its_place_is_refused(home: Path, tmp_path: Path):
    (home / ".claude").mkdir()
    target = tmp_path / "elsewhere"
    try:
        os.symlink(target, _creds(home))
    except (OSError, NotImplementedError):
        pytest.skip("Symbolic links are not available here")
    with pytest.raises(RuntimeError, match="symbolic link"):
        subscription.provision_credentials_file()
    assert not target.exists()


def test_a_file_that_appears_mid_write_is_left_alone(
    home: Path, monkeypatch: pytest.MonkeyPatch
):
    """The create is exclusive: a file that turns up between the check and the
    write is neither written through nor overwritten."""
    (home / ".claude").mkdir()
    _creds(home).write_text("raced")
    real_lstat = os.lstat

    def lstat(path, *args, **kwargs):
        if Path(path) == _creds(home):
            raise FileNotFoundError(path)
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(subscription.os, "lstat", lstat)
    with pytest.raises(RuntimeError, match="appeared"):
        subscription.provision_credentials_file()
    assert _creds(home).read_text() == "raced"


def test_the_auth_check_logs_no_account_identity(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    def run(args, **_kwargs):
        stdout = (
            json.dumps(
                {
                    "loggedIn": True,
                    "authMethod": "claude.ai",
                    "email": "person@example.com",
                }
            )
            if "auth" in args
            else "2.1.284"
        )
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subscription, "find_bundled_cli", lambda: "claude-under-test")
    monkeypatch.setattr(subscription, "provision_credentials_file", lambda: None)
    monkeypatch.setattr(subscription.subprocess, "run", run)
    caplog.set_level("INFO", logger=subscription.logger.name)

    subscription.validate_subscription.__wrapped__()

    assert "method=claude.ai" in caplog.text
    assert "person@example.com" not in caplog.text
