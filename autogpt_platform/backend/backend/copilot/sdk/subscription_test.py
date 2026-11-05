"""The subscription check and the headless Claude credentials file.

Every credential and account here is fake.
"""

import json
import logging
import os
import stat
import subprocess
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.copilot.sdk import subscription

_ACCESS = "sk-ant-oat01-FAKE-access"
_REFRESH = "sk-ant-ort01-FAKE-refresh"
ACCOUNT_EMAIL = "synthetic-account-privacy-16237@example.invalid"
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


@pytest.fixture
def cli_run() -> Iterator[MagicMock]:
    subscription.validate_subscription.cache_clear()
    try:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(subscription, "provision_credentials_file") as provision,
            patch.object(subscription, "find_bundled_cli", return_value="fake-claude"),
            patch.object(subscription.subprocess, "run") as run,
        ):
            run.side_effect = [
                subprocess.CompletedProcess([], 0, "1.2.3", ""),
                subprocess.CompletedProcess(
                    [],
                    0,
                    json.dumps(
                        {
                            "loggedIn": True,
                            "authMethod": "claude.ai",
                            "email": ACCOUNT_EMAIL,
                        }
                    ),
                    "",
                ),
            ]
            yield run
            provision.assert_called_once_with()
    finally:
        subscription.validate_subscription.cache_clear()


def test_validate_subscription_keeps_auth_method_without_account_email(
    cli_run: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger=subscription.__name__):
        assert subscription.validate_subscription() is None
        assert subscription.validate_subscription() is None

    assert cli_run.call_count == 2
    assert ACCOUNT_EMAIL not in caplog.text
    assert all(ACCOUNT_EMAIL not in repr(vars(record)) for record in caplog.records)
    assert "Claude Code subscription mode: CLI version 1.2.3" in caplog.messages
    assert "Claude subscription auth: method=claude.ai" in caplog.messages


@pytest.mark.parametrize(
    "version_code, auth_code, logged_in, error, calls",
    [
        (1, 0, True, "Claude CLI check failed", 1),
        (0, 1, True, "Claude CLI is not authenticated", 2),
        (0, 0, False, "Claude CLI reports loggedIn=false", 2),
    ],
)
def test_validate_subscription_still_rejects_failed_checks(
    cli_run: MagicMock,
    caplog: pytest.LogCaptureFixture,
    version_code: int,
    auth_code: int,
    logged_in: bool,
    error: str,
    calls: int,
) -> None:
    cli_run.side_effect = [
        subprocess.CompletedProcess([], version_code, "1.2.3", "version unavailable"),
        subprocess.CompletedProcess(
            [],
            auth_code,
            json.dumps(
                {
                    "loggedIn": logged_in,
                    "authMethod": "claude.ai",
                    "email": ACCOUNT_EMAIL,
                }
            ),
            "",
        ),
    ]
    with (
        caplog.at_level(logging.INFO, logger=subscription.__name__),
        pytest.raises(RuntimeError, match=error),
    ):
        subscription.validate_subscription()

    assert cli_run.call_count == calls
    assert subscription.validate_subscription.cache_info().currsize == 0
    assert not any(
        "Claude subscription auth:" in message for message in caplog.messages
    )
    assert ACCOUNT_EMAIL not in caplog.text
    assert all(ACCOUNT_EMAIL not in repr(vars(record)) for record in caplog.records)


def test_validate_subscription_still_warns_on_unparseable_auth_status(
    cli_run: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    cli_run.side_effect = [
        subprocess.CompletedProcess([], 0, "1.2.3", ""),
        subprocess.CompletedProcess([], 0, f"invalid JSON: {ACCOUNT_EMAIL}", ""),
    ]
    with caplog.at_level(logging.INFO, logger=subscription.__name__):
        assert subscription.validate_subscription() is None

    assert cli_run.call_count == 2
    assert "Could not parse `claude auth status` output" in caplog.messages
    assert ACCOUNT_EMAIL not in caplog.text
    assert all(ACCOUNT_EMAIL not in repr(vars(record)) for record in caplog.records)
