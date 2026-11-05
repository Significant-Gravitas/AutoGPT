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
from unittest.mock import MagicMock, Mock, patch

import pytest
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

from backend.copilot.sdk import cli_version, subscription


@pytest.fixture(autouse=True)
def isolated_subscription(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in ("CHAT_CLAUDE_AGENT_CLI_PATH", "CLAUDE_AGENT_CLI_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(subscription, "provision_credentials_file", Mock())
    monkeypatch.setattr(subscription, "host_login_env", dict)
    monkeypatch.setattr(subscription, "inherited_env", dict)
    monkeypatch.setattr(
        subscription,
        "_cli_the_sdk_finds",
        Mock(side_effect=AssertionError("unexpected fallback")),
    )
    monkeypatch.setattr(
        subscription.subprocess,
        "run",
        Mock(side_effect=AssertionError("unexpected spawn")),
    )
    subscription._validate_subscription.cache_clear()
    yield
    subscription._validate_subscription.cache_clear()


@pytest.fixture
def cli(tmp_path: Path) -> Path:
    path = tmp_path / "2.1.284.exe"
    path.write_bytes(b"test executable, never run")
    path.chmod(0o755)
    return path


@pytest.fixture
def runs(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        output = '{"loggedIn": true}' if argv[1:] == ["auth", "status"] else "2.1.284"
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(subscription.subprocess, "run", run)
    return calls


def test_configured_cli_drives_both_probes(cli: Path, runs: list[list[str]]):
    subscription.validate_subscription(str(cli))
    assert runs == [[str(cli), "--version"], [str(cli), "auth", "status"]]


@pytest.mark.parametrize(
    "kind", ["missing", "directory", "not-executable", "whitespace"]
)
def test_invalid_configured_cli_fails_before_provisioning_or_fallback(
    cli: Path, monkeypatch: pytest.MonkeyPatch, kind: str
):
    path = {
        "missing": str(cli.with_name("missing.exe")),
        "directory": str(cli.parent),
        "not-executable": str(cli),
        "whitespace": " ",
    }[kind]
    if kind == "not-executable":
        monkeypatch.setattr(subscription.os, "access", lambda path, mode: False)
    with pytest.raises(RuntimeError, match="Configured Claude CLI") as error:
        subscription.validate_subscription(path)
    assert str(cli.parent) not in str(error.value)
    subscription.provision_credentials_file.assert_not_called()
    subscription._cli_the_sdk_finds.assert_not_called()
    subscription.subprocess.run.assert_not_called()


def test_unset_cli_keeps_existing_discovery(
    cli: Path, runs: list[list[str]], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(subscription, "_cli_the_sdk_finds", lambda: str(cli))
    subscription.validate_subscription()
    assert [call[0] for call in runs] == [str(cli)] * 2


def test_launcher_retargeting_does_not_redirect_auth(
    cli: Path, monkeypatch: pytest.MonkeyPatch
):
    launcher = cli.parent / "claude.exe"
    replacement = cli.with_name("2.1.247.exe")
    replacement.write_bytes(b"different executable")
    try:
        launcher.symlink_to(cli)
    except OSError:
        pytest.skip("symlink permission is unavailable")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[1:] == ["--version"]:
            launcher.unlink()
            launcher.symlink_to(replacement)
            output = "2.1.284"
        else:
            output = '{"loggedIn": true}'
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(subscription.subprocess, "run", run)
    subscription.validate_subscription(str(launcher))
    assert calls == [[str(cli), "--version"], [str(cli), "auth", "status"]]


def test_validation_cache_distinguishes_configured_paths(
    cli: Path, runs: list[list[str]]
):
    second = cli.with_name("2.1.285.exe")
    second.write_bytes(b"another executable")
    second.chmod(0o755)
    for path in (cli, cli, second):
        subscription.validate_subscription(os.fspath(path))
    assert [call[0] for call in runs] == [str(cli), str(cli), str(second), str(second)]


def test_validation_cache_follows_retargeted_launcher(cli: Path, runs: list[list[str]]):
    second = cli.with_name("2.1.285.exe")
    second.write_bytes(b"another executable")
    second.chmod(0o755)
    launcher = cli.with_name("claude.exe")
    try:
        launcher.symlink_to(cli)
    except OSError:
        pytest.skip("symlink permission is unavailable")
    subscription.validate_subscription(str(launcher))
    launcher.unlink()
    launcher.symlink_to(second)
    subscription.validate_subscription(str(launcher))
    subscription.validate_subscription(str(second))
    assert [call[0] for call in runs] == [str(cli), str(cli), str(second), str(second)]


def test_cached_selection_still_rejects_a_removed_executable(
    cli: Path, runs: list[list[str]], monkeypatch: pytest.MonkeyPatch
):
    subscription.validate_subscription(str(cli))
    cli.unlink()
    provision = Mock(side_effect=AssertionError("unexpected credential write"))
    monkeypatch.setattr(subscription, "provision_credentials_file", provision)
    with pytest.raises(RuntimeError, match="Configured Claude CLI"):
        subscription.validate_subscription(str(cli))
    provision.assert_not_called()
    assert len(runs) == 2


@pytest.mark.parametrize("failure", ["version", "auth", "logged-out"])
def test_failed_validation_is_not_cached(
    cli: Path, monkeypatch: pytest.MonkeyPatch, failure: str
):
    version = subprocess.CompletedProcess([], 0, "2.1.284", "")
    logged_in = subprocess.CompletedProcess([], 0, '{"loggedIn": true}', "")
    refused = subprocess.CompletedProcess([], 1, "", "probe failed")
    logged_out = subprocess.CompletedProcess([], 0, '{"loggedIn": false}', "")
    failures = {
        "version": [refused],
        "auth": [version, refused],
        "logged-out": [version, logged_out],
    }[failure]
    run = Mock(side_effect=[*failures, version, logged_in])
    monkeypatch.setattr(subscription.subprocess, "run", run)
    with pytest.raises(RuntimeError):
        subscription.validate_subscription(str(cli))
    subscription.validate_subscription(str(cli))
    subscription.validate_subscription(str(cli))
    assert run.call_count == len(failures) + 2


_ACCESS = "sk-ant-oat01-FAKE-access"
_REFRESH = "sk-ant-ort01-FAKE-refresh"
ACCOUNT_EMAIL = "synthetic-account-privacy-16237@example.invalid"
posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
_PROVISION = subscription.provision_credentials_file


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A home for the credentials file, written by the real provisioning:
    ``isolated_subscription`` stubs it out for the subscription checks."""
    monkeypatch.setattr(subscription, "provision_credentials_file", _PROVISION)
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
def cli_run(cli: Path) -> Iterator[MagicMock]:
    subscription._validate_subscription.cache_clear()
    try:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(subscription, "provision_credentials_file") as provision,
            patch.object(subscription, "_cli_the_sdk_finds", return_value=str(cli)),
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
        subscription._validate_subscription.cache_clear()


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
    assert subscription._validate_subscription.cache_info().currsize == 0
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


def test_unset_cli_reuses_sdk_discovery(
    cli: Path, runs: list[list[str]], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        subscription, "_cli_the_sdk_finds", cli_version._cli_the_sdk_finds
    )
    discovery = Mock(return_value=str(cli))
    monkeypatch.setattr(SubprocessCLITransport, "_find_cli", discovery)
    cli_version._cli_the_sdk_finds.cache_clear()
    try:
        for _ in range(5):
            subscription.validate_subscription()
        discovery.assert_called_once_with()
        assert runs == [[str(cli), "--version"], [str(cli), "auth", "status"]]
    finally:
        cli_version._cli_the_sdk_finds.cache_clear()
