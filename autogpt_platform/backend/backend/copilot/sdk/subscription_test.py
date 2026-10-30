"""Subscription validation uses the configured executable, without real auth."""

import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest

from backend.copilot.sdk import subscription


@pytest.fixture(autouse=True)
def isolated_subscription(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in ("CHAT_CLAUDE_AGENT_CLI_PATH", "CLAUDE_AGENT_CLI_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(subscription, "provision_credentials_file", Mock())
    monkeypatch.setattr(subscription, "host_login_env", dict)
    monkeypatch.setattr(subscription, "inherited_env", dict)
    monkeypatch.setattr(
        subscription,
        "find_bundled_cli",
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


@pytest.mark.parametrize(
    "setting", ["explicit", "CHAT_CLAUDE_AGENT_CLI_PATH", "CLAUDE_AGENT_CLI_PATH"]
)
def test_configured_cli_drives_both_probes(
    cli: Path, runs: list[list[str]], monkeypatch: pytest.MonkeyPatch, setting: str
):
    if setting == "explicit":
        subscription.validate_subscription(str(cli))
    else:
        monkeypatch.setenv(setting, str(cli))
        subscription.validate_subscription()
    assert runs == [[str(cli), "--version"], [str(cli), "auth", "status"]]


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_setting_precedence(
    cli: Path, runs: list[list[str]], monkeypatch: pytest.MonkeyPatch, explicit: bool
):
    monkeypatch.setenv("CLAUDE_AGENT_CLI_PATH", "missing-unprefixed-cli")
    monkeypatch.setenv(
        "CHAT_CLAUDE_AGENT_CLI_PATH", "missing-prefixed-cli" if explicit else str(cli)
    )
    subscription.validate_subscription(str(cli) if explicit else None)
    assert [call[0] for call in runs] == [str(cli), str(cli)]


def test_empty_prefixed_setting_uses_unprefixed_alias(
    cli: Path, runs: list[list[str]], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("CHAT_CLAUDE_AGENT_CLI_PATH", "")
    monkeypatch.setenv("CLAUDE_AGENT_CLI_PATH", str(cli))
    subscription.validate_subscription()
    assert [call[0] for call in runs] == [str(cli), str(cli)]


@pytest.mark.parametrize(
    "kind", ["missing", "directory", "not-executable", "whitespace"]
)
@pytest.mark.parametrize("explicit", [False, True])
def test_invalid_configured_cli_fails_before_provisioning_or_fallback(
    cli: Path, monkeypatch: pytest.MonkeyPatch, kind: str, explicit: bool
):
    path = {
        "missing": str(cli.with_name("missing.exe")),
        "directory": str(cli.parent),
        "not-executable": str(cli),
        "whitespace": " ",
    }[kind]
    if kind == "not-executable":
        monkeypatch.setattr(subscription.os, "access", lambda path, mode: False)
    monkeypatch.setenv("CLAUDE_AGENT_CLI_PATH", str(cli))
    if not explicit:
        monkeypatch.setenv("CHAT_CLAUDE_AGENT_CLI_PATH", path)
    with pytest.raises(RuntimeError, match="Configured Claude CLI") as error:
        subscription.validate_subscription(path if explicit else None)
    assert str(cli.parent) not in str(error.value)
    subscription.provision_credentials_file.assert_not_called()
    subscription.find_bundled_cli.assert_not_called()
    subscription.subprocess.run.assert_not_called()


def test_unset_cli_keeps_existing_discovery(
    runs: list[list[str]], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        subscription, "find_bundled_cli", lambda: "existing-bundled-or-path-cli"
    )
    subscription.validate_subscription()
    assert [call[0] for call in runs] == [
        os.path.realpath("existing-bundled-or-path-cli")
    ] * 2


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


@pytest.mark.parametrize(
    "setting", ["CHAT_CLAUDE_AGENT_CLI_PATH", "CLAUDE_AGENT_CLI_PATH"]
)
def test_validation_cache_follows_environment_selection(
    cli: Path, runs: list[list[str]], monkeypatch: pytest.MonkeyPatch, setting: str
):
    second = cli.with_name("2.1.285.exe")
    second.write_bytes(b"another executable")
    second.chmod(0o755)
    for path in (cli, cli, second, cli):
        monkeypatch.setenv(setting, str(path))
        subscription.validate_subscription()
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
