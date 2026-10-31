import json
import logging
import os
import subprocess
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from backend.copilot.sdk import subscription

ACCOUNT_EMAIL = "synthetic-account-privacy-16237@example.invalid"


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
