"""revoke_connected_apps: every app the user connected is revoked at its provider."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from pydantic import SecretStr
from pytest_mock import MockerFixture

from backend.api.features.account_deletion.connected_apps import revoke_connected_apps
from backend.data.model import APIKeyCredentials, OAuth2Credentials
from backend.integrations.credentials_store import ollama_credentials
from backend.integrations.providers import ProviderName

MODULE = "backend.api.features.account_deletion.connected_apps"
USER_ID = "user-1"
SECRETS = ("ghp_access_secret", "ghp_refresh_secret", "sk-user-secret")


def _github(provider: str = "github") -> OAuth2Credentials:
    return OAuth2Credentials(
        id="cred-1",
        provider=provider,
        title="GitHub",
        access_token=SecretStr("ghp_access_secret"),
        refresh_token=SecretStr("ghp_refresh_secret"),
        scopes=["repo"],
        username="octocat",
    )


def _openai_key() -> APIKeyCredentials:
    return APIKeyCredentials(
        id="cred-2",
        provider="openai",
        title="My OpenAI key",
        api_key=SecretStr("sk-user-secret"),
    )


def _managed() -> APIKeyCredentials:
    return APIKeyCredentials(
        id="cred-managed",
        provider="agent_mail",
        title="AgentMail (managed by AutoGPT)",
        api_key=SecretStr("sk-managed-secret"),
        is_managed=True,
    )


def _sdk_default() -> APIKeyCredentials:
    return APIKeyCredentials(
        id="exa-default",
        provider="exa",
        title="Exa (default)",
        api_key=SecretStr("sk-platform-secret"),
    )


@pytest.fixture
def get_all_creds(mocker: MockerFixture) -> AsyncMock:
    manager = mocker.patch(f"{MODULE}.creds_manager")
    manager.store.get_all_creds = AsyncMock(return_value=[])
    return manager.store.get_all_creds


@pytest.fixture
def revoke(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch(
        f"{MODULE}.delete_and_revoke_credentials", new_callable=AsyncMock
    )


@pytest.fixture
def cleanup(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch(
        f"{MODULE}.cleanup_user_managed_credentials", new_callable=AsyncMock
    )


@pytest.fixture
def call_order(revoke: AsyncMock, cleanup: AsyncMock) -> MagicMock:
    order = MagicMock()
    order.attach_mock(revoke, "revoke")
    order.attach_mock(cleanup, "cleanup")
    return order


@pytest.mark.asyncio
async def test_each_user_owned_credential_is_revoked_once_with_force(
    get_all_creds: AsyncMock, revoke: AsyncMock, cleanup: AsyncMock
):
    github, openai_key = _github(), _openai_key()
    get_all_creds.return_value = [
        github,
        _managed(),
        ollama_credentials,
        _sdk_default(),
        openai_key,
    ]

    await revoke_connected_apps(USER_ID)

    get_all_creds.assert_awaited_once_with(USER_ID)
    assert revoke.await_args_list == [
        call(USER_ID, github, ProviderName.GITHUB, force=True, skip_failed_prunes=True),
        call(
            USER_ID,
            openai_key,
            ProviderName.OPENAI,
            force=True,
            skip_failed_prunes=True,
        ),
    ]


@pytest.mark.asyncio
async def test_managed_system_and_sdk_default_credentials_are_skipped(
    get_all_creds: AsyncMock, revoke: AsyncMock, cleanup: AsyncMock
):
    get_all_creds.return_value = [_managed(), ollama_credentials, _sdk_default()]

    await revoke_connected_apps(USER_ID)

    revoke.assert_not_awaited()
    cleanup.assert_awaited_once_with(USER_ID)


@pytest.mark.asyncio
async def test_one_failure_is_logged_without_secrets_and_the_rest_still_run(
    get_all_creds: AsyncMock,
    revoke: AsyncMock,
    cleanup: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    github, openai_key = _github(), _openai_key()
    get_all_creds.return_value = [github, openai_key]
    revoke.side_effect = [RuntimeError("provider rejected ghp_access_secret"), None]
    caplog.set_level(logging.WARNING, logger=MODULE)

    await revoke_connected_apps(USER_ID)

    assert revoke.await_count == 2
    assert revoke.await_args_list[1].args[1] is openai_key
    cleanup.assert_awaited_once_with(USER_ID)
    records = [r for r in caplog.records if r.name == MODULE]
    assert len(records) == 1
    assert records[0].levelno >= logging.WARNING
    message = records[0].getMessage()
    assert "cred-1" in message
    assert "github" in message
    assert "cred-2" not in caplog.text
    for secret in SECRETS:
        assert secret not in caplog.text


@pytest.mark.asyncio
async def test_a_provider_refusing_the_revoke_is_logged_without_secrets(
    get_all_creds: AsyncMock,
    revoke: AsyncMock,
    cleanup: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    """revoke_tokens reports a refusal (expired token, no revoke endpoint)
    as False rather than raising; the grant may still be live, so say so."""
    get_all_creds.return_value = [_github(), _openai_key()]
    revoke.side_effect = [False, None]
    caplog.set_level(logging.WARNING, logger=MODULE)

    await revoke_connected_apps(USER_ID)

    cleanup.assert_awaited_once_with(USER_ID)
    [record] = [r for r in caplog.records if r.name == MODULE]
    assert record.levelno == logging.WARNING
    assert "cred-1" in record.getMessage()
    assert "github" in record.getMessage()
    assert "cred-2" not in caplog.text
    for secret in SECRETS:
        assert secret not in caplog.text


@pytest.mark.asyncio
async def test_a_legacy_stored_provider_is_revoked_under_its_canonical_name(
    get_all_creds: AsyncMock, revoke: AsyncMock, cleanup: AsyncMock
):
    legacy = _github(provider="ProviderName.GITHUB")
    get_all_creds.return_value = [legacy]

    await revoke_connected_apps(USER_ID)

    revoke.assert_awaited_once_with(
        USER_ID, legacy, ProviderName.GITHUB, force=True, skip_failed_prunes=True
    )


@pytest.mark.asyncio
async def test_managed_cleanup_runs_after_every_revoke(
    get_all_creds: AsyncMock, call_order: MagicMock
):
    get_all_creds.return_value = [_github(), _openai_key()]

    await revoke_connected_apps(USER_ID)

    assert call_order.mock_calls == [
        call.revoke(
            USER_ID,
            get_all_creds.return_value[0],
            ProviderName.GITHUB,
            force=True,
            skip_failed_prunes=True,
        ),
        call.revoke(
            USER_ID,
            get_all_creds.return_value[1],
            ProviderName.OPENAI,
            force=True,
            skip_failed_prunes=True,
        ),
        call.cleanup(USER_ID),
    ]


@pytest.mark.asyncio
async def test_managed_cleanup_runs_even_when_every_revoke_failed(
    get_all_creds: AsyncMock,
    revoke: AsyncMock,
    cleanup: AsyncMock,
    call_order: MagicMock,
):
    get_all_creds.return_value = [_github(), _openai_key()]
    revoke.side_effect = RuntimeError("provider down")

    await revoke_connected_apps(USER_ID)

    assert revoke.await_count == 2
    cleanup.assert_awaited_once_with(USER_ID)
    assert call_order.mock_calls[-1] == call.cleanup(USER_ID)


@pytest.mark.asyncio
async def test_an_error_listing_credentials_propagates_and_nothing_is_revoked(
    get_all_creds: AsyncMock, revoke: AsyncMock, cleanup: AsyncMock
):
    """The account must not close while the apps it connected are unknown."""
    get_all_creds.side_effect = RuntimeError("database down")

    with pytest.raises(RuntimeError, match="database down"):
        await revoke_connected_apps(USER_ID)

    revoke.assert_not_awaited()
    cleanup.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_error_in_the_managed_cleanup_propagates(
    get_all_creds: AsyncMock, revoke: AsyncMock, cleanup: AsyncMock
):
    get_all_creds.return_value = [_github()]
    cleanup.side_effect = RuntimeError("cleanup failed")

    with pytest.raises(RuntimeError, match="cleanup failed"):
        await revoke_connected_apps(USER_ID)

    revoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_stalled_provider_is_given_up_on_and_the_rest_still_run(
    mocker: MockerFixture,
    get_all_creds: AsyncMock,
    revoke: AsyncMock,
    cleanup: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    mocker.patch(f"{MODULE}.REVOKE_TIMEOUT_SECONDS", 0.05)
    github, openai_key = _github(), _openai_key()
    get_all_creds.return_value = [github, openai_key]

    async def stall_on_github(user_id, creds, provider, *, force, skip_failed_prunes):
        if creds is github:
            await asyncio.Event().wait()

    revoke.side_effect = stall_on_github
    caplog.set_level(logging.WARNING, logger=MODULE)

    await asyncio.wait_for(revoke_connected_apps(USER_ID), timeout=5)

    assert [c.args[1] for c in revoke.await_args_list] == [github, openai_key]
    cleanup.assert_awaited_once_with(USER_ID)
    [record] = [r for r in caplog.records if r.name == MODULE]
    assert "cred-1" in record.getMessage()
    assert "TimeoutError" in record.getMessage()


@pytest.mark.asyncio
async def test_providers_are_waited_on_together(
    mocker: MockerFixture,
    get_all_creds: AsyncMock,
    revoke: AsyncMock,
    cleanup: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    """Each revoke here only returns once both have started, so revoking one
    at a time would time both out."""
    mocker.patch(f"{MODULE}.REVOKE_TIMEOUT_SECONDS", 1.0)
    get_all_creds.return_value = [_github(), _openai_key()]
    started: list[str] = []
    both_started = asyncio.Event()

    async def wait_for_the_other(
        user_id, creds, provider, *, force, skip_failed_prunes
    ):
        started.append(creds.id)
        if len(started) == 2:
            both_started.set()
        await both_started.wait()

    revoke.side_effect = wait_for_the_other
    caplog.set_level(logging.WARNING, logger=MODULE)

    await asyncio.wait_for(revoke_connected_apps(USER_ID), timeout=5)

    assert started == ["cred-1", "cred-2"]
    assert not [r for r in caplog.records if r.name == MODULE]


@pytest.mark.asyncio
async def test_at_most_ten_providers_are_called_at_once(
    get_all_creds: AsyncMock, revoke: AsyncMock, cleanup: AsyncMock
):
    get_all_creds.return_value = [_github() for _ in range(25)]
    running = 0
    peak = 0

    async def slow_revoke(*_args, **_kwargs):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        running -= 1
        return True

    revoke.side_effect = slow_revoke

    await revoke_connected_apps(USER_ID)

    assert revoke.await_count == 25
    assert peak == 10
