"""revoke_connected_apps: every app the user connected is revoked at its provider."""

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
        call(USER_ID, github, ProviderName.GITHUB, force=True),
        call(USER_ID, openai_key, ProviderName.OPENAI, force=True),
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
async def test_a_legacy_stored_provider_is_revoked_under_its_canonical_name(
    get_all_creds: AsyncMock, revoke: AsyncMock, cleanup: AsyncMock
):
    legacy = _github(provider="ProviderName.GITHUB")
    get_all_creds.return_value = [legacy]

    await revoke_connected_apps(USER_ID)

    revoke.assert_awaited_once_with(USER_ID, legacy, ProviderName.GITHUB, force=True)


@pytest.mark.asyncio
async def test_managed_cleanup_runs_after_every_revoke(
    get_all_creds: AsyncMock, call_order: MagicMock
):
    get_all_creds.return_value = [_github(), _openai_key()]

    await revoke_connected_apps(USER_ID)

    assert call_order.mock_calls == [
        call.revoke(
            USER_ID, get_all_creds.return_value[0], ProviderName.GITHUB, force=True
        ),
        call.revoke(
            USER_ID, get_all_creds.return_value[1], ProviderName.OPENAI, force=True
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
