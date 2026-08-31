import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.oauth.microsoft_365_copilot import (
    MICROSOFT_365_COPILOT_PUBLIC_CLIENT_ID,
    Microsoft365CopilotDeviceAuthHandler,
)


def _mock_client(mocker, responses: list[MagicMock]) -> AsyncMock:
    client = AsyncMock()
    client.post = AsyncMock(side_effect=responses)
    client.get = AsyncMock()
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=None)
    mocker.patch(
        "backend.integrations.oauth.microsoft_365_copilot.httpx.AsyncClient",
        return_value=context,
    )
    return client


@pytest.mark.asyncio
async def test_device_auth_uses_public_client_without_secret(
    mocker, monkeypatch
) -> None:
    monkeypatch.delenv("MICROSOFT_CLIENT_ID", raising=False)
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "device_code": "device-code",
        "user_code": "ABCD-EFGH",
        "verification_uri": "https://microsoft.com/devicelogin",
        "expires_in": 900,
        "interval": 5,
    }
    client = _mock_client(mocker, [response])

    result = await Microsoft365CopilotDeviceAuthHandler().initiate_device_auth([])

    assert result.user_code == "ABCD-EFGH"
    body = client.post.await_args.kwargs["data"]
    assert body["client_id"] == MICROSOFT_365_COPILOT_PUBLIC_CLIENT_ID
    assert "client_secret" not in body
    assert "offline_access" in body["scope"].split()
    assert "User.Read" in body["scope"].split()
    assert set(Microsoft365CopilotDeviceAuthHandler.DEFAULT_SCOPES).issubset(
        body["scope"].split()
    )


@pytest.mark.asyncio
async def test_poll_and_refresh_rotate_tokens_without_secret(
    mocker,
) -> None:
    mocker.patch(
        "backend.integrations.oauth.microsoft_365_copilot._client_id",
        return_value="configured-client",
    )
    poll = MagicMock(status_code=200)
    poll.json.return_value = {
        "access_token": "access-one",
        "refresh_token": "refresh-one",
        "expires_in": 3600,
        "scope": "User.Read Sites.Read.All Mail.Read",
    }
    profile = MagicMock(status_code=200)
    profile.json.return_value = {
        "id": "microsoft-user-id",
        "displayName": "Nick Example",
        "mail": "nick@example.com",
        "userPrincipalName": "nick@example.onmicrosoft.com",
    }
    refresh = MagicMock(status_code=200)
    refresh.json.return_value = {
        "access_token": "access-two",
        "refresh_token": "refresh-two",
        "expires_in": 7200,
        "scope": "User.Read Sites.Read.All Mail.Read",
    }
    client = _mock_client(mocker, [poll, refresh])
    client.get.return_value = profile
    mocker.patch.object(time, "time", return_value=1_000)
    handler = Microsoft365CopilotDeviceAuthHandler()

    result = await handler.poll_for_tokens("device-code")
    assert result.status == "approved"
    assert result.credentials is not None
    assert result.credentials.username == "nick@example.com"
    assert result.credentials.title == "Nick Example"
    assert result.credentials.metadata["microsoft_account_id"] == "microsoft-user-id"
    result.credentials.id = "credential-id"
    refreshed = await handler.refresh_tokens(result.credentials)

    assert refreshed.id == "credential-id"
    assert refreshed.access_token.get_secret_value() == "access-two"
    assert refreshed.refresh_token
    assert refreshed.refresh_token.get_secret_value() == "refresh-two"
    assert refreshed.access_token_expires_at == 8_200
    for call in client.post.await_args_list:
        assert call.kwargs["data"]["client_id"] == "configured-client"
        assert "client_secret" not in call.kwargs["data"]
    client.get.assert_awaited_once()


@pytest.mark.asyncio
async def test_profile_failure_does_not_discard_authorized_tokens(mocker) -> None:
    poll = MagicMock(status_code=200)
    poll.json.return_value = {
        "access_token": "access-one",
        "refresh_token": "refresh-one",
        "expires_in": 3600,
        "scope": "User.Read Sites.Read.All",
    }
    client = _mock_client(mocker, [poll])
    client.get.side_effect = RuntimeError("profile unavailable")

    result = await Microsoft365CopilotDeviceAuthHandler().poll_for_tokens("code")

    assert result.status == "approved"
    assert result.credentials is not None
    assert result.credentials.username is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "status"),
    [
        ("authorization_pending", "pending"),
        ("slow_down", "slow_down"),
        ("authorization_declined", "denied"),
        ("expired_token", "expired"),
    ],
)
async def test_poll_maps_microsoft_device_statuses(mocker, error, status) -> None:
    response = MagicMock(status_code=400)
    response.json.return_value = {"error": error}
    _mock_client(mocker, [response])

    result = await Microsoft365CopilotDeviceAuthHandler().poll_for_tokens("code")

    assert result.status == status


@pytest.mark.asyncio
async def test_refresh_keeps_existing_refresh_token_when_omitted(mocker) -> None:
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "access_token": "new-access",
        "expires_in": 3600,
        "scope": "Sites.Read.All",
    }
    _mock_client(mocker, [response])
    credentials = OAuth2Credentials(
        id="credential-id",
        provider="microsoft_365_copilot",
        access_token=SecretStr("old-access"),
        refresh_token=SecretStr("old-refresh"),
        scopes=["Sites.Read.All"],
    )

    refreshed = await Microsoft365CopilotDeviceAuthHandler().refresh_tokens(credentials)

    assert refreshed.refresh_token
    assert refreshed.refresh_token.get_secret_value() == "old-refresh"


def test_initial_token_without_scope_keeps_requested_default_scopes() -> None:
    credentials = (
        Microsoft365CopilotDeviceAuthHandler()._credentials_from_token_response(
            {
                "access_token": "access",
                "refresh_token": "refresh",
                "expires_in": 3600,
            }
        )
    )

    assert credentials.scopes == Microsoft365CopilotDeviceAuthHandler.DEFAULT_SCOPES


def test_malformed_token_expiry_does_not_crash_authentication() -> None:
    credentials = (
        Microsoft365CopilotDeviceAuthHandler()._credentials_from_token_response(
            {
                "access_token": "access",
                "refresh_token": "refresh",
                "expires_in": "not-a-number",
            }
        )
    )

    assert credentials.access_token_expires_at is None
