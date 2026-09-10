"""Tests for the Stripe Link device-code handler.

The polling branch is the important one: RFC 8628 signals `pending`,
`slow_down`, `expired_token` and `access_denied` all as HTTP 400, so
misreading the body silently turns a normal wait into a hard failure.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.oauth.stripe_link import StripeLinkDeviceAuthHandler


def mock_http(status_code: int, payload: dict, text: str = ""):
    """Patch the module's httpx client to return one canned response."""
    response = MagicMock()
    response.status_code = status_code
    response.json = MagicMock(return_value=payload)
    response.text = text
    response.raise_for_status = MagicMock()

    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return (
        patch(
            "backend.integrations.oauth.stripe_link.httpx.AsyncClient",
            return_value=ctx,
        ),
        client,
    )


def credentials(refresh_token: str | None = "old-refresh") -> OAuth2Credentials:
    return OAuth2Credentials(
        provider="stripe_link",
        access_token=SecretStr("old-access"),
        refresh_token=SecretStr(refresh_token) if refresh_token else None,
        access_token_expires_at=int(time.time()) + 60,
        scopes=["userinfo:read", "payment_methods.agentic"],
        title="Stripe Link",
    )


@pytest.mark.asyncio
async def test_initiate_maps_the_device_code_response():
    patcher, client = mock_http(
        200,
        {
            "device_code": "lwldevice_abc",
            "user_code": "glow-relish-chaste-soft",
            "verification_uri": "https://app.link.com/device/setup",
            "verification_uri_complete": "https://app.link.com/device/setup?code=glow",
            "expires_in": 600,
            "interval": 5,
        },
    )
    with patcher:
        result = await StripeLinkDeviceAuthHandler().initiate_device_auth([])

    assert result.device_code == "lwldevice_abc"
    assert result.user_code == "glow-relish-chaste-soft"
    assert result.verification_url == "https://app.link.com/device/setup"
    assert result.expires_in == 600 and result.interval == 5

    sent = client.post.call_args.kwargs["data"]
    # Defaults must be applied, since an empty scope list would be rejected
    assert sent["scope"] == "userinfo:read payment_methods.agentic"
    assert sent["client_id"].startswith("lwlpk_")


@pytest.mark.asyncio
async def test_initiate_defaults_the_poll_interval_when_absent():
    """`interval` is optional in RFC 8628; polling must not crash without it."""
    patcher, _ = mock_http(
        200,
        {
            "device_code": "d",
            "user_code": "u",
            "verification_uri": "https://app.link.com/device/setup",
            "expires_in": 600,
        },
    )
    with patcher:
        result = await StripeLinkDeviceAuthHandler().initiate_device_auth([])
    assert result.interval == 5
    assert result.verification_url_complete is None


@pytest.mark.asyncio
async def test_poll_returns_credentials_once_approved():
    patcher, _ = mock_http(
        200,
        {
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_in": 3600,
        },
    )
    with patcher:
        result = await StripeLinkDeviceAuthHandler().poll_for_tokens("dc")

    assert result.status == "approved"
    assert result.credentials is not None
    assert result.credentials.access_token.get_secret_value() == "new-access"
    assert result.credentials.provider == "stripe_link"
    assert result.credentials.access_token_expires_at is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_code, expected",
    [
        ("authorization_pending", "pending"),
        ("slow_down", "slow_down"),
        ("expired_token", "expired"),
        ("access_denied", "denied"),
    ],
)
async def test_poll_maps_each_rfc8628_error_code(error_code, expected):
    """All four arrive as HTTP 400 and must not be conflated."""
    patcher, _ = mock_http(400, {"error": error_code})
    with patcher:
        result = await StripeLinkDeviceAuthHandler().poll_for_tokens("dc")
    assert result.status == expected
    assert result.credentials is None


@pytest.mark.asyncio
async def test_poll_slow_down_backs_off():
    patcher, _ = mock_http(400, {"error": "slow_down"})
    with patcher:
        result = await StripeLinkDeviceAuthHandler().poll_for_tokens("dc")
    assert result.next_poll_interval == 10


@pytest.mark.asyncio
async def test_poll_raises_on_an_unrecognised_response():
    """An unknown failure must be loud, not silently treated as pending."""
    patcher, _ = mock_http(500, {}, text="upstream exploded")
    with patcher, pytest.raises(RuntimeError, match="Unexpected response"):
        await StripeLinkDeviceAuthHandler().poll_for_tokens("dc")


@pytest.mark.asyncio
async def test_refresh_replaces_both_tokens():
    """Link rotates the refresh token, so keeping the old one locks the user out."""
    patcher, _ = mock_http(
        200,
        {
            "access_token": "rotated-access",
            "refresh_token": "rotated-refresh",
            "expires_in": 3600,
        },
    )
    creds = credentials()
    with patcher:
        updated = await StripeLinkDeviceAuthHandler()._refresh_tokens(creds)

    assert updated.access_token.get_secret_value() == "rotated-access"
    assert updated.refresh_token is not None
    assert updated.refresh_token.get_secret_value() == "rotated-refresh"
    assert updated.access_token_expires_at > int(time.time())


@pytest.mark.asyncio
async def test_refresh_without_a_refresh_token_is_an_error():
    with pytest.raises(RuntimeError, match="No refresh token"):
        await StripeLinkDeviceAuthHandler()._refresh_tokens(credentials(None))


@pytest.mark.asyncio
async def test_revoke_reports_success_and_failure():
    patcher, _ = mock_http(200, {})
    with patcher:
        assert await StripeLinkDeviceAuthHandler().revoke_tokens(credentials()) is True

    patcher, _ = mock_http(400, {})
    with patcher:
        assert await StripeLinkDeviceAuthHandler().revoke_tokens(credentials()) is False


@pytest.mark.asyncio
async def test_revoke_without_a_refresh_token_is_a_noop():
    assert await StripeLinkDeviceAuthHandler().revoke_tokens(credentials(None)) is False


@pytest.mark.asyncio
async def test_credentials_manager_resolves_the_device_handler():
    """The device registry is the seam between the handler and the platform:
    if this lookup misses, refresh silently falls through to the OAuth path
    and fails for every device-code provider."""
    from backend.integrations.creds_manager import IntegrationCredentialsManager

    handler = await IntegrationCredentialsManager()._get_oauth_handler(credentials())
    assert isinstance(handler, StripeLinkDeviceAuthHandler)


@pytest.mark.parametrize(
    "host, expected",
    [
        ("platform.agpt.co", "AutoGPT on platform.agpt.co"),
        # Link truncates at 32 chars, and this is the approval sheet's headline —
        # a clean name beats "AutoGPT on prompt-neat-flea.ngro".
        ("prompt-neat-flea.ngrok-free.app", "AutoGPT"),
        ("", "AutoGPT"),
    ],
)
def test_connection_label_never_exceeds_links_limit(host, expected):
    from backend.integrations.oauth.stripe_link import (
        CONNECTION_LABEL_MAX_LEN,
        _connection_label,
    )

    label = _connection_label(host)
    assert label == expected
    assert len(label) <= CONNECTION_LABEL_MAX_LEN
