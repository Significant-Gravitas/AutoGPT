"""Tests for the RMFG device-code handler.

RFC 8628 signals `pending`, `slow_down`, `expired_token` and `access_denied`
all as HTTP 400, so misreading the body turns a normal wait into a failure.
The scope handling matters too: a block that needs one opt-in permission
must not end up with a token that lacks the base permissions.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.oauth.rmfg import (
    BASE_SCOPES,
    RMFG_CLIENT_ID,
    RMFGDeviceAuthHandler,
)


def _response(status_code: int, payload: dict, text: str = "") -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json = MagicMock(return_value=payload)
    response.text = text
    response.raise_for_status = MagicMock()
    return response


def mock_http(
    post: tuple[int, dict] | None = None, get: tuple[int, dict] | None = None
):
    """Patch the module's httpx client with canned POST and GET responses."""
    client = MagicMock()
    if post:
        client.post = AsyncMock(return_value=_response(*post))
    if get:
        client.get = AsyncMock(return_value=_response(*get))
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return (
        patch("backend.integrations.oauth.rmfg.httpx.AsyncClient", return_value=ctx),
        client,
    )


def credentials(refresh_token: str | None = "old-refresh") -> OAuth2Credentials:
    return OAuth2Credentials(
        provider="rmfg",
        access_token=SecretStr("old-access"),
        refresh_token=SecretStr(refresh_token) if refresh_token else None,
        access_token_expires_at=int(time.time()) + 60,
        scopes=BASE_SCOPES,
        title="RMFG",
    )


DEVICE_PAYLOAD = {
    "device_code": "dev_abc",
    "user_code": "ZULF-5PT3",
    "verification_uri": "https://www.rmfg.com/connect",
    "verification_uri_complete": "https://www.rmfg.com/connect?user_code=ZULF-5PT3",
    "expires_in": 600,
    "interval": 5,
}


class TestInitiate:
    async def test_maps_the_device_code_response(self):
        patcher, client = mock_http(post=(200, DEVICE_PAYLOAD))
        with patcher:
            result = await RMFGDeviceAuthHandler().initiate_device_auth([])

        assert result.device_code == "dev_abc"
        assert result.user_code == "ZULF-5PT3"
        assert result.verification_url == "https://www.rmfg.com/connect"
        assert result.verification_url_complete is not None
        assert (result.expires_in, result.interval) == (600, 5)

        sent = client.post.call_args.kwargs["data"]
        assert sent["client_id"] == RMFG_CLIENT_ID
        # Nothing requested: ask for everything, including the opt-in scopes
        # RMFG shows as checkboxes on its approval page.
        assert sent["scope"] == "designs dfm quotes carts orders webhooks payments"

    async def test_base_scopes_ride_along_with_a_narrow_request(self):
        # The Pay Cart block requires only `payments`; the frontend forwards
        # exactly that. Without the base set the token could not read a cart.
        patcher, client = mock_http(post=(200, DEVICE_PAYLOAD))
        with patcher:
            await RMFGDeviceAuthHandler().initiate_device_auth(["payments"])

        sent = client.post.call_args.kwargs["data"]
        assert sent["scope"] == "designs dfm quotes carts orders payments"

    async def test_interval_defaults_when_absent(self):
        payload = {k: v for k, v in DEVICE_PAYLOAD.items() if k != "interval"}
        patcher, _ = mock_http(post=(200, payload))
        with patcher:
            result = await RMFGDeviceAuthHandler().initiate_device_auth([])
        assert result.interval == 5


class TestPoll:
    async def test_approved_records_granted_scopes_and_account(self):
        patcher, client = mock_http(
            post=(
                200,
                {
                    "access_token": "acc",
                    "refresh_token": "ref",
                    "expires_in": 900,
                    # The person left "Also allow paid orders" unchecked.
                    "scope": "designs dfm quotes carts orders webhooks",
                },
            ),
            get=(200, {"email": "ada@example.com", "auth_method": "api_key"}),
        )
        with patcher:
            result = await RMFGDeviceAuthHandler().poll_for_tokens("dev_abc")

        assert result.status == "approved"
        assert result.credentials is not None
        creds = result.credentials
        assert creds.access_token.get_secret_value() == "acc"
        assert creds.refresh_token is not None
        assert creds.refresh_token.get_secret_value() == "ref"
        assert creds.scopes == [
            "designs",
            "dfm",
            "quotes",
            "carts",
            "orders",
            "webhooks",
        ]
        assert "payments" not in creds.scopes
        assert creds.username == "ada@example.com"
        assert creds.access_token_expires_at is not None
        assert creds.access_token_expires_at > int(time.time()) + 800

        sent = client.post.call_args.kwargs["data"]
        assert sent["grant_type"] == "urn:ietf:params:oauth:grant-type:device_code"
        assert sent["device_code"] == "dev_abc"

    async def test_account_lookup_failure_does_not_lose_the_grant(self):
        patcher, _ = mock_http(
            post=(
                200,
                {"access_token": "acc", "refresh_token": "ref", "expires_in": 900},
            ),
            get=(500, {}),
        )
        with patcher:
            result = await RMFGDeviceAuthHandler().poll_for_tokens("dev_abc")
        assert result.status == "approved"
        assert result.credentials is not None
        assert result.credentials.username is None

    @pytest.mark.parametrize(
        "error, status",
        [
            ("authorization_pending", "pending"),
            ("slow_down", "slow_down"),
            ("expired_token", "expired"),
            ("access_denied", "denied"),
        ],
    )
    async def test_maps_rfc8628_400_states(self, error: str, status: str):
        patcher, _ = mock_http(post=(400, {"error": error}))
        with patcher:
            result = await RMFGDeviceAuthHandler().poll_for_tokens("dev_abc")
        assert result.status == status
        assert result.credentials is None

    async def test_slow_down_lengthens_the_interval(self):
        patcher, _ = mock_http(post=(400, {"error": "slow_down"}))
        with patcher:
            result = await RMFGDeviceAuthHandler().poll_for_tokens("dev_abc")
        assert result.next_poll_interval == 10

    async def test_unexpected_status_raises(self):
        patcher, _ = mock_http(post=(503, {}, "down"))
        with patcher:
            with pytest.raises(RuntimeError, match="503"):
                await RMFGDeviceAuthHandler().poll_for_tokens("dev_abc")


class TestRefresh:
    async def test_rotates_the_refresh_token(self):
        patcher, client = mock_http(
            post=(
                200,
                {
                    "access_token": "new-access",
                    "refresh_token": "new-refresh",
                    "expires_in": 900,
                },
            )
        )
        with patcher:
            updated = await RMFGDeviceAuthHandler().refresh_tokens(credentials())

        assert updated.access_token.get_secret_value() == "new-access"
        assert updated.refresh_token is not None
        assert updated.refresh_token.get_secret_value() == "new-refresh"
        assert updated.access_token_expires_at is not None
        assert updated.access_token_expires_at > int(time.time()) + 800
        sent = client.post.call_args.kwargs["data"]
        assert sent["grant_type"] == "refresh_token"
        assert sent["refresh_token"] == "old-refresh"
        assert sent["client_id"] == RMFG_CLIENT_ID

    async def test_a_missing_replacement_refresh_token_is_a_failed_refresh(self):
        # The old token has been consumed; saving it would make the next
        # refresh look like a replay and get the connection revoked.
        patcher, _ = mock_http(post=(200, {"access_token": "new-access"}))
        creds = credentials()
        with patcher, pytest.raises(RuntimeError, match="replacement refresh token"):
            await RMFGDeviceAuthHandler().refresh_tokens(creds)
        assert creds.access_token.get_secret_value() == "old-access"
        assert creds.refresh_token is not None
        assert creds.refresh_token.get_secret_value() == "old-refresh"

    async def test_a_missing_access_token_is_a_failed_refresh(self):
        patcher, _ = mock_http(post=(200, {"refresh_token": "new-refresh"}))
        with patcher, pytest.raises(RuntimeError, match="no access token"):
            await RMFGDeviceAuthHandler().refresh_tokens(credentials())

    async def test_requires_a_refresh_token(self):
        with pytest.raises(RuntimeError, match="No refresh token"):
            await RMFGDeviceAuthHandler().refresh_tokens(
                credentials(refresh_token=None)
            )

    async def test_refuses_other_providers(self):
        other = credentials()
        other.provider = "slant3d"
        with pytest.raises(ValueError, match="cannot refresh"):
            await RMFGDeviceAuthHandler().refresh_tokens(other)

    def test_is_marked_as_rotating(self):
        # The credentials manager serializes refreshes only for handlers that
        # say so; RMFG revokes the connection on refresh-token replay.
        assert RMFGDeviceAuthHandler.ROTATES_REFRESH_TOKEN is True


class TestRevoke:
    async def test_revokes_the_refresh_token(self):
        patcher, client = mock_http(post=(200, {}))
        with patcher:
            assert await RMFGDeviceAuthHandler().revoke_tokens(credentials()) is True
        sent = client.post.call_args.kwargs["data"]
        assert sent["token"] == "old-refresh"

    async def test_nothing_to_revoke(self):
        assert (
            await RMFGDeviceAuthHandler().revoke_tokens(credentials(refresh_token=None))
            is False
        )
