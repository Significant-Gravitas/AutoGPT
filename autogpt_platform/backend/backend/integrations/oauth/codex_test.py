from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.codex.auth_bundle import CodexAuthBundleError
from backend.integrations.codex.chatgpt_auth import (
    ChatGPTDeviceCode,
    ChatGPTPollResult,
    CodexAuthError,
)
from backend.integrations.oauth import DEVICE_HANDLERS_BY_NAME
from backend.integrations.oauth.codex import (
    CodexDeviceAuthHandler,
    _pack_poll_key,
    _unpack_poll_key,
)


def test_codex_is_registered_as_a_device_auth_provider() -> None:
    assert DEVICE_HANDLERS_BY_NAME["codex"] is CodexDeviceAuthHandler


def test_refreshes_are_serialized_because_openai_rotates_the_token() -> None:
    """Without this the credentials manager takes the unlocked path and two
    workers can replay the same rotated token, revoking the whole grant."""
    assert CodexDeviceAuthHandler.ROTATES_REFRESH_TOKEN is True


# --------------------------------------------------------------------------- #
# Poll key
#
# ChatGPT polls on a (device_auth_id, user_code) pair while the base contract
# carries one opaque string, so the pair is packed into it.
# --------------------------------------------------------------------------- #


def test_the_poll_key_round_trips() -> None:
    packed = _pack_poll_key("deviceauth_6a907a91", "FWK1-91A0I")
    assert _unpack_poll_key(packed) == ("deviceauth_6a907a91", "FWK1-91A0I")


@pytest.mark.parametrize("bad", ["", "deviceauth_only", "|FWK1-91A0I", "abc|"])
def test_a_malformed_poll_key_is_rejected(bad: str) -> None:
    with pytest.raises(CodexAuthError):
        _unpack_poll_key(bad)


@pytest.mark.asyncio
async def test_initiate_reports_the_verify_url_and_a_usable_interval() -> None:
    device = ChatGPTDeviceCode.model_validate(
        {
            "device_auth_id": "deviceauth_abc",
            "user_code": "FWK1-91A0I",
            "interval": "5",
        }
    )
    with patch(
        "backend.integrations.oauth.codex.request_device_code",
        new=AsyncMock(return_value=device),
    ):
        initiation = await CodexDeviceAuthHandler().initiate_device_auth([])

    assert initiation.user_code == "FWK1-91A0I"
    assert initiation.verification_url == "https://auth.openai.com/codex/device"
    # There is no pre-filled variant; claiming one would send users to a 404.
    assert initiation.verification_url_complete is None
    assert initiation.interval == 5
    assert initiation.expires_in > 0
    assert _unpack_poll_key(initiation.device_code) == (
        "deviceauth_abc",
        "FWK1-91A0I",
    )


@pytest.mark.asyncio
async def test_a_pending_poll_does_not_attempt_the_token_exchange() -> None:
    exchange = AsyncMock()
    with (
        patch(
            "backend.integrations.oauth.codex.poll_device_code",
            new=AsyncMock(return_value=ChatGPTPollResult(status="pending")),
        ),
        patch(
            "backend.integrations.oauth.codex.exchange_authorization_code",
            new=exchange,
        ),
    ):
        result = await CodexDeviceAuthHandler().poll_for_tokens(
            _pack_poll_key("deviceauth_abc", "FWK1-91A0I")
        )

    assert result.status == "pending"
    assert result.credentials is None
    exchange.assert_not_awaited()


@pytest.mark.asyncio
async def test_malformed_approved_tokens_become_a_normal_auth_error() -> None:
    with (
        patch(
            "backend.integrations.oauth.codex.poll_device_code",
            new=AsyncMock(
                return_value=ChatGPTPollResult(
                    status="approved",
                    authorization_code="auth-code",
                    code_verifier="verifier",
                )
            ),
        ),
        patch(
            "backend.integrations.oauth.codex.exchange_authorization_code",
            new=AsyncMock(return_value=object()),
        ),
        patch(
            "backend.integrations.oauth.codex.bundle_from_tokens",
            side_effect=CodexAuthBundleError("invalid jwt"),
        ),
    ):
        with pytest.raises(CodexAuthError, match="invalid token data"):
            await CodexDeviceAuthHandler().poll_for_tokens(
                _pack_poll_key("deviceauth_abc", "FWK1-91A0I")
            )


@pytest.mark.asyncio
async def test_malformed_refresh_tokens_become_a_normal_auth_error() -> None:
    credentials = OAuth2Credentials(
        provider="codex",
        access_token=SecretStr("access"),
        refresh_token=SecretStr("refresh"),
        scopes=[],
        refresh_strategy="oauth_handler",
    )
    with (
        patch(
            "backend.integrations.oauth.codex.refresh_access_token",
            new=AsyncMock(return_value=object()),
        ),
        patch(
            "backend.integrations.oauth.codex.bundle_from_tokens",
            side_effect=CodexAuthBundleError("invalid jwt"),
        ),
    ):
        with pytest.raises(CodexAuthError, match="invalid token data"):
            await CodexDeviceAuthHandler()._refresh_tokens(credentials)


@pytest.mark.asyncio
async def test_revocation_reports_false_rather_than_claiming_success() -> None:
    """OpenAI publishes no revocation endpoint for this client."""
    handler = CodexDeviceAuthHandler()
    assert await handler.revoke_tokens(None) is False  # type: ignore[arg-type]
