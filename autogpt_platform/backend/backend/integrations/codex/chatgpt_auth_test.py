from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.integrations.codex import chatgpt_auth
from backend.integrations.codex.chatgpt_auth import (
    ChatGPTDeviceCode,
    CodexAuthError,
    exchange_authorization_code,
    poll_device_code,
    refresh_access_token,
    request_device_code,
)


class _FakeResponse:
    def __init__(self, status: int, payload: object) -> None:
        self.status = status
        self._payload = payload

    def json(self) -> object:
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _patch_post(response: _FakeResponse):
    requests = AsyncMock()
    requests.post = AsyncMock(return_value=response)
    return patch.object(chatgpt_auth, "Requests", return_value=requests)


def _error(code: str) -> dict[str, object]:
    return {"error": {"message": "…", "type": "invalid_request_error", "code": code}}


# --------------------------------------------------------------------------- #
# Interval coercion
#
# ChatGPT sends `interval` as a *string*. A naive int() would raise, and a
# `or default` fallback would turn 0 into a busy loop against the auth server.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("5", 5),
        (5, 5),
        ("2.7", 2),
        ("0", 5),
        (-3, 5),
        ("", 5),
        (None, 5),
        ("banana", 5),
        (99999, 60),
    ],
)
def test_interval_is_coerced_to_a_sane_number_of_seconds(raw, expected) -> None:
    device = ChatGPTDeviceCode.model_validate(
        {"device_auth_id": "deviceauth_x", "user_code": "AAAA-BBBBB", "interval": raw}
    )
    assert device.interval == expected


def test_verification_url_is_the_fixed_codex_device_page() -> None:
    device = ChatGPTDeviceCode(device_auth_id="deviceauth_x", user_code="AAAA-BBBBB")
    assert device.verification_url == "https://auth.openai.com/codex/device"


def test_seconds_remaining_uses_the_absolute_expiry() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    device = ChatGPTDeviceCode(
        device_auth_id="deviceauth_x",
        user_code="AAAA-BBBBB",
        expires_at=now + timedelta(seconds=600),
    )
    assert device.seconds_remaining(now=now) == 600


# --------------------------------------------------------------------------- #
# Poll classification
#
# The load-bearing case: pending answers 403 and an unknown/expired login
# answers 404. Switching on HTTP status instead of the error code would either
# poll a dead login until the deadline or abort one the user is mid-approval on.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_403_authorization_pending_is_pending_not_a_failure() -> None:
    with _patch_post(_FakeResponse(403, _error("deviceauth_authorization_pending"))):
        result = await poll_device_code("deviceauth_x", "AAAA-BBBBB")
    assert result.status == "pending"


@pytest.mark.asyncio
async def test_404_not_found_is_expired_rather_than_pending() -> None:
    with _patch_post(_FakeResponse(404, _error("deviceauth_not_found"))):
        result = await poll_device_code("deviceauth_x", "AAAA-BBBBB")
    assert result.status == "expired"


@pytest.mark.asyncio
async def test_approved_poll_returns_the_code_and_verifier() -> None:
    payload = {"authorization_code": "ac_123", "code_verifier": "cv_456"}
    with _patch_post(_FakeResponse(200, payload)):
        result = await poll_device_code("deviceauth_x", "AAAA-BBBBB")
    assert result.status == "approved"
    assert result.authorization_code == "ac_123"
    assert result.code_verifier == "cv_456"


@pytest.mark.asyncio
async def test_approval_without_a_code_raises_instead_of_looking_approved() -> None:
    with _patch_post(_FakeResponse(200, {"authorization_code": "ac_123"})):
        with pytest.raises(CodexAuthError):
            await poll_device_code("deviceauth_x", "AAAA-BBBBB")


@pytest.mark.asyncio
async def test_unrecognised_429_backs_off_rather_than_killing_the_login() -> None:
    with _patch_post(_FakeResponse(429, {"error": {"code": "something_new"}})):
        result = await poll_device_code("deviceauth_x", "AAAA-BBBBB")
    assert result.status == "slow_down"


@pytest.mark.asyncio
async def test_denied_is_terminal() -> None:
    with _patch_post(_FakeResponse(403, _error("access_denied"))):
        result = await poll_device_code("deviceauth_x", "AAAA-BBBBB")
    assert result.status == "denied"


@pytest.mark.asyncio
async def test_an_unknown_error_raises_rather_than_polling_forever() -> None:
    with _patch_post(_FakeResponse(500, _error("internal_error"))):
        with pytest.raises(CodexAuthError):
            await poll_device_code("deviceauth_x", "AAAA-BBBBB")


@pytest.mark.asyncio
async def test_a_non_json_body_is_a_failure_not_a_pending_state() -> None:
    with _patch_post(_FakeResponse(502, None)):
        with pytest.raises(CodexAuthError):
            await poll_device_code("deviceauth_x", "AAAA-BBBBB")


@pytest.mark.asyncio
async def test_every_auth_http_call_has_a_finite_retry_budget() -> None:
    cases = [
        (
            request_device_code,
            (),
            _FakeResponse(
                200,
                {"device_auth_id": "deviceauth_x", "user_code": "AAAA-BBBBB"},
            ),
            {"retry_max_attempts": 3},
        ),
        (
            poll_device_code,
            ("deviceauth_x", "AAAA-BBBBB"),
            _FakeResponse(403, _error("deviceauth_authorization_pending")),
            {"raise_for_status": False, "retry_max_attempts": 1},
        ),
        (
            exchange_authorization_code,
            ("auth-code", "verifier"),
            _FakeResponse(
                200,
                {"id_token": "id", "access_token": "at", "refresh_token": "rt"},
            ),
            {"raise_for_status": False, "retry_max_attempts": 3},
        ),
        (
            refresh_access_token,
            (SecretStr("old-refresh"),),
            _FakeResponse(
                200,
                {"id_token": "id", "access_token": "at", "refresh_token": "rt"},
            ),
            {"raise_for_status": False, "retry_max_attempts": 3},
        ),
    ]

    for operation, args, response, expected_kwargs in cases:
        requests = AsyncMock()
        requests.post = AsyncMock(return_value=response)
        factory = MagicMock(return_value=requests)
        with patch.object(chatgpt_auth, "Requests", factory):
            await operation(*args)
        factory.assert_called_once_with(**expected_kwargs)


# --------------------------------------------------------------------------- #
# Token exchange
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_authorization_code_exchange_uses_oauth_form_encoding() -> None:
    response = _FakeResponse(
        200,
        {"id_token": "id", "access_token": "at", "refresh_token": "rt"},
    )
    requests = AsyncMock()
    requests.post = AsyncMock(return_value=response)

    with patch.object(chatgpt_auth, "Requests", return_value=requests):
        await exchange_authorization_code("auth-code", "pkce-verifier")

    _, kwargs = requests.post.await_args
    assert "json" not in kwargs
    assert kwargs["headers"]["content-type"] == "application/x-www-form-urlencoded"
    assert kwargs["data"] == {
        "grant_type": "authorization_code",
        "client_id": chatgpt_auth.CLIENT_ID,
        "code": "auth-code",
        "code_verifier": "pkce-verifier",
        "redirect_uri": chatgpt_auth.DEVICE_REDIRECT_URI,
    }


@pytest.mark.asyncio
async def test_authorization_code_exchange_reports_a_safe_provider_error() -> None:
    with _patch_post(_FakeResponse(400, _error("invalid_grant"))):
        with pytest.raises(CodexAuthError, match="HTTP 400, code=invalid_grant"):
            await exchange_authorization_code("auth-code", "pkce-verifier")


# --------------------------------------------------------------------------- #
# Refresh
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_refresh_keeps_the_old_token_when_none_is_returned() -> None:
    """Dropping it would leave the credential unable to refresh ever again."""
    payload = {"id_token": "id", "access_token": "at"}
    with _patch_post(_FakeResponse(200, payload)):
        tokens = await refresh_access_token(SecretStr("old-refresh"))
    assert tokens.refresh_token.get_secret_value() == "old-refresh"


@pytest.mark.asyncio
async def test_refresh_keeps_the_old_id_token_when_none_is_returned() -> None:
    payload = {"access_token": "at", "refresh_token": "new-refresh"}
    with _patch_post(_FakeResponse(200, payload)):
        tokens = await refresh_access_token(
            SecretStr("old-refresh"),
            current_id_token=SecretStr("old-id"),
        )
    assert tokens.id_token.get_secret_value() == "old-id"


@pytest.mark.asyncio
async def test_refresh_reports_a_safe_error_without_any_id_token() -> None:
    payload = {"access_token": "at", "refresh_token": "new-refresh"}
    with _patch_post(_FakeResponse(200, payload)):
        with pytest.raises(CodexAuthError, match="omitted the ID token"):
            await refresh_access_token(SecretStr("old-refresh"))


@pytest.mark.asyncio
async def test_refresh_prefers_a_rotated_token_when_one_arrives() -> None:
    payload = {"id_token": "id", "access_token": "at", "refresh_token": "new-refresh"}
    with _patch_post(_FakeResponse(200, payload)):
        tokens = await refresh_access_token(SecretStr("old-refresh"))
    assert tokens.refresh_token.get_secret_value() == "new-refresh"


@pytest.mark.asyncio
async def test_refresh_uses_oauth_form_encoding() -> None:
    response = _FakeResponse(
        200,
        {"id_token": "id", "access_token": "at", "refresh_token": "new-refresh"},
    )
    requests = AsyncMock()
    requests.post = AsyncMock(return_value=response)

    with patch.object(chatgpt_auth, "Requests", return_value=requests):
        await refresh_access_token(SecretStr("old-refresh"))

    _, kwargs = requests.post.await_args
    assert "json" not in kwargs
    assert kwargs["headers"]["content-type"] == "application/x-www-form-urlencoded"
    assert kwargs["data"] == {
        "grant_type": "refresh_token",
        "client_id": chatgpt_auth.CLIENT_ID,
        "refresh_token": "old-refresh",
    }


@pytest.mark.asyncio
async def test_refresh_failure_raises() -> None:
    with _patch_post(_FakeResponse(400, _error("invalid_grant"))):
        with pytest.raises(CodexAuthError):
            await refresh_access_token(SecretStr("old-refresh"))
