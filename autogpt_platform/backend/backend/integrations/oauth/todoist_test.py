import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr
from pytest_mock import MockerFixture

from backend.data.model import OAuth2Credentials
from backend.integrations.oauth.todoist import TodoistOAuthHandler
from backend.integrations.providers import ProviderName


@pytest.mark.parametrize(
    "token_response,refresh_token,expires",
    [
        pytest.param(
            {"access_token": "a", "expires_in": 3600, "refresh_token": "r"},
            "r",
            True,
            id="refresh-enabled app",
        ),
        pytest.param(
            {"access_token": "a", "expires_in": 315360000},
            None,
            False,
            id="legacy app",
        ),
    ],
)
async def test_exchange_stores_the_refresh_token_and_expiry_only_when_issued(
    mocker: MockerFixture, token_response, refresh_token, expires
):
    _mock_token_endpoint(mocker, token_response)

    credentials = await _handler().exchange_code_for_tokens("code", [], None)

    stored = credentials.refresh_token
    assert (stored.get_secret_value() if stored else None) == refresh_token
    assert (credentials.access_token_expires_at is not None) == expires


async def test_refresh_sends_the_refresh_token_and_stores_the_rotated_one(
    mocker: MockerFixture,
):
    post = _mock_token_endpoint(
        mocker, {"access_token": "a2", "expires_in": 3600, "refresh_token": "r2"}
    )

    fresh = await _handler().refresh_tokens(_credentials())

    assert post.await_args.kwargs["data"]["grant_type"] == "refresh_token"
    assert post.await_args.kwargs["data"]["refresh_token"] == "r1"
    assert fresh.id == "credential-id"
    assert fresh.access_token.get_secret_value() == "a2"
    assert fresh.refresh_token is not None
    assert fresh.refresh_token.get_secret_value() == "r2"
    assert fresh.access_token_expires_at is not None
    assert fresh.access_token_expires_at > time.time()


async def test_replayed_refresh_drops_the_spent_token_and_fails_after_expiry(
    mocker: MockerFixture,
):
    # Todoist answers a reused refresh token inside its 60 s window without one.
    post = _mock_token_endpoint(mocker, {"access_token": "a2", "expires_in": 3600})
    clock = mocker.patch("time.time", return_value=1_750_000_000)
    handler = _handler()

    fresh = await handler.refresh_tokens(_credentials())

    assert fresh.refresh_token is None
    assert fresh.access_token_expires_at == 1_750_003_600
    assert await handler.get_access_token(fresh) == "a2"

    clock.return_value = 1_750_003_601
    with pytest.raises(ValueError, match="No refresh token available"):
        await handler.get_access_token(fresh)
    post.assert_awaited_once()


async def test_legacy_access_token_remains_usable_without_a_refresh_token(
    mocker: MockerFixture,
):
    post = _mock_token_endpoint(mocker, {})
    credentials = _credentials().model_copy(
        update={"refresh_token": None, "access_token_expires_at": None}
    )

    assert await _handler().get_access_token(credentials) == "a1"
    post.assert_not_awaited()


def _handler() -> TodoistOAuthHandler:
    return TodoistOAuthHandler(
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri="https://example.com/callback",
    )


def _credentials() -> OAuth2Credentials:
    return OAuth2Credentials(
        id="credential-id",
        provider=ProviderName.TODOIST,
        access_token=SecretStr("a1"),
        refresh_token=SecretStr("r1"),
        access_token_expires_at=1,
        scopes=["data:read"],
    )


def _mock_token_endpoint(mocker: MockerFixture, payload: dict) -> AsyncMock:
    def respond(url: str, **_kwargs) -> MagicMock:
        if url == TodoistOAuthHandler.TOKEN_URL:
            return MagicMock(json=MagicMock(return_value=payload))
        return MagicMock(json=MagicMock(return_value={"user": {"email": "e@x.io"}}))

    post = AsyncMock(side_effect=respond)
    mocker.patch(
        "backend.integrations.oauth.todoist.Requests",
        return_value=MagicMock(post=post),
    )
    return post
