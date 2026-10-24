"""The external API rejects a key or OAuth token whose user closed their account.

Deletion revokes every key and grant, but one minted while the account was
closing, or by an admin acting as the user afterwards, would otherwise keep
working until the account's data is erased.

A key or token whose owner is banned is refused with a 403.
"""

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import fastapi
import pytest
from autogpt_libs.auth import forget_account_state, remember_account_closed
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
from prisma.enums import APIKeyPermission, APIKeyStatus
from pytest_mock import MockerFixture

from backend.api.external import middleware
from backend.api.external.middleware import (
    require_access_token,
    require_api_key,
    require_auth,
)
from backend.data.auth.api_key import APIKeyInfo
from backend.data.auth.oauth import OAuthAccessTokenInfo

MODULE = "backend.api.external.middleware"
OPEN_USER = "open-external-user"
CLOSED_USER = "closed-external-user"
BEARER = HTTPAuthorizationCredentials(scheme="Bearer", credentials="agpt_xt_token")
CLOSED = "This account is closed."


@pytest.fixture(autouse=True)
def closed_account() -> Iterator[None]:
    remember_account_closed(CLOSED_USER)
    yield
    for user_id in (OPEN_USER, CLOSED_USER):
        forget_account_state(user_id)


@pytest.mark.parametrize(
    "state, status",
    [("banned", 403), ("until_tomorrow", 403), ("lapsed", 200), ("unbanned", 200)],
)
@pytest.mark.parametrize(
    "path, headers",
    [
        ("/auth", {"X-API-Key": "key"}),
        ("/auth", {"Authorization": "Bearer token"}),
        ("/api-key", {"X-API-Key": "key"}),
        ("/access-token", {"Authorization": "Bearer token"}),
    ],
)
def test_a_credential_answers_by_its_owners_ban_state(
    mocker, auth_row, state, status, path, headers
):
    user_id = auth_row(state)
    mocker.patch.object(middleware, "validate_api_key", return_value=_key(user_id))
    mocker.patch.object(
        middleware, "validate_access_token", return_value=(_token(user_id), None)
    )

    response = _client().get(path, headers=headers)

    assert response.status_code == status
    if status == 403:
        assert response.json()["detail"]["code"] == "account_banned"
    else:
        assert response.json() == {"user_id": user_id}


def _client() -> TestClient:
    app = fastapi.FastAPI()

    @app.get("/auth")
    async def by_auth(auth=fastapi.Security(middleware.require_auth)):
        return {"user_id": auth.user_id}

    @app.get("/api-key")
    async def by_api_key(auth=fastapi.Security(middleware.require_api_key)):
        return {"user_id": auth.user_id}

    @app.get("/access-token")
    async def by_access_token(auth=fastapi.Security(middleware.require_access_token)):
        return {"user_id": auth.user_id}

    return TestClient(app)


def _key(user_id: str) -> APIKeyInfo:
    return APIKeyInfo(
        id="key-1",
        name="integration",
        head="agpt_",
        tail="abcd",
        status=APIKeyStatus.ACTIVE,
        user_id=user_id,
        scopes=[APIKeyPermission.IDENTITY],
        created_at=datetime.now(timezone.utc),
    )


def _token(user_id: str) -> OAuthAccessTokenInfo:
    now = datetime.now(timezone.utc)
    return OAuthAccessTokenInfo(
        id="token-1",
        user_id=user_id,
        scopes=[APIKeyPermission.IDENTITY],
        created_at=now,
        expires_at=now + timedelta(hours=1),
        application_id="app-1",
    )


def _validate_key(mocker: MockerFixture, user_id: str) -> None:
    mocker.patch(
        f"{MODULE}.validate_api_key",
        new_callable=AsyncMock,
        return_value=_key(user_id),
    )


def _validate_token(mocker: MockerFixture, user_id: str) -> None:
    mocker.patch(
        f"{MODULE}.validate_access_token",
        new_callable=AsyncMock,
        return_value=(_token(user_id), MagicMock()),
    )


async def _by_key() -> object:
    return await require_api_key(api_key="agpt_key")


async def _by_token() -> object:
    return await require_access_token(bearer=BEARER)


async def _auth_by_key() -> object:
    return await require_auth(api_key="agpt_key", bearer=None)


async def _auth_by_token() -> object:
    return await require_auth(api_key=None, bearer=BEARER)


KEY_CHECKS = [_by_key, _auth_by_key]
TOKEN_CHECKS = [_by_token, _auth_by_token]


@pytest.mark.asyncio
@pytest.mark.parametrize("check", KEY_CHECKS)
async def test_a_closed_accounts_api_key_is_refused(mocker: MockerFixture, check):
    _validate_key(mocker, CLOSED_USER)

    with pytest.raises(HTTPException) as raised:
        await check()

    assert (raised.value.status_code, raised.value.detail) == (401, CLOSED)


@pytest.mark.asyncio
@pytest.mark.parametrize("check", TOKEN_CHECKS)
async def test_a_closed_accounts_oauth_token_is_refused(mocker: MockerFixture, check):
    _validate_token(mocker, CLOSED_USER)

    with pytest.raises(HTTPException) as raised:
        await check()

    assert (raised.value.status_code, raised.value.detail) == (401, CLOSED)


@pytest.mark.asyncio
@pytest.mark.parametrize("check", KEY_CHECKS)
async def test_an_open_accounts_api_key_still_works(mocker: MockerFixture, check):
    _validate_key(mocker, OPEN_USER)

    info = await check()

    assert isinstance(info, APIKeyInfo)
    assert info.user_id == OPEN_USER


@pytest.mark.asyncio
@pytest.mark.parametrize("check", TOKEN_CHECKS)
async def test_an_open_accounts_oauth_token_still_works(mocker: MockerFixture, check):
    _validate_token(mocker, OPEN_USER)

    info = await check()

    assert isinstance(info, OAuthAccessTokenInfo)
    assert info.user_id == OPEN_USER
