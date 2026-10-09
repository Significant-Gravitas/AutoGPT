"""Which lookup a credential gets, and so what checking it costs.

An API key is matched on its head and then hashed with Scrypt; an OAuth access
token is looked up by its digest. A bearer in the token format must never be
tried as a key: one old key headed like a token would otherwise make every
forged token cost a hash, ahead of every rate limit.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
import pytest_mock
from autogpt_libs.api_key.keysmith import APIKeySmith
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from backend.api.external.middleware import resolve_auth_info
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.auth.oauth import (
    ACCESS_TOKEN_PREFIX,
    REFRESH_TOKEN_PREFIX,
    InvalidTokenError,
)


def _bearer(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def _principal() -> APIAuthorizationInfo:
    return APIAuthorizationInfo(
        user_id="user-1", scopes=[], type="api_key", created_at=datetime.now(UTC)
    )


async def test_a_token_shaped_bearer_is_never_tried_as_an_api_key(
    mocker: pytest_mock.MockFixture,
) -> None:
    validate_key = mocker.patch(
        "backend.api.external.middleware.validate_api_key", new_callable=AsyncMock
    )
    mocker.patch(
        "backend.api.external.middleware.validate_access_token",
        new_callable=AsyncMock,
        side_effect=InvalidTokenError("not found"),
    )

    with pytest.raises(HTTPException) as rejected:
        await resolve_auth_info(api_key=None, bearer=_bearer("agpt_xt_forged"))

    assert rejected.value.status_code == 401
    validate_key.assert_not_awaited()


async def test_an_api_key_sent_as_a_bearer_still_authenticates(
    mocker: pytest_mock.MockFixture,
) -> None:
    """MCP clients can only send a key as a bearer."""
    principal = _principal()
    mocker.patch(
        "backend.api.external.middleware.validate_api_key",
        new_callable=AsyncMock,
        return_value=principal,
    )
    validate_token = mocker.patch(
        "backend.api.external.middleware.validate_access_token",
        new_callable=AsyncMock,
    )

    assert await resolve_auth_info(api_key=None, bearer=_bearer("agpt_key")) is (
        principal
    )
    validate_token.assert_not_awaited()


async def test_a_token_shaped_value_sent_as_an_api_key_is_tried_as_one(
    mocker: pytest_mock.MockFixture,
) -> None:
    """The header says what it is; the v2 limiter counts it as a key too."""
    validate_key = mocker.patch(
        "backend.api.external.middleware.validate_api_key",
        new_callable=AsyncMock,
        return_value=None,
    )

    with pytest.raises(HTTPException):
        await resolve_auth_info(api_key="agpt_xt_value", bearer=None)

    validate_key.assert_awaited_once_with("agpt_xt_value")


def test_no_api_key_is_issued_with_an_oauth_token_prefix() -> None:
    assert ACCESS_TOKEN_PREFIX in APIKeySmith.RESERVED_PREFIXES
    assert REFRESH_TOKEN_PREFIX in APIKeySmith.RESERVED_PREFIXES
