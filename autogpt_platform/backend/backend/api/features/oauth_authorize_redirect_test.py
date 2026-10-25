from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import HTTPException
from prisma.enums import APIKeyPermission

from backend.api.features import oauth
from backend.data.auth.oauth import OAuthApplicationInfo

REGISTERED = "https://client.example/callback"
UNREGISTERED = "https://evil.example/collect"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "app_state, response_type",
    [("unknown", "code"), ("inactive", "code"), ("active", "token")],
)
async def test_an_error_found_before_the_redirect_uri_checks_out_is_not_redirected(
    app_state, response_type
):
    with pytest.raises(HTTPException) as error:
        await _authorize(app_state, UNREGISTERED, response_type=response_type)

    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_an_unexpected_failure_before_the_check_is_not_redirected():
    failing_lookup = AsyncMock(side_effect=RuntimeError("database unavailable"))
    with patch.object(oauth, "get_oauth_application", failing_lookup), pytest.raises(
        HTTPException
    ) as error:
        await oauth.authorize(request=_request(UNREGISTERED), user_id="user")

    assert error.value.status_code == 500


@pytest.mark.asyncio
async def test_an_error_found_once_the_redirect_uri_checks_out_goes_back_to_it():
    response = await _authorize("active", REGISTERED, response_type="token")

    sent_to = urlparse(response.redirect_url)
    assert f"{sent_to.scheme}://{sent_to.netloc}{sent_to.path}" == REGISTERED
    assert parse_qs(sent_to.query)["error"] == ["unsupported_response_type"]


@pytest.mark.asyncio
async def test_an_unexpected_failure_once_the_redirect_uri_checks_out_goes_back_to_it():
    failing_issue = AsyncMock(side_effect=RuntimeError("database unavailable"))
    with patch.object(oauth, "create_authorization_code", failing_issue):
        response = await _authorize("active", REGISTERED, response_type="code")

    sent_to = urlparse(response.redirect_url)
    assert f"{sent_to.scheme}://{sent_to.netloc}{sent_to.path}" == REGISTERED
    assert parse_qs(sent_to.query)["error"] == ["server_error"]


async def _authorize(app_state: str, redirect_uri: str, *, response_type: str):
    app = None if app_state == "unknown" else _app(is_active=app_state == "active")
    with patch.object(oauth, "get_oauth_application", AsyncMock(return_value=app)):
        return await oauth.authorize(
            request=_request(redirect_uri, response_type=response_type),
            user_id="user",
        )


def _request(redirect_uri: str, *, response_type: str = "code"):
    return oauth.AuthorizeRequest(
        client_id="client-123",
        redirect_uri=redirect_uri,
        scopes=[APIKeyPermission.READ_GRAPH.value],
        state="state-1",
        response_type=response_type,
        code_challenge="challenge",
    )


def _app(*, is_active: bool) -> OAuthApplicationInfo:
    now = datetime.now(UTC)
    return OAuthApplicationInfo(
        id="app-123",
        name="Test app",
        client_id="client-123",
        redirect_uris=[REGISTERED],
        grant_types=["authorization_code"],
        scopes=[APIKeyPermission.READ_GRAPH],
        owner_id="owner",
        is_active=is_active,
        created_at=now,
        updated_at=now,
    )
