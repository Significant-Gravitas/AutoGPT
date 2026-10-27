from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from prisma.enums import APIKeyPermission

from backend.api.features import oauth
from backend.data.auth.oauth import OAuthApplicationInfo

REGISTERED = "https://client.example/callback"


@pytest.mark.asyncio
@pytest.mark.parametrize("redirect_uri", [REGISTERED, None])
async def test_app_info_answers_for_a_registered_or_absent_redirect_uri(redirect_uri):
    info = await _get_app_info(redirect_uri)

    assert info.name == "Test app"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "redirect_uri",
    [
        "https://evil.example/phish",
        f"{REGISTERED}/",
        "javascript:alert(document.domain)//",
    ],
)
async def test_app_info_refuses_a_redirect_uri_the_app_did_not_register(
    redirect_uri,
):
    with pytest.raises(HTTPException) as error:
        await _get_app_info(redirect_uri)

    assert error.value.status_code == 400


async def _get_app_info(redirect_uri: str | None):
    now = datetime.now(UTC)
    app = OAuthApplicationInfo(
        id="app-123",
        name="Test app",
        client_id="client-123",
        redirect_uris=[REGISTERED],
        grant_types=["authorization_code"],
        scopes=[APIKeyPermission.READ_GRAPH],
        owner_id="owner",
        is_active=True,
        created_at=now,
        updated_at=now,
    )
    with patch.object(oauth, "get_oauth_application", AsyncMock(return_value=app)):
        return await oauth.get_oauth_app_info(
            "client-123", redirect_uri=redirect_uri, user_id="user"
        )
