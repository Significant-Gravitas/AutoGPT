"""An API key or OAuth access token is refused when its owner is banned."""

from datetime import datetime, timedelta, timezone

import fastapi
import pytest
from fastapi.testclient import TestClient
from prisma.enums import APIKeyPermission, APIKeyStatus

from backend.api.external import middleware
from backend.data.auth.api_key import APIKeyInfo
from backend.data.auth.oauth import OAuthAccessTokenInfo

_NOW = datetime.now(timezone.utc)


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
        name="Key",
        head="agpt_",
        tail="abcd",
        status=APIKeyStatus.ACTIVE,
        user_id=user_id,
        scopes=[APIKeyPermission.EXECUTE_GRAPH],
        created_at=_NOW,
    )


def _token(user_id: str) -> OAuthAccessTokenInfo:
    return OAuthAccessTokenInfo(
        id="token-1",
        user_id=user_id,
        scopes=[APIKeyPermission.EXECUTE_GRAPH],
        created_at=_NOW,
        expires_at=_NOW + timedelta(hours=1),
        application_id="app-1",
    )
