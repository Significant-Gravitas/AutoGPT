from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, Security
from httpx import ASGITransport, AsyncClient
from prisma.enums import APIKeyPermission, APIKeyStatus

from backend.api import org_rollout
from backend.api.external import middleware
from backend.api.external.v2 import tenancy as v2_tenancy
from backend.api.features.orgs import db as org_db
from backend.data.auth.api_key import APIKeyInfo
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.auth.oauth import OAuthAccessTokenInfo, OAuthApplicationInfo
from backend.util.exceptions import NotAuthorizedError


@pytest.mark.asyncio
@pytest.mark.parametrize("credential_type", ["key", "oauth"])
@pytest.mark.parametrize(
    "enabled,org,team,status",
    [
        (False, "shared", "shared-team", 403),
        (False, "someone-elses-personal", "their-default", 403),
        (False, "personal", "another-team", 403),
        (False, "personal", "default-team", 200),
        (False, "personal", None, 200),
        (False, None, None, 200),
        (True, "shared", "shared-team", 200),
    ],
)
async def test_external_auth_preserves_personal_scope_and_rejects_shared(
    credential_type, enabled, org, team, status, monkeypatch, mocker
):
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", str(enabled).lower())
    default = AsyncMock(return_value=("personal", "default-team"))
    mocker.patch.object(org_rollout, "get_user_default_team", default)
    mocker.patch.object(org_db, "get_user_default_team", default)
    now = datetime.now(UTC)
    if credential_type == "key":
        principal = APIKeyInfo(
            id="key",
            name="test",
            head="agpt_x",
            tail="tail",
            status=APIKeyStatus.ACTIVE,
            scopes=[APIKeyPermission.READ_GRAPH],
            created_at=now,
            user_id="owner",
            organization_id=org,
            team_id_restriction=team,
        )
        validation = mocker.patch.object(
            middleware, "validate_api_key", AsyncMock(return_value=principal)
        )
        headers = {"X-API-Key": "test-key"}
    else:
        principal = OAuthAccessTokenInfo(
            id="token",
            user_id="owner",
            scopes=[APIKeyPermission.READ_GRAPH],
            created_at=now,
            expires_at=now + timedelta(hours=1),
            application_id="app",
        )
        application = OAuthApplicationInfo(
            id="app",
            name="test",
            client_id="client",
            redirect_uris=[],
            grant_types=["authorization_code"],
            scopes=[APIKeyPermission.READ_GRAPH],
            owner_id="app-owner",
            is_active=True,
            created_at=now,
            updated_at=now,
            organization_id=org,
            team_id_restriction=team,
        )
        validation = mocker.patch.object(
            middleware,
            "validate_access_token",
            AsyncMock(return_value=(principal, application)),
        )
        headers = {"Authorization": "Bearer test-token"}

    before = principal.model_dump()
    app = FastAPI()

    @app.get("/resource")
    async def resource(auth=Security(middleware.require_auth)):
        return {"org": auth.organization_id, "team": auth.team_id_restriction}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/resource", headers=headers)
    assert response.status_code == status
    validation.assert_awaited_once()
    assert principal.model_dump() == before
    if status == 200:
        assert response.json() == {
            "org": org if org is not None else "personal",
            "team": team if org is not None else "default-team",
        }
    else:
        assert "not enabled" in response.json()["detail"]
    if not enabled:
        default.assert_awaited_once_with("owner")


@pytest.mark.asyncio
async def test_invalid_api_key_stays_unauthorized_before_rollout(mocker):
    validation = mocker.patch.object(
        middleware, "validate_api_key", AsyncMock(return_value=None)
    )
    gate = mocker.patch.object(
        middleware, "require_personal_scope_without_collaboration", AsyncMock()
    )
    app = FastAPI()

    @app.get("/resource")
    async def resource(auth=Security(middleware.require_auth)):
        return {"org": auth.organization_id}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/resource", headers={"X-API-Key": "invalid-key"})
    assert response.status_code == 401
    validation.assert_awaited_once_with("invalid-key")
    gate.assert_not_awaited()


@pytest.fixture
def v2_membership(mocker):
    """Every membership v2 checks is active, so only the rollout gate decides."""
    default = AsyncMock(return_value=("personal", "default-team"))
    mocker.patch.object(org_rollout, "get_user_default_team", default)
    mocker.patch.object(org_db, "get_user_default_team", default)
    membership = mocker.patch.object(v2_tenancy, "prisma")
    membership.orgmember.find_unique = AsyncMock(
        return_value=MagicMock(status="ACTIVE", Org=MagicMock(deletedAt=None))
    )
    return membership


def _v2_credential(org: str | None, team: str | None) -> APIAuthorizationInfo:
    return APIAuthorizationInfo(
        user_id="owner",
        scopes=[APIKeyPermission.READ_GRAPH],
        type="api_key",
        created_at=datetime.now(UTC),
        organization_id=org,
        team_id_restriction=team,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "enabled,org,team,header,allowed",
    [
        (False, "shared", None, None, False),
        (False, "shared", "shared-team", None, False),
        (False, "personal", "another-team", None, False),
        (False, "personal", None, "another-team", False),
        (False, None, None, "another-team", False),
        (False, "personal", "default-team", None, True),
        (False, "personal", None, "default-team", True),
        (False, None, None, "default-team", True),
        (False, "personal", None, None, True),
        (False, None, None, None, True),
        (True, "shared", "shared-team", None, True),
    ],
)
async def test_v2_tenant_preserves_personal_scope_and_rejects_shared(
    enabled, org, team, header, allowed, monkeypatch, v2_membership
):
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", str(enabled).lower())
    v2_membership.teammember.find_unique = AsyncMock(
        return_value=MagicMock(status="ACTIVE", Team=MagicMock(orgId=org or "personal"))
    )
    credential = _v2_credential(org, team)

    if not allowed:
        with pytest.raises(NotAuthorizedError, match="not enabled"):
            await v2_tenancy.resolve_tenant(credential, requested_team=header)
        return
    tenant = await v2_tenancy.resolve_tenant(credential, requested_team=header)
    assert (tenant.organization_id, tenant.team_id) == (
        org or "personal",
        header or team,
    )


@pytest.mark.asyncio
async def test_v2_mcp_credential_tenancy_rejects_shared_scope(
    monkeypatch, v2_membership
):
    """The MCP server resolves its tenant outside `resolve_tenant`."""
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", "false")

    with pytest.raises(NotAuthorizedError, match="not enabled"):
        await v2_tenancy.resolve_credential_tenancy(_v2_credential("shared", None))
    assert await v2_tenancy.resolve_credential_tenancy(
        _v2_credential("personal", None)
    ) == ("personal", None)
