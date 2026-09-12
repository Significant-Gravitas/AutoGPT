from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from autogpt_libs.auth.dependencies import get_request_context
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from autogpt_libs.auth.models import RequestContext
from fastapi import FastAPI, HTTPException, Request, Security
from httpx import ASGITransport, AsyncClient
from prisma.enums import APIKeyPermission, APIKeyStatus

from backend.api import org_rollout, rest_api
from backend.api.external import middleware
from backend.api.features import v1
from backend.data.auth.api_key import APIKeyInfo


@pytest.fixture
def rollout_boundary(monkeypatch, mocker):
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", "false")
    default = mocker.patch.object(
        org_rollout,
        "get_user_default_team",
        AsyncMock(return_value=("personal", "default-team")),
    )
    app = FastAPI()
    org_rollout.install_org_rollout_boundary(app)
    app.dependency_overrides[get_jwt_payload] = lambda: {"sub": "owner"}

    async def resolve(request: Request, jwt_payload: dict):
        return RequestContext(
            user_id=jwt_payload["sub"],
            org_id=request.headers.get("X-Org-Id", "personal"),
            team_id=request.headers.get("X-Team-Id") or None,
            is_org_owner=True,
            is_org_admin=True,
            is_org_billing_manager=False,
            is_team_admin=True,
            is_team_billing_manager=False,
            seat_status="ACTIVE",
        )

    auth = mocker.patch.object(
        org_rollout, "resolve_request_context", AsyncMock(side_effect=resolve)
    )

    @app.post("/resource")
    @app.delete("/resource")
    async def resource(ctx: RequestContext = Security(get_request_context)):
        return {"org": ctx.org_id, "team": ctx.team_id}

    @app.delete("/cleanup")
    @org_rollout.org_rollout_cleanup
    async def cleanup(ctx: RequestContext = Security(get_request_context)):
        return {"org": ctx.org_id, "team": ctx.team_id}

    @app.get("/public")
    async def public():
        return {"public": True}

    return app, auth, default


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["POST", "DELETE"])
@pytest.mark.parametrize(
    "org,team",
    [
        ("shared", "shared-team"),
        ("someone-elses-personal", "their-default"),
        ("personal", "another-team"),
        ("shared", ""),
    ],
)
async def test_disabled_boundary_rejects_explicit_shared_headers(
    rollout_boundary, method, org, team
):
    app, auth, _ = rollout_boundary
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.request(
            method, "/resource", headers={"X-Org-Id": org, "X-Team-Id": team}
        )
    assert response.status_code == 403
    assert "not enabled" in response.json()["detail"]
    auth.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"X-Org-Id": "personal"},
        {"X-Org-Id": "personal", "X-Team-Id": "default-team"},
        {"X-Team-Id": "default-team"},
    ],
)
async def test_disabled_boundary_preserves_personal_context(rollout_boundary, headers):
    app, auth, _ = rollout_boundary
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/resource", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"org": "personal", "team": headers.get("X-Team-Id")}
    auth.assert_awaited_once()


@pytest.mark.asyncio
async def test_enabled_boundary_keeps_shared_scope(rollout_boundary, monkeypatch):
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", "true")
    app, auth, default = rollout_boundary
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/resource", headers={"X-Org-Id": "shared", "X-Team-Id": "shared-team"}
        )
    assert response.status_code == 200
    assert response.json() == {"org": "shared", "team": "shared-team"}
    auth.assert_awaited_once()
    default.assert_not_awaited()


@pytest.mark.asyncio
async def test_only_exact_cleanup_handler_is_exempt(rollout_boundary):
    app, auth, default = rollout_boundary
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.delete("/cleanup", headers={"X-Org-Id": "shared"})
    assert response.status_code == 200
    assert response.json()["org"] == "shared"
    auth.assert_awaited_once()
    default.assert_not_awaited()


@pytest.mark.asyncio
async def test_rollout_does_not_bypass_auth_even_for_cleanup(rollout_boundary):
    app, auth, default = rollout_boundary
    auth.side_effect = HTTPException(403, "Membership revoked")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.delete("/cleanup", headers={"X-Org-Id": "shared"})
    assert response.status_code == 403
    assert response.json()["detail"] == "Membership revoked"
    default.assert_not_awaited()


@pytest.mark.asyncio
async def test_public_endpoints_do_not_gain_auth_or_rollout_dependency(
    rollout_boundary,
):
    app, auth, default = rollout_boundary
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/public", headers={"X-Org-Id": "shared"})
    assert response.status_code == 200
    auth.assert_not_awaited()
    default.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["require_api_key", "require_auth"])
async def test_scoped_api_key_cannot_reenable_shared_resources(
    rollout_boundary, mocker, method
):
    app, _, _ = rollout_boundary
    auth = APIKeyInfo(
        id="key",
        name="test",
        head="agpt_x",
        tail="tail",
        status=APIKeyStatus.ACTIVE,
        scopes=[APIKeyPermission.READ_GRAPH],
        created_at=datetime.now(UTC),
        user_id="owner",
        organization_id="shared",
        team_id_restriction="shared-team",
    )
    validate = mocker.patch.object(
        middleware, "validate_api_key", AsyncMock(return_value=auth)
    )
    dependency = (
        middleware.require_api_key
        if method == "require_api_key"
        else middleware.require_auth
    )

    @app.get("/external")
    async def external(principal=Security(dependency)):
        return {"org": principal.organization_id}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/external", headers={"X-API-Key": "test-key"})
    assert response.status_code == 403
    validate.assert_awaited_once_with("test-key")


@pytest.mark.asyncio
async def test_production_app_uses_boundary_before_resource_handler(
    rollout_boundary, mocker
):
    _, auth, _ = rollout_boundary
    assert (
        rest_api.app.dependency_overrides[get_request_context]
        is org_rollout.get_rollout_request_context
    )
    mocker.patch.dict(
        rest_api.app.dependency_overrides, {get_jwt_payload: lambda: {"sub": "owner"}}
    )
    async with AsyncClient(
        transport=ASGITransport(app=rest_api.app), base_url="http://test"
    ) as client:
        response = await client.get("/api/home", headers={"X-Org-Id": "shared"})
    assert response.status_code == 403
    assert "not enabled" in response.json()["detail"]
    auth.assert_awaited_once()


@pytest.mark.asyncio
async def test_body_team_override_cannot_select_shared_team(rollout_boundary, mocker):
    membership = mocker.patch.object(
        v1, "get_user_team_ids", AsyncMock(return_value=["another-team"])
    )
    with pytest.raises(HTTPException) as error:
        await v1._resolve_write_team_id("owner", "personal", "another-team")
    assert error.value.status_code == 403
    membership.assert_awaited_once_with("owner", "personal")


@pytest.mark.asyncio
async def test_disabled_boundary_fails_closed_without_canonical_org(rollout_boundary):
    _, _, default = rollout_boundary
    default.return_value = (None, None)
    with pytest.raises(HTTPException) as error:
        await org_rollout.require_personal_scope_without_collaboration(
            "owner", None, None
        )
    assert error.value.status_code == 403
