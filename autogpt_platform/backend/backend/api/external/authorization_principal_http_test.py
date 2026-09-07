import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
from prisma.enums import APIKeyPermission, APIKeyStatus

from backend.api.external import authorization_principal as principal
from backend.api.external import middleware
from backend.data.auth.api_key import APIKeyInfo


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def server():
    return None


@pytest_asyncio.fixture(scope="session", loop_scope="session", autouse=True)
async def graph_cleanup():
    yield


@pytest.mark.asyncio
async def test_revocation_cancels_fastapi_handler_and_returns_403(mocker):
    auth = APIKeyInfo(
        id="key",
        name="test",
        head="agpt_x",
        tail="tail",
        status=APIKeyStatus.ACTIVE,
        scopes=[APIKeyPermission.READ_GRAPH],
        created_at=datetime.now(UTC),
        user_id="user",
        organization_id="org",
        team_id_restriction="team",
    )
    allowed = True
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def validate(*_args):
        if not allowed:
            raise HTTPException(403, "API key permissions changed")
        return principal._PrincipalScope(
            organization_id="org", team_id="team", owner_id="user"
        )

    mocker.patch.object(principal, "_validate_principal", side_effect=validate)
    mocker.patch.object(principal, "PRINCIPAL_POLL_INTERVAL_SECONDS", 0.01)
    mocker.patch.object(
        middleware, "has_live_resource_access", AsyncMock(return_value=True)
    )
    app = FastAPI()
    app.dependency_overrides[middleware.require_auth] = lambda: auth

    @app.get("/long-action")
    async def action(_auth=middleware.require_permission(APIKeyPermission.READ_GRAPH)):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        request = asyncio.create_task(client.get("/long-action"))
        await asyncio.wait_for(started.wait(), 1)
        allowed = False
        response = await asyncio.wait_for(request, 1)
    assert response.status_code == 403
    assert response.json() == {"detail": "API key permissions changed"}
    assert cancelled.is_set()
