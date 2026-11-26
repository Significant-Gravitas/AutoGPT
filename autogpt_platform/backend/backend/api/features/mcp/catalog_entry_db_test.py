"""An MCP sign-in names the catalog entry a skill asks for (#17342).

A Marketplace skill names an MCP service by its catalog entry
(``mcp_amplitude``), while the OAuth callback stores the sign-in as
``provider="mcp"`` with only the server's URL in its metadata, so a skill page
reading providers alone kept offering to connect it. This drives the real
callback, credential store and listing through PostgreSQL and Redis; only the
vendor's token endpoint is faked.
"""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock
from uuid import uuid4

import fastapi
import httpx
import pytest
import pytest_asyncio
from autogpt_libs.auth import get_user_id
from prisma.models import Organization, User
from pydantic import SecretStr
from pytest_mock import MockerFixture

from backend.api.features.integrations.router import router as integrations_router
from backend.api.features.mcp import routes as mcp_routes
from backend.api.rest_api import app as rest_app
from backend.data import db
from backend.data.model import OAuth2Credentials
from backend.data.org_migration import ensure_personal_org

# One of Amplitude's region options rather than a single server URL, so the
# sign-in only finds its entry through the catalog's own URL matching.
AMPLITUDE_EU = "https://mcp.eu.amplitude.com/mcp"
LINEAR = "https://mcp.linear.app/mcp"
UNCATALOGUED = "https://mcp.uncatalogued.example.com/mcp"

app = fastapi.FastAPI(exception_handlers=rest_app.exception_handlers)
# Mounted where rest_api.py mounts them.
app.include_router(mcp_routes.router, prefix="/api/mcp")
app.include_router(integrations_router, prefix="/api/integrations")


@pytest_asyncio.fixture(loop_scope="session")
async def database() -> AsyncIterator[None]:
    owned_connection = not db.is_connected()
    if owned_connection:
        await db.connect()
    try:
        yield
    finally:
        if owned_connection:
            await db.disconnect()


@pytest.mark.asyncio(loop_scope="session")
async def test_an_mcp_sign_in_names_its_catalog_entry_to_its_owner_only(
    database: None, mocker: MockerFixture
):
    mocker.patch.object(
        mcp_routes.settings.config, "frontend_base_url", "https://platform.example"
    )
    mocker.patch.object(
        mcp_routes.MCPOAuthHandler,
        "exchange_code_for_tokens",
        new=AsyncMock(side_effect=_issued_token),
    )
    amplitude_user, linear_user = str(uuid4()), str(uuid4())
    try:
        for user_id in (amplitude_user, linear_user):
            await User.prisma().create(
                data={"id": user_id, "email": f"{user_id}@example.com"}
            )
            await ensure_personal_org(user_id)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as api:
            await _sign_in(api, amplitude_user, AMPLITUDE_EU)
            await _sign_in(api, amplitude_user, UNCATALOGUED)
            await _sign_in(api, linear_user, LINEAR)

            amplitude_listing = await _list_credentials(api, amplitude_user)
            linear_listing = await _list_credentials(api, linear_user)

        assert _sign_ins(amplitude_listing) == {
            AMPLITUDE_EU: "mcp_amplitude",
            UNCATALOGUED: None,
        }
        assert _sign_ins(linear_listing) == {LINEAR: "mcp_linear"}
        # Only a sign-in names a catalog entry, never a block credential.
        others = [c for c in amplitude_listing if c["provider"] != "mcp"]
        assert others
        assert all(c["mcp_catalog_entry"] is None for c in others)
    finally:
        app.dependency_overrides.clear()
        # The personal org has no FK to its User; deleting it takes the
        # credentials stored under it.
        await Organization.prisma().delete_many(
            where={"bootstrapUserId": {"in": [amplitude_user, linear_user]}}
        )
        await User.prisma().delete_many(
            where={"id": {"in": [amplitude_user, linear_user]}}
        )


async def _sign_in(api: httpx.AsyncClient, user_id: str, server_url: str) -> None:
    """What the login route leaves for the callback, then the callback itself."""
    origin = server_url.removesuffix("/mcp")
    state_token, _ = await mcp_routes.creds_manager.store.store_state_token(
        user_id,
        "mcp",
        [],
        state_metadata={
            "authorize_url": f"{origin}/authorize",
            "token_url": f"{origin}/token",
            "revoke_url": None,
            "resource_url": server_url,
            "server_url": server_url,
            "client_id": "autogpt-platform",
            "client_secret": "",
            "token_endpoint_auth_method": "none",
            "issuer": origin,
            "iss_required": False,
        },
    )
    app.dependency_overrides[get_user_id] = lambda: user_id
    response = await api.post(
        "/api/mcp/oauth/callback", json={"code": "code", "state_token": state_token}
    )
    assert response.status_code == 200, response.text


async def _list_credentials(api: httpx.AsyncClient, user_id: str) -> list[dict]:
    app.dependency_overrides[get_user_id] = lambda: user_id
    response = await api.get("/api/integrations/credentials")
    assert response.status_code == 200, response.text
    return response.json()


def _sign_ins(listing: list[dict]) -> dict[str, str | None]:
    return {
        c["host"]: c["mcp_catalog_entry"] for c in listing if c["provider"] == "mcp"
    }


async def _issued_token(*_args, **_kwargs) -> OAuth2Credentials:
    return OAuth2Credentials(
        provider="mcp",
        access_token=SecretStr("vendor-access-token"),
        scopes=[],
        title=None,
    )
