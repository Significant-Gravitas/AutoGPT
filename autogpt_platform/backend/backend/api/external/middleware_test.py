import contextlib
from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi import HTTPException
from fastapi.params import Depends as DependsParameter
from prisma.enums import APIKeyPermission, APIKeyStatus

from backend.api.external import middleware
from backend.data.auth.api_key import APIKeyInfo
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.auth.oauth import OAuthAccessTokenInfo, OAuthApplicationInfo


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def server():
    return None


@pytest_asyncio.fixture(scope="session", loop_scope="session", autouse=True)
async def graph_cleanup():
    yield


@pytest.fixture
def allow_live_authorization_principal(mocker):
    @contextlib.asynccontextmanager
    async def allow(*_args, **_kwargs):
        yield

    mocker.patch.object(middleware, "_live_authorization_principal", new=allow)


async def resolve_permission_dependency(permission, auth):
    marker = cast(DependsParameter, middleware.require_permission(permission))
    assert marker.scope == "function"
    assert marker.dependency is not None
    dependency = marker.dependency(auth)
    result = await anext(dependency)
    await dependency.aclose()
    return result


def scoped_auth(permission: APIKeyPermission) -> APIAuthorizationInfo:
    return APIAuthorizationInfo(
        user_id="user-1",
        scopes=[permission],
        type="api_key",
        created_at=datetime.now(UTC),
        organization_id="org-1",
        team_id_restriction="team-1",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("permission", "access"),
    [
        (APIKeyPermission.READ_GRAPH, "view"),
        (APIKeyPermission.WRITE_GRAPH, "create"),
        (APIKeyPermission.EXECUTE_GRAPH, "execute"),
    ],
)
async def test_scoped_key_rechecks_live_resource_access(
    mocker,
    allow_live_authorization_principal,
    permission: APIKeyPermission,
    access: str,
) -> None:
    live_access = mocker.patch.object(
        middleware,
        "has_live_resource_access",
        new=AsyncMock(return_value=False),
    )

    with pytest.raises(HTTPException) as exc:
        await resolve_permission_dependency(permission, scoped_auth(permission))

    assert exc.value.status_code == 403
    live_access.assert_awaited_once_with(
        "user-1",
        "org-1",
        "team-1",
        access,
    )


@pytest.mark.asyncio
async def test_identity_key_does_not_require_resource_access(
    mocker, allow_live_authorization_principal
) -> None:
    live_access = mocker.patch.object(
        middleware,
        "has_live_resource_access",
        new=AsyncMock(),
    )
    auth = scoped_auth(APIKeyPermission.IDENTITY)

    assert await resolve_permission_dependency(APIKeyPermission.IDENTITY, auth) == auth
    live_access.assert_not_awaited()


def api_key_auth(*permissions: APIKeyPermission) -> APIKeyInfo:
    return APIKeyInfo(
        id="key-1",
        name="test",
        head="agpt_x",
        tail="tail",
        status=APIKeyStatus.ACTIVE,
        scopes=list(permissions),
        created_at=datetime.now(UTC),
        user_id="user-1",
        organization_id="org-1",
        team_id_restriction="team-1",
    )


@pytest.mark.asyncio
async def test_api_key_principal_row_is_locked_through_action(mocker) -> None:
    tx = MagicMock()
    tx.query_raw = AsyncMock(return_value=[{"id": "key-1"}])
    transaction = MagicMock()
    transaction.__aenter__ = AsyncMock(return_value=tx)
    transaction.__aexit__ = AsyncMock(return_value=False)
    mocker.patch.object(
        middleware, "prisma", MagicMock(tx=MagicMock(return_value=transaction))
    )
    delegate = MagicMock(
        find_unique=AsyncMock(
            return_value=MagicMock(
                status=APIKeyStatus.ACTIVE,
                userId="user-1",
                permissions=[APIKeyPermission.READ_GRAPH],
            )
        )
    )
    mocker.patch.object(middleware.PrismaAPIKey, "prisma", return_value=delegate)

    async with middleware._live_authorization_principal(
        api_key_auth(APIKeyPermission.READ_GRAPH),
        (APIKeyPermission.READ_GRAPH,),
    ):
        transaction.__aexit__.assert_not_awaited()

    tx.query_raw.assert_awaited_once_with(
        'SELECT "id" FROM "APIKey" WHERE "id" = $1 FOR SHARE',
        "key-1",
    )
    transaction.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_api_key_permission_downgrade_fails_after_lock(mocker) -> None:
    tx = MagicMock(query_raw=AsyncMock(return_value=[{"id": "key-1"}]))
    transaction = MagicMock()
    transaction.__aenter__ = AsyncMock(return_value=tx)
    transaction.__aexit__ = AsyncMock(return_value=False)
    mocker.patch.object(
        middleware, "prisma", MagicMock(tx=MagicMock(return_value=transaction))
    )
    delegate = MagicMock(
        find_unique=AsyncMock(
            return_value=MagicMock(
                status=APIKeyStatus.ACTIVE,
                userId="user-1",
                permissions=[],
            )
        )
    )
    mocker.patch.object(middleware.PrismaAPIKey, "prisma", return_value=delegate)

    with pytest.raises(HTTPException) as exc:
        async with middleware._live_authorization_principal(
            api_key_auth(APIKeyPermission.READ_GRAPH),
            (APIKeyPermission.READ_GRAPH,),
        ):
            pass

    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_oauth_without_app_scope_uses_live_default_context(mocker) -> None:
    token = OAuthAccessTokenInfo(
        id="token-1",
        user_id="user-1",
        scopes=[APIKeyPermission.READ_GRAPH],
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC),
        application_id="app-1",
    )
    application = OAuthApplicationInfo(
        id="app-1",
        name="App",
        client_id="client-1",
        redirect_uris=[],
        grant_types=[],
        scopes=[APIKeyPermission.READ_GRAPH],
        owner_id="owner-1",
        is_active=True,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    default_context = mocker.patch(
        "backend.api.features.orgs.db.get_user_default_team",
        new=AsyncMock(return_value=("org-1", "team-1")),
    )

    scoped = await middleware._scope_oauth_token(token, application)

    assert scoped.organization_id == "org-1"
    assert scoped.team_id_restriction == "team-1"
    default_context.assert_awaited_once_with("user-1")


@pytest.mark.asyncio
async def test_oauth_app_scope_is_preserved(mocker) -> None:
    token = OAuthAccessTokenInfo(
        id="token-1",
        user_id="user-1",
        scopes=[APIKeyPermission.READ_GRAPH],
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC),
        application_id="app-1",
    )
    application = OAuthApplicationInfo(
        id="app-1",
        name="App",
        client_id="client-1",
        redirect_uris=[],
        grant_types=[],
        scopes=[APIKeyPermission.READ_GRAPH],
        owner_id="owner-1",
        is_active=True,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        organization_id="org-2",
        team_id_restriction="team-2",
    )
    default_context = mocker.patch(
        "backend.api.features.orgs.db.get_user_default_team",
        new=AsyncMock(),
    )

    scoped = await middleware._scope_oauth_token(token, application)

    assert scoped.organization_id == "org-2"
    assert scoped.team_id_restriction == "team-2"
    default_context.assert_not_awaited()
