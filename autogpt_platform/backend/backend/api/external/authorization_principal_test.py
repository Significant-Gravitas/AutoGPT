import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi import HTTPException
from prisma.enums import APIKeyPermission, APIKeyStatus

from backend.api.external import authorization_principal as principal
from backend.data.auth.api_key import APIKeyInfo
from backend.data.auth.oauth import OAuthAccessTokenInfo

PERMISSIONS = (APIKeyPermission.EXECUTE_GRAPH,)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def server():
    return None


@pytest_asyncio.fixture(scope="session", loop_scope="session", autouse=True)
async def graph_cleanup():
    yield


@pytest.fixture
def authorization(mocker):
    key = SimpleNamespace(
        status=APIKeyStatus.ACTIVE,
        revokedAt=None,
        userId="user",
        permissions=list(PERMISSIONS),
        organizationId="org",
        teamIdRestriction="team",
        ownerType=None,
    )
    app = SimpleNamespace(
        isActive=True,
        scopes=list(PERMISSIONS),
        organizationId="org",
        teamIdRestriction="team",
        ownerId="app-owner",
        ownerType=None,
    )
    token = SimpleNamespace(
        applicationId="app",
        userId="user",
        revokedAt=None,
        expiresAt=datetime.now(UTC) + timedelta(hours=1),
        scopes=list(PERMISSIONS),
    )
    tx = MagicMock(query_raw=AsyncMock(return_value=[{"id": "principal"}]))
    transactions = SimpleNamespace(active=0, completed=0)

    @asynccontextmanager
    async def transaction(*_args, **_kwargs):
        transactions.active += 1
        try:
            yield tx
        finally:
            transactions.active -= 1
            transactions.completed += 1

    mocker.patch.object(principal, "prisma", MagicMock(tx=transaction))
    for model, row in (
        (principal.PrismaAPIKey, key),
        (principal.PrismaOAuthApplication, app),
        (principal.PrismaOAuthAccessToken, token),
    ):
        mocker.patch.object(
            model,
            "prisma",
            return_value=MagicMock(find_unique=AsyncMock(return_value=row)),
        )
    mocker.patch.object(principal, "PRINCIPAL_POLL_INTERVAL_SECONDS", 0.01)
    return SimpleNamespace(
        key=key, app=app, token=token, tx=tx, transactions=transactions
    )


def key_auth():
    return APIKeyInfo(
        id="key",
        name="test",
        head="agpt_x",
        tail="tail",
        status=APIKeyStatus.ACTIVE,
        scopes=list(PERMISSIONS),
        created_at=datetime.now(UTC),
        user_id="user",
        organization_id="org",
        team_id_restriction="team",
    )


def oauth_auth():
    return OAuthAccessTokenInfo(
        id="token",
        application_id="app",
        user_id="user",
        scopes=list(PERMISSIONS),
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        organization_id="org",
        team_id_restriction="team",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row,field,value,status_code",
    [
        ("key", "status", APIKeyStatus.REVOKED, 401),
        ("key", "revokedAt", datetime.now(UTC), 401),
        ("key", "permissions", [], 403),
        ("key", "organizationId", "another-org", 403),
        ("key", "teamIdRestriction", "another-team", 403),
        ("key", "userId", "another-user", 401),
        ("key", "ownerType", "TEAM", 401),
        ("app", "isActive", False, 401),
        ("app", "scopes", [], 403),
        ("app", "organizationId", "another-org", 403),
        ("app", "teamIdRestriction", None, 403),
        ("app", "ownerId", "another-owner", 403),
        ("app", "ownerType", "TEAM", 403),
        ("token", "revokedAt", datetime.now(UTC), 401),
        ("token", "expiresAt", datetime.now(UTC) - timedelta(hours=1), 401),
        ("token", "scopes", [], 403),
    ],
)
async def test_principal_change_cancels_inflight_action(
    authorization, row, field, value, status_code
):
    started, cancelled = asyncio.Event(), asyncio.Event()
    auth = key_auth() if row == "key" else oauth_auth()

    async def action():
        async with principal.live_authorization_principal(auth, PERMISSIONS):
            assert authorization.transactions.active == 0
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    task = asyncio.create_task(action())
    await started.wait()
    setattr(vars(authorization)[row], field, value)
    with pytest.raises(HTTPException) as exc:
        await asyncio.wait_for(task, timeout=1)
    assert exc.value.status_code == status_code
    assert cancelled.is_set()
    assert authorization.transactions.active == 0
    assert task.cancelling() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_factory,row", [(key_auth, "key"), (oauth_auth, "app")])
async def test_normalized_default_scope_preserves_original_null_scope(
    authorization, auth_factory, row
):
    record = vars(authorization)[row]
    record.organizationId = record.teamIdRestriction = None
    async with principal.live_authorization_principal(auth_factory(), PERMISSIONS):
        await asyncio.sleep(0.03)
        assert authorization.transactions.active == 0
    assert authorization.transactions.completed >= 3


@pytest.mark.asyncio
async def test_rechecks_revocation_after_action_between_polls(authorization, mocker):
    mocker.patch.object(principal, "PRINCIPAL_POLL_INTERVAL_SECONDS", 60)
    with pytest.raises(HTTPException) as exc:
        async with principal.live_authorization_principal(key_auth(), PERMISSIONS):
            authorization.key.permissions = []
    assert exc.value.status_code == 403
    assert authorization.transactions.completed == 2


@pytest.mark.asyncio
async def test_original_null_scope_cannot_change_midrequest(authorization):
    authorization.key.organizationId = authorization.key.teamIdRestriction = None
    with pytest.raises(HTTPException) as exc:
        async with principal.live_authorization_principal(key_auth(), PERMISSIONS):
            authorization.key.organizationId = "org"
            authorization.key.teamIdRestriction = "team"
    assert exc.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [False, True])
async def test_database_validation_failure_cancels_work_without_error_details(
    authorization, mocker, timeout
):
    mocker.patch.object(principal, "PRINCIPAL_VALIDATION_TIMEOUT_SECONDS", 0.02)

    async def hung_query(*_args):
        await asyncio.Event().wait()

    with pytest.raises(HTTPException) as exc:
        async with principal.live_authorization_principal(key_auth(), PERMISSIONS):
            authorization.tx.query_raw.side_effect = (
                hung_query if timeout else RuntimeError("secret-token-do-not-expose")
            )
            await asyncio.wait_for(asyncio.Event().wait(), 1)
    assert exc.value.status_code == 503
    assert exc.value.detail == "Authorization could not be validated"
    assert authorization.transactions.active == 0


@pytest.mark.asyncio
async def test_default_polling_cancels_within_about_one_second(authorization, mocker):
    mocker.patch.object(principal, "PRINCIPAL_POLL_INTERVAL_SECONDS", 0.5)
    with pytest.raises(HTTPException):
        async with principal.live_authorization_principal(key_auth(), PERMISSIONS):
            authorization.key.status = APIKeyStatus.REVOKED
            await asyncio.wait_for(asyncio.Event().wait(), timeout=1.2)


@pytest.mark.asyncio
async def test_client_cancellation_is_preserved_and_monitor_stops(authorization):
    started = asyncio.Event()

    async def action():
        async with principal.live_authorization_principal(key_auth(), PERMISSIONS):
            started.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(action())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    completed = authorization.transactions.completed
    await asyncio.sleep(0.03)
    assert authorization.transactions.completed == completed


@pytest.mark.asyncio
async def test_three_long_actions_release_connections_for_other_requests(authorization):
    started = asyncio.Queue()
    finish = asyncio.Event()

    async def action():
        async with principal.live_authorization_principal(key_auth(), PERMISSIONS):
            started.put_nowait(True)
            await finish.wait()

    tasks = [asyncio.create_task(action()) for _ in range(3)]
    try:
        for _ in tasks:
            await asyncio.wait_for(started.get(), 1)
        assert authorization.transactions.active == 0
    finally:
        finish.set()
        await asyncio.gather(*tasks)
