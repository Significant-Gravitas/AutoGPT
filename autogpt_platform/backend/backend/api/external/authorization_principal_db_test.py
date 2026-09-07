"""Run with EXTERNAL_AUTH_DB_TESTS=1 and DATABASE_URL set to a disposable local DB."""

import asyncio
import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from urllib.parse import urlparse
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi import HTTPException
from prisma import Prisma
from prisma.enums import APIKeyPermission, APIKeyStatus

from backend.api.external import authorization_principal as principal
from backend.api.external.middleware import _scope_api_key
from backend.api.features.orgs.db import get_user_default_team
from backend.data import db
from backend.data.auth.api_key import APIKeyInfo, revoke_api_key
from backend.data.auth.oauth import OAuthAccessTokenInfo
from backend.data.org_migration import create_personal_org
from backend.util.exceptions import NotAuthorizedError

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("EXTERNAL_AUTH_DB_TESTS") != "1",
        reason="Requires an explicitly selected disposable PostgreSQL database",
    ),
]
PERMISSIONS = (APIKeyPermission.EXECUTE_GRAPH,)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def server():
    return None


@pytest_asyncio.fixture(scope="session", loop_scope="session", autouse=True)
async def graph_cleanup():
    yield


@pytest_asyncio.fixture
async def live_database():
    assert urlparse(db.DATABASE_URL).hostname in {"localhost", "127.0.0.1", "::1"}
    connected_here = not db.prisma.is_connected()
    if connected_here:
        await db.prisma.connect()
    writer = Prisma(auto_register=False, datasource={"url": db.DATABASE_URL})
    await writer.connect()
    user_id = str(uuid4())
    try:
        await writer.user.create(
            data={"id": user_id, "email": f"{user_id}@test.invalid"}
        )
        await create_personal_org(
            user_id, f"external-auth-{user_id}", "External auth test"
        )
        organization_id, team_id = await get_user_default_team(user_id)
        assert organization_id and team_id
        key = await writer.apikey.create(
            data={
                "name": "External authorization integration test",
                "head": "test",
                "tail": "test",
                "hash": str(uuid4()),
                "userId": user_id,
                "permissions": list(PERMISSIONS),
                "organizationId": organization_id,
                "teamIdRestriction": team_id,
                "teamId": team_id,
            }
        )
        yield SimpleNamespace(
            writer=writer,
            auth=APIKeyInfo.from_db(key),
            user_id=user_id,
            organization_id=organization_id,
            team_id=team_id,
        )
    finally:
        await writer.organization.delete_many(where={"bootstrapUserId": user_id})
        await writer.user.delete_many(where={"id": user_id})
        await writer.disconnect()
        if connected_here:
            await db.prisma.disconnect()


@pytest.mark.asyncio
async def test_three_blocked_handlers_do_not_hold_key_locks_or_connections(
    live_database,
):
    auth = live_database.auth
    started = asyncio.Queue()

    async def action():
        async with principal.live_authorization_principal(auth, PERMISSIONS):
            started.put_nowait(True)
            await asyncio.Event().wait()

    tasks = [asyncio.create_task(action()) for _ in range(3)]
    try:
        for _ in tasks:
            try:
                await asyncio.wait_for(started.get(), 2)
            except TimeoutError:
                failures = [
                    (task.exception(), type(task.exception().__context__).__name__)
                    for task in tasks
                    if task.done()
                ]
                pytest.fail(f"Handlers did not start: {failures!r}")
        assert await asyncio.wait_for(
            db.prisma.query_raw("SELECT 1 AS available"), 0.5
        ) == [{"available": 1}]
        async with live_database.writer.tx(timeout=timedelta(seconds=1)) as tx:
            await tx.execute_raw("SET LOCAL lock_timeout = '200ms'")
            await tx.execute_raw(
                'UPDATE "APIKey" SET "name" = $1 WHERE "id" = $2',
                "Concurrent update",
                auth.id,
            )
        revoked = await asyncio.wait_for(
            revoke_api_key(
                auth.id,
                auth.user_id,
                organization_id=auth.organization_id,
                team_id_restriction=auth.team_id_restriction,
                exact_scope=True,
            ),
            0.5,
        )
        assert revoked.status == APIKeyStatus.REVOKED
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), 1.2
        )
        assert all(
            isinstance(result, HTTPException) and result.status_code == 401
            for result in results
        )
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong_scope", ["owner", "organization", "team"])
async def test_live_key_revocation_preserves_owner_and_exact_scope_guards(
    live_database, wrong_scope
):
    auth = live_database.auth
    with pytest.raises(NotAuthorizedError):
        await revoke_api_key(
            auth.id,
            str(uuid4()) if wrong_scope == "owner" else auth.user_id,
            organization_id=(
                str(uuid4()) if wrong_scope == "organization" else auth.organization_id
            ),
            team_id_restriction=(
                str(uuid4()) if wrong_scope == "team" else auth.team_id_restriction
            ),
            exact_scope=True,
        )
    key = await live_database.writer.apikey.find_unique(where={"id": auth.id})
    assert key is not None and key.status == APIKeyStatus.ACTIVE


@pytest.mark.asyncio
async def test_live_null_scope_uses_default_and_detects_persisted_scope_change(
    live_database,
):
    key = await live_database.writer.apikey.update(
        where={"id": live_database.auth.id},
        data={"organizationId": None, "teamIdRestriction": None, "teamId": None},
    )
    assert key is not None
    auth = await _scope_api_key(APIKeyInfo.from_db(key))
    assert (auth.organization_id, auth.team_id_restriction) == (
        live_database.organization_id,
        live_database.team_id,
    )
    async with principal.live_authorization_principal(auth, PERMISSIONS):
        await asyncio.sleep(0.55)
    with pytest.raises(HTTPException) as exc:
        async with principal.live_authorization_principal(auth, PERMISSIONS):
            await live_database.writer.apikey.update(
                where={"id": auth.id},
                data={
                    "organizationId": auth.organization_id,
                    "teamIdRestriction": auth.team_id_restriction,
                },
            )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["revoke", "permissions", "scope"])
async def test_live_oauth_app_changes_cancel_blocked_handlers(live_database, change):
    writer = live_database.writer
    auth = live_database.auth
    app = await writer.oauthapplication.create(
        data={
            "name": "External auth integration test",
            "clientId": str(uuid4()),
            "clientSecret": str(uuid4()),
            "clientSecretSalt": str(uuid4()),
            "redirectUris": [],
            "scopes": list(PERMISSIONS),
            "ownerId": auth.user_id,
            "organizationId": auth.organization_id,
            "teamIdRestriction": auth.team_id_restriction,
        }
    )
    token = await writer.oauthaccesstoken.create(
        data={
            "token": str(uuid4()),
            "applicationId": app.id,
            "userId": auth.user_id,
            "scopes": list(PERMISSIONS),
            "expiresAt": datetime.now(UTC) + timedelta(hours=1),
        }
    )
    oauth_auth = OAuthAccessTokenInfo.from_db(token).model_copy(
        update={
            "organization_id": auth.organization_id,
            "team_id_restriction": auth.team_id_restriction,
        }
    )
    started = asyncio.Event()

    async def action():
        async with principal.live_authorization_principal(oauth_auth, PERMISSIONS):
            started.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(action())
    try:
        await asyncio.wait_for(started.wait(), 2)
        if change == "revoke":
            await writer.oauthapplication.update(
                where={"id": app.id}, data={"isActive": False}
            )
        elif change == "permissions":
            await writer.oauthapplication.update(
                where={"id": app.id}, data={"scopes": []}
            )
        else:
            await writer.oauthapplication.update(
                where={"id": app.id}, data={"teamIdRestriction": None}
            )
        with pytest.raises(HTTPException) as exc:
            await asyncio.wait_for(task, 1.2)
        assert exc.value.status_code == (401 if change == "revoke" else 403)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
