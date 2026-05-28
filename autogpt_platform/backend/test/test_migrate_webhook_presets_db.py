"""
DB-backed integration tests for migrate_webhook_presets_to_new_version.

Unlike test_migrate_webhook_presets.py (which mocks prisma and only asserts the
shape of the WHERE clause), these tests run the real `update_many` against the
test Postgres so a query that is syntactically valid but semantically wrong
(e.g. a missing userId scope, or migrating newer-pinned/deleted/non-webhook
rows) is actually caught.

Requires a running Postgres (the infra `deps` stack); connects via the
`db_conn` fixture.
"""

import uuid

import prisma.models
import pytest
import pytest_asyncio

from backend.api.features.library.db import migrate_webhook_presets_to_new_version
from backend.data import db as backend_db
from backend.data.user import get_or_create_user


@pytest_asyncio.fixture(loop_scope="session")
async def db_conn():
    # When the full suite runs, the session-scoped `server` fixture already
    # holds the shared Prisma connection; only tear it down if we opened it,
    # otherwise the session-teardown graph_cleanup hits ClientNotConnectedError.
    opened_here = not backend_db.is_connected()
    if opened_here:
        await backend_db.connect()
    try:
        yield
    finally:
        if opened_here:
            await backend_db.disconnect()


async def _mk_user() -> str:
    user_id = str(uuid.uuid4())
    await get_or_create_user(
        {
            "sub": user_id,
            "email": f"{user_id}@example.com",
            "user_metadata": {"name": "Preset Test User"},
        }
    )
    return user_id


async def _mk_graph(user_id: str, graph_id: str, version: int) -> None:
    await prisma.models.AgentGraph.prisma().create(
        data={
            "id": graph_id,
            "version": version,
            "name": f"g-{version}",
            "description": "",
            "isActive": version == 2,
            "userId": user_id,
        }
    )


async def _mk_webhook(user_id: str) -> str:
    wh = await prisma.models.IntegrationWebhook.prisma().create(
        data={
            "userId": user_id,
            "provider": "telegram",
            "credentialsId": str(uuid.uuid4()),
            "webhookType": "message",
            "resource": "",
            "events": ["message"],
            "config": prisma.Json({}),
            "secret": "s",
            "providerWebhookId": str(uuid.uuid4()),
        }
    )
    return wh.id


async def _mk_preset(
    *,
    user_id: str,
    graph_id: str,
    version: int,
    webhook_id: str | None,
    is_deleted: bool = False,
    name: str = "preset",
) -> str:
    p = await prisma.models.AgentPreset.prisma().create(
        data={
            "name": name,
            "description": "desc",
            "userId": user_id,
            "agentGraphId": graph_id,
            "agentGraphVersion": version,
            "webhookId": webhook_id,
            "isDeleted": is_deleted,
        }
    )
    return p.id


async def _version_of(preset_id: str) -> int:
    p = await prisma.models.AgentPreset.prisma().find_unique(where={"id": preset_id})
    assert p is not None
    return p.agentGraphVersion


@pytest.mark.asyncio(loop_scope="session")
async def test_migrate_db_scopes_and_preserves_payload(db_conn):
    owner = await _mk_user()
    other = await _mk_user()
    graph_id = str(uuid.uuid4())
    await _mk_graph(owner, graph_id, 1)
    await _mk_graph(owner, graph_id, 2)
    await _mk_graph(owner, graph_id, 3)
    owner_wh = await _mk_webhook(owner)
    other_wh = await _mk_webhook(other)
    # 'other' needs the same graph_id rows to attach a preset to (FK).
    # Graph is owner's; AgentPreset.userId is independent of graph ownership,
    # so 'other' can have a preset row on the same graph_id+version.

    # owner: webhook preset pinned to old v1 -> SHOULD migrate to 2
    p_migrate = await _mk_preset(
        user_id=owner, graph_id=graph_id, version=1, webhook_id=owner_wh, name="keep-me"
    )
    # owner: non-webhook preset on v1 -> SHOULD NOT migrate
    p_no_webhook = await _mk_preset(
        user_id=owner, graph_id=graph_id, version=1, webhook_id=None
    )
    # owner: webhook preset pinned to NEWER v3 -> SHOULD NOT migrate (lt filter)
    p_newer = await _mk_preset(
        user_id=owner, graph_id=graph_id, version=3, webhook_id=owner_wh
    )
    # owner: deleted webhook preset on v1 -> SHOULD NOT migrate
    p_deleted = await _mk_preset(
        user_id=owner,
        graph_id=graph_id,
        version=1,
        webhook_id=owner_wh,
        is_deleted=True,
    )
    # OTHER user: webhook preset on the SAME graph_id+v1 -> MUST NOT migrate
    p_other = await _mk_preset(
        user_id=other, graph_id=graph_id, version=1, webhook_id=other_wh
    )

    count = await migrate_webhook_presets_to_new_version(
        user_id=owner, graph_id=graph_id, new_version=2
    )

    # Exactly one row (the owner's old webhook preset) migrated.
    assert count == 1, f"expected 1 migrated, got {count}"

    assert await _version_of(p_migrate) == 2, "owner webhook preset should be repinned"
    assert await _version_of(p_no_webhook) == 1, "non-webhook preset must be untouched"
    assert await _version_of(p_newer) == 3, "newer-pinned preset must be untouched"
    assert await _version_of(p_deleted) == 1, "deleted preset must be untouched"
    # NEGATIVE / SECURITY: other user's preset on the same graph must be untouched.
    assert await _version_of(p_other) == 1, "OTHER user's preset must NOT migrate"

    # Payload integrity: only the version pin changed on the migrated row.
    migrated = await prisma.models.AgentPreset.prisma().find_unique(
        where={"id": p_migrate}
    )
    assert migrated is not None
    assert migrated.name == "keep-me"
    assert migrated.description == "desc"
    assert migrated.webhookId == owner_wh
    assert migrated.isDeleted is False


@pytest.mark.asyncio(loop_scope="session")
async def test_migrate_db_returns_zero_when_no_webhook_presets(db_conn):
    owner = await _mk_user()
    graph_id = str(uuid.uuid4())
    await _mk_graph(owner, graph_id, 1)
    await _mk_graph(owner, graph_id, 2)
    await _mk_preset(user_id=owner, graph_id=graph_id, version=1, webhook_id=None)

    count = await migrate_webhook_presets_to_new_version(
        user_id=owner, graph_id=graph_id, new_version=2
    )
    assert count == 0
