import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio

from backend.api.features.memory import routes
from backend.copilot.dream.batch_submit import persist_input_bundle, read_input_bundle
from backend.copilot.dream.fetch import DreamInput
from backend.copilot.model import ChatSession
from backend.copilot.tools import graphiti_forget, graphiti_store
from backend.data import db, memory_scope

from . import ingest
from .falkordb_driver import open_driver
from .scope import MemoryScope
from .scope_resolution import resolve_memory_scope

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="session")]


@pytest_asyncio.fixture(loop_scope="session")
async def private_orgs(falkordb_available, monkeypatch):
    # The client is shared with the rest of the session: leave it as found.
    connected_here = not db.is_connected()
    await db.connect()
    user_id = str(uuid4())
    organizations = [str(uuid4()) for _ in range(2)]
    await db.prisma.user.create(
        data={"id": user_id, "email": f"{user_id}@example.invalid"}
    )
    for org_id in organizations:
        await db.prisma.organization.create(
            data={"id": org_id, "name": org_id, "slug": org_id}
        )
    await db.query_raw_with_schema(
        'INSERT INTO {schema_prefix}"PersonalMemoryOrganization" ("userId", "organizationId") VALUES ($1, $2)',
        user_id,
        organizations[0],
    )
    scopes = [
        await resolve_memory_scope(user_id, organization_id=org)
        for org in organizations
    ]
    monkeypatch.setattr(
        graphiti_store, "resolve_user_name", AsyncMock(return_value="Owner")
    )
    monkeypatch.setattr(ingest, "ensure_indices_once", AsyncMock())
    monkeypatch.setattr(ingest, "ensure_scope_scheduled", AsyncMock())

    class Lease:
        async def run(self, action):
            return await action

    @asynccontextmanager
    async def lease(*args):
        yield Lease()

    monkeypatch.setattr(ingest, "live_resource_lease", lease)
    try:
        yield user_id, organizations, scopes
    finally:
        state = ingest._get_loop_state()
        tasks = list(state.group_workers.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for scope in scopes:
            driver = open_driver(scope)
            await driver.execute_query("MATCH (n) DETACH DELETE n")
            await driver.close()
        await db.prisma.user.delete(where={"id": user_id})
        await db.prisma.organization.delete_many(where={"id": {"in": organizations}})
        if connected_here:
            await db.disconnect()


async def test_default_store_delayed_worker_recall_and_hard_forget(
    private_orgs, monkeypatch
):
    user_id, orgs, scopes = private_orgs
    release = asyncio.Event()
    clients = {}

    async def client_for(group_id):
        if group_id in clients:
            return clients[group_id]
        scope = next(scope for scope in scopes if scope.group_id == group_id)
        driver = open_driver(scope)

        async def add_episode(**payload):
            await release.wait()
            await driver.execute_query(
                "CREATE (a:Entity {name: 'Owner', group_id: $g})-[:RELATES_TO "
                "{uuid: 'same-fact-id', group_id: $g, fact: $body}]->"
                "(b:Entity {name: 'Business', group_id: $g})",
                g=group_id,
                body=payload["episode_body"],
            )
            return SimpleNamespace(episode=SimpleNamespace(uuid="episode"), edges=[])

        client = SimpleNamespace(driver=driver, add_episode=add_episode)
        clients[group_id] = client
        return client

    monkeypatch.setattr(ingest, "get_graphiti_client", client_for)
    session = ChatSession.new(user_id, dry_run=False, organization_id=orgs[0])
    store = graphiti_store.MemoryStoreTool()
    await store._execute(user_id, session, name="AutoGPT", content="AUTOGPT_PRIVATE")
    session.organization_id = orgs[1]
    await store._execute(user_id, session, name="Denton", content="DENTON_PRIVATE")
    release.set()
    await asyncio.wait_for(
        asyncio.gather(
            *(q.join() for q in ingest._get_loop_state().group_queues.values())
        ),
        timeout=15,
    )
    for org, own, foreign in (
        (orgs[0], "AUTOGPT_PRIVATE", "DENTON_PRIVATE"),
        (orgs[1], "DENTON_PRIVATE", "AUTOGPT_PRIVATE"),
    ):
        facts = await routes._list_facts_impl(user_id, None, 10, org)
        assert len(facts.items) == 1
        assert own in facts.items[0].fact
        assert foreign not in facts.items[0].fact
    registered = await memory_scope.list_memory_scopes(user_id)
    assert set(registered) == set(scopes)
    assert await memory_scope.list_memory_scopes(str(uuid4())) == []
    await graphiti_forget.MemoryForgetConfirmTool()._execute(
        user_id, session, uuids=["same-fact-id"], hard_delete=True
    )
    assert (await routes._list_facts_impl(user_id, None, 10, orgs[1])).items == []
    assert len((await routes._list_facts_impl(user_id, None, 10, orgs[0])).items) == 1
    for client in clients.values():
        await client.driver.close()


async def test_persisted_dream_scope_and_original_org_survive_restart(
    private_orgs, monkeypatch
):
    from datetime import datetime, timezone

    from backend.copilot.dream.batch_submit import delete_input_bundle

    user_id, orgs, scopes = private_orgs
    now = datetime.now(timezone.utc)
    pass_id = str(uuid4())
    bundle = DreamInput(
        user_id=user_id,
        organization_id=orgs[1],
        group_id=scopes[1].group_id,
        window_start=now,
        window_end=now,
    )
    await persist_input_bundle(pass_id, bundle, lock_token="test-owned-lock")
    try:
        restored = await read_input_bundle(pass_id)
        assert restored is not None
        assert restored.memory_scope == scopes[1]
        await db.prisma.organization.update(
            where={"id": orgs[0]},
            data={"name": "Original business", "isPersonal": False},
        )
        monkeypatch.setattr(
            memory_scope,
            "get_user_default_team",
            AsyncMock(side_effect=AssertionError("must not rebind personal memory")),
        )
        await db.disconnect()
        await db.connect()
        assert await memory_scope.get_personal_memory_organization(user_id) == orgs[0]
        assert await resolve_memory_scope(
            user_id, organization_id=orgs[0]
        ) == MemoryScope.for_user(user_id)
        assert (await read_input_bundle(pass_id)).memory_scope == scopes[1]
    finally:
        await delete_input_bundle(pass_id)


async def test_migration_retains_legacy_personal_and_expert_namespaces(private_orgs):
    from datetime import timedelta
    from pathlib import Path

    schema = "memory_migration_" + uuid4().hex
    migration = Path(
        "migrations/20261016180000_org_private_memory/migration.sql"
    ).read_text()
    async with db.prisma.tx(timeout=timedelta(seconds=30)) as tx:
        await tx.execute_raw(f'CREATE SCHEMA "{schema}"')
        await tx.execute_raw(f'SET LOCAL search_path TO "{schema}"')
        for statement in (
            'CREATE TABLE "User" (id TEXT PRIMARY KEY)',
            'CREATE TABLE "Organization" (id TEXT PRIMARY KEY, "isPersonal" BOOLEAN, "deletedAt" TIMESTAMP)',
            'CREATE TABLE "OrgMember" (id TEXT, "userId" TEXT, "orgId" TEXT, "isOwner" BOOLEAN, "createdAt" TIMESTAMP)',
            'CREATE TABLE "Expert" (id TEXT, "ownerUserId" TEXT, "isTemplate" BOOLEAN)',
            "INSERT INTO \"User\" VALUES ('owner')",
        ):
            await tx.execute_raw(statement)
        await tx.execute_raw(
            "INSERT INTO \"Organization\" VALUES ('personal', true, NULL), ('business', false, NULL)"
        )
        await tx.execute_raw(
            "INSERT INTO \"OrgMember\" VALUES ('1', 'owner', 'personal', true, now()), ('2', 'owner', 'business', true, now())"
        )
        await tx.execute_raw(
            "INSERT INTO \"Expert\" VALUES ('expert', 'owner', false), ('template', 'owner', true)"
        )
        for statement in migration.split(";"):
            if statement.strip():
                await tx.execute_raw(statement)
        anchors = await tx.query_raw('SELECT * FROM "PersonalMemoryOrganization"')
        rows = await tx.query_raw('SELECT * FROM "MemoryScopeRegistry"')
        assert anchors == [{"userId": "owner", "organizationId": "personal"}]
        assert {row["groupId"] for row in rows} == {
            MemoryScope.for_user("owner").group_id,
            MemoryScope.for_expert("owner", "expert").group_id,
        }
        assert all(row["organizationId"] is None for row in rows)
        await tx.execute_raw(f'DROP SCHEMA "{schema}" CASCADE')
