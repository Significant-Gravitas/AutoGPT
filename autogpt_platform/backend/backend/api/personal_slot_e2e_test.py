"""A personal organization's work, from two slots to one, end to end.

Prod holds a personal organization's rows in two places: the web app saved
some at its org-home, and the startup backfill and internal writers put the
rest in its default team. This seeds rows in that shape with the database's
triggers off, runs migration 20261103180000's SQL, and drives the real app with
organizations off: the API keys page, a run started with a key, new versions
saved on the web and through v2, and a team organization's key on v1 and v2.

Seeding switches triggers off with ``session_replication_role``, which needs a
superuser, as the local and CI test databases (the image's ``postgres``) are.
"""

import asyncio
import os
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlsplit, urlunsplit

import httpx
import psycopg2
import pytest
import pytest_asyncio
from autogpt_libs.auth import get_optional_user_id, get_user_id
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from prisma.enums import APIKeyPermission
from pydantic import BaseModel

from backend.api.features.orgs.db import create_org, get_user_default_team
from backend.api.model import CreateGraph
from backend.api.rest_api import app
from backend.data import graph as graph_db
from backend.data.auth import api_key as api_key_db
from backend.data.db import prisma
from backend.data.user import get_or_create_user
from backend.executor import utils as executor_utils
from backend.usecases.sample import create_test_graph
from backend.util.test import SpinTestServer

pytestmark = pytest.mark.asyncio(loop_scope="session")

MIGRATION = (
    Path(__file__).parents[2]
    / "migrations"
    / "20261103180000_one_personal_team_slot"
    / "migration.sql"
)
TENANT_TABLES = (
    "AgentGraph",
    "AgentGraphExecution",
    "AgentPreset",
    "AlertCondition",
    "APIKey",
    "BuilderSearchHistory",
    "ChatSession",
    "Expert",
    "IntegrationWebhook",
    "LibraryAgent",
    "LibraryFolder",
    "PendingHumanReview",
    "StoreListingVersion",
    "UserNotificationBatch",
    "UserWorkspaceFile",
    "UserWorkspaceFolder",
)
KEY_PERMISSIONS = [
    APIKeyPermission.IDENTITY,
    APIKeyPermission.READ_GRAPH,
    APIKeyPermission.WRITE_GRAPH,
    APIKeyPermission.EXECUTE_GRAPH,
]


class Owner(BaseModel):
    user_id: str
    email: str
    personal_org: str
    default_team: str
    team_org: str

    def web_headers(self) -> dict[str, str]:
        """What the web app sends with organizations off."""
        return {"X-Org-Id": self.personal_org, "X-Team-Id": self.default_team}


class Seeded(BaseModel):
    graph_id: str
    key_a: str
    key_a_id: str
    key_b_id: str
    team_org_key: str
    team_org_key_id: str


@pytest.fixture
def organizations_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", "false")


@pytest_asyncio.fixture(loop_scope="session")
async def owner(server: SpinTestServer) -> Owner:
    email = f"slot-e2e-{uuid.uuid4().hex[:8]}@example.com"
    user = await get_or_create_user({"sub": str(uuid.uuid4()), "email": email})
    personal_org, default_team = await get_user_default_team(user.id)
    assert personal_org and default_team
    team_org = await create_org("Acme", f"acme-{uuid.uuid4().hex[:8]}", user.id)
    return Owner(
        user_id=user.id,
        email=email,
        personal_org=personal_org,
        default_team=default_team,
        team_org=team_org.id,
    )


@pytest_asyncio.fixture(loop_scope="session")
async def client(owner: Owner):
    previous = dict(app.dependency_overrides)
    app.dependency_overrides[get_jwt_payload] = lambda: {
        "sub": owner.user_id,
        "role": "user",
        "email": owner.email,
    }
    app.dependency_overrides[get_user_id] = lambda: owner.user_id
    app.dependency_overrides[get_optional_user_id] = lambda: owner.user_id
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
    app.dependency_overrides.update(previous)


async def test_personal_work_in_both_slots_is_one_slot_after_the_migration(
    organizations_off, owner: Owner, client: httpx.AsyncClient, monkeypatch
):
    queued: list[str] = []

    class _Queue:
        async def publish_message(self, *, message: str, **_: object) -> None:
            queued.append(message)

    async def _execution_queue() -> _Queue:
        return _Queue()

    monkeypatch.setattr(executor_utils, "get_async_execution_queue", _execution_queue)
    seeded = await _seed_prod_shape(owner, client)
    assert not {seeded.key_a_id, seeded.key_b_id} & await _listed_keys(owner, client)

    await asyncio.to_thread(_run_sql, MIGRATION.read_text(encoding="utf-8"))

    # 1. The API keys page lists both keys again, and not the team org's.
    listed = await _listed_keys(owner, client)
    assert {seeded.key_a_id, seeded.key_b_id} <= listed
    assert seeded.team_org_key_id not in listed

    # 2. A run started with key A is stored in the default team.
    started = await client.post(
        f"/external-api/v1/graphs/{seeded.graph_id}/execute/1",
        headers={"X-API-Key": seeded.key_a},
        json={"node_input": {"input_1": "a", "input_2": "b"}},
    )
    assert started.status_code == 200, started.text
    run = await prisma.agentgraphexecution.find_unique_or_raise(
        where={"id": started.json()["id"]}
    )
    assert (run.organizationId, run.teamId) == (owner.personal_org, owner.default_team)
    assert len(queued) == 1

    # 3. New versions save on the web and through v2.
    saved = await client.put(
        f"/api/graphs/{seeded.graph_id}",
        headers=owner.web_headers(),
        json=_version_of(seeded.graph_id, 3).model_dump(mode="json"),
    )
    assert saved.status_code == 200, saved.text
    saved_v2 = await client.put(
        f"/api/v2/graphs/{seeded.graph_id}",
        headers={"X-API-Key": seeded.key_a},
        json={"name": "Slot", "nodes": [], "links": []},
    )
    assert saved_v2.status_code == 200, saved_v2.text

    # 4. A team organization's key acts nowhere for now.
    for path in ("/external-api/v1/me", "/api/v2/me"):
        refused = await client.get(path, headers={"X-API-Key": seeded.team_org_key})
        assert refused.status_code == 403, (path, refused.text)

    # 5. Nothing of the personal organization is left at its org-home.
    assert await _org_home_rows(owner.personal_org) == {}


async def _seed_prod_shape(owner: Owner, client: httpx.AsyncClient) -> Seeded:
    """Graph v1, its library entry and key A in the default team; v2, a chat
    and key B at org-home; key A missing its restriction, as before keys
    carried one."""
    created = await client.post(
        "/api/graphs",
        headers=owner.web_headers(),
        json=CreateGraph(graph=create_test_graph()).model_dump(mode="json"),
    )
    assert created.status_code == 200, created.text
    graph_id = created.json()["id"]
    v2 = graph_db.make_graph_model(_version_of(graph_id, 2), owner.user_id)
    v2.reassign_ids(user_id=owner.user_id, reassign_graph_id=False)
    await graph_db.create_graph(
        v2,
        owner.user_id,
        organization_id=owner.personal_org,
        team_id=owner.default_team,
    )
    key_a, plain_a = await _key(owner, owner.personal_org, owner.default_team)
    key_b, _ = await _key(owner, owner.personal_org, owner.default_team)
    team_key, plain_team = await _key(owner, owner.team_org, None)
    chat = await prisma.chatsession.create(
        data={"userId": owner.user_id, "organizationId": owner.personal_org}
    )
    await asyncio.to_thread(
        _run_sql,
        "BEGIN; SET LOCAL session_replication_role = replica;"
        'UPDATE "AgentGraph" SET "teamId" = NULL WHERE id = %(graph)s AND version = 2;'
        'UPDATE "APIKey" SET "teamIdRestriction" = NULL WHERE id = %(key_a)s;'
        'UPDATE "APIKey" SET "teamId" = NULL, "teamIdRestriction" = NULL '
        "WHERE id = %(key_b)s;"
        'UPDATE "ChatSession" SET "teamId" = NULL WHERE id = %(chat)s;'
        "COMMIT;",
        {"graph": graph_id, "key_a": key_a, "key_b": key_b, "chat": chat.id},
    )
    return Seeded(
        graph_id=graph_id,
        key_a=plain_a,
        key_a_id=key_a,
        key_b_id=key_b,
        team_org_key=plain_team,
        team_org_key_id=team_key,
    )


def _version_of(graph_id: str, version: int) -> graph_db.Graph:
    return create_test_graph().model_copy(update={"id": graph_id, "version": version})


async def _key(owner: Owner, org: str, team: str | None) -> tuple[str, str]:
    info, plain = await api_key_db.create_api_key(
        name="Slot",
        user_id=owner.user_id,
        permissions=KEY_PERMISSIONS,
        organization_id=org,
        owner_type="USER",
        team_id_restriction=team,
    )
    return info.id, plain


async def _listed_keys(owner: Owner, client: httpx.AsyncClient) -> set[str]:
    response = await client.get("/api/api-keys", headers=owner.web_headers())
    assert response.status_code == 200, response.text
    return {key["id"] for key in response.json()}


async def _org_home_rows(org_id: str) -> dict[str, int]:
    counts = {}
    for table in TENANT_TABLES:
        rows = await prisma.query_raw(
            f'SELECT count(*)::int AS n FROM "{table}" '
            'WHERE "organizationId" = $1 AND "teamId" IS NULL',
            org_id,
        )
        if rows[0]["n"]:
            counts[table] = rows[0]["n"]
    return counts


def _run_sql(sql: str, params: dict[str, str] | None = None) -> None:
    """One script on its own connection, as ``prisma migrate deploy`` runs a
    migration: its own BEGIN and COMMIT, NOTICEs ignored. A lock still held
    elsewhere after 30s fails the script rather than hanging the test."""
    url = urlsplit(os.environ["DATABASE_URL"])
    schema = parse_qs(url.query).get("schema", ["public"])[0]
    connection = psycopg2.connect(
        urlunsplit(url._replace(query="")),
        options=f"-c search_path={schema} -c lock_timeout=30s",
    )
    try:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
    finally:
        connection.close()
