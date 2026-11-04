"""Every write of a personal organization's row lands in its default team.

Migration 20261103180000 installs ``a0_normalize_personal_team`` on each
tenant table: a row in a personal organization with no team gets the default
team, before the table's own guards run. Team-organization rows and rows with
no organization are left as written.
"""

from uuid import uuid4

import pytest
import pytest_asyncio
from prisma.errors import RawQueryError
from pydantic import BaseModel

from backend.api.features.orgs.db import create_org, get_user_default_team
from backend.data.db import prisma
from backend.data.user import get_or_create_user
from backend.util.test import SpinTestServer

pytestmark = pytest.mark.asyncio(loop_scope="session")


class Tenant(BaseModel):
    user_id: str
    personal_org: str
    default_team: str
    team_org: str
    workspace_id: str

    def org(self, scope: str) -> str | None:
        return {"personal": self.personal_org, "team_org": self.team_org}.get(scope)


@pytest_asyncio.fixture(loop_scope="session")
async def tenant(server: SpinTestServer) -> Tenant:
    user = await get_or_create_user(
        {"sub": str(uuid4()), "email": f"slot-{uuid4().hex[:8]}@example.com"}
    )
    personal_org, default_team = await get_user_default_team(user.id)
    assert personal_org and default_team
    team_org = await create_org("Slot", f"slot-{uuid4().hex[:8]}", user.id)
    workspace = await prisma.userworkspace.create(data={"userId": user.id})
    return Tenant(
        user_id=user.id,
        personal_org=personal_org,
        default_team=default_team,
        team_org=team_org.id,
        workspace_id=workspace.id,
    )


async def _returning(sql: str, *params: object, key: str = "id") -> str:
    rows = await prisma.query_raw(f'{sql} RETURNING "{key}" AS id', *params)
    return rows[0]["id"]


async def _graph(t: Tenant, org: str | None) -> str:
    return await _returning(
        'INSERT INTO "AgentGraph" (id, version, "userId", "organizationId") '
        "VALUES (gen_random_uuid()::text, 1, $1, $2)",
        t.user_id,
        org,
    )


async def _execution(t: Tenant, org: str | None) -> str:
    return await _returning(
        'INSERT INTO "AgentGraphExecution" '
        '(id, "agentGraphId", "agentGraphVersion", "userId", "organizationId") '
        "VALUES (gen_random_uuid()::text, $1, 1, $2, $3)",
        await _graph(t, org),
        t.user_id,
        org,
    )


async def _listing_version(t: Tenant, org: str | None) -> str:
    graph = await _graph(t, org)
    listing = await _returning(
        'INSERT INTO "StoreListing" (id, "agentGraphId", "owningUserId", slug, '
        '"owningOrgId") VALUES (gen_random_uuid()::text, $1, $2, $3, $4)',
        graph,
        t.user_id,
        f"slot-{uuid4().hex}",
        org,
    )
    return await _returning(
        'INSERT INTO "StoreListingVersion" (id, "agentGraphId", "agentGraphVersion", '
        'name, "subHeading", "imageUrls", description, categories, '
        '"storeListingId", "organizationId") VALUES (gen_random_uuid()::text, '
        "$1, 1, 'Slot', '', ARRAY[]::text[], '', ARRAY[]::text[], $2, $3)",
        graph,
        listing,
        org,
    )


def _simple(sql: str, key: str = "id"):
    async def insert(t: Tenant, org: str | None) -> str:
        return await _returning(sql, t.user_id, org, key=key)

    return insert


def _on_graph(sql: str):
    async def insert(t: Tenant, org: str | None) -> str:
        return await _returning(sql, t.user_id, org, await _graph(t, org))

    return insert


def _in_workspace(sql: str):
    async def insert(t: Tenant, org: str | None) -> str:
        return await _returning(sql, t.workspace_id, org)

    return insert


INSERTS = {
    "AgentGraph": _graph,
    "AgentGraphExecution": _execution,
    "AgentPreset": _on_graph(
        'INSERT INTO "AgentPreset" (id, name, description, "userId", '
        '"organizationId", "agentGraphId", "agentGraphVersion") '
        "VALUES (gen_random_uuid()::text, 'Slot', '', $1, $2, $3, 1)"
    ),
    "BuilderSearchHistory": _simple(
        'INSERT INTO "BuilderSearchHistory" (id, "userId", "organizationId", '
        "\"searchQuery\") VALUES (gen_random_uuid()::text, $1, $2, 'slot')"
    ),
    "ChatSession": _simple(
        'INSERT INTO "ChatSession" (id, "userId", "organizationId") '
        "VALUES (gen_random_uuid()::text, $1, $2)"
    ),
    "Expert": _simple(
        'INSERT INTO "Expert" (id, "ownerUserId", "organizationId", name, role, '
        "identity) VALUES (gen_random_uuid()::text, $1, $2, 'Slot', 'Slot', 'Slot')"
    ),
    "IntegrationWebhook": _simple(
        'INSERT INTO "IntegrationWebhook" (id, "userId", "organizationId", provider, '
        '"credentialsId", "webhookType", resource, events, config, secret, '
        '"providerWebhookId") VALUES (gen_random_uuid()::text, $1, $2, '
        "'generic_webhook', '', 'plain', '', ARRAY[]::text[], '{}', '', '')"
    ),
    # An entry of the user's own graph, so its owned-entry guard runs too.
    "LibraryAgent": _on_graph(
        'INSERT INTO "LibraryAgent" (id, "userId", "organizationId", '
        '"agentGraphId", "agentGraphVersion", "isCreatedByUser") '
        "VALUES (gen_random_uuid()::text, $1, $2, $3, 1, true)"
    ),
    "LibraryFolder": _simple(
        'INSERT INTO "LibraryFolder" (id, "userId", "organizationId", name) '
        "VALUES (gen_random_uuid()::text, $1, $2, 'Slot')"
    ),
    "PendingHumanReview": _simple(
        'INSERT INTO "PendingHumanReview" ("nodeExecId", "userId", '
        '"organizationId", payload, "graphExecId") VALUES '
        "(gen_random_uuid()::text, $1, $2, '{}', gen_random_uuid()::text)",
        key="nodeExecId",
    ),
    "StoreListingVersion": _listing_version,
    "UserNotificationBatch": _simple(
        'INSERT INTO "UserNotificationBatch" (id, "userId", "organizationId", type) '
        "VALUES (gen_random_uuid()::text, $1, $2, 'BRIEFING')"
    ),
    "UserWorkspaceFile": _in_workspace(
        'INSERT INTO "UserWorkspaceFile" (id, "updatedAt", "workspaceId", '
        '"organizationId", name, path, "storagePath", "mimeType", "sizeBytes") '
        "VALUES (gen_random_uuid()::text, NOW(), $1, $2, 'slot.txt', '/slot.txt', "
        "'slot/slot.txt', 'text/plain', 1)"
    ),
    "UserWorkspaceFolder": _in_workspace(
        'INSERT INTO "UserWorkspaceFolder" (id, "workspaceId", "organizationId", '
        "name) VALUES (gen_random_uuid()::text, $1, $2, 'Slot')"
    ),
}


async def _team_of(table: str, row_id: str) -> str | None:
    key = '"nodeExecId"' if table == "PendingHumanReview" else "id"
    rows = await prisma.query_raw(
        f'SELECT "teamId" FROM "{table}" WHERE {key} = $1', row_id
    )
    return rows[0]["teamId"]


@pytest.mark.parametrize("table", sorted(INSERTS))
@pytest.mark.parametrize("scope", ["personal", "team_org", "no_org"])
async def test_a_write_without_a_team_lands_in_the_personal_default_team(
    tenant: Tenant, table: str, scope: str
):
    row_id = await INSERTS[table](tenant, tenant.org(scope))

    expected = tenant.default_team if scope == "personal" else None
    assert await _team_of(table, row_id) == expected


@pytest.mark.parametrize("table", ["AgentGraph", "ChatSession", "LibraryFolder"])
async def test_moving_a_personal_row_to_org_home_keeps_it_in_the_default_team(
    tenant: Tenant, table: str
):
    row_id = await INSERTS[table](tenant, tenant.personal_org)

    await prisma.execute_raw(
        f'UPDATE "{table}" SET "teamId" = NULL WHERE id = $1', row_id
    )

    assert await _team_of(table, row_id) == tenant.default_team


async def test_a_workspace_file_in_the_default_team_is_keyed_to_it(tenant: Tenant):
    file_id = await INSERTS["UserWorkspaceFile"](tenant, tenant.personal_org)

    file = await prisma.userworkspacefile.find_unique_or_raise(where={"id": file_id})
    assert file.scopeKey == f"{tenant.personal_org}:{tenant.default_team}"


@pytest.mark.parametrize(
    "team, restriction, expected",
    [
        (None, None, "default"),
        ("default", None, "default"),
        (None, "default", "default"),
    ],
)
async def test_a_personal_api_key_is_restricted_to_the_default_team(
    tenant: Tenant, team: str | None, restriction: str | None, expected: str
):
    resolve = {None: None, "default": tenant.default_team}
    key = await _api_key(
        tenant, tenant.personal_org, resolve[team], resolve[restriction]
    )

    assert await _key_teams(key) == (resolve[expected], resolve[expected])


@pytest.mark.parametrize("scope", ["team_org", "no_org"])
async def test_other_api_keys_keep_their_scope(tenant: Tenant, scope: str):
    key = await _api_key(tenant, tenant.org(scope), None, None)

    assert await _key_teams(key) == (None, None)


async def test_an_alert_names_its_executions_exact_scope(tenant: Tenant):
    """Its scope guard runs before the normalizer (uppercase names sort first),
    so an alert stays pinned to its source execution, which is normalized."""
    execution = await _execution(tenant, tenant.personal_org)

    stored = await _alert(tenant, tenant.personal_org, tenant.default_team, execution)
    assert await _team_of("AlertCondition", stored) == tenant.default_team
    with pytest.raises(RawQueryError, match="alert source execution scope mismatch"):
        await _alert(tenant, tenant.personal_org, None, execution)


async def test_a_team_organization_alert_stays_at_org_home(tenant: Tenant):
    execution = await _execution(tenant, tenant.team_org)

    alert = await _alert(tenant, tenant.team_org, None, execution)

    assert await _team_of("AlertCondition", alert) is None


async def _api_key(
    t: Tenant, org: str | None, team: str | None, restriction: str | None
) -> str:
    return await _returning(
        'INSERT INTO "APIKey" (id, name, head, tail, hash, permissions, "userId", '
        '"organizationId", "teamId", "teamIdRestriction") VALUES '
        "(gen_random_uuid()::text, 'Slot', 'agpt_', 'tail', gen_random_uuid()::text, "
        'ARRAY[]::"APIKeyPermission"[], $1, $2, $3, $4)',
        t.user_id,
        org,
        team,
        restriction,
    )


async def _key_teams(key_id: str) -> tuple[str | None, str | None]:
    key = await prisma.apikey.find_unique_or_raise(where={"id": key_id})
    return key.teamId, key.teamIdRestriction


async def _alert(
    t: Tenant, org: str | None, team: str | None, execution: str | None
) -> str:
    return await _returning(
        'INSERT INTO "AlertCondition" (id, "userId", cause, "causeKey", data, '
        '"organizationId", "teamId", "sourceGraphExecutionId") VALUES '
        "(gen_random_uuid()::text, $1, 'BLOCK_FAILED', $2, '{}', $3, $4, $5)",
        t.user_id,
        f"slot-{uuid4().hex}",
        org,
        team,
        execution,
    )
