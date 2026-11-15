"""Leaving an organization or team ends what the member's v1 keys can do there.

Against Postgres, through the real External API app: real keys, memberships,
removals, workflows and library rows. Only the rate limiter's Redis counter,
the library's background work (images, embeddings, schedule badges) and the
scheduling tool behind `/tools/run-agent` are stubbed; the tool stands in for
the schedule a request would create.
"""

import uuid
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from prisma.enums import APIKeyPermission, APIKeyStatus
from prisma.models import (
    AgentGraph,
    APIKey,
    LibraryAgent,
    Organization,
    OrgMember,
    Team,
    TeamMember,
    User,
)
from pydantic import BaseModel

from backend.api.external import middleware
from backend.api.external.fastapi_app import external_api
from backend.api.external.v1 import tools as v1_tools
from backend.api.features.library import db as library_db
from backend.api.features.orgs import db as org_db
from backend.api.features.orgs import team_db
from backend.copilot.tools.models import ErrorResponse
from backend.data import db
from backend.data.auth import api_key

PERMISSIONS = [
    APIKeyPermission.WRITE_GRAPH,
    APIKeyPermission.WRITE_LIBRARY,
    APIKeyPermission.USE_TOOLS,
]
GRAPH = {"name": "Weekly report", "description": "", "nodes": [], "links": []}
SCHEDULE = {
    "username_agent_slug": "someone/weekly-report",
    "inputs": {},
    "schedule_name": "Weekly",
    "cron": "0 9 * * 1",
}


class Tenants(BaseModel):
    """An organization with a default team and a second team ("squad"), its
    owner, and a member who also has a personal organization."""

    owner: str
    member: str
    shared: str
    squad: str
    personal: str


class Key(BaseModel):
    id: str
    plaintext: str


@pytest_asyncio.fixture(loop_scope="session")
async def tenants() -> AsyncIterator[Tenants]:
    owned_connection = not db.is_connected()
    if owned_connection:
        await db.connect()
    owner, member = _user_id(), _user_id()
    for user_id in (owner, member):
        await User.prisma().create(
            data={"id": user_id, "email": f"{user_id}@example.com"}
        )
    shared, shared_home = await _org("Shared", owner, admin_of_home=True)
    squad = await Team.prisma().create(
        data={"name": "Squad", "orgId": shared, "isDefault": False}
    )
    await _membership(shared, member, teams=(shared_home, squad.id))
    await TeamMember.prisma().create(
        data={"teamId": squad.id, "userId": owner, "isAdmin": True, "status": "ACTIVE"}
    )
    personal, _ = await _org("Personal", member, personal=True)
    try:
        yield Tenants(
            owner=owner,
            member=member,
            shared=shared,
            squad=squad.id,
            personal=personal,
        )
    finally:
        users = {"userId": {"in": [owner, member]}}
        await LibraryAgent.prisma().delete_many(where=users)
        await AgentGraph.prisma().delete_many(where=users)
        await User.prisma().delete_many(where={"id": {"in": [owner, member]}})
        await Organization.prisma().delete_many(
            where={"id": {"in": [shared, personal]}}
        )
        if owned_connection:
            await db.disconnect()


@pytest.fixture(autouse=True)
def fresh_membership_answers():
    middleware.holds_membership.cache_clear()
    yield
    middleware.holds_membership.cache_clear()


@pytest.fixture(autouse=True)
def no_library_background_work(mocker):
    """Images, embeddings and schedule badges call out to other services."""
    for side_effect in ("add_generated_agent_image", "_fetch_schedule_info"):
        mocker.patch.object(library_db, side_effect, AsyncMock(return_value={}))
    mocker.patch.object(library_db, "schedule_library_agent_embedding", MagicMock())


@pytest.fixture
def scheduler(mocker) -> AsyncMock:
    """The tool behind `/tools/run-agent`, which would create the schedule."""
    return mocker.patch.object(
        v1_tools.run_agent_tool,
        "_execute",
        AsyncMock(return_value=ErrorResponse(message="scheduled")),
    )


@pytest_asyncio.fixture(loop_scope="session")
async def client(v1_counter, scheduler) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=external_api), base_url="http://test"
    ) as client:
        yield client


async def test_a_removed_member_can_no_longer_create_in_the_former_organization(
    tenants: Tenants, client: AsyncClient, scheduler: AsyncMock
):
    shared_key = await _key(tenants.member, tenants.shared)
    suspended_key = await _key(tenants.member, tenants.shared)
    await api_key.suspend_api_key(suspended_key.id, tenants.member)
    revoked_key = await _key(tenants.member, tenants.shared)
    revoked_at = (
        await api_key.revoke_api_key(revoked_key.id, tenants.member)
    ).revoked_at
    personal_key = await _key(tenants.member, tenants.personal)
    pre_org_key = await _key(tenants.member, None)
    owner_key = await _key(tenants.owner, tenants.shared)

    created = await _create(client, shared_key)
    scheduled = await _schedule(client, shared_key)

    assert created.status_code == 201, created.text
    assert scheduled.status_code == 200, scheduled.text
    assert scheduler.await_args.kwargs["session"].organization_id == tenants.shared
    assert await _graphs(tenants.member, tenants.shared) == 1
    assert await _library_rows(tenants.member, tenants.shared) == 1

    await org_db.remove_org_member(
        tenants.shared, tenants.member, requesting_user_id=tenants.owner
    )

    for key in (shared_key, suspended_key):
        assert await _status(key) == APIKeyStatus.REVOKED
    assert (await _row(revoked_key)).revokedAt == revoked_at
    for refused in (
        await _create(client, shared_key),
        await _schedule(client, shared_key),
    ):
        assert refused.status_code == 401
    assert scheduler.await_count == 1
    assert await _graphs(tenants.member, tenants.shared) == 1
    assert await _library_rows(tenants.member, tenants.shared) == 1

    # Keys for anywhere else, and the colleagues' keys, are untouched.
    for key in (personal_key, pre_org_key, owner_key):
        assert await _status(key) == APIKeyStatus.ACTIVE
        assert (await _create(client, key)).status_code == 201
    assert await _graphs(tenants.member, tenants.personal) == 1
    assert await _graphs(tenants.owner, tenants.shared) == 1


@pytest.mark.parametrize("ended", ["organization", "team"])
async def test_a_key_that_outlived_its_membership_is_refused(
    tenants: Tenants, client: AsyncClient, scheduler: AsyncMock, ended: str
):
    """A key whose membership ended without the key being revoked, such as one
    from before removals revoked keys, is refused when it is used."""
    team = tenants.squad if ended == "team" else None
    key = await _key(tenants.member, tenants.shared, team)
    if ended == "team":
        await TeamMember.prisma().delete(
            where={"teamId_userId": {"teamId": tenants.squad, "userId": tenants.member}}
        )
    else:
        await OrgMember.prisma().delete(
            where={"orgId_userId": {"orgId": tenants.shared, "userId": tenants.member}}
        )

    for refused in (await _create(client, key), await _schedule(client, key)):
        assert refused.status_code == 403
        assert refused.json()["detail"] == middleware.FORMER_MEMBER_DETAIL
    scheduler.assert_not_awaited()
    assert await _graphs(tenants.member, tenants.shared) == 0
    assert await _status(key) == APIKeyStatus.ACTIVE


async def test_a_team_key_ends_with_the_team_membership(
    tenants: Tenants, client: AsyncClient
):
    team_key = await _key(tenants.member, tenants.shared, tenants.squad)
    org_key = await _key(tenants.member, tenants.shared)

    created = await _create(client, team_key)

    assert created.status_code == 201, created.text
    graph = await AgentGraph.prisma().find_first(where={"id": created.json()["id"]})
    assert graph is not None
    assert (graph.organizationId, graph.teamId) == (tenants.shared, tenants.squad)

    await team_db.remove_team_member(tenants.squad, tenants.member)

    assert await _status(team_key) == APIKeyStatus.REVOKED
    assert (await _create(client, team_key)).status_code == 401
    # Still a member of the organization, so its key is kept.
    assert await _status(org_key) == APIKeyStatus.ACTIVE
    assert (await _create(client, org_key)).status_code == 201


async def test_leaving_or_deleting_a_team_revokes_the_keys_restricted_to_it(
    tenants: Tenants,
):
    members_team_key = await _key(tenants.member, tenants.shared, tenants.squad)
    owners_team_key = await _key(tenants.owner, tenants.shared, tenants.squad)
    owners_org_key = await _key(tenants.owner, tenants.shared)

    await team_db.leave_team(tenants.squad, tenants.member)

    assert await _status(members_team_key) == APIKeyStatus.REVOKED
    assert await _status(owners_team_key) == APIKeyStatus.ACTIVE

    await team_db.delete_team(tenants.squad)

    assert await _status(owners_team_key) == APIKeyStatus.REVOKED
    assert await _status(owners_org_key) == APIKeyStatus.ACTIVE


def _user_id() -> str:
    return f"key-membership-{uuid.uuid4()}"


async def _org(
    name: str, owner: str, *, personal: bool = False, admin_of_home: bool = False
) -> tuple[str, str]:
    """Create an organization owned by ``owner``; returns (org id, home team id)."""
    suffix = uuid.uuid4().hex[:8]
    org = await Organization.prisma().create(
        data={"name": name, "slug": f"key-membership-{suffix}", "isPersonal": personal}
    )
    home = await Team.prisma().create(
        data={"name": "General", "orgId": org.id, "isDefault": True}
    )
    await OrgMember.prisma().create(
        data={
            "orgId": org.id,
            "userId": owner,
            "isOwner": True,
            "isAdmin": True,
            "status": "ACTIVE",
        }
    )
    await TeamMember.prisma().create(
        data={
            "teamId": home.id,
            "userId": owner,
            "isAdmin": admin_of_home or personal,
            "status": "ACTIVE",
        }
    )
    return org.id, home.id


async def _membership(org_id: str, user_id: str, *, teams: tuple[str, ...]) -> None:
    await OrgMember.prisma().create(
        data={"orgId": org_id, "userId": user_id, "status": "ACTIVE"}
    )
    for team_id in teams:
        await TeamMember.prisma().create(
            data={"teamId": team_id, "userId": user_id, "status": "ACTIVE"}
        )


async def _key(
    user_id: str, organization_id: str | None, team_id: str | None = None
) -> Key:
    info, plaintext = await api_key.create_api_key(
        name="membership",
        user_id=user_id,
        permissions=PERMISSIONS,
        organization_id=organization_id,
        team_id_restriction=team_id,
    )
    return Key(id=info.id, plaintext=plaintext)


async def _create(client: AsyncClient, key: Key) -> httpx.Response:
    return await client.post(
        "/v1/graphs", headers={"X-API-Key": key.plaintext}, json=GRAPH
    )


async def _schedule(client: AsyncClient, key: Key) -> httpx.Response:
    return await client.post(
        "/v1/tools/run-agent", headers={"X-API-Key": key.plaintext}, json=SCHEDULE
    )


async def _row(key: Key) -> APIKey:
    row = await APIKey.prisma().find_unique(where={"id": key.id})
    assert row is not None
    return row


async def _status(key: Key) -> APIKeyStatus:
    return (await _row(key)).status


async def _graphs(user_id: str, organization_id: str) -> int:
    return await AgentGraph.prisma().count(
        where={"userId": user_id, "organizationId": organization_id}
    )


async def _library_rows(user_id: str, organization_id: str) -> int:
    return await LibraryAgent.prisma().count(
        where={"userId": user_id, "organizationId": organization_id}
    )
