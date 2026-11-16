"""Org and team roles on the External API, end to end: the real v1 and v2 apps,
real API keys, and membership rows in Postgres.

A membership whose only role is billing manager covers spend, not workflows,
runs and the library. An API key acting in such an organization or team is
refused those, as the web app refuses its owner, and still reads its owner's
identity and the organization's credits. Ordinary members' keys are unchanged.

These tests need only Postgres, so they connect Prisma themselves.
"""

import uuid
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from prisma.enums import APIKeyPermission
from pydantic import BaseModel

from backend.api.external import rate_limit
from backend.api.external.fastapi_app import external_api
from backend.api.external.v2.app import v2_app
from backend.api.external.v2.mcp_server import (
    ExternalAPITokenVerifier,
    TenantedAccessToken,
)
from backend.data import execution as execution_db
from backend.data import graph as graph_db
from backend.data.auth import api_key
from backend.data.db import prisma
from backend.data.org_migration import create_personal_org
from backend.data.tenancy import BILLING_ONLY_ORG_ROLE, BILLING_ONLY_TEAM_ROLE

GRAPH = {"name": "From a key", "description": "", "nodes": [], "links": []}
SECRET = "notes only the owner should read"


class World(BaseModel):
    """One shared org with one team. ``org_billing`` manages the org's billing
    and is a plain member of the team; ``team_billing`` is a plain org member
    who manages only the team's billing."""

    org: str
    team: str
    owner: str
    member: str
    org_billing: str
    team_billing: str
    owner_graph: str
    owner_team_run: str
    owner_home_run: str
    keys: dict[str, str]
    team_keys: dict[str, str]


@pytest_asyncio.fixture(loop_scope="session")
async def world() -> AsyncIterator[World]:
    connected_here = not prisma.is_connected()
    if connected_here:
        await prisma.connect()
    users: list[str] = []
    orgs: list[str] = []
    try:
        yield await _build_world(users, orgs)
    finally:
        try:
            if prisma.is_connected():
                await prisma.libraryagent.delete_many(where={"userId": {"in": users}})
                await prisma.agentgraph.delete_many(where={"userId": {"in": users}})
                await prisma.user.delete_many(where={"id": {"in": users}})
                await prisma.organization.delete_many(where={"id": {"in": orgs}})
        finally:
            if connected_here and prisma.is_connected():
                await prisma.disconnect()


async def _build_world(users: list[str], orgs: list[str]) -> World:
    suffix = uuid.uuid4().hex[:8]
    owner, member, org_billing, team_billing = [
        await _user(f"{role}-{suffix}", users)
        for role in ("owner", "member", "orgbill", "teambill")
    ]
    for user_id in users:
        orgs.append((await create_personal_org(user_id, user_id, user_id)).id)
    org = await prisma.organization.create(
        data={"name": f"Key roles {suffix}", "slug": f"key-roles-{suffix}"}
    )
    orgs.append(org.id)
    team = await prisma.team.create(data={"name": "Squad", "orgId": org.id})
    for user_id, org_flags, team_flags in (
        (owner, {"isOwner": True, "isAdmin": True}, {"isAdmin": True}),
        (member, {}, {}),
        (org_billing, {"isBillingManager": True}, {}),
        (team_billing, {}, {"isBillingManager": True}),
    ):
        await prisma.orgmember.create(
            data={"orgId": org.id, "userId": user_id, "status": "ACTIVE", **org_flags}
        )
        await prisma.teammember.create(
            data={
                "teamId": team.id,
                "userId": user_id,
                "status": "ACTIVE",
                **team_flags,
            }
        )
    graph = await graph_db.create_graph(
        graph_db.make_graph_model(
            graph_db.Graph(name="Briefing", description="Summarises the week"), owner
        ),
        owner,
        organization_id=org.id,
        team_id=team.id,
    )
    return World(
        org=org.id,
        team=team.id,
        owner=owner,
        member=member,
        org_billing=org_billing,
        team_billing=team_billing,
        owner_graph=graph.id,
        owner_team_run=await _run(owner, graph, org.id, team.id),
        owner_home_run=await _run(owner, graph, org.id, None),
        keys={
            who: await _key(user_id, org.id)
            for who, user_id in (
                ("member", member),
                ("org_billing", org_billing),
                ("team_billing", team_billing),
            )
        },
        team_keys={
            who: await _key(user_id, org.id, team.id)
            for who, user_id in (("member", member), ("team_billing", team_billing))
        },
    )


async def _user(tag: str, users: list[str]) -> str:
    user_id = f"key-roles-{tag}"
    await prisma.user.create(data={"id": user_id, "email": f"{user_id}@example.com"})
    users.append(user_id)
    return user_id


async def _run(user_id: str, graph, organization_id: str, team_id: str | None) -> str:
    run = await execution_db.create_graph_execution(
        graph_id=graph.id,
        graph_version=graph.version,
        starting_nodes_input=[],
        inputs={"brief": SECRET},
        user_id=user_id,
        organization_id=organization_id,
        team_id=team_id,
    )
    return run.id


async def _key(user_id: str, organization_id: str, team_id: str | None = None) -> str:
    _, plaintext = await api_key.create_api_key(
        name="roles",
        user_id=user_id,
        permissions=list(APIKeyPermission),
        organization_id=organization_id,
        team_id_restriction=team_id,
    )
    return plaintext


@pytest_asyncio.fixture(loop_scope="session")
async def v2(world: World) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=v2_app), base_url="http://test"
    ) as client:
        yield client


@pytest_asyncio.fixture(loop_scope="session")
async def v1(world: World) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=external_api), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture(autouse=True)
def no_background_work(mocker):
    mocker.patch("backend.api.features.library.db.add_generated_agent_image")
    mocker.patch("backend.api.features.library.db.schedule_library_agent_embedding")
    # v1's per-key request counter lives in Redis; its own tests cover it.
    mocker.patch.object(rate_limit, "_incr_window", return_value=1)


@pytest.mark.asyncio(loop_scope="session")
async def test_an_org_billing_managers_key_keeps_billing_and_loses_the_rest(
    v2: httpx.AsyncClient, world: World
):
    key = {"X-API-Key": world.keys["org_billing"]}

    identity = await v2.get("/me", headers=key)
    balance = await v2.get("/credits", headers=key)
    refused = {
        call: await v2.request(*call.split(" ", 1), headers=key)
        for call in (
            "GET /graphs",
            f"GET /graphs/{world.owner_graph}",
            "GET /runs",
            f"GET /runs/{world.owner_home_run}",
            "GET /library/agents",
            "GET /library/folders",
            "GET /library/presets",
            "GET /schedules",
            "GET /runs/reviews",
        )
    }
    created = await v2.post("/graphs", headers=key, json=GRAPH)
    in_team = await v2.post(
        "/graphs", headers={**key, "X-Team-Id": world.team}, json=GRAPH
    )

    assert identity.status_code == 200, identity.text
    assert balance.status_code == 200, balance.text
    for call, response in refused.items():
        assert response.status_code == 403, f"{call}: {response.text}"
        assert BILLING_ONLY_ORG_ROLE in response.text, call
        assert SECRET not in response.text, call
    for response in (created, in_team):
        assert response.status_code == 403, response.text
        assert BILLING_ONLY_ORG_ROLE in response.text
    assert await _workflows(world.org_billing, world.org) == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_a_team_billing_managers_key_is_kept_out_of_the_team(
    v2: httpx.AsyncClient, world: World
):
    """Whether the team comes from the request or is pinned on the key. In the
    org's home they are a plain member, and the team's rows stay hidden."""
    team_refusal = BILLING_ONLY_TEAM_ROLE.format(team_id=world.team)
    by_header = {"X-API-Key": world.keys["team_billing"], "X-Team-Id": world.team}
    pinned = {"X-API-Key": world.team_keys["team_billing"]}
    home = {"X-API-Key": world.keys["team_billing"]}

    for headers in (by_header, pinned):
        listed = await v2.get("/runs", headers=headers)
        read = await v2.get(f"/runs/{world.owner_team_run}", headers=headers)
        created = await v2.post("/graphs", headers=headers, json=GRAPH)
        for response in (listed, read, created):
            assert response.status_code == 403, response.text
            assert team_refusal in response.text
            assert SECRET not in response.text
    assert await _workflows(world.team_billing, world.org, world.team) == 0

    listed_at_home = await v2.get("/runs", headers=home)
    read_at_home = await v2.get(f"/runs/{world.owner_team_run}", headers=home)
    assert listed_at_home.status_code == 200, listed_at_home.text
    assert world.owner_team_run not in listed_at_home.text
    assert read_at_home.status_code == 404, read_at_home.text
    assert SECRET not in read_at_home.text


@pytest.mark.asyncio(loop_scope="session")
async def test_a_plain_members_key_keeps_what_it_had(
    v2: httpx.AsyncClient, world: World
):
    key = {"X-API-Key": world.keys["member"]}
    in_team = {**key, "X-Team-Id": world.team}

    listed = await v2.get("/runs", headers=in_team)
    read = await v2.get(f"/runs/{world.owner_team_run}", headers=in_team)
    workflows = await v2.get("/graphs", headers=key)
    created_at_home = await v2.post("/graphs", headers=key, json=GRAPH)
    created_in_team = await v2.post("/graphs", headers=in_team, json=GRAPH)

    assert listed.status_code == 200 and world.owner_team_run in listed.text
    assert read.status_code == 200 and SECRET in read.text
    assert workflows.status_code == 200, workflows.text
    assert created_at_home.status_code == 201, created_at_home.text
    assert created_in_team.status_code == 201, created_in_team.text
    assert await _workflows(world.member, world.org, world.team) == 1
    assert await _workflows(world.member, world.org) == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_v1_refuses_a_billing_managers_key_its_workflow_and_run_routes(
    v1: httpx.AsyncClient, world: World
):
    """And keeps `/v1/me`; a plain member's key creates as before."""
    org_billing = {"X-API-Key": world.keys["org_billing"]}
    team_billing = {"X-API-Key": world.team_keys["team_billing"]}
    member = {"X-API-Key": world.keys["member"]}

    identity = await v1.get("/v1/me", headers=org_billing)
    refused = [
        await v1.post("/v1/graphs", headers=org_billing, json=GRAPH),
        await v1.post(
            f"/v1/graphs/{world.owner_graph}/execute/1",
            headers=org_billing,
            json={"node_input": {}},
        ),
        await v1.get(
            f"/v1/graphs/{world.owner_graph}/executions/"
            f"{world.owner_home_run}/results",
            headers=org_billing,
        ),
        await v1.post(
            "/v1/tools/run-agent",
            headers=org_billing,
            json={"username_agent_slug": "someone/something"},
        ),
    ]
    in_team = await v1.post("/v1/graphs", headers=team_billing, json=GRAPH)
    created = await v1.post("/v1/graphs", headers=member, json=GRAPH)

    assert identity.status_code == 200, identity.text
    for response in refused:
        assert response.status_code == 403, response.text
        assert BILLING_ONLY_ORG_ROLE in response.text
    assert in_team.status_code == 403, in_team.text
    assert BILLING_ONLY_TEAM_ROLE.format(team_id=world.team) in in_team.text
    assert await _workflows(world.org_billing, world.org) == 0
    assert await _workflows(world.team_billing, world.org) == 0
    assert created.status_code == 201, created.text


@pytest.mark.asyncio(loop_scope="session")
async def test_an_mcp_token_carries_the_role_its_tools_are_held_to(world: World):
    """The MCP server reads the role with the tenant, once per request; its
    tools on workflows, runs, library and schedules refuse a token that
    carries a refusal (mcp_server_test.py)."""
    verifier = ExternalAPITokenVerifier()

    org_billing = await verifier.verify_token(world.keys["org_billing"])
    team_billing = await verifier.verify_token(world.team_keys["team_billing"])
    member = await verifier.verify_token(world.team_keys["member"])

    assert isinstance(org_billing, TenantedAccessToken)
    assert org_billing.role_refusal == BILLING_ONLY_ORG_ROLE
    assert isinstance(team_billing, TenantedAccessToken)
    assert team_billing.role_refusal == BILLING_ONLY_TEAM_ROLE.format(
        team_id=world.team
    )
    assert isinstance(member, TenantedAccessToken)
    assert (member.team_id, member.role_refusal) == (world.team, None)


async def _workflows(user_id: str, organization_id: str, team_id: str | None = None):
    where = {"userId": user_id, "organizationId": organization_id}
    if team_id:
        where["teamId"] = team_id
    return await prisma.agentgraph.count(where=where)
