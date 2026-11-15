"""Org and team roles on the workflow, run, library and marketplace routes, end
to end: the real routes, the real request-context resolution, and membership
rows in Postgres.

A membership whose only role is billing manager covers spend, not workflows.
Such a member must not see the team's workflows or runs, nor create, run,
change, delete, publish or share workflows and runs there, whichever org or
team a request names. Ordinary members, admins and owners keep what they had.

Lives outside ``orgs/``, whose conftest stubs the ``server`` fixture; these
tests need only Postgres, so they connect Prisma themselves.
"""

import uuid
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import fastapi
import httpx
import pytest
import pytest_asyncio
from autogpt_libs.auth.dependencies import get_unbanned_jwt_payload
from pydantic import BaseModel

from backend.api.features.graph_executions.routes import router as executions_router
from backend.api.features.graphs.routes import router as graphs_router
from backend.api.features.library.routes import router as library_router
from backend.api.features.orgs.db import resolve_member_tenancy
from backend.api.features.store.routes import router as store_router
from backend.api.utils.exceptions import add_exception_handlers
from backend.copilot.rate_limit import enforce_payment_paywall
from backend.data import execution as execution_db
from backend.data import graph as graph_db
from backend.data.db import prisma
from backend.data.execution import GraphExecutionMeta
from backend.data.org_migration import create_personal_org

USER_HEADER = "X-Test-User"


class World(BaseModel):
    """One shared org with one team. ``team_billing`` is a plain org member who
    only manages the team's billing, ``org_billing`` manages the org's billing
    and is a plain member of the team, as adding them to the org leaves them
    on its default team."""

    org: str
    team: str
    owner: str
    member: str
    team_billing: str
    org_billing: str
    personal: dict[str, tuple[str, str]]
    owner_graph: str
    owner_run: str
    owner_home_run: str
    member_graph: str
    member_run: str
    billing_graph: str
    billing_run: str
    billing_preset: str


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
                await prisma.agentpreset.delete_many(where={"userId": {"in": users}})
                await prisma.user.delete_many(where={"id": {"in": users}})
                await prisma.organization.delete_many(where={"id": {"in": orgs}})
        finally:
            if connected_here and prisma.is_connected():
                await prisma.disconnect()


async def _build_world(users: list[str], orgs: list[str]) -> World:
    suffix = uuid.uuid4().hex[:8]
    owner, member, team_billing, org_billing = [
        await _user(f"{role}-{suffix}", users)
        for role in ("owner", "member", "teambill", "orgbill")
    ]
    personal = {uid: await _personal_org(uid, orgs) for uid in users}
    org = await prisma.organization.create(
        data={"name": f"Roles {suffix}", "slug": f"roles-{suffix}"}
    )
    orgs.append(org.id)
    team = await prisma.team.create(data={"name": "Squad", "orgId": org.id})
    for uid, flags in (
        (owner, {"isOwner": True, "isAdmin": True}),
        (member, {}),
        (team_billing, {}),
        (org_billing, {"isBillingManager": True}),
    ):
        await prisma.orgmember.create(
            data={"orgId": org.id, "userId": uid, "status": "ACTIVE", **flags}
        )
        await prisma.teammember.create(
            data={
                "teamId": team.id,
                "userId": uid,
                "status": "ACTIVE",
                "isAdmin": uid == owner,
            }
        )

    owner_graph = await _workflow(owner, org.id, team.id)
    member_graph = await _workflow(member, org.id, team.id)
    billing_graph = await _workflow(team_billing, org.id, team.id)
    world = World(
        org=org.id,
        team=team.id,
        owner=owner,
        member=member,
        team_billing=team_billing,
        org_billing=org_billing,
        personal=personal,
        owner_graph=owner_graph.id,
        owner_run=await _run(owner, owner_graph, org.id, team.id),
        owner_home_run=await _run(owner, owner_graph, org.id, None),
        member_graph=member_graph.id,
        member_run=await _run(member, member_graph, org.id, team.id),
        billing_graph=billing_graph.id,
        billing_run=await _run(team_billing, billing_graph, org.id, team.id),
        billing_preset=(
            await prisma.agentpreset.create(
                data={
                    "name": "Weekly digest",
                    "description": "",
                    "userId": team_billing,
                    "agentGraphId": billing_graph.id,
                    "agentGraphVersion": billing_graph.version,
                    "organizationId": org.id,
                    "teamId": team.id,
                }
            )
        ).id,
    )
    # What team_billing made above they made as a plain member; an admin has
    # since made billing their only role in the team.
    await prisma.teammember.update(
        where={"teamId_userId": {"teamId": team.id, "userId": team_billing}},
        data={"isBillingManager": True},
    )
    return world


async def _user(tag: str, users: list[str]) -> str:
    user_id = f"roles-{tag}"
    await prisma.user.create(data={"id": user_id, "email": f"{user_id}@example.com"})
    users.append(user_id)
    return user_id


async def _personal_org(user_id: str, orgs: list[str]) -> tuple[str, str]:
    org = await create_personal_org(user_id, user_id, user_id)
    orgs.append(org.id)
    team = await prisma.team.find_first(where={"orgId": org.id, "isDefault": True})
    assert team is not None
    return org.id, team.id


async def _workflow(user_id: str, org_id: str, team_id: str) -> graph_db.GraphModel:
    graph = graph_db.make_graph_model(
        graph_db.Graph(name="Briefing", description="Summarises the week"), user_id
    )
    return await graph_db.create_graph(
        graph, user_id, organization_id=org_id, team_id=team_id
    )


async def _run(
    user_id: str, graph: graph_db.GraphModel, org_id: str, team_id: str | None
) -> str:
    run = await execution_db.create_graph_execution(
        graph_id=graph.id,
        graph_version=graph.version,
        starting_nodes_input=[],
        inputs={"brief": f"confidential notes of {user_id}"},
        user_id=user_id,
        organization_id=org_id,
        team_id=team_id,
    )
    return run.id


@pytest_asyncio.fixture(loop_scope="session")
async def api(world: World) -> AsyncIterator[httpx.AsyncClient]:
    app = fastapi.FastAPI()
    app.include_router(graphs_router)
    app.include_router(executions_router)
    app.include_router(library_router, prefix="/library")
    app.include_router(store_router, prefix="/store")
    add_exception_handlers(app)

    async def token(request: fastapi.Request) -> dict:
        user_id = request.headers[USER_HEADER]
        return {"sub": user_id, "role": "authenticated", "email": f"{user_id}@x.io"}

    app.dependency_overrides[get_unbanned_jwt_payload] = token
    app.dependency_overrides[enforce_payment_paywall] = lambda: None
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
def other_services(mocker) -> None:
    """The scheduler runs as its own service, and a new library entry starts
    image and embedding jobs that call out; this test runs none of them."""
    scheduler = AsyncMock()
    scheduler.get_graph_execution_schedules = AsyncMock(return_value=[])
    for site in ("db", "_schedule_info"):
        mocker.patch(
            f"backend.api.features.library.{site}.get_scheduler_client",
            return_value=scheduler,
        )
    mocker.patch("backend.api.features.library.db.add_generated_agent_image")
    mocker.patch("backend.api.features.library.db.schedule_library_agent_embedding")


async def _call(
    api: httpx.AsyncClient,
    method: str,
    path: str,
    user: str,
    org: str | None = None,
    team: str | None = None,
    json: dict | None = None,
) -> httpx.Response:
    headers = {USER_HEADER: user}
    if org:
        headers["X-Org-Id"] = org
    if team:
        headers["X-Team-Id"] = team
    return await api.request(method, path, headers=headers, json=json)


def _refused_by_role(response: httpx.Response) -> bool:
    """A 403 from the role checks, not from something else turning the
    request away."""
    return response.status_code == 403 and "role in this" in str(
        response.json().get("detail")
    )


async def _run_ids(api: httpx.AsyncClient, user: str, org: str, team=None) -> set:
    response = await _call(api, "GET", "/executions", user, org, team)
    assert response.status_code == 200, response.text
    return {run["id"] for run in response.json()}


@pytest.mark.asyncio(loop_scope="session")
async def test_a_team_billing_manager_does_not_see_the_teams_runs(api, world):
    w = world
    run_path = f"/graphs/{w.owner_graph}/executions/{w.owner_run}"

    listed = await _run_ids(api, w.team_billing, w.org)
    assert w.owner_run not in listed
    assert w.member_run not in listed
    detail = await _call(api, "GET", run_path, w.team_billing, w.org)
    assert detail.status_code == 404
    graph_runs = await _call(
        api, "GET", f"/graphs/{w.owner_graph}/executions", w.team_billing, w.org
    )
    assert graph_runs.status_code == 200
    assert w.owner_run not in {r["id"] for r in graph_runs.json()["executions"]}

    in_team = [
        await _call(api, "GET", path, w.team_billing, w.org, w.team)
        for path in ("/executions", run_path, f"/graphs/{w.owner_graph}/executions")
    ]
    assert all(_refused_by_role(r) for r in in_team), [r.text for r in in_team]
    assert "confidential" not in "".join(r.text for r in [detail, *in_team])


@pytest.mark.asyncio(loop_scope="session")
async def test_team_members_and_admins_still_see_the_teams_runs(api, world):
    w = world
    for user in (w.member, w.owner):
        for team in (None, w.team):
            assert {w.owner_run, w.member_run} <= await _run_ids(api, user, w.org, team)
    detail = await _call(
        api,
        "GET",
        f"/graphs/{w.owner_graph}/executions/{w.owner_run}",
        w.member,
        w.org,
        w.team,
    )
    assert detail.status_code == 200, detail.text
    assert detail.json()["inputs"] == {"brief": f"confidential notes of {w.owner}"}


@pytest.mark.asyncio(loop_scope="session")
async def test_a_team_billing_manager_does_not_see_the_teams_workflows(api, world):
    w = world
    listed = await _call(api, "GET", "/graphs", w.team_billing, w.org)
    assert listed.status_code == 200
    assert w.owner_graph not in {g["id"] for g in listed.json()}
    assert w.billing_graph in {g["id"] for g in listed.json()}
    read = await _call(api, "GET", f"/graphs/{w.owner_graph}", w.team_billing, w.org)
    assert read.status_code == 404
    in_team = await _call(
        api, "GET", f"/graphs/{w.owner_graph}", w.team_billing, w.org, w.team
    )
    assert _refused_by_role(in_team), in_team.text

    teammate = await _call(api, "GET", f"/graphs/{w.owner_graph}", w.member, w.org)
    assert teammate.status_code == 200, teammate.text


@pytest.mark.asyncio(loop_scope="session")
async def test_a_team_billing_manager_cannot_create_or_run_workflows_in_the_team(
    api, world, other_services
):
    w = world
    body = {"graph": {"name": "New", "description": "", "nodes": [], "links": []}}
    created = await _call(api, "POST", "/graphs", w.team_billing, w.org, w.team, body)
    run = await _call(
        api,
        "POST",
        f"/graphs/{w.billing_graph}/execute/1",
        w.team_billing,
        w.org,
        w.team,
        {"inputs": {}},
    )

    assert _refused_by_role(created), created.text
    assert _refused_by_role(run), run.text
    assert await prisma.agentgraph.count(where={"userId": w.team_billing}) == 1
    assert await prisma.agentgraphexecution.count(where={"userId": w.team_billing}) == 1

    by_member = await _call(api, "POST", "/graphs", w.member, w.org, w.team, body)
    assert by_member.status_code == 200, by_member.text
    stored = await prisma.agentgraph.find_first(where={"id": by_member.json()["id"]})
    assert stored is not None and stored.teamId == w.team


@pytest.mark.asyncio(loop_scope="session")
async def test_a_team_billing_manager_cannot_change_or_delete_their_team_workflow(
    api, world, other_services
):
    """Not from the org's home either: the workflow lives in the team."""
    w = world
    for team in (w.team, None):
        deleted = await _call(
            api, "DELETE", f"/graphs/{w.billing_graph}", w.team_billing, w.org, team
        )
        updated = await _call(
            api,
            "PUT",
            f"/graphs/{w.billing_graph}",
            w.team_billing,
            w.org,
            team,
            {"id": w.billing_graph, "name": "Renamed", "description": ""},
        )
        assert _refused_by_role(deleted), deleted.text
        assert _refused_by_role(updated), updated.text
    assert await prisma.agentgraph.count(where={"id": w.billing_graph}) == 1

    by_member = await _call(api, "DELETE", f"/graphs/{w.member_graph}", w.member, w.org)
    assert by_member.status_code == 200, by_member.text
    assert by_member.json()["version_counts"] == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_a_team_billing_manager_cannot_share_or_delete_their_team_run(api, world):
    """Not from their personal org either: the run lives in the team."""
    w = world
    share_path = f"/graphs/{w.billing_graph}/executions/{w.billing_run}/share"
    shared = await _call(api, "POST", share_path, w.team_billing)
    deleted = await _call(api, "DELETE", f"/executions/{w.billing_run}", w.team_billing)

    assert _refused_by_role(shared), shared.text
    assert _refused_by_role(deleted), deleted.text
    run = await prisma.agentgraphexecution.find_unique(where={"id": w.billing_run})
    assert run is not None and not run.isShared and not run.isDeleted

    member_path = f"/graphs/{w.member_graph}/executions/{w.member_run}/share"
    by_member = await _call(api, "POST", member_path, w.member)
    assert by_member.status_code == 200, by_member.text
    withdrawn = await _call(api, "DELETE", member_path, w.member)
    assert withdrawn.status_code == 204


@pytest.mark.asyncio(loop_scope="session")
async def test_a_team_billing_manager_cannot_run_their_team_preset(api, world, mocker):
    """Its run would land in the preset's team, whatever the request names."""
    w = world
    queued = mocker.patch(
        "backend.api.features.library.routes.presets.add_graph_execution",
        AsyncMock(),
    )

    for org in (None, w.org):
        refused = await _call(
            api,
            "POST",
            f"/library/presets/{w.billing_preset}/execute",
            w.team_billing,
            org,
            json={"inputs": {}},
        )
        assert _refused_by_role(refused), refused.text
    queued.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="session")
async def test_a_member_still_runs_their_team_preset(api, world, mocker):
    w = world
    preset = await prisma.agentpreset.create(
        data={
            "name": "Daily digest",
            "description": "",
            "userId": w.member,
            "agentGraphId": w.member_graph,
            "agentGraphVersion": 1,
            "organizationId": w.org,
            "teamId": w.team,
        }
    )
    queued = mocker.patch(
        "backend.api.features.library.routes.presets.add_graph_execution",
        AsyncMock(
            return_value=GraphExecutionMeta.model_validate(
                {
                    "id": "queued-run",
                    "user_id": w.member,
                    "graph_id": w.member_graph,
                    "graph_version": 1,
                    "inputs": {},
                    "credential_inputs": None,
                    "nodes_input_masks": None,
                    "preset_id": preset.id,
                    "status": "QUEUED",
                    "stats": None,
                }
            )
        ),
    )

    response = await _call(
        api, "POST", f"/library/presets/{preset.id}/execute", w.member, json={}
    )

    assert response.status_code == 200, response.text
    assert queued.await_args.kwargs["organization_id"] == w.org
    assert queued.await_args.kwargs["team_id"] == w.team


@pytest.mark.asyncio(loop_scope="session")
async def test_a_team_billing_manager_cannot_publish_their_team_workflow(api, world):
    w = world
    submission = {
        "graph_id": w.billing_graph,
        "graph_version": 1,
        "slug": f"briefing-{uuid.uuid4().hex[:6]}",
        "name": "Briefing",
        "sub_heading": "Summarises the week",
        "categories": ["research"],
    }
    for team in (w.team, None):
        response = await _call(
            api, "POST", "/store/submissions", w.team_billing, w.org, team, submission
        )
        assert _refused_by_role(response), response.text
    assert await prisma.storelisting.count(where={"agentGraphId": w.billing_graph}) == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_an_org_billing_manager_is_refused_the_orgs_workflows_and_runs(
    api, world
):
    """Even on a team where they're a plain member."""
    w = world
    calls = [
        ("GET", "/executions", None),
        ("GET", f"/graphs/{w.owner_graph}/executions/{w.owner_home_run}", None),
        ("GET", "/graphs", None),
        ("POST", "/graphs", {"graph": {"name": "N", "description": "", "nodes": []}}),
        ("GET", "/library/agents", None),
        ("GET", "/store/submissions", None),
    ]
    for team in (None, w.team):
        for method, path, body in calls:
            response = await _call(api, method, path, w.org_billing, w.org, team, body)
            assert _refused_by_role(response), (method, path, team, response.text)
    assert await prisma.agentgraph.count(where={"userId": w.org_billing}) == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_runs_presets_and_triggers_land_only_where_the_role_covers_workflows(
    world,
):
    """Whatever starts them: a schedule, a trigger, a preset or another run."""
    w = world
    assert await resolve_member_tenancy(w.team_billing, w.org, w.team) == (
        w.personal[w.team_billing]
    )
    assert await resolve_member_tenancy(w.org_billing, w.org, None) == (
        w.personal[w.org_billing]
    )
    assert await resolve_member_tenancy(w.member, w.org, w.team) == (w.org, w.team)
    assert await resolve_member_tenancy(w.owner, w.org, None) == (w.org, None)
