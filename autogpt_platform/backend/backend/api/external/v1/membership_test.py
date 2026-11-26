"""A v1 key made in an organization works only while its owner is a member.

Through the real External API app: real key verification and the real auth
dependencies, with the key rows, the membership lookup and the writes behind
the routes stubbed. Every kind of route is covered: one reached only through
`require_auth` (the store), the rate limiter's, and `require_permission`'s.
The Postgres-backed journey is in `membership_db_test.py`.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.external import middleware
from backend.api.external.v1 import routes as v1_routes
from backend.api.external.v1 import tools as v1_tools
from backend.api.features.library import db as library_db
from backend.copilot.tools.models import ErrorResponse

ORG = "org-1"
TEAM = "team-1"
GRAPH = {"name": "Weekly report", "description": "", "nodes": [], "links": []}
SCHEDULE = {
    "username_agent_slug": "someone/weekly-report",
    "inputs": {},
    "schedule_name": "Weekly",
    "cron": "0 9 * * 1",
}
ROUTES = [
    ("GET", "/v1/me", None),
    ("GET", "/v1/store/agents", None),
    ("POST", "/v1/graphs", GRAPH),
    ("POST", "/v1/graphs/graph-1/execute/1", {"node_input": {}}),
    ("POST", "/v1/tools/run-agent", SCHEDULE),
]


@pytest.fixture(autouse=True)
def fresh_membership_answers():
    middleware.holds_membership.cache_clear()
    yield
    middleware.holds_membership.cache_clear()


@pytest.fixture
def key(v1_generated_keys):
    """The plaintext of the first stored key (`v1_key_rows[0]`)."""
    return v1_generated_keys[0].key


@pytest.fixture
def owner(v1_key_rows) -> str:
    return v1_key_rows[0].user_id


@pytest.fixture
def scoped(v1_key_rows):
    """Store the key as made in an organization, and possibly a team."""

    def scope(organization_id: str | None, team_id: str | None = None) -> None:
        v1_key_rows[0].organization_id = organization_id
        v1_key_rows[0].team_id_restriction = team_id

    return scope


@pytest.fixture
def membership(mocker):
    return mocker.patch.object(
        middleware, "is_active_member", AsyncMock(return_value=True)
    )


@pytest.fixture
def writes(mocker):
    """What the routes would write: workflows, library rows, runs, schedules."""

    async def same_graph(graph, user_id):
        return graph

    mocker.patch.object(v1_routes, "before_graph_activate", side_effect=same_graph)
    return {
        "graph": mocker.patch.object(v1_routes.graph_db, "create_graph", AsyncMock()),
        "library": mocker.patch.object(
            library_db, "create_library_agent", AsyncMock(return_value=[])
        ),
        "run": mocker.patch.object(
            v1_routes,
            "add_graph_execution",
            AsyncMock(return_value=MagicMock(id="run-1")),
        ),
        "schedule": mocker.patch.object(
            v1_tools.run_agent_tool,
            "_execute",
            AsyncMock(return_value=ErrorResponse(message="stub")),
        ),
    }


@pytest.mark.parametrize("method, path, body", ROUTES, ids=[r[1] for r in ROUTES])
@pytest.mark.parametrize("team_id", [None, TEAM], ids=["org-key", "team-key"])
async def test_a_former_members_key_is_refused_before_the_route_runs(
    v1_client,
    v1_counter,
    key,
    owner,
    scoped,
    membership,
    writes,
    method,
    path,
    body,
    team_id,
):
    scoped(ORG, team_id)
    membership.return_value = False

    response = await v1_client.request(
        method, path, headers={"X-API-Key": key}, json=body
    )

    assert response.status_code == 403
    assert response.json()["detail"] == middleware.FORMER_MEMBER_DETAIL
    membership.assert_awaited_with(owner, ORG, team_id)
    for write in writes.values():
        write.assert_not_awaited()
    v1_counter.assert_not_awaited()


@pytest.mark.parametrize(
    "organization_id, team_id, acts_in",
    [(ORG, TEAM, ORG), (ORG, None, ORG), (None, None, "personal-org")],
    ids=["team-key", "org-key", "key-without-org"],
)
async def test_a_members_key_creates_where_it_was_issued(
    mocker,
    v1_client,
    v1_counter,
    key,
    owner,
    scoped,
    membership,
    writes,
    organization_id,
    team_id,
    acts_in,
):
    scoped(organization_id, team_id)
    mocker.patch.object(
        middleware,
        "get_user_default_team",
        AsyncMock(return_value=("personal-org", "personal-team")),
    )

    created = await v1_client.post("/v1/graphs", headers={"X-API-Key": key}, json=GRAPH)
    scheduled = await v1_client.post(
        "/v1/tools/run-agent", headers={"X-API-Key": key}, json=SCHEDULE
    )

    assert created.status_code == 201, created.text
    assert scheduled.status_code == 200, scheduled.text
    for write in (writes["graph"], writes["library"]):
        assert write.await_args.kwargs["organization_id"] == acts_in
        assert write.await_args.kwargs["team_id"] == team_id
    session = writes["schedule"].await_args.kwargs["session"]
    assert (session.user_id, session.organization_id) == (owner, acts_in)
    if organization_id:
        membership.assert_awaited_with(owner, organization_id, team_id)
    else:
        membership.assert_not_awaited()


async def test_the_membership_is_read_once_per_request_and_then_reused(
    v1_client, v1_counter, key, scoped, membership
):
    """The rate limiter and the route each resolve the caller; one read serves
    both, and the requests after it within the cache window."""
    scoped(ORG, TEAM)

    for _ in range(3):
        response = await v1_client.get("/v1/me", headers={"X-API-Key": key})
        assert response.status_code == 200, response.text
    assert membership.await_count == 1

    middleware.holds_membership.cache_clear()
    response = await v1_client.get("/v1/me", headers={"X-API-Key": key})

    assert response.status_code == 200
    assert membership.await_count == 2
