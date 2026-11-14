"""A deleted workspace's agents and runs stay where they were; real database.

The foreign keys from content to `Team` decide what a delete does to it, and a
mocked Prisma has none. Outside `orgs/`, whose conftest no-ops `server`.
"""

import uuid

import prisma.models
import pytest

from backend.api.features.orgs import db as orgs_db
from backend.api.features.orgs import team_db
from backend.data.execution import (
    create_graph_execution,
    get_graph_executions_paginated,
)
from backend.data.graph import Graph, create_graph, list_graphs_paginated
from backend.data.user import get_or_create_user
from backend.util.exceptions import NotFoundError
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_a_deleted_private_teams_agents_and_runs_stay_with_their_authors(
    server: SpinTestServer,
):
    author, teammate, outsider = await _user(), await _user(), await _user()
    org = await orgs_db.create_org("Team Delete", f"td-{uuid.uuid4().hex[:12]}", author)
    for user_id in (teammate, outsider):
        await orgs_db.add_org_member(org.id, user_id)
    team = await team_db.create_team(org.id, "Private", author, join_policy="PRIVATE")
    await team_db.add_team_member(team.id, teammate, org.id)
    graph = await create_graph(
        Graph(name="Team agent", description="", nodes=[], links=[]),
        author,
        organization_id=org.id,
        team_id=team.id,
    )
    run = await create_graph_execution(
        graph_id=graph.id,
        graph_version=graph.version,
        starting_nodes_input=[],
        inputs={},
        user_id=author,
        organization_id=org.id,
        team_id=team.id,
    )

    assert await _sees(teammate, org.id, graph.id, run.id) == (True, True)
    assert await _sees(outsider, org.id, graph.id, run.id) == (False, False)

    await team_db.delete_team(team.id)

    assert await _sees(outsider, org.id, graph.id, run.id) == (False, False)
    assert await _team_id_of(graph.id) == team.id
    assert await _sees(author, org.id, graph.id, run.id) == (True, True)
    # Shared through membership only, which the delete ends.
    assert await _sees(teammate, org.id, graph.id, run.id) == (False, False)
    with pytest.raises(NotFoundError):
        await team_db.get_team(team.id, expected_org_id=org.id)
    # The name is free for a new team.
    await team_db.create_team(org.id, "Private", author, join_policy="PRIVATE")


async def _sees(
    user_id: str, org_id: str, graph_id: str, run_id: str
) -> tuple[bool, bool]:
    graphs, _ = await list_graphs_paginated(
        user_id, page_size=100, organization_id=org_id
    )
    runs = await get_graph_executions_paginated(
        user_id, graph_id=graph_id, organization_id=org_id
    )
    return (
        graph_id in {g.id for g in graphs},
        run_id in {r.id for r in runs.executions},
    )


async def _user() -> str:
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"team-delete-{uuid.uuid4().hex[:8]}@example.com",
            "name": "Team Delete",
        }
    )
    return user.id


async def _team_id_of(graph_id: str) -> str | None:
    row = await prisma.models.AgentGraph.prisma().find_first_or_raise(
        where={"id": graph_id}
    )
    return row.teamId
