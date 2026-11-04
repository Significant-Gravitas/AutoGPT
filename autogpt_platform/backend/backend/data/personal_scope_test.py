"""A scope a caller builds by hand matches what the database stored.

The database moves a personal organization's rows to its default team, so a
``(personal org, None)`` pair handed to an exact-match check must be read as
that team, or the check refuses the caller's own graph.
"""

from uuid import uuid4

import prisma.models
import pytest
import pytest_asyncio
from pydantic import BaseModel

from backend.api.features.library import db as library_db
from backend.api.features.orgs.db import create_org, get_user_default_team
from backend.api.features.store import db as store_db
from backend.data import graph as graph_db
from backend.data.personal_scope import canonical_personal_scope
from backend.data.user import get_or_create_user
from backend.util.test import SpinTestServer

pytestmark = pytest.mark.asyncio(loop_scope="session")


class Owner(BaseModel):
    user_id: str
    personal_org: str
    default_team: str


@pytest_asyncio.fixture(loop_scope="session")
async def owner(server: SpinTestServer) -> Owner:
    user = await get_or_create_user(
        {"sub": str(uuid4()), "email": f"scope-{uuid4().hex[:8]}@example.com"}
    )
    personal_org, default_team = await get_user_default_team(user.id)
    assert personal_org and default_team
    return Owner(user_id=user.id, personal_org=personal_org, default_team=default_team)


async def test_a_personal_org_without_a_team_is_its_default_team(owner: Owner):
    assert await canonical_personal_scope(owner.personal_org, None) == (
        owner.personal_org,
        owner.default_team,
    )


async def test_other_scopes_are_returned_as_given(owner: Owner):
    team_org = await create_org("Scope", f"scope-{uuid4().hex[:8]}", owner.user_id)

    assert await canonical_personal_scope(team_org.id, None) == (team_org.id, None)
    assert await canonical_personal_scope(None, None) == (None, None)
    assert await canonical_personal_scope(owner.personal_org, "team-x") == (
        owner.personal_org,
        "team-x",
    )


async def test_versions_saved_at_personal_org_home_join_the_default_team(
    owner: Owner,
):
    v1 = await graph_db.create_graph(
        _graph(owner, str(uuid4()), 1),
        owner.user_id,
        organization_id=owner.personal_org,
    )

    v2 = await graph_db.create_graph(
        _graph(owner, v1.id, 2), owner.user_id, organization_id=owner.personal_org
    )

    assert (v1.team_id, v2.team_id) == (owner.default_team, owner.default_team)


async def test_a_library_entry_added_at_personal_org_home_is_the_default_teams(
    owner: Owner,
):
    graph = await graph_db.create_graph(
        _graph(owner, str(uuid4()), 1),
        owner.user_id,
        organization_id=owner.personal_org,
    )

    [entry] = await library_db.create_library_agent(
        graph, owner.user_id, organization_id=owner.personal_org, team_id=None
    )
    moved = await library_db.update_agent_version_in_library(
        owner.user_id,
        graph.id,
        graph.version,
        organization_id=owner.personal_org,
        team_id=None,
    )

    assert entry.team_id == owner.default_team
    assert moved.id == entry.id


async def test_a_store_submission_named_at_personal_org_home_finds_the_graph(
    owner: Owner,
):
    """Internal callers name a personal org without its team; the submission's
    exact graph lookup must still find the graph the database moved."""
    graph = await graph_db.create_graph(
        _graph(owner, str(uuid4()), 1),
        owner.user_id,
        organization_id=owner.personal_org,
    )
    # A submission needs a marketplace profile; sign-up usually made one.
    await prisma.models.Profile.prisma().upsert(
        where={"userId": owner.user_id},
        data={
            "create": {
                "userId": owner.user_id,
                "name": "Scope",
                "username": f"scope-{uuid4().hex[:8]}",
                "description": "",
                "links": [],
            },
            "update": {},
        },
    )

    submission = await store_db.create_store_submission(
        owner.user_id,
        graph.id,
        graph.version,
        slug=f"scope-{uuid4().hex[:8]}",
        name="Scope",
        sub_heading="",
        organization_id=owner.personal_org,
        team_id_restriction=None,
    )

    assert (submission.graph_id, submission.graph_version) == (graph.id, 1)


def _graph(owner: Owner, graph_id: str, version: int) -> graph_db.Graph:
    graph = graph_db.make_graph_model(
        graph_db.Graph(id=graph_id, version=version, name="Scope", description=""),
        owner.user_id,
    )
    graph.reassign_ids(user_id=owner.user_id, reassign_graph_id=False)
    return graph
