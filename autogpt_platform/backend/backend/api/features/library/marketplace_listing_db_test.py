"""`get_library_agent` reports the marketplace listing of the agent's graph."""

import uuid

import prisma.models
import pytest
from autogpt_libs.auth.models import RequestContext

import backend.api.features.store.model as store_model
from backend.api.features.graphs import routes as graphs_routes
from backend.api.features.library import db as library_db
from backend.api.model import CreateGraph
from backend.blocks.io import AgentInputBlock
from backend.data.graph import Graph, Node
from backend.data.user import get_or_create_user
from backend.util.test import SpinTestServer

_seeded_user_ids: list[str] = []


@pytest.fixture(autouse=True)
async def _clean_seeded_rows():
    yield
    user_ids = list(_seeded_user_ids)
    _seeded_user_ids.clear()
    if not user_ids:
        return
    # A listing pointing at a seed user's graph blocks the cascade from User.
    await prisma.models.StoreListing.prisma().delete_many(
        where={"owningUserId": {"in": user_ids}}
    )
    await prisma.models.LibraryAgent.prisma().delete_many(
        where={"userId": {"in": user_ids}}
    )
    await prisma.models.User.prisma().delete_many(where={"id": {"in": user_ids}})


async def test_a_published_agent_reports_its_listing(server: SpinTestServer):
    owner, graph, listing_name = await _publish_graph(server)

    agent = await library_db.get_library_agent(
        await _library_agent_id(owner.id, graph.id), owner.id
    )

    listing = agent.marketplace_listing
    assert listing is not None
    assert listing.name == listing_name
    assert listing.slug == graph.id
    profile = await prisma.models.Profile.prisma().find_unique(
        where={"userId": owner.id}
    )
    assert profile is not None
    assert listing.creator.slug == profile.username


async def test_a_version_saved_after_publishing_still_reports_the_listing(
    server: SpinTestServer,
):
    """Saving moves the owner's library agent to a version no listing names."""
    owner, graph, listing_name = await _publish_graph(server)
    library_agent_id = await _library_agent_id(owner.id, graph.id)

    await graphs_routes.update_graph(
        graph.id,
        Graph(id=graph.id, name=graph.name, description="Edited", nodes=_nodes()),
        owner.id,
        _ctx(owner.id),
    )

    agent = await library_db.get_library_agent(library_agent_id, owner.id)
    assert agent.graph_version == graph.version + 1
    assert agent.marketplace_listing is not None
    assert agent.marketplace_listing.name == listing_name


async def _publish_graph(server: SpinTestServer):
    owner = await _create_user()
    admin = await _create_user()
    graph = await server.agent_server.test_create_graph(
        CreateGraph(
            graph=Graph(
                name=f"Listed graph {uuid.uuid4().hex[:8]}",
                description="Published to the marketplace",
                nodes=_nodes(),
                links=[],
            )
        ),
        owner.id,
    )
    listing_name = f"Listing {uuid.uuid4().hex[:8]}"
    submission = await server.agent_server.test_create_store_listing(
        store_model.StoreSubmissionRequest(
            graph_id=graph.id,
            graph_version=graph.version,
            slug=graph.id,
            name=listing_name,
            sub_heading="Sub heading",
            video_url=None,
            image_urls=[],
            description="Description",
            categories=["operations"],
        ),
        owner.id,
    )
    assert submission.listing_version_id is not None
    await server.agent_server.test_review_store_listing(
        store_model.ReviewSubmissionRequest(
            store_listing_version_id=submission.listing_version_id,
            is_approved=True,
            comments="ok",
        ),
        user_id=admin.id,
    )
    return owner, graph, listing_name


async def _library_agent_id(user_id: str, graph_id: str) -> str:
    library_agent = await prisma.models.LibraryAgent.prisma().find_first(
        where={"userId": user_id, "agentGraphId": graph_id}
    )
    assert library_agent is not None
    return library_agent.id


async def _create_user():
    suffix = uuid.uuid4().hex[:8]
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"library-listing-{suffix}@example.com",
            "name": "Listing Owner",
        }
    )
    _seeded_user_ids.append(user.id)
    return user


def _nodes() -> list[Node]:
    return [Node(block_id=AgentInputBlock().id, input_default={"name": "input_1"})]


def _ctx(user_id: str) -> RequestContext:
    return RequestContext(
        user_id=user_id,
        org_id=f"test-org-{user_id}",
        team_id=None,
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=False,
        is_team_admin=True,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )
