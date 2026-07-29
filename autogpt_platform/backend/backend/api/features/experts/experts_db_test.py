import uuid
from unittest.mock import AsyncMock, patch

import prisma.models
import pytest

import backend.api.features.store.model as store_model
from backend.api.features.experts import experts_db, seed
from backend.api.model import CreateGraph
from backend.blocks.io import AgentInputBlock
from backend.data.graph import Graph, Node
from backend.data.user import get_or_create_user
from backend.usecases.sample import create_test_user
from backend.util.test import SpinTestServer


@pytest.fixture(scope="session", autouse=True)
def mock_embedding_functions():
    """Mock embedding functions to avoid database/API dependencies
    (mirrors backend/data/graph_test.py)."""
    with patch(
        "backend.api.features.store.db.ensure_embedding",
        new_callable=AsyncMock,
        return_value=True,
    ):
        yield


@pytest.fixture
async def test_user():
    return await create_test_user()


@pytest.fixture
async def other_user():
    return await create_test_user(alt_user=True)


async def _create_seed_user():
    suffix = uuid.uuid4().hex[:8]
    return await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"expert-seed-{suffix}@example.com",
            "name": "Seed Owner",
        }
    )


async def _seed_store_listing(server: SpinTestServer) -> str:
    """Create a graph plus an APPROVED store listing on top of it.

    Returns the StoreListingVersion ID, ready for
    ``add_store_agent_to_library``. Mirrors the seeding pattern from
    ``backend/data/graph_test.py::test_access_store_listing_graph``.
    """
    owner = await _create_seed_user()
    admin = await _create_seed_user()

    graph = Graph(
        name=f"Expert seed graph {uuid.uuid4().hex[:8]}",
        description="Seed graph for expert workflow installs",
        nodes=[
            Node(
                block_id=AgentInputBlock().id,
                input_default={"name": "input_1"},
            ),
        ],
        links=[],
    )
    created_graph = await server.agent_server.test_create_graph(
        CreateGraph(graph=graph), owner.id
    )

    # A Profile is required for store submissions; get_or_create_user
    # already ensures one for every user.
    listing = await server.agent_server.test_create_store_listing(
        store_model.StoreSubmissionRequest(
            graph_id=created_graph.id,
            graph_version=created_graph.version,
            slug=created_graph.id,
            name=f"Seed listing {uuid.uuid4().hex[:8]}",
            sub_heading="Seed sub heading",
            video_url=None,
            image_urls=[],
            description="Seed description",
            categories=[],
        ),
        owner.id,
    )
    slv_id = listing.listing_version_id
    assert slv_id is not None, "Failed to create store listing"

    await server.agent_server.test_review_store_listing(
        store_model.ReviewSubmissionRequest(
            store_listing_version_id=slv_id,
            is_approved=True,
            comments="seed",
        ),
        user_id=admin.id,
    )
    return slv_id


async def _seed_template(
    name: str, preload_listings: list[str]
) -> prisma.models.Expert:
    """Create an Expert roster template plus ExpertWorkflow preload rows.

    The stored name gets a unique suffix so ad-hoc test templates never
    collide with seed.ROSTER's real roster names — seed._upsert_template
    resolves templates by name, and a bare "Maria" here would be silently
    adopted and overwritten by test_seed_roster_round_trip.
    """
    template = await prisma.models.Expert.prisma().create(
        data={
            "name": f"{name} {uuid.uuid4().hex[:8]}",
            "role": f"{name}'s role",
            "identity": f"You are {name}, an expert.",
            "isTemplate": True,
        }
    )
    for slv_id in preload_listings:
        await prisma.models.ExpertWorkflow.prisma().create(
            data={"expertId": template.id, "storeListingVersionId": slv_id}
        )
    return template


@pytest.mark.asyncio(loop_scope="session")
async def test_hire_expert_is_idempotent(server: SpinTestServer, test_user):
    template = await _seed_template(name="Maria", preload_listings=[])
    first = await experts_db.hire_expert(test_user.id, template.id, None)
    second = await experts_db.hire_expert(test_user.id, template.id, None)
    assert first.expert.id == second.expert.id
    assert not first.expert.is_template
    assert first.expert.source_template_id == template.id


@pytest.mark.asyncio(loop_scope="session")
async def test_rehire_after_archive_revives_expert(server: SpinTestServer, test_user):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    await experts_db.archive_expert(test_user.id, hired.expert.id)

    revived = await experts_db.hire_expert(test_user.id, template.id, None)

    assert revived.expert.id == hired.expert.id
    assert not revived.expert.is_archived
    assert hired.expert.id in {
        e.id for e in await experts_db.list_experts(test_user.id)
    }


@pytest.mark.asyncio(loop_scope="session")
async def test_install_workflow_on_archived_expert_raises(
    server: SpinTestServer, test_user
):
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    await experts_db.archive_expert(test_user.id, hired.expert.id)

    with pytest.raises(experts_db.ExpertNotFoundError):
        await experts_db.install_workflow(test_user.id, hired.expert.id, slv_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_hire_installs_preloads_into_library(server: SpinTestServer, test_user):
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(name="Maria", preload_listings=[slv_id])
    result = await experts_db.hire_expert(test_user.id, template.id, None)
    wf = result.expert.workflows[0]
    assert wf.library_agent_id is not None
    assert wf.store_listing_version_id == slv_id
    assert result.failed_preloads == []


@pytest.mark.asyncio(loop_scope="session")
async def test_hire_reports_failed_preload_without_sinking_hire(
    server: SpinTestServer, test_user
):
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(name="Maria", preload_listings=[slv_id])
    with patch.object(
        experts_db.library_db,
        "add_store_agent_to_library",
        new_callable=AsyncMock,
        side_effect=RuntimeError("install exploded"),
    ):
        result = await experts_db.hire_expert(test_user.id, template.id, None)
    assert not result.expert.is_template
    assert result.expert.workflows == []
    assert len(result.failed_preloads) == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_get_expert_scopes_by_owner(
    server: SpinTestServer, test_user, other_user
):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    assert await experts_db.get_expert(other_user.id, hired.expert.id) is None
    assert await experts_db.get_expert(test_user.id, hired.expert.id) is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_install_workflow_duplicate_returns_existing(
    server: SpinTestServer, test_user
):
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    a = await experts_db.install_workflow(test_user.id, hired.expert.id, slv_id)
    b = await experts_db.install_workflow(test_user.id, hired.expert.id, slv_id)
    assert a.id == b.id
    assert a.library_agent_id is not None
    assert a.store_listing_version_id == slv_id


@pytest.mark.asyncio(loop_scope="session")
async def test_seed_roster_round_trip(server: SpinTestServer):
    first_ids = await seed.seed_roster()
    assert len(first_ids) == 3

    templates = await experts_db.list_templates()
    seeded = {t.name: t for t in templates if t.id in first_ids}
    assert {e["name"] for e in seed.ROSTER} == set(seeded)
    for entry in seed.ROSTER:
        template = seeded[entry["name"]]
        assert template.is_template
        assert template.role == entry["role"]
        assert template.identity == entry["identity"]

    second_ids = await seed.seed_roster()
    assert first_ids == second_ids

    templates_after = await experts_db.list_templates()
    seeded_after = [t for t in templates_after if t.id in second_ids]
    assert len(seeded_after) == 3


@pytest.mark.asyncio(loop_scope="session")
async def test_seed_backfills_presentation_fields_onto_hired_copies(
    server: SpinTestServer, test_user
):
    """A re-seed must reach experts hired before the roster changed."""
    # Unique name so _upsert_template resolves to this template and not to a
    # same-named one left behind by another test in the shared session DB.
    template = await _seed_template(
        name=f"Maria {uuid.uuid4().hex[:8]}", preload_listings=[]
    )
    hired = await experts_db.hire_expert(test_user.id, template.id, "My Maria")
    assert hired.expert.avatar_url is None
    assert hired.expert.bio is None
    assert hired.expert.skills == []

    entry: seed.RosterEntry = {
        "name": template.name,
        "role": template.role,
        "tagline": "Refreshed tagline",
        "avatar_url": "/experts/maria.svg",
        "bio": "Maria is a senior marketing strategist.",
        "skills": ["Content strategy", "SEO writing"],
        "identity": template.identity,
        "preload_slugs": [],
    }
    refreshed_template = await seed._upsert_template(entry)
    assert await seed._backfill_hired_copies(refreshed_template) == 1

    refreshed = await experts_db.get_expert(test_user.id, hired.expert.id)
    assert refreshed is not None
    assert refreshed.avatar_url == "/experts/maria.svg"
    assert refreshed.bio == "Maria is a senior marketing strategist."
    assert refreshed.skills == ["Content strategy", "SEO writing"]
    # A user's rename of their own hire survives the refresh.
    assert refreshed.name == "My Maria"
