import asyncio
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from test import load_store_agents as store_assets
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import prisma.enums
import prisma.errors
import prisma.models
import pydantic
import pytest

import backend.api.features.store.model as store_model
from backend.api.features.experts import experts_db, scheduling, seed
from backend.api.features.experts.models import (
    ExpertSoulFieldsPatch,
    ExpertSoulUpdate,
    HireResult,
    RaiseAttachment,
    VoiceSample,
    encode_voice_preferences,
)
from backend.api.features.library import db as library_db
from backend.api.features.library import model as library_model
from backend.api.model import CreateGraph
from backend.blocks.io import AgentInputBlock
from backend.copilot.model import create_chat_session
from backend.data.db import prisma as db_client
from backend.data.graph import Graph, GraphSettings, Node
from backend.data.model import User
from backend.data.user import get_or_create_user
from backend.util.exceptions import ExpertRunPausedError, NotFoundError
from backend.util.json import SafeJson
from backend.util.test import SpinTestServer

EXPECTED_ROSTER_PRELOAD_SLUGS = {
    "ai-webpage-copy-improver",
    "automated-blog-writer",
    "automated-support-ai",
    "business-ownerceo-finder",
    "email-address-finder",
    "lead-finder-local-businesses",
    "linkedin-post-generator",
    "personalized-morning-coffee-newsletter",
    "smart-meeting-brief",
}
EXPECTED_ROSTER_SCHEDULE = (
    "Frankie",
    "personalized-morning-coffee-newsletter",
    "40 7 * * *",
)


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
    return await _create_seed_user()


@pytest.fixture
async def other_user():
    return await _create_seed_user()


def _marketplace_workflow(listing_id: str) -> list[RaiseAttachment]:
    return [RaiseAttachment(kind="workflow", source="marketplace", id=listing_id)]


def _library_workflow(library_agent_id: str) -> list[RaiseAttachment]:
    return [RaiseAttachment(kind="workflow", source="library", id=library_agent_id)]


def _library_skill(slug: str) -> list[RaiseAttachment]:
    return [RaiseAttachment(kind="skill", source="library", id=slug)]


def _marketplace_skill(listing_id: str) -> list[RaiseAttachment]:
    return [RaiseAttachment(kind="skill", source="marketplace", id=listing_id)]


async def _create_seed_user():
    suffix = uuid.uuid4().hex[:8]
    return await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"expert-seed-{suffix}@example.com",
            "name": "Seed Owner",
        }
    )


async def _seed_store_listing(server: SpinTestServer, approved: bool = True) -> str:
    """Create a graph plus a store listing on top of it.

    Returns the StoreListingVersion ID, ready for
    ``add_store_agent_to_library``. With ``approved=False`` the version is
    left in its submitted PENDING state. Mirrors the seeding pattern from
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

    if approved:
        await server.agent_server.test_review_store_listing(
            store_model.ReviewSubmissionRequest(
                store_listing_version_id=slv_id,
                is_approved=True,
                comments="seed",
            ),
            user_id=admin.id,
        )
    return slv_id


async def _load_roster_store_assets() -> dict[str, str]:
    """Load the checked-in production store assets (StoreAgent_rows.csv plus
    the matching graph JSONs) for every ROSTER preload slug into the test DB,
    published under the official creator — the exact data ``load-store-agents``
    deploys. Idempotent: the loaders skip rows that already exist.

    Returns slug -> the CSV's StoreListingVersion id, the version a hire is
    expected to install. A ROSTER slug with no checked-in asset fails here
    instead of being silently substituted by a synthetic listing.
    """
    await store_assets.create_user_and_profile(db_client)
    metadata = await store_assets.load_csv_metadata()
    by_slug = {m["slug"]: m for m in metadata.values() if m["is_available"]}
    expected: dict[str, str] = {}
    for slug in EXPECTED_ROSTER_PRELOAD_SLUGS:
        assert slug in by_slug, f"Expected roster slug '{slug}' has no store asset"
        meta = by_slug[slug]
        version_id = meta["store_listing_version_id"]
        agent_json = await store_assets.load_agent_json(
            store_assets.AGENTS_DIR / f"agent_{version_id}.json"
        )
        graph_id, graph_version = await store_assets.create_agent_graph(
            db_client, agent_json, set()
        )
        await store_assets.create_store_listing(
            db_client, graph_id, graph_version, meta
        )
        expected[slug] = version_id
    return expected


async def _hire_roster_and_assert_preloads(
    hire_user: User,
    templates: dict[str, prisma.models.Expert],
    expected: dict[str, str],
) -> dict[str, HireResult]:
    scheduler = AsyncMock()
    scheduler.add_execution_schedule = AsyncMock(
        return_value=SimpleNamespace(id="sched-1")
    )
    results: dict[str, HireResult] = {}
    with patch.object(scheduling, "get_scheduler_client", return_value=scheduler):
        for entry in seed.ROSTER:
            result = await experts_db.hire_expert(
                hire_user.id, templates[entry["name"]].id, None
            )
            assert result.failed_preloads == []
            assert {w.store_listing_version_id for w in result.expert.workflows} == {
                expected[p["slug"]] for p in entry["preloads"]
            }
            results[entry["name"]] = result
    return results


async def _transfer_listing_to_official_creator(slv_id: str) -> None:
    """Re-own an ad-hoc test listing to the official creator so seed slug
    resolution (creator-scoped) can see it."""
    await store_assets.create_user_and_profile(db_client)
    listing = await prisma.models.StoreListing.prisma().find_first(
        where={"activeVersionId": slv_id}
    )
    assert listing is not None
    await prisma.models.StoreListing.prisma().update(
        where={"id": listing.id},
        data={"owningUserId": store_assets.AUTOGPT_USER_ID},
    )


async def _seed_template(
    name: str,
    preload_listings: list[str],
    preload_crons: dict[str, str] | None = None,
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
            data={
                "expertId": template.id,
                "storeListingVersionId": slv_id,
                "scheduleCron": (preload_crons or {}).get(slv_id),
            }
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
async def test_hire_expert_is_idempotent_at_active_cap(server: SpinTestServer):
    owner = await _create_seed_user()
    template = await _seed_template(name="Maria", preload_listings=[])
    first = await experts_db.hire_expert(owner.id, template.id, None)
    await prisma.models.Expert.prisma().create_many(
        data=[
            {
                "ownerUserId": owner.id,
                "name": f"Filler {i}",
                "role": "",
                "identity": f"I'm Filler {i}.",
            }
            for i in range(experts_db.ACTIVE_EXPERT_LIMIT - 1)
        ]
    )

    second = await experts_db.hire_expert(owner.id, template.id, None)

    assert second.expert.id == first.expert.id


@pytest.mark.asyncio(loop_scope="session")
async def test_hire_expert_enforces_active_expert_cap(server: SpinTestServer):
    owner = await _create_seed_user()
    template = await _seed_template(name="Maria", preload_listings=[])
    await prisma.models.Expert.prisma().create_many(
        data=[
            {
                "ownerUserId": owner.id,
                "name": f"Filler {i}",
                "role": "",
                "identity": f"I'm Filler {i}.",
            }
            for i in range(experts_db.ACTIVE_EXPERT_LIMIT)
        ]
    )

    with pytest.raises(experts_db.ExpertLimitExceededError):
        await experts_db.hire_expert(owner.id, template.id, None)

    assert (
        await prisma.models.Expert.prisma().count(
            where={"ownerUserId": owner.id, "sourceTemplateId": template.id}
        )
        == 0
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_hire_and_raise_share_active_expert_cap(server: SpinTestServer):
    owner = await _create_seed_user()
    template = await _seed_template(name="Maria", preload_listings=[])
    await prisma.models.Expert.prisma().create_many(
        data=[
            {
                "ownerUserId": owner.id,
                "name": f"Filler {i}",
                "role": "",
                "identity": f"I'm Filler {i}.",
            }
            for i in range(experts_db.ACTIVE_EXPERT_LIMIT - 1)
        ]
    )

    results = await asyncio.gather(
        experts_db.hire_expert(owner.id, template.id, None),
        experts_db.create_raised_expert(owner.id, "Nova", None, None),
        return_exceptions=True,
    )

    assert sum(not isinstance(result, BaseException) for result in results) == 1
    assert (
        sum(
            isinstance(result, experts_db.ExpertLimitExceededError)
            for result in results
        )
        == 1
    )
    assert (
        await prisma.models.Expert.prisma().count(
            where={
                "ownerUserId": owner.id,
                "isTemplate": False,
                "isArchived": False,
            }
        )
        == experts_db.ACTIVE_EXPERT_LIMIT
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_creates_blank_owned_expert(server: SpinTestServer):
    owner = await _create_seed_user()
    raised = await experts_db.create_raised_expert(
        owner.id,
        name="Otto",
        role=None,
        voice_preferences=None,
    )
    assert not raised.expert.is_template
    assert raised.expert.source_template_id is None
    assert raised.expert.name == "Otto"
    assert "Otto" in raised.expert.identity
    assert raised.expert.workflows == []
    assert raised.failed_attachments == []
    assert raised.expert.id in {e.id for e in await experts_db.list_experts(owner.id)}


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_persists_avatar_and_color(server: SpinTestServer):
    owner = await _create_seed_user()
    raised = await experts_db.create_raised_expert(
        owner.id,
        name="Nova",
        role=None,
        voice_preferences=None,
        avatar_url="https://storage.googleapis.com/bucket/nova.png",
        color="sky-300",
    )
    assert raised.expert.avatar_url == "https://storage.googleapis.com/bucket/nova.png"
    assert raised.expert.color == "sky-300"

    reloaded = await experts_db.get_expert(owner.id, raised.expert.id)
    assert reloaded is not None
    assert reloaded.avatar_url == "https://storage.googleapis.com/bucket/nova.png"
    assert reloaded.color == "sky-300"


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_stores_about_as_identity(server: SpinTestServer):
    owner = await _create_seed_user()
    raised = await experts_db.create_raised_expert(
        owner.id,
        name="Nova",
        role=None,
        voice_preferences=None,
        about="Keeps replies short and always cites a source.",
    )
    assert raised.expert.identity == "Keeps replies short and always cites a source."

    reloaded = await experts_db.get_expert(owner.id, raised.expert.id)
    assert reloaded is not None
    assert reloaded.identity == "Keeps replies short and always cites a source."


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_falls_back_to_default_identity_without_about(
    server: SpinTestServer,
):
    owner = await _create_seed_user()
    raised = await experts_db.create_raised_expert(
        owner.id,
        name="Otto",
        role=None,
        voice_preferences=None,
    )
    assert raised.expert.identity == experts_db._raised_identity("Otto")


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_defaults_avatar_and_color_when_omitted(
    server: SpinTestServer,
):
    owner = await _create_seed_user()
    raised = await experts_db.create_raised_expert(
        owner.id,
        name="Otto",
        role=None,
        voice_preferences=None,
    )
    assert raised.expert.avatar_url is None
    assert raised.expert.color == ""


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_allows_multiple_per_owner(server: SpinTestServer):
    owner = await _create_seed_user()
    first = await experts_db.create_raised_expert(owner.id, "Otto", None, None)
    second = await experts_db.create_raised_expert(owner.id, "Nova", None, None)
    assert first.expert.id != second.expert.id
    owned = {e.id for e in await experts_db.list_experts(owner.id)}
    assert {first.expert.id, second.expert.id} <= owned


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_is_scoped_to_owner(server: SpinTestServer, other_user):
    owner = await _create_seed_user()
    raised = await experts_db.create_raised_expert(owner.id, "Otto", None, None)
    assert await experts_db.get_expert(other_user.id, raised.expert.id) is None
    assert await experts_db.get_expert(owner.id, raised.expert.id) is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_installs_marketplace_workflow(server: SpinTestServer):
    owner = await _create_seed_user()
    slv_id = await _seed_store_listing(server)
    raised = await experts_db.create_raised_expert(
        owner.id,
        name="Nova",
        role="Research Assistant",
        voice_preferences="Warm and detailed.",
        attachments=_marketplace_workflow(slv_id),
    )
    assert raised.expert.role == "Research Assistant"
    assert raised.expert.voice_preferences == "Warm and detailed."
    assert raised.failed_attachments == []
    assert len(raised.expert.workflows) == 1
    assert raised.expert.workflows[0].store_listing_version_id == slv_id


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_restores_existing_first_job_library_agent(
    server: SpinTestServer,
):
    owner = await _create_seed_user()
    slv_id = await _seed_store_listing(server)
    existing = await library_db.add_store_agent_to_library(slv_id, owner.id)
    existing_row = await prisma.models.LibraryAgent.prisma().find_unique(
        where={"id": existing.id}
    )
    assert existing_row is not None
    original_tenancy = (existing_row.organizationId, existing_row.teamId)
    await prisma.models.LibraryAgent.prisma().update(
        where={"id": existing.id},
        data={"isDeleted": True, "isArchived": True},
    )

    raised = await experts_db.create_raised_expert(
        owner.id, "Nova", None, None, attachments=_marketplace_workflow(slv_id)
    )

    restored = await prisma.models.LibraryAgent.prisma().find_unique(
        where={"id": existing.id}
    )
    assert restored is not None
    assert restored.isDeleted is False
    assert restored.isArchived is False
    assert (restored.organizationId, restored.teamId) == original_tenancy
    assert raised.expert.workflows[0].library_agent_id == existing.id


@pytest.mark.asyncio(loop_scope="session")
async def test_first_job_install_rolls_back_library_agent_on_link_race(
    server: SpinTestServer,
):
    """A concurrent raise already attached this listing, so the link insert
    loses the unique constraint: the transaction — including the library agent
    it created — rolls back, and the caller still sees success because the
    workflow is attached."""
    owner = await _create_seed_user()
    slv_id = await _seed_store_listing(server)
    expert = await prisma.models.Expert.prisma().create(
        data={
            "ownerUserId": owner.id,
            "name": "Nova",
            "role": "",
            "identity": experts_db._raised_identity("Nova"),
        }
    )
    await prisma.models.ExpertWorkflow.prisma().create(
        data={
            "expertId": expert.id,
            "storeListingVersionId": slv_id,
        }
    )

    await experts_db._install_first_job(owner.id, expert.id, slv_id)

    listing = await prisma.models.StoreListingVersion.prisma().find_unique(
        where={"id": slv_id}, include={"AgentGraph": True}
    )
    assert listing is not None
    assert listing.AgentGraph is not None
    assert (
        await prisma.models.LibraryAgent.prisma().count(
            where={
                "userId": owner.id,
                "agentGraphId": listing.AgentGraph.id,
                "agentGraphVersion": listing.AgentGraph.version,
            }
        )
        == 0
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_reports_failed_first_job(server: SpinTestServer):
    owner = await _create_seed_user()
    slv_id = await _seed_store_listing(server)
    with patch.object(
        experts_db.library_db,
        "add_store_agent_to_library_in_transaction",
        new_callable=AsyncMock,
        side_effect=RuntimeError("install exploded"),
    ):
        raised = await experts_db.create_raised_expert(
            owner.id, "Otto", None, None, attachments=_marketplace_workflow(slv_id)
        )
    assert not raised.expert.is_template
    assert raised.expert.workflows == []
    assert len(raised.failed_attachments) == 1
    assert raised.failed_attachments[0].reason == "installation_failed"


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_reports_vanished_graph_as_unavailable(
    server: SpinTestServer,
):
    """A graph that disappears during install is 'unavailable', not a failure."""
    owner = await _create_seed_user()
    slv_id = await _seed_store_listing(server)
    with patch.object(
        experts_db.library_db,
        "add_store_agent_to_library_in_transaction",
        new_callable=AsyncMock,
        side_effect=NotFoundError("Graph #x v1 not found or accessible"),
    ):
        raised = await experts_db.create_raised_expert(
            owner.id, "Otto", None, None, attachments=_marketplace_workflow(slv_id)
        )
    assert raised.expert.workflows == []
    assert len(raised.failed_attachments) == 1
    assert raised.failed_attachments[0].reason == "unavailable"


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_handles_braces_in_name(server: SpinTestServer):
    owner = await _create_seed_user()
    raised = await experts_db.create_raised_expert(owner.id, "a{b", None, None)
    assert raised.expert.name == "a{b"
    assert "a{b" in raised.expert.identity


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_degrades_when_first_job_vanishes_mid_flight(
    server: SpinTestServer,
):
    """A real withdrawal between precheck and locked install is rejected."""
    owner = await _create_seed_user()
    slv_id = await _seed_store_listing(server)
    create_row = experts_db._create_raised_expert_row

    async def create_then_withdraw(
        user_id: str,
        name: str,
        role: str | None,
        voice_preferences: str | None,
        **kwargs,
    ) -> prisma.models.Expert:
        expert = await create_row(user_id, name, role, voice_preferences, **kwargs)
        await prisma.models.StoreListingVersion.prisma().update(
            where={"id": slv_id}, data={"isAvailable": False}
        )
        return expert

    with patch.object(
        experts_db,
        "_create_raised_expert_row",
        new=create_then_withdraw,
    ):
        raised = await experts_db.create_raised_expert(
            owner.id, "Otto", None, None, attachments=_marketplace_workflow(slv_id)
        )
    assert raised.expert.workflows == []
    assert len(raised.failed_attachments) == 1
    assert raised.failed_attachments[0].reason == "unavailable"
    assert (
        await prisma.models.ExpertWorkflow.prisma().count(
            where={"expertId": raised.expert.id}
        )
        == 0
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_rejects_unapproved_first_job(server: SpinTestServer):
    owner = await _create_seed_user()
    pending_slv_id = await _seed_store_listing(server, approved=False)

    with pytest.raises(experts_db.FirstJobUnavailableError):
        await experts_db.create_raised_expert(
            owner.id,
            "Otto",
            None,
            None,
            attachments=_marketplace_workflow(pending_slv_id),
        )

    assert await experts_db.list_experts(owner.id) == []


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_rejects_withdrawn_first_job(server: SpinTestServer):
    owner = await _create_seed_user()
    withdrawn_slv_id = await _seed_store_listing(server)
    await prisma.models.StoreListingVersion.prisma().update(
        where={"id": withdrawn_slv_id}, data={"isAvailable": False}
    )

    with pytest.raises(experts_db.FirstJobUnavailableError):
        await experts_db.create_raised_expert(
            owner.id,
            "Otto",
            None,
            None,
            attachments=_marketplace_workflow(withdrawn_slv_id),
        )

    assert await experts_db.list_experts(owner.id) == []


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_rejects_pending_version_of_approved_graph(
    server: SpinTestServer,
):
    owner = await _create_seed_user()
    approved_slv_id = await _seed_store_listing(server)
    approved = await prisma.models.StoreListingVersion.prisma().find_unique(
        where={"id": approved_slv_id}
    )
    assert approved is not None
    pending = await prisma.models.StoreListingVersion.prisma().create(
        data={
            "version": approved.version + 1,
            "agentGraphId": approved.agentGraphId,
            "agentGraphVersion": approved.agentGraphVersion,
            "name": approved.name,
            "subHeading": approved.subHeading,
            "videoUrl": approved.videoUrl,
            "agentOutputDemoUrl": approved.agentOutputDemoUrl,
            "imageUrls": approved.imageUrls,
            "description": approved.description,
            "instructions": approved.instructions,
            "categories": approved.categories,
            "submissionStatus": prisma.enums.SubmissionStatus.PENDING,
            "storeListingId": approved.storeListingId,
        }
    )

    with pytest.raises(experts_db.FirstJobUnavailableError):
        await experts_db.create_raised_expert(
            owner.id, "Otto", None, None, attachments=_marketplace_workflow(pending.id)
        )

    assert await experts_db.list_experts(owner.id) == []


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_persists_weekly_budget(server: SpinTestServer):
    owner = await _create_seed_user()
    raised = await experts_db.create_raised_expert(
        owner.id,
        name="Otto",
        role=None,
        voice_preferences=None,
        weekly_budget=250,
    )
    row = await prisma.models.Expert.prisma().find_unique(
        where={"id": raised.expert.id}
    )
    assert row is not None
    assert row.weeklyBudget == 250
    assert raised.expert.weekly_budget == 250


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_omitted_weekly_budget_uses_platform_default(
    server: SpinTestServer,
):
    owner = await _create_seed_user()
    raised = await experts_db.create_raised_expert(
        owner.id, name="Otto", role=None, voice_preferences=None
    )
    row = await prisma.models.Expert.prisma().find_unique(
        where={"id": raised.expert.id}
    )
    assert row is not None
    assert row.weeklyBudget is None
    assert raised.expert.weekly_budget == scheduling.effective_weekly_budget(row)


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_attaches_library_workflow(server: SpinTestServer):
    owner = await _create_seed_user()
    slv_id = await _seed_store_listing(server)
    library_agent = await library_db.add_store_agent_to_library(slv_id, owner.id)

    raised = await experts_db.create_raised_expert(
        owner.id,
        name="Nova",
        role=None,
        voice_preferences=None,
        attachments=_library_workflow(library_agent.id),
    )

    assert raised.failed_attachments == []
    assert len(raised.expert.workflows) == 1
    assert raised.expert.workflows[0].library_agent_id == library_agent.id


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_rejects_other_users_library_workflow(
    server: SpinTestServer, other_user
):
    owner = await _create_seed_user()
    slv_id = await _seed_store_listing(server)
    library_agent = await library_db.add_store_agent_to_library(slv_id, other_user.id)

    with pytest.raises(experts_db.FirstJobUnavailableError):
        await experts_db.create_raised_expert(
            owner.id,
            "Otto",
            None,
            None,
            attachments=_library_workflow(library_agent.id),
        )
    assert await experts_db.list_experts(owner.id) == []


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_attaches_library_skill(server: SpinTestServer):
    owner = await _create_seed_user()
    skill = SimpleNamespace(name="oauth_flow")
    with (
        patch.object(
            experts_db.raise_attachments,
            "get_default_skill_with_body",
            return_value=None,
        ),
        patch.object(
            experts_db.raise_attachments,
            "read_user_skill_with_body",
            new_callable=AsyncMock,
            return_value=skill,
        ),
    ):
        raised = await experts_db.create_raised_expert(
            owner.id,
            name="Nova",
            role=None,
            voice_preferences=None,
            attachments=_library_skill("oauth_flow"),
        )
    assert raised.expert.skills == ["oauth_flow"]
    assert raised.failed_attachments == []


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_rejects_missing_library_skill(server: SpinTestServer):
    owner = await _create_seed_user()
    with (
        patch.object(
            experts_db.raise_attachments,
            "get_default_skill_with_body",
            return_value=None,
        ),
        patch.object(
            experts_db.raise_attachments,
            "read_user_skill_with_body",
            new_callable=AsyncMock,
            return_value=None,
        ),
        pytest.raises(experts_db.FirstJobUnavailableError),
    ):
        await experts_db.create_raised_expert(
            owner.id,
            "Otto",
            None,
            None,
            attachments=_library_skill("missing_skill"),
        )
    assert await experts_db.list_experts(owner.id) == []


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_attaches_marketplace_skill_name(server: SpinTestServer):
    owner = await _create_seed_user()
    slv_id = await _seed_store_listing(server)
    listing = await prisma.models.StoreListingVersion.prisma().find_unique(
        where={"id": slv_id}
    )
    assert listing is not None

    raised = await experts_db.create_raised_expert(
        owner.id,
        name="Nova",
        role=None,
        voice_preferences=None,
        attachments=_marketplace_skill(slv_id),
    )
    assert raised.expert.skills == [listing.name]
    assert raised.expert.workflows == []


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_enforces_active_expert_cap(server: SpinTestServer):
    owner = await _create_seed_user()
    await prisma.models.Expert.prisma().create_many(
        data=[
            {
                "ownerUserId": owner.id,
                "name": f"Filler {i}",
                "role": "",
                "identity": f"I'm Filler {i}.",
            }
            for i in range(experts_db.ACTIVE_EXPERT_LIMIT)
        ]
    )

    with pytest.raises(experts_db.ExpertLimitExceededError):
        await experts_db.create_raised_expert(owner.id, "One Too Many", None, None)

    filler = await prisma.models.Expert.prisma().find_first(
        where={"ownerUserId": owner.id}
    )
    assert filler is not None
    await experts_db.archive_expert(owner.id, filler.id)

    raised = await experts_db.create_raised_expert(owner.id, "Fits Now", None, None)
    assert raised.expert.name == "Fits Now"


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_serializes_concurrent_cap_checks(server: SpinTestServer):
    owner = await _create_seed_user()
    await prisma.models.Expert.prisma().create_many(
        data=[
            {
                "ownerUserId": owner.id,
                "name": f"Filler {i}",
                "role": "",
                "identity": f"I'm Filler {i}.",
            }
            for i in range(experts_db.ACTIVE_EXPERT_LIMIT - 1)
        ]
    )

    results = await asyncio.gather(
        experts_db.create_raised_expert(owner.id, "Alpha", None, None),
        experts_db.create_raised_expert(owner.id, "Beta", None, None),
        return_exceptions=True,
    )

    assert sum(not isinstance(result, BaseException) for result in results) == 1
    assert (
        sum(
            isinstance(result, experts_db.ExpertLimitExceededError)
            for result in results
        )
        == 1
    )
    assert (
        await prisma.models.Expert.prisma().count(
            where={
                "ownerUserId": owner.id,
                "isTemplate": False,
                "isArchived": False,
            }
        )
        == experts_db.ACTIVE_EXPERT_LIMIT
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_locks_are_independent_per_owner(server: SpinTestServer):
    first_owner = await _create_seed_user()
    second_owner = await _create_seed_user()
    first_lock_acquired = asyncio.Event()
    second_lock_acquired = asyncio.Event()
    release_first = asyncio.Event()
    lock_creation = experts_db._lock_expert_creation

    async def lock_and_hold_first(tx: prisma.Prisma, user_id: str) -> None:
        await lock_creation(tx, user_id)
        if user_id == first_owner.id:
            first_lock_acquired.set()
            await release_first.wait()
        elif user_id == second_owner.id:
            second_lock_acquired.set()

    with patch.object(experts_db, "_lock_expert_creation", new=lock_and_hold_first):
        first = asyncio.create_task(
            experts_db.create_raised_expert(first_owner.id, "Alpha", None, None)
        )
        await asyncio.wait_for(first_lock_acquired.wait(), timeout=5)
        second = asyncio.create_task(
            experts_db.create_raised_expert(second_owner.id, "Beta", None, None)
        )
        try:
            await asyncio.wait_for(second_lock_acquired.wait(), timeout=5)
        finally:
            release_first.set()
        first_result, second_result = await asyncio.gather(first, second)

    assert first_result.expert.name == "Alpha"
    assert second_result.expert.name == "Beta"


@pytest.mark.asyncio(loop_scope="session")
async def test_raise_expert_enforces_lifetime_cap(server: SpinTestServer):
    owner = await _create_seed_user()
    await prisma.models.Expert.prisma().create_many(
        data=[
            {
                "ownerUserId": owner.id,
                "name": f"Archived {i}",
                "role": "",
                "identity": f"I'm Archived {i}.",
                "isArchived": True,
            }
            for i in range(experts_db.LIFETIME_RAISED_EXPERT_LIMIT)
        ]
    )

    with pytest.raises(experts_db.RaisedExpertLifetimeLimitExceededError):
        await experts_db.create_raised_expert(owner.id, "One Too Many", None, None)


@pytest.mark.asyncio(loop_scope="session")
async def test_hired_experts_do_not_consume_raised_lifetime_cap(
    server: SpinTestServer,
):
    owner = await _create_seed_user()
    template = await _seed_template(name="Maria", preload_listings=[])
    await experts_db.hire_expert(owner.id, template.id, None)

    with patch.object(experts_db, "LIFETIME_RAISED_EXPERT_LIMIT", 1):
        raised = await experts_db.create_raised_expert(owner.id, "Nova", None, None)

    assert raised.expert.source_template_id is None


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
async def test_rehire_archived_expert_respects_active_cap(server: SpinTestServer):
    owner = await _create_seed_user()
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(owner.id, template.id, None)
    await experts_db.archive_expert(owner.id, hired.expert.id)
    await prisma.models.Expert.prisma().create_many(
        data=[
            {
                "ownerUserId": owner.id,
                "name": f"Filler {i}",
                "role": "",
                "identity": f"I'm Filler {i}.",
            }
            for i in range(experts_db.ACTIVE_EXPERT_LIMIT)
        ]
    )

    with pytest.raises(experts_db.ExpertLimitExceededError):
        await experts_db.hire_expert(owner.id, template.id, None)

    archived = await prisma.models.Expert.prisma().find_unique(
        where={"id": hired.expert.id}
    )
    assert archived is not None
    assert archived.isArchived


@pytest.mark.asyncio(loop_scope="session")
async def test_list_expert_identities_is_lightweight_and_includes_archived(
    server: SpinTestServer, test_user, other_user
):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    active_template = await _seed_template(name="Max", preload_listings=[])
    active_hired = await experts_db.hire_expert(test_user.id, active_template.id, None)
    other_hired = await experts_db.hire_expert(other_user.id, template.id, None)
    await experts_db.archive_expert(test_user.id, hired.expert.id)

    with (
        patch.object(experts_db, "_latest_runs", new_callable=AsyncMock) as latest_runs,
        patch.object(
            experts_db, "_weekly_spends", new_callable=AsyncMock
        ) as weekly_spend,
    ):
        identities = await experts_db.list_expert_identities(test_user.id)

    identity_ids = {item.id for item in identities}
    identity = next(item for item in identities if item.id == hired.expert.id)
    assert identity.name == hired.expert.name
    assert identity.is_archived is True
    active_identity = next(
        item for item in identities if item.id == active_hired.expert.id
    )
    assert active_identity.is_archived is False
    assert template.id not in identity_ids
    assert active_template.id not in identity_ids
    assert other_hired.expert.id not in identity_ids
    latest_runs.assert_not_awaited()
    weekly_spend.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="session")
async def test_list_experts_reads_weekly_spend_for_each_expert(
    server: SpinTestServer, test_user
):
    hired = [
        await experts_db.hire_expert(
            test_user.id,
            (await _seed_template(name=f"Expert{i}", preload_listings=[])).id,
            None,
        )
        for i in range(3)
    ]
    spends = {h.expert.id: (i + 1) * 100 for i, h in enumerate(hired)}

    with patch.object(
        experts_db,
        "_weekly_spends",
        new=AsyncMock(return_value=spends),
    ):
        experts = await experts_db.list_experts(test_user.id)

    assert {e.id: e.weekly_spend for e in experts if e.id in spends} == spends


@pytest.mark.asyncio(loop_scope="session")
async def test_list_experts_defaults_a_missing_spend_entry_to_zero(
    server: SpinTestServer, test_user
):
    """A spend read that degraded to a missing key must not break the roster."""
    hired = [
        await experts_db.hire_expert(
            test_user.id,
            (await _seed_template(name=f"Gap{i}", preload_listings=[])).id,
            None,
        )
        for i in range(2)
    ]
    present, missing = hired[0].expert.id, hired[1].expert.id

    with patch.object(
        experts_db,
        "_weekly_spends",
        new=AsyncMock(return_value={present: 700}),
    ):
        experts = await experts_db.list_experts(test_user.id)

    by_id = {e.id: e.weekly_spend for e in experts}
    assert by_id[present] == 700
    assert by_id[missing] == 0


def test_expert_identity_projection_columns_exist_in_schema():
    """Guard the hand-written projection in ``list_expert_identities``.

    The raw SQL aliases physical column names, so a ``schema.prisma`` rename
    would otherwise only surface as a runtime query error.
    """
    schema = (Path(__file__).parents[4] / "schema.prisma").read_text()
    model = re.search(r"^model Expert \{(.*?)^\}", schema, re.S | re.M)
    assert model is not None, "Expert model not found in schema.prisma"
    fields = set(re.findall(r"^\s{2}(\w+)", model.group(1), re.M))
    assert {"id", "name", "avatarUrl", "role", "isArchived"} <= fields
    assert {"ownerUserId", "isTemplate"} <= fields


@pytest.mark.asyncio
async def test_weekly_spends_degrades_an_unexpected_read_failure_to_zero():
    async def read(expert_id: str) -> int:
        if expert_id == "expert-bad":
            raise RuntimeError("redis read failed")
        return 125

    with patch.object(experts_db, "get_weekly_spend", side_effect=read):
        spends = await experts_db._weekly_spends(["expert-ok", "expert-bad"])

    assert spends == {"expert-ok": 125, "expert-bad": 0}


@pytest.mark.asyncio
async def test_weekly_spends_limits_concurrent_reads():
    active_reads = 0
    peak_reads = 0

    async def read(_: str) -> int:
        nonlocal active_reads, peak_reads
        active_reads += 1
        peak_reads = max(peak_reads, active_reads)
        await asyncio.sleep(0)
        active_reads -= 1
        return 125

    expert_ids = [f"expert-{index}" for index in range(25)]
    with patch.object(experts_db, "get_weekly_spend", side_effect=read):
        spends = await experts_db._weekly_spends(expert_ids)

    assert peak_reads == experts_db._WEEKLY_SPEND_READ_CONCURRENCY
    assert spends == {expert_id: 125 for expert_id in expert_ids}


@pytest.mark.asyncio(loop_scope="session")
async def test_owns_active_expert_scopes_owner_and_archive_state(
    server: SpinTestServer, test_user, other_user
):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)

    assert await experts_db.owns_active_expert(test_user.id, hired.expert.id)
    assert not await experts_db.owns_active_expert(test_user.id, template.id)
    assert not await experts_db.owns_active_expert(other_user.id, hired.expert.id)

    await experts_db.archive_expert(test_user.id, hired.expert.id)
    assert not await experts_db.owns_active_expert(test_user.id, hired.expert.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_expert_rejects_cross_user(
    server: SpinTestServer, test_user, other_user
):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)

    with pytest.raises(experts_db.ExpertNotFoundError):
        await experts_db.archive_expert(other_user.id, hired.expert.id)

    assert await experts_db.owns_active_expert(test_user.id, hired.expert.id)


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
    assert await experts_db.get_expert(test_user.id, template.id) is None
    assert await experts_db.get_expert(other_user.id, hired.expert.id) is None
    assert await experts_db.get_expert(test_user.id, hired.expert.id) is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_get_expert_excludes_archived_experts(server: SpinTestServer, test_user):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    await experts_db.archive_expert(test_user.id, hired.expert.id)

    assert await experts_db.get_expert(test_user.id, hired.expert.id) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_get_expert_only_reads_private_owner_scope():
    find_first = AsyncMock(return_value=None)
    manager = SimpleNamespace(find_first=find_first)
    with patch.object(prisma.models.Expert, "prisma", return_value=manager):
        assert await experts_db.get_expert("owner-1", "shared-expert") is None

    find_first.assert_awaited_once_with(
        where={
            "id": "shared-expert",
            "ownerUserId": "owner-1",
            "isTemplate": False,
            "isArchived": False,
            "visibility": prisma.enums.ResourceVisibility.PRIVATE,
        },
        include=experts_db._WORKFLOW_INCLUDE,
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_existing_non_private_hire_is_never_revived():
    """The reservation must fail closed on a non-PRIVATE existing row —
    no revive, no create."""
    shared = SimpleNamespace(
        id="shared-expert",
        visibility=prisma.enums.ResourceVisibility.TEAM,
        isArchived=True,
    )
    tx = SimpleNamespace(
        execute_raw=AsyncMock(),
        expert=SimpleNamespace(
            find_first=AsyncMock(return_value=shared),
            update=AsyncMock(),
            create=AsyncMock(),
            count=AsyncMock(return_value=0),
        ),
    )

    @asynccontextmanager
    async def fake_transaction(*args, **kwargs):
        yield tx

    with (
        patch.object(experts_db, "transaction", fake_transaction),
        pytest.raises(experts_db.ExpertNotFoundError),
    ):
        await experts_db._reserve_hired_expert("owner-1", "template-1", {})

    tx.expert.update.assert_not_awaited()
    tx.expert.create.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="session")
async def test_hire_existing_team_expert_fails_closed():
    template = SimpleNamespace(
        id="template-1",
        name="Maria",
        avatarUrl=None,
        color="",
        role="Marketing Specialist",
        tagline=None,
        bio=None,
        skills=[],
        identity="You are Maria.",
        voicePreferences=None,
        boundaries=None,
        toolProfile=None,
        Workflows=[],
    )
    shared = SimpleNamespace(
        id="shared-expert",
        visibility=prisma.enums.ResourceVisibility.TEAM,
        isArchived=False,
    )
    expert_client = SimpleNamespace(find_first=AsyncMock(return_value=template))
    tx = SimpleNamespace(
        execute_raw=AsyncMock(),
        expert=SimpleNamespace(
            find_first=AsyncMock(return_value=shared),
            update=AsyncMock(),
            create=AsyncMock(),
            count=AsyncMock(return_value=0),
        ),
    )

    @asynccontextmanager
    async def fake_transaction(*args, **kwargs):
        yield tx

    with (
        patch.object(prisma.models.Expert, "prisma", return_value=expert_client),
        patch.object(experts_db, "transaction", fake_transaction),
        pytest.raises(experts_db.ExpertNotFoundError) as exc_info,
    ):
        await experts_db.hire_expert("owner-1", "template-1", None)

    assert exc_info.value.expert_id == "shared-expert"
    tx.expert.create.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="session")
async def test_hire_raced_org_expert_fails_closed():
    """Losing the create race to a row that is (now) non-PRIVATE must fail
    closed on the retry instead of returning the shared row."""
    template = SimpleNamespace(
        id="template-1",
        name="Maria",
        avatarUrl=None,
        color="",
        role="Marketing Specialist",
        tagline=None,
        bio=None,
        skills=[],
        identity="You are Maria.",
        voicePreferences=None,
        boundaries=None,
        toolProfile=None,
        Workflows=[],
    )
    raced = SimpleNamespace(
        id="shared-expert",
        visibility=prisma.enums.ResourceVisibility.ORG,
        isArchived=False,
    )
    expert_client = SimpleNamespace(find_first=AsyncMock(return_value=template))
    tx = SimpleNamespace(
        execute_raw=AsyncMock(),
        expert=SimpleNamespace(
            # First reservation: no existing row → create races and loses.
            # Retry reservation: the winner's row is found — and is shared.
            find_first=AsyncMock(side_effect=[None, raced]),
            update=AsyncMock(),
            create=AsyncMock(side_effect=prisma.errors.UniqueViolationError({})),
            count=AsyncMock(return_value=0),
        ),
    )

    @asynccontextmanager
    async def fake_transaction(*args, **kwargs):
        yield tx

    with (
        patch.object(prisma.models.Expert, "prisma", return_value=expert_client),
        patch.object(experts_db, "transaction", fake_transaction),
        pytest.raises(experts_db.ExpertNotFoundError) as exc_info,
    ):
        await experts_db.hire_expert("owner-1", "template-1", None)

    assert exc_info.value.expert_id == "shared-expert"
    tx.expert.create.assert_awaited_once()


@pytest.mark.asyncio(loop_scope="session")
async def test_rehire_missing_private_tenancy_rolls_back_to_archived():
    """The reservation unarchives in-transaction; a missing personal
    workspace must roll the row back to archived and surface as retryable —
    never return a revived expert without a workspace."""
    row = SimpleNamespace(
        id="expert-1",
        ownerUserId="owner-1",
        isArchived=False,
        visibility=prisma.enums.ResourceVisibility.PRIVATE,
    )
    expert_client = SimpleNamespace(update=AsyncMock(), find_unique=AsyncMock())
    with (
        patch.object(prisma.models.Expert, "prisma", return_value=expert_client),
        patch.object(
            experts_db,
            "get_user_default_team",
            new=AsyncMock(return_value=(None, None)),
        ),
        patch.object(
            scheduling, "resume_expert_schedules", new_callable=AsyncMock
        ) as resume,
        patch.object(
            scheduling, "reattach_expert_triggers", new_callable=AsyncMock
        ) as reattach,
        patch.object(
            scheduling, "pause_expert_schedules", new_callable=AsyncMock
        ) as pause,
        patch.object(
            scheduling, "detach_expert_triggers", new_callable=AsyncMock
        ) as detach,
        pytest.raises(experts_db.ExpertPrivateTenancyNotFoundError),
    ):
        await experts_db._resume_revived_hire(row)

    resume.assert_not_awaited()
    reattach.assert_not_awaited()
    pause.assert_awaited_once_with(
        "owner-1", "expert-1", reason="Expert re-hire did not complete"
    )
    expert_client.update.assert_awaited_once_with(
        where={"id": "expert-1"}, data={"isArchived": True}
    )
    detach.assert_awaited_once_with("owner-1", "expert-1")


@pytest.mark.asyncio(loop_scope="session")
async def test_rehire_reattach_failure_restores_archived_state():
    row = SimpleNamespace(
        id="expert-1",
        ownerUserId="owner-1",
        isArchived=False,
        visibility=prisma.enums.ResourceVisibility.PRIVATE,
    )
    expert_client = SimpleNamespace(update=AsyncMock(), find_unique=AsyncMock())
    with (
        patch.object(prisma.models.Expert, "prisma", return_value=expert_client),
        patch.object(
            experts_db,
            "get_user_default_team",
            new=AsyncMock(return_value=("personal-org", "personal-team")),
        ),
        patch.object(
            scheduling,
            "resume_expert_schedules",
            new=AsyncMock(return_value=True),
        ),
        patch.object(
            scheduling,
            "reattach_expert_triggers",
            new=AsyncMock(side_effect=RuntimeError("scheduler unavailable")),
        ),
        patch.object(
            scheduling, "pause_expert_schedules", new_callable=AsyncMock
        ) as pause,
        patch.object(
            scheduling, "detach_expert_triggers", new_callable=AsyncMock
        ) as detach,
        pytest.raises(experts_db.ExpertHireUnavailableError) as exc_info,
    ):
        await experts_db._resume_revived_hire(row)

    assert exc_info.value.expert_id == "expert-1"
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    expert_client.update.assert_awaited_once_with(
        where={"id": "expert-1"}, data={"isArchived": True}
    )
    pause.assert_awaited_once_with(
        "owner-1", "expert-1", reason="Expert re-hire did not complete"
    )
    detach.assert_awaited_once_with("owner-1", "expert-1")


@pytest.mark.asyncio(loop_scope="session")
async def test_resolve_private_expert_tenancy_uses_owner_personal_scope():
    find_first = AsyncMock(return_value=SimpleNamespace(id="expert-1"))
    manager = SimpleNamespace(find_first=find_first)
    lookup = AsyncMock(return_value=("personal-org", "personal-team"))

    with (
        patch.object(prisma.models.Expert, "prisma", return_value=manager),
        patch.object(experts_db, "get_user_default_team", lookup),
    ):
        result = await experts_db.resolve_private_expert_tenancy("owner-1", "expert-1")

    assert result == ("personal-org", "personal-team")
    find_first.assert_awaited_once_with(
        where={
            "id": "expert-1",
            "ownerUserId": "owner-1",
            "isTemplate": False,
            "isArchived": False,
            "visibility": prisma.enums.ResourceVisibility.PRIVATE,
        }
    )
    lookup.assert_awaited_once_with("owner-1")


@pytest.mark.asyncio(loop_scope="session")
async def test_resolve_private_expert_tenancy_rejects_unsupported_experts():
    find_first = AsyncMock(side_effect=[None, None, None, None, None])
    manager = SimpleNamespace(find_first=find_first)
    lookup = AsyncMock(return_value=("org-should-not-leak", "team-should-not-leak"))
    with (
        patch.object(prisma.models.Expert, "prisma", return_value=manager),
        patch.object(experts_db, "get_user_default_team", lookup),
    ):
        for rejected_id in (
            "other-owners-expert",
            "template",
            "archived",
            "team-expert",
            "org-expert",
        ):
            with pytest.raises(experts_db.ExpertNotFoundError):
                await experts_db.resolve_private_expert_tenancy("attacker", rejected_id)

    lookup.assert_not_awaited()
    assert all(
        call.kwargs["where"]["ownerUserId"] == "attacker"
        and call.kwargs["where"]["isTemplate"] is False
        and call.kwargs["where"]["isArchived"] is False
        and call.kwargs["where"]["visibility"]
        == prisma.enums.ResourceVisibility.PRIVATE
        for call in find_first.await_args_list
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_resolve_private_expert_tenancy_fails_without_personal_org():
    manager = SimpleNamespace(
        find_first=AsyncMock(return_value=SimpleNamespace(id="expert-1"))
    )
    with (
        patch.object(prisma.models.Expert, "prisma", return_value=manager),
        patch.object(
            experts_db,
            "get_user_default_team",
            new_callable=AsyncMock,
            return_value=(None, None),
        ),
        pytest.raises(experts_db.ExpertPrivateTenancyNotFoundError),
    ):
        await experts_db.resolve_private_expert_tenancy("owner-1", "expert-1")


@pytest.mark.asyncio(loop_scope="session")
async def test_resolve_private_expert_tenancy_allows_missing_default_team():
    manager = SimpleNamespace(
        find_first=AsyncMock(return_value=SimpleNamespace(id="expert-1"))
    )
    with (
        patch.object(prisma.models.Expert, "prisma", return_value=manager),
        patch.object(
            experts_db,
            "get_user_default_team",
            new_callable=AsyncMock,
            return_value=("personal-org", None),
        ),
    ):
        assert await experts_db.resolve_private_expert_tenancy(
            "owner-1", "expert-1"
        ) == ("personal-org", None)


@pytest.mark.asyncio(loop_scope="session")
async def test_owner_can_update_expert_soul(server: SpinTestServer, test_user):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)

    updated = await experts_db.update_soul(
        test_user.id,
        hired.expert.id,
        ExpertSoulUpdate(
            name="Mara",
            identity="You are Mara, a thoughtful strategist.",
            voice_preferences="Warm, concise, and direct.",
            boundaries="Never invent customer evidence.",
        ),
    )

    assert updated.name == "Mara"
    assert updated.identity == "You are Mara, a thoughtful strategist."
    assert updated.voice_preferences == "Warm, concise, and direct."
    assert updated.boundaries == "Never invent customer evidence."


@pytest.mark.asyncio(loop_scope="session")
async def test_other_user_cannot_update_expert_soul(
    server: SpinTestServer, test_user, other_user
):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)

    with pytest.raises(experts_db.ExpertNotFoundError):
        await experts_db.update_soul(
            other_user.id,
            hired.expert.id,
            ExpertSoulUpdate(
                name="Stolen",
                identity=hired.expert.identity,
                voice_preferences="",
                boundaries="",
            ),
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_templates_and_archived_experts_cannot_update_soul(
    server: SpinTestServer, test_user
):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    await experts_db.archive_expert(test_user.id, hired.expert.id)
    soul = ExpertSoulUpdate(
        name="Mara",
        identity="You are Mara.",
        voice_preferences="Direct.",
        boundaries="Ask before sending.",
    )

    with pytest.raises(experts_db.ExpertNotFoundError):
        await experts_db.update_soul(test_user.id, template.id, soul)
    with pytest.raises(experts_db.ExpertNotFoundError):
        await experts_db.update_soul(test_user.id, hired.expert.id, soul)


@pytest.mark.asyncio(loop_scope="session")
async def test_hire_copies_soul_fields_from_template(server: SpinTestServer, test_user):
    template = await prisma.models.Expert.prisma().create(
        data={
            "name": f"Otto {uuid.uuid4().hex[:8]}",
            "role": "Writer",
            "identity": "You are Otto, a playful writer.",
            "voicePreferences": "Direct, playful, and concise.",
            "boundaries": "Never publish without approval.",
            "isTemplate": True,
        }
    )

    hired = await experts_db.hire_expert(test_user.id, template.id, None)

    assert hired.expert.identity == "You are Otto, a playful writer."
    assert hired.expert.voice_preferences == "Direct, playful, and concise."
    assert hired.expert.boundaries == "Never publish without approval."


@pytest.mark.asyncio(loop_scope="session")
async def test_hire_from_template_with_samples_stores_plain_voice(
    server: SpinTestServer, test_user
):
    """A hire copies the plain description, never the template's sample
    envelope, so a skipped voice pick can't leave raw JSON in the prompt."""
    envelope = encode_voice_preferences(
        "Direct and concise.",
        [
            VoiceSample(label="Punchy", text="Ship it."),
            VoiceSample(label="Warm", text="Let's start with a story."),
        ],
    )
    template = await prisma.models.Expert.prisma().create(
        data={
            "name": f"Vox {uuid.uuid4().hex[:8]}",
            "role": "Writer",
            "identity": "You are Vox.",
            "voicePreferences": envelope,
            "isTemplate": True,
        }
    )

    listed = next(t for t in await experts_db.list_templates() if t.id == template.id)
    assert len(listed.voice_samples) == 2
    assert listed.voice_preferences == "Direct and concise."

    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    assert hired.expert.voice_samples == []
    assert hired.expert.voice_preferences == "Direct and concise."
    assert "{" not in hired.expert.voice_preferences


@pytest.mark.asyncio(loop_scope="session")
async def test_seed_roster_exposes_two_voice_samples_per_template(
    server: SpinTestServer,
):
    await _load_roster_store_assets()
    ids = await seed.seed_roster()
    seeded = {t.name: t for t in await experts_db.list_templates() if t.id in ids}
    for entry in seed.ROSTER:
        template = seeded[entry["name"]]
        assert len(template.voice_samples) == 2
        assert template.voice_preferences == entry["voice_preferences"]


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
async def test_install_workflow_returns_concurrent_winner(
    server: SpinTestServer, test_user
):
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    library_agent = await experts_db.library_db.add_store_agent_to_library(
        slv_id, test_user.id
    )
    winner = await prisma.models.ExpertWorkflow.prisma().create(
        data={
            "expertId": hired.expert.id,
            "storeListingVersionId": slv_id,
            "libraryAgentId": library_agent.id,
        },
        include={"LibraryAgent": True, "StoreListingVersion": True},
    )
    workflow_client = SimpleNamespace(
        find_first=AsyncMock(side_effect=[None, winner]),
        create=AsyncMock(side_effect=prisma.errors.UniqueViolationError({})),
    )

    with patch.object(
        prisma.models.ExpertWorkflow, "prisma", return_value=workflow_client
    ):
        installed = await experts_db.install_workflow(
            test_user.id, hired.expert.id, slv_id
        )

    assert installed.id == winner.id
    assert workflow_client.find_first.await_count == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_install_workflow_reraises_race_without_winner(
    server: SpinTestServer, test_user
):
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    workflow_client = SimpleNamespace(
        find_first=AsyncMock(side_effect=[None, None]),
        create=AsyncMock(side_effect=prisma.errors.UniqueViolationError({})),
    )

    with (
        patch.object(
            prisma.models.ExpertWorkflow, "prisma", return_value=workflow_client
        ),
        pytest.raises(prisma.errors.UniqueViolationError),
    ):
        await experts_db.install_workflow(test_user.id, hired.expert.id, slv_id)

    assert workflow_client.find_first.await_count == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_install_workflow_reuses_library_agent_without_resetting_settings(
    server: SpinTestServer, test_user
):
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    library_agent = await experts_db.library_db.add_store_agent_to_library(
        slv_id, test_user.id
    )
    expected_settings = GraphSettings(
        human_in_the_loop_safe_mode=False,
        sensitive_action_safe_mode=True,
        builder_chat_session_id="builder-session",
    )
    await prisma.models.LibraryAgent.prisma().update(
        where={"id": library_agent.id},
        data={"settings": SafeJson(expected_settings.model_dump())},
    )
    before_count = await prisma.models.LibraryAgent.prisma().count(
        where={"userId": test_user.id}
    )

    installed = await experts_db.install_workflow(test_user.id, hired.expert.id, slv_id)

    persisted = await prisma.models.LibraryAgent.prisma().find_unique(
        where={"id": library_agent.id}
    )
    after_count = await prisma.models.LibraryAgent.prisma().count(
        where={"userId": test_user.id}
    )
    assert persisted is not None
    assert installed.library_agent_id == library_agent.id
    assert after_count == before_count
    assert GraphSettings.model_validate(persisted.settings) == expected_settings


@pytest.mark.asyncio(loop_scope="session")
async def test_install_workflow_restores_archived_deleted_library_agent(
    server: SpinTestServer, test_user
):
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    library_agent = await experts_db.library_db.add_store_agent_to_library(
        slv_id, test_user.id
    )
    expected_settings = GraphSettings(
        human_in_the_loop_safe_mode=False,
        sensitive_action_safe_mode=False,
        builder_chat_session_id="restored-builder-session",
    )
    await prisma.models.LibraryAgent.prisma().update(
        where={"id": library_agent.id},
        data={
            "isArchived": True,
            "isDeleted": True,
            "settings": SafeJson(expected_settings.model_dump()),
        },
    )

    installed = await experts_db.install_workflow(test_user.id, hired.expert.id, slv_id)

    restored = await prisma.models.LibraryAgent.prisma().find_unique(
        where={"id": library_agent.id}
    )
    assert restored is not None
    assert installed.library_agent_id == library_agent.id
    assert not restored.isArchived
    assert not restored.isDeleted
    assert GraphSettings.model_validate(restored.settings) == expected_settings


@pytest.mark.asyncio(loop_scope="session")
async def test_install_workflow_rejects_unapproved_store_version(
    server: SpinTestServer, test_user
):
    slv_id = await _seed_store_listing(server)
    await prisma.models.StoreListingVersion.prisma().update(
        where={"id": slv_id},
        data={"submissionStatus": prisma.enums.SubmissionStatus.DRAFT},
    )
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)

    with pytest.raises(NotFoundError):
        await experts_db.install_workflow(test_user.id, hired.expert.id, slv_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_resolve_expert_for_graph_unique_match(
    server: SpinTestServer, test_user, other_user
):
    """Manually scheduling an expert-installed workflow must keep the
    attribution: a unique (user, graph) → expert match resolves."""
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(name="Maria", preload_listings=[slv_id])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    graph_id = hired.expert.workflows[0].graph_id
    assert graph_id is not None

    resolved = await experts_db.resolve_expert_for_graph(test_user.id, graph_id)
    assert resolved == hired.expert.id
    assert await experts_db.resolve_expert_for_graph(other_user.id, graph_id) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_resolve_expert_for_graph_ambiguous_returns_none(
    server: SpinTestServer, test_user
):
    """Two experts sharing one installed workflow (same LibraryAgent) make
    the join ambiguous — resolution must decline rather than guess."""
    slv_id = await _seed_store_listing(server)
    template_a = await _seed_template(name="Maria", preload_listings=[slv_id])
    template_b = await _seed_template(name="Max", preload_listings=[slv_id])
    hired_a = await experts_db.hire_expert(test_user.id, template_a.id, None)
    await experts_db.hire_expert(test_user.id, template_b.id, None)
    graph_id = hired_a.expert.workflows[0].graph_id
    assert graph_id is not None

    assert await experts_db.resolve_expert_for_graph(test_user.id, graph_id) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_resolve_expert_for_graph_fails_closed_on_non_private_expert(
    server: SpinTestServer, test_user
):
    """A graph mapped to a TEAM/ORG expert must error (mirroring the 404 the
    explicit-id path gives) instead of silently detaching attribution — an
    unattributed run would bypass the expert budget guard entirely."""
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(name="Maria", preload_listings=[slv_id])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    graph_id = hired.expert.workflows[0].graph_id
    assert graph_id is not None
    await prisma.models.Expert.prisma().update(
        where={"id": hired.expert.id},
        data={"visibility": prisma.enums.ResourceVisibility.ORG},
    )

    with pytest.raises(experts_db.ExpertNotFoundError):
        await experts_db.resolve_expert_for_graph(test_user.id, graph_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_expert_row_exists_is_lenient_about_archive_state(
    server: SpinTestServer, test_user, other_user
):
    """The scheduler's recovery check must see archived rows (so schedules
    survive) but not other users' rows or vanished ids."""
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    await experts_db.archive_expert(test_user.id, hired.expert.id)

    assert await experts_db.expert_row_exists(test_user.id, hired.expert.id) is True
    assert await experts_db.expert_row_exists(other_user.id, hired.expert.id) is False
    assert await experts_db.expert_row_exists(test_user.id, "no-such-expert") is False


@pytest.mark.asyncio(loop_scope="session")
async def test_hire_creates_schedule_from_template_cadence(
    server: SpinTestServer, test_user
):
    """A roster preload with a cadence gets its schedule created at hire
    time, attributed to the expert, with the schedule id recorded."""
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(
        name="Frankie",
        preload_listings=[slv_id],
        preload_crons={slv_id: "40 7 * * *"},
    )
    mock_scheduler = AsyncMock()
    mock_scheduler.add_execution_schedule = AsyncMock(
        return_value=SimpleNamespace(id="sched-1")
    )
    with patch.object(scheduling, "get_scheduler_client", return_value=mock_scheduler):
        result = await experts_db.hire_expert(test_user.id, template.id, None)

    wf = result.expert.workflows[0]
    assert wf.schedule_cron == "40 7 * * *"
    assert wf.schedule_id == "sched-1"
    assert result.failed_preloads == []
    call_kwargs = mock_scheduler.add_execution_schedule.call_args.kwargs
    assert call_kwargs["cron"] == "40 7 * * *"
    assert call_kwargs["expert_id"] == result.expert.id


@pytest.mark.asyncio(loop_scope="session")
async def test_hire_schedule_failure_marks_needs_setup(
    server: SpinTestServer, test_user
):
    """Schedule creation failing (e.g. missing credentials) must not sink
    the hire or the install: the cadence is kept, schedule_id stays None."""
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(
        name="Frankie",
        preload_listings=[slv_id],
        preload_crons={slv_id: "40 7 * * *"},
    )
    mock_scheduler = AsyncMock()
    mock_scheduler.add_execution_schedule = AsyncMock(
        side_effect=RuntimeError("graph validation failed")
    )
    with patch.object(scheduling, "get_scheduler_client", return_value=mock_scheduler):
        result = await experts_db.hire_expert(test_user.id, template.id, None)

    wf = result.expert.workflows[0]
    assert wf.library_agent_id is not None
    assert wf.schedule_cron == "40 7 * * *"
    assert wf.schedule_id is None
    assert result.failed_preloads == []


@pytest.mark.asyncio(loop_scope="session")
async def test_attributed_writes_fail_closed_when_archived_after_validation(
    server: SpinTestServer,
):
    """An archive between an earlier lookup and either durable write wins.

    Both ChatSession and AgentPreset creation refuse the stale expert id
    (fail closed) rather than silently persisting detached/attributed work —
    the caller gets a not-found it can surface, not a session or preset in an
    unexpected memory scope.
    """
    slv_id = await _seed_store_listing(server)
    owner = await _create_seed_user()
    template = await _seed_template(name="Maria", preload_listings=[slv_id])
    hired = await experts_db.hire_expert(owner.id, template.id, None)
    expert_id = hired.expert.id
    workflow = hired.expert.workflows[0]
    assert workflow.library_agent_id is not None
    library_agent = await prisma.models.LibraryAgent.prisma().find_unique(
        where={"id": workflow.library_agent_id}
    )
    assert library_agent is not None

    assert (
        await experts_db.resolve_attributable_expert(owner.id, expert_id) == expert_id
    )
    await prisma.models.Expert.prisma().update(
        where={"id": expert_id}, data={"isArchived": True}
    )

    with pytest.raises(experts_db.ExpertNotFoundError):
        await create_chat_session(
            owner.id,
            dry_run=False,
            expert_id=expert_id,
        )

    with pytest.raises(NotFoundError):
        await library_db.create_preset(
            owner.id,
            library_model.LibraryAgentPresetCreatable(
                graph_id=library_agent.agentGraphId,
                graph_version=library_agent.agentGraphVersion,
                inputs={},
                credentials={},
                name="Atomic attribution fallback",
                description="",
            ),
            expert_id=expert_id,
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_pauses_detaches_and_revive_reattaches(
    server: SpinTestServer, test_user
):
    """Archiving must leave no orphaned firing: presets deactivate, schedules
    pause, and the pause is logged. Re-hiring reverses all of it."""
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(
        name="Frankie",
        preload_listings=[slv_id],
        preload_crons={slv_id: "40 7 * * *"},
    )
    sched = AsyncMock()
    sched.add_execution_schedule = AsyncMock(return_value=SimpleNamespace(id="sched-1"))
    sched.pause_schedule = AsyncMock()
    sched.resume_schedule = AsyncMock()

    with patch.object(scheduling, "get_scheduler_client", return_value=sched):
        hired = await experts_db.hire_expert(test_user.id, template.id, None)
        expert_id = hired.expert.id
        wf = hired.expert.workflows[0]
        assert wf.library_agent_id is not None
        library_agent = await prisma.models.LibraryAgent.prisma().find_unique(
            where={"id": wf.library_agent_id}
        )
        assert library_agent is not None
        preset = await prisma.models.AgentPreset.prisma().create(
            data={
                "userId": test_user.id,
                "name": "Email trigger",
                "description": "",
                "agentGraphId": library_agent.agentGraphId,
                "agentGraphVersion": library_agent.agentGraphVersion,
                "expertId": expert_id,
            }
        )
        disabled_preset = await prisma.models.AgentPreset.prisma().create(
            data={
                "userId": test_user.id,
                "name": "Manually disabled trigger",
                "description": "",
                "agentGraphId": library_agent.agentGraphId,
                "agentGraphVersion": library_agent.agentGraphVersion,
                "expertId": expert_id,
                "isActive": False,
            }
        )
        sched.get_execution_schedules = AsyncMock(
            return_value=[
                SimpleNamespace(
                    kind="graph", id="sched-1", name="n", expert_id=expert_id
                )
            ]
        )
        await experts_db.archive_expert(test_user.id, expert_id)

    # Keyword args matter: passed positionally, user_id binds to graph_id
    # and the filter silently matches nothing.
    sched.get_execution_schedules.assert_awaited_with(
        user_id=test_user.id, kind="graph", include_paused=False
    )
    preset_row = await prisma.models.AgentPreset.prisma().find_unique(
        where={"id": preset.id}
    )
    assert preset_row is not None and preset_row.isActive is False
    assert preset_row.deactivatedByExpertArchive is True
    disabled_row = await prisma.models.AgentPreset.prisma().find_unique(
        where={"id": disabled_preset.id}
    )
    assert disabled_row is not None and disabled_row.isActive is False
    assert disabled_row.deactivatedByExpertArchive is False
    expert_row = await prisma.models.Expert.prisma().find_unique(
        where={"id": expert_id}
    )
    assert expert_row is not None and expert_row.schedulesPausedAt is not None
    events = await prisma.models.ExpertPauseEvent.prisma().find_many(
        where={"expertId": expert_id}
    )
    assert any(e.clearedAt is None for e in events)
    sched.pause_schedule.assert_awaited_once_with("sched-1", user_id=test_user.id)
    # Paused, not deleted: the pointer survives so the same job is resumed
    # rather than a second one being created alongside it.
    wf_row = await prisma.models.ExpertWorkflow.prisma().find_first(
        where={"expertId": expert_id}
    )
    assert wf_row is not None and wf_row.scheduleId == "sched-1"

    with patch.object(scheduling, "get_scheduler_client", return_value=sched):
        revived = await experts_db.hire_expert(test_user.id, template.id, None)

    sched.resume_schedule.assert_awaited_once_with("sched-1", user_id=test_user.id)
    # Resume has to look through the paused jobs; the default read hides them.
    sched.get_execution_schedules.assert_awaited_with(
        user_id=test_user.id, kind="graph", include_paused=True
    )
    sched.add_execution_schedule.assert_awaited_once()
    assert revived.expert.schedules_paused_at is None
    preset_row = await prisma.models.AgentPreset.prisma().find_unique(
        where={"id": preset.id}
    )
    assert preset_row is not None and preset_row.isActive is True
    assert preset_row.deactivatedByExpertArchive is False
    # The preset the user disabled before archiving must not come back on.
    disabled_row = await prisma.models.AgentPreset.prisma().find_unique(
        where={"id": disabled_preset.id}
    )
    assert disabled_row is not None and disabled_row.isActive is False
    wf_row = await prisma.models.ExpertWorkflow.prisma().find_first(
        where={"expertId": expert_id}
    )
    assert wf_row is not None and wf_row.scheduleId == "sched-1"


@pytest.mark.asyncio(loop_scope="session")
async def test_detach_survives_a_schedule_that_cannot_pause(
    server: SpinTestServer, test_user
):
    """A schedule the scheduler refuses to pause must not abort the archive:
    the remaining schedules still pause and the run-time gate stops the
    survivor from actually executing."""
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(
        name="Frankie",
        preload_listings=[slv_id],
        preload_crons={slv_id: "40 7 * * *"},
    )
    sched = AsyncMock()
    sched.add_execution_schedule = AsyncMock(return_value=SimpleNamespace(id="sched-1"))
    sched.pause_schedule = AsyncMock(side_effect=[RuntimeError("scheduler down"), None])

    with patch.object(scheduling, "get_scheduler_client", return_value=sched):
        hired = await experts_db.hire_expert(test_user.id, template.id, None)
        expert_id = hired.expert.id
        sched.get_execution_schedules = AsyncMock(
            return_value=[
                SimpleNamespace(
                    kind="graph", id="sched-stuck", name="n", expert_id=expert_id
                ),
                SimpleNamespace(
                    kind="graph", id="sched-1", name="n", expert_id=expert_id
                ),
            ]
        )
        await scheduling.detach_expert_triggers(test_user.id, expert_id)

    assert [c.args[0] for c in sched.pause_schedule.await_args_list] == [
        "sched-stuck",
        "sched-1",
    ]
    wf_row = await prisma.models.ExpertWorkflow.prisma().find_first(
        where={"expertId": expert_id}
    )
    assert wf_row is not None and wf_row.scheduleId == "sched-1"


@pytest.mark.asyncio(loop_scope="session")
async def test_user_created_schedule_survives_fire_and_rehire(
    server: SpinTestServer, test_user
):
    """The schedule a user set up themselves has no ExpertWorkflow cadence
    backing it, so it is unrecoverable if archiving deletes it. Firing must
    pause it by expert attribution alone and re-hiring must bring back that
    same job — inputs and all — not a reconstruction."""
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(name="Frankie", preload_listings=[slv_id])
    sched = AsyncMock()
    sched.pause_schedule = AsyncMock()
    sched.resume_schedule = AsyncMock()

    with patch.object(scheduling, "get_scheduler_client", return_value=sched):
        hired = await experts_db.hire_expert(test_user.id, template.id, None)
        expert_id = hired.expert.id
        # No preload cron, so nothing was scheduled at hire time and no
        # ExpertWorkflow row carries a cadence to rebuild from.
        sched.add_execution_schedule.assert_not_awaited()
        wf_row = await prisma.models.ExpertWorkflow.prisma().find_first(
            where={"expertId": expert_id}
        )
        assert wf_row is not None and wf_row.scheduleCron is None

        # The user then creates their own schedule through the scheduling
        # UI or chat; it carries expert attribution but nothing else.
        user_schedule = SimpleNamespace(
            kind="graph", id="user-sched", name="Weekly report", expert_id=expert_id
        )
        sched.get_execution_schedules = AsyncMock(return_value=[user_schedule])

        await experts_db.archive_expert(test_user.id, expert_id)
        revived = await experts_db.hire_expert(test_user.id, template.id, None)

    sched.pause_schedule.assert_awaited_once_with("user-sched", user_id=test_user.id)
    sched.resume_schedule.assert_awaited_once_with("user-sched", user_id=test_user.id)
    # Restored by resuming the original job, never by creating a new one.
    sched.add_execution_schedule.assert_not_awaited()
    assert revived.expert.is_archived is False
    assert revived.expert.schedules_paused_at is None


@pytest.mark.asyncio(loop_scope="session")
async def test_enforce_budget_pauses_blocks_and_resumes(
    server: SpinTestServer, test_user
):
    template = await _seed_template(name="Max", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    await prisma.models.Expert.prisma().update(
        where={"id": hired.expert.id}, data={"weeklyBudget": 100}
    )

    with (
        patch.object(scheduling, "get_weekly_spend", new=AsyncMock(return_value=150)),
        pytest.raises(ExpertRunPausedError),
    ):
        await scheduling.enforce_expert_run_budget(test_user.id, hired.expert.id)

    row = await prisma.models.Expert.prisma().find_unique(where={"id": hired.expert.id})
    assert row is not None and row.schedulesPausedAt is not None

    with pytest.raises(ExpertRunPausedError):
        await scheduling.enforce_expert_run_budget(test_user.id, hired.expert.id)

    # Resume must forgive the week's tracked spend, or the still-breached
    # counter re-pauses her on the very next fire and Resume is a no-op
    # until the ISO week rolls over.
    with patch.object(scheduling, "reset_weekly_spend", new=AsyncMock()) as reset_spend:
        assert await scheduling.resume_expert_schedules(test_user.id, hired.expert.id)
    reset_spend.assert_awaited_once_with(hired.expert.id)
    events = await prisma.models.ExpertPauseEvent.prisma().find_many(
        where={"expertId": hired.expert.id}
    )
    assert events and all(e.clearedAt is not None for e in events)
    with patch.object(scheduling, "get_weekly_spend", new=AsyncMock(return_value=10)):
        await scheduling.enforce_expert_run_budget(test_user.id, hired.expert.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_seed_roster_round_trip(server: SpinTestServer):
    await _load_roster_store_assets()
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
async def test_seed_roster_rejects_missing_preloads_before_template_mutation(
    monkeypatch: pytest.MonkeyPatch,
):
    resolve = AsyncMock(return_value=None)
    upsert = AsyncMock()
    monkeypatch.setattr(seed, "_resolve_active_version_id", resolve)
    monkeypatch.setattr(seed, "_upsert_template", upsert)

    with pytest.raises(RuntimeError, match="Load marketplace store assets"):
        await seed.seed_roster()

    assert resolve.await_count == len(EXPECTED_ROSTER_PRELOAD_SLUGS)
    upsert.assert_not_awaited()


def test_roster_assigns_two_to_four_workflows_with_one_scheduled_cadence():
    """Launch invariant, checked without a DB: every persona ships 2-4
    preloads, and exactly one scheduled cadence exists across the whole
    roster (Frankie's daily ops digest), so schedule attribution has a
    single unambiguous real case."""
    for entry in seed.ROSTER:
        assert 2 <= len(entry["preloads"]) <= 4, entry["name"]

    assert {
        preload["slug"] for entry in seed.ROSTER for preload in entry["preloads"]
    } == EXPECTED_ROSTER_PRELOAD_SLUGS
    scheduled = [
        (entry["name"], preload["slug"], preload["cron"])
        for entry in seed.ROSTER
        for preload in entry["preloads"]
        if preload["cron"] is not None
    ]
    assert scheduled == [EXPECTED_ROSTER_SCHEDULE]


@pytest.mark.asyncio(loop_scope="session")
async def test_roster_preloads_resolve_and_hire_installs_cleanly(
    server: SpinTestServer,
):
    """Launch acceptance gate against the real checked-in store assets: every
    ROSTER preload slug resolves to the exact StoreListingVersion the CSV
    ships, each persona's template carries exactly its preloads, and hiring
    installs all of them with zero failed preloads. A fictional or renamed
    ROSTER slug fails the asset lookup instead of being papered over by a
    synthetic listing."""
    expected = await _load_roster_store_assets()

    for slug, version_id in expected.items():
        assert await seed._resolve_active_version_id(slug) == version_id, slug

    template_ids = await seed.seed_roster()
    templates = {
        t.name: t for t in await experts_db.list_templates() if t.id in template_ids
    }
    for entry in seed.ROSTER:
        expected_versions = {expected[p["slug"]] for p in entry["preloads"]}
        assert len(expected_versions) == len(entry["preloads"])
        assert {
            w.store_listing_version_id for w in templates[entry["name"]].workflows
        } == expected_versions

    # A fresh user per run: a reused fixture user would make hire_expert
    # short-circuit to a previous run's copy and skip _install_preloads.
    hire_user = await _create_seed_user()
    results = await _hire_roster_and_assert_preloads(hire_user, templates, expected)

    frankie_crons = [
        w.schedule_cron for w in results["Frankie"].expert.workflows if w.schedule_cron
    ]
    assert frankie_crons == ["40 7 * * *"]
    for name in ("Maria", "Max"):
        assert all(w.schedule_cron is None for w in results[name].expert.workflows)


@pytest.mark.asyncio(loop_scope="session")
async def test_roster_slug_resolution_rejects_impostor_creator(
    server: SpinTestServer,
):
    expected = await _load_roster_store_assets()
    slug = "automated-blog-writer"
    impostor_version_id = await _seed_store_listing(server)
    impostor_listing = await prisma.models.StoreListing.prisma().find_first(
        where={"activeVersionId": impostor_version_id}
    )
    assert impostor_listing is not None
    await prisma.models.StoreListing.prisma().update(
        where={"id": impostor_listing.id}, data={"slug": slug}
    )

    resolved = await seed._resolve_active_version_id(slug)

    assert resolved == expected[slug]
    assert resolved != impostor_version_id


@pytest.mark.asyncio(loop_scope="session")
async def test_seed_clears_removed_cadences_on_hired_copies(server: SpinTestServer):
    """Hires made before the single-cadence decision still carry the removed
    template cadences with live scheduler jobs; the seed's migration must
    delete the job (owner-scoped) and clear the row, while a user-customized
    cadence on the same listing is left alone."""
    expected = await _load_roster_store_assets()
    removed_slug, removed_cron = seed.REMOVED_TEMPLATE_CADENCES[0]
    owner = await _create_seed_user()
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(owner.id, template.id, None)
    legacy = await prisma.models.ExpertWorkflow.prisma().create(
        data={
            "expertId": hired.expert.id,
            "storeListingVersionId": expected[removed_slug],
            "scheduleCron": removed_cron,
            "scheduleId": "sched-legacy",
        }
    )
    custom = await prisma.models.ExpertWorkflow.prisma().create(
        data={
            "expertId": hired.expert.id,
            "storeListingVersionId": expected["lead-finder-local-businesses"],
            "scheduleCron": "0 12 * * 1",
            "scheduleId": "sched-custom",
        }
    )
    sched = AsyncMock()
    sched.get_execution_schedules = AsyncMock(
        return_value=[
            SimpleNamespace(id="sched-legacy"),
            SimpleNamespace(id="sched-custom"),
        ]
    )
    sched.delete_schedule = AsyncMock()
    with patch.object(seed, "get_scheduler_client", return_value=sched):
        assert await seed._clear_removed_cadences() >= 1

    sched.delete_schedule.assert_any_await("sched-legacy", user_id=owner.id)
    deleted_ids = [c.args[0] for c in sched.delete_schedule.await_args_list]
    assert "sched-custom" not in deleted_ids
    legacy_after = await prisma.models.ExpertWorkflow.prisma().find_unique(
        where={"id": legacy.id}
    )
    assert legacy_after is not None
    assert legacy_after.scheduleId is None
    assert legacy_after.scheduleCron is None
    custom_after = await prisma.models.ExpertWorkflow.prisma().find_unique(
        where={"id": custom.id}
    )
    assert custom_after is not None
    assert custom_after.scheduleId == "sched-custom"
    assert custom_after.scheduleCron == "0 12 * * 1"


@pytest.mark.asyncio(loop_scope="session")
async def test_seed_clears_removed_cadences_without_live_jobs(server: SpinTestServer):
    expected = await _load_roster_store_assets()
    removed_slug, removed_cron = seed.REMOVED_TEMPLATE_CADENCES[0]
    template = await _seed_template(name="Maria", preload_listings=[])
    owner_without_id = await _create_seed_user()
    owner_with_missing_job = await _create_seed_user()
    hired_without_id = await experts_db.hire_expert(
        owner_without_id.id, template.id, None
    )
    hired_with_missing_job = await experts_db.hire_expert(
        owner_with_missing_job.id, template.id, None
    )
    without_id = await prisma.models.ExpertWorkflow.prisma().create(
        data={
            "expertId": hired_without_id.expert.id,
            "storeListingVersionId": expected[removed_slug],
            "scheduleCron": removed_cron,
        }
    )
    with_missing_job = await prisma.models.ExpertWorkflow.prisma().create(
        data={
            "expertId": hired_with_missing_job.expert.id,
            "storeListingVersionId": expected[removed_slug],
            "scheduleCron": removed_cron,
            "scheduleId": "already-gone",
        }
    )
    scheduler = AsyncMock()
    scheduler.get_execution_schedules = AsyncMock(return_value=[])
    scheduler.delete_schedule = AsyncMock()

    with patch.object(seed, "get_scheduler_client", return_value=scheduler):
        assert await seed._clear_removed_cadences() >= 2

    scheduler.get_execution_schedules.assert_awaited_once_with(
        user_id=owner_with_missing_job.id, kind="graph"
    )
    scheduler.delete_schedule.assert_not_awaited()
    for workflow_id in (without_id.id, with_missing_job.id):
        row = await prisma.models.ExpertWorkflow.prisma().find_unique(
            where={"id": workflow_id}
        )
        assert row is not None
        assert row.scheduleId is None
        assert row.scheduleCron is None


@pytest.mark.asyncio(loop_scope="session")
async def test_seed_clears_removed_cadence_after_listing_version_rotation(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    previous_version_id = await _seed_store_listing(server)
    await _transfer_listing_to_official_creator(previous_version_id)
    listing = await prisma.models.StoreListing.prisma().find_first(
        where={"activeVersionId": previous_version_id}
    )
    previous = await prisma.models.StoreListingVersion.prisma().find_unique(
        where={"id": previous_version_id}
    )
    assert listing is not None
    assert previous is not None

    current = await prisma.models.StoreListingVersion.prisma().create(
        data={
            "version": previous.version + 1,
            "agentGraphId": previous.agentGraphId,
            "agentGraphVersion": previous.agentGraphVersion,
            "name": previous.name,
            "subHeading": previous.subHeading,
            "imageUrls": previous.imageUrls,
            "description": previous.description,
            "categories": previous.categories,
            "submissionStatus": prisma.enums.SubmissionStatus.APPROVED,
            "storeListingId": listing.id,
        }
    )
    await prisma.models.StoreListing.prisma().update(
        where={"id": listing.id}, data={"activeVersionId": current.id}
    )

    removed_cron = "0 9 * * 1"
    monkeypatch.setattr(
        seed, "REMOVED_TEMPLATE_CADENCES", [(listing.slug, removed_cron)]
    )
    owner = await _create_seed_user()
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(owner.id, template.id, None)
    legacy = await prisma.models.ExpertWorkflow.prisma().create(
        data={
            "expertId": hired.expert.id,
            "storeListingVersionId": previous_version_id,
            "scheduleCron": removed_cron,
            "scheduleId": "sched-previous-version",
        }
    )
    sched = AsyncMock()
    sched.get_execution_schedules = AsyncMock(
        return_value=[SimpleNamespace(id="sched-previous-version")]
    )
    sched.delete_schedule = AsyncMock()

    with patch.object(seed, "get_scheduler_client", return_value=sched):
        assert await seed._clear_removed_cadences() == 1

    sched.delete_schedule.assert_awaited_once_with(
        "sched-previous-version", user_id=owner.id
    )
    legacy_after = await prisma.models.ExpertWorkflow.prisma().find_unique(
        where={"id": legacy.id}
    )
    assert legacy_after is not None
    assert legacy_after.scheduleId is None
    assert legacy_after.scheduleCron is None


@pytest.mark.asyncio(loop_scope="session")
async def test_seed_clears_removed_cadence_on_soft_deleted_listing(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    """Soft-deleting the listing must not strand the schedule: the hired copy
    still fires the removed cron, so the migration has to reach it even though
    the listing is gone from the marketplace."""
    version_id = await _seed_store_listing(server)
    await _transfer_listing_to_official_creator(version_id)
    listing = await prisma.models.StoreListing.prisma().find_first(
        where={"activeVersionId": version_id}
    )
    assert listing is not None
    await prisma.models.StoreListing.prisma().update(
        where={"id": listing.id}, data={"isDeleted": True}
    )

    removed_cron = "0 9 * * 1"
    monkeypatch.setattr(
        seed, "REMOVED_TEMPLATE_CADENCES", [(listing.slug, removed_cron)]
    )
    owner = await _create_seed_user()
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(owner.id, template.id, None)
    legacy = await prisma.models.ExpertWorkflow.prisma().create(
        data={
            "expertId": hired.expert.id,
            "storeListingVersionId": version_id,
            "scheduleCron": removed_cron,
            "scheduleId": "sched-soft-deleted-listing",
        }
    )
    sched = AsyncMock()
    sched.get_execution_schedules = AsyncMock(
        return_value=[SimpleNamespace(id="sched-soft-deleted-listing")]
    )
    sched.delete_schedule = AsyncMock()

    with patch.object(seed, "get_scheduler_client", return_value=sched):
        assert await seed._clear_removed_cadences() == 1

    sched.delete_schedule.assert_awaited_once_with(
        "sched-soft-deleted-listing", user_id=owner.id
    )
    legacy_after = await prisma.models.ExpertWorkflow.prisma().find_unique(
        where={"id": legacy.id}
    )
    assert legacy_after is not None
    assert legacy_after.scheduleId is None
    assert legacy_after.scheduleCron is None


@pytest.mark.asyncio(loop_scope="session")
async def test_delete_live_schedule_updates_owner_cache():
    scheduler = AsyncMock()
    scheduler.get_execution_schedules = AsyncMock(
        return_value=[SimpleNamespace(id="sched-shared")]
    )
    scheduler.delete_schedule = AsyncMock()
    live_by_owner: dict[str, set[str]] = {}

    with patch.object(seed, "get_scheduler_client", return_value=scheduler):
        assert await seed._delete_live_schedule(
            "owner-1", "sched-shared", live_by_owner
        )
        assert await seed._delete_live_schedule(
            "owner-1", "sched-shared", live_by_owner
        )

    scheduler.get_execution_schedules.assert_awaited_once_with(
        user_id="owner-1", kind="graph"
    )
    scheduler.delete_schedule.assert_awaited_once_with(
        "sched-shared", user_id="owner-1"
    )
    assert live_by_owner == {"owner-1": set()}


@pytest.mark.asyncio(loop_scope="session")
async def test_seed_preserves_cadence_when_schedule_delete_fails(
    server: SpinTestServer,
):
    """A scheduler failure must keep scheduleId/scheduleCron on the hired row
    so the next seed run retries — clearing first would leave the live job
    firing with nothing pointing at it. The retry then clears the row."""
    expected = await _load_roster_store_assets()
    removed_slug, removed_cron = seed.REMOVED_TEMPLATE_CADENCES[0]
    owner = await _create_seed_user()
    template = await _seed_template(name="Frankie", preload_listings=[])
    hired = await experts_db.hire_expert(owner.id, template.id, None)
    row = await prisma.models.ExpertWorkflow.prisma().create(
        data={
            "expertId": hired.expert.id,
            "storeListingVersionId": expected[removed_slug],
            "scheduleCron": removed_cron,
            "scheduleId": "sched-stuck",
        }
    )
    broken = AsyncMock()
    broken.get_execution_schedules = AsyncMock(
        return_value=[SimpleNamespace(id="sched-stuck")]
    )
    broken.delete_schedule = AsyncMock(side_effect=RuntimeError("scheduler down"))
    with patch.object(seed, "get_scheduler_client", return_value=broken):
        await seed._clear_removed_cadences()

    stuck = await prisma.models.ExpertWorkflow.prisma().find_unique(
        where={"id": row.id}
    )
    assert stuck is not None
    assert stuck.scheduleId == "sched-stuck"
    assert stuck.scheduleCron == removed_cron

    healed = AsyncMock()
    healed.get_execution_schedules = AsyncMock(
        return_value=[SimpleNamespace(id="sched-stuck")]
    )
    healed.delete_schedule = AsyncMock()
    with patch.object(seed, "get_scheduler_client", return_value=healed):
        await seed._clear_removed_cadences()

    healed.delete_schedule.assert_any_await("sched-stuck", user_id=owner.id)
    cleared = await prisma.models.ExpertWorkflow.prisma().find_unique(
        where={"id": row.id}
    )
    assert cleared is not None
    assert cleared.scheduleId is None
    assert cleared.scheduleCron is None


@pytest.mark.asyncio(loop_scope="session")
async def test_list_experts_includes_last_run(server: SpinTestServer, test_user):
    """The /team card needs last-run status: the latest expert-attributed
    execution surfaces as last_run_at / last_run_status."""
    slv_id = await _seed_store_listing(server)
    template = await _seed_template(name="Maria", preload_listings=[slv_id])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    wf = hired.expert.workflows[0]
    assert wf.library_agent_id is not None
    library_agent = await prisma.models.LibraryAgent.prisma().find_unique(
        where={"id": wf.library_agent_id}
    )
    assert library_agent is not None
    await prisma.models.AgentGraphExecution.prisma().create(
        data={
            "agentGraphId": library_agent.agentGraphId,
            "agentGraphVersion": library_agent.agentGraphVersion,
            "userId": test_user.id,
            "executionStatus": prisma.enums.AgentExecutionStatus.COMPLETED,
            "expertId": hired.expert.id,
        }
    )

    experts = await experts_db.list_experts(test_user.id)
    me = next(e for e in experts if e.id == hired.expert.id)
    assert me.last_run_at is not None
    assert me.last_run_status == "COMPLETED"

    fetched = await experts_db.get_expert(test_user.id, hired.expert.id)
    assert fetched is not None
    assert fetched.last_run_status == "COMPLETED"


@pytest.mark.asyncio(loop_scope="session")
async def test_sync_preloads_updates_template_cadence(server: SpinTestServer):
    """Re-seeding must propagate roster cadence changes onto existing
    template preload rows — the old sync was create-only."""
    slv_id = await _seed_store_listing(server)
    # Resolution is creator-scoped, so the ad-hoc listing must belong to the
    # official creator for _sync_preloads to see it.
    await _transfer_listing_to_official_creator(slv_id)
    listing = await prisma.models.StoreListing.prisma().find_first(
        where={"activeVersionId": slv_id}
    )
    assert listing is not None
    template = await _seed_template(name="Frankie", preload_listings=[])
    entry: seed.RosterEntry = {
        "name": template.name,
        "role": template.role,
        "tagline": "",
        "avatar_url": None,
        "bio": "",
        "skills": [],
        "identity": template.identity,
        "preloads": [{"slug": listing.slug, "cron": "40 7 * * *"}],
    }

    await seed._sync_preloads(template.id, entry)
    row = await prisma.models.ExpertWorkflow.prisma().find_first(
        where={"expertId": template.id, "storeListingVersionId": slv_id}
    )
    assert row is not None
    assert row.scheduleCron == "40 7 * * *"

    entry["preloads"] = [{"slug": listing.slug, "cron": "0 8 * * *"}]
    await seed._sync_preloads(template.id, entry)
    row = await prisma.models.ExpertWorkflow.prisma().find_first(
        where={"expertId": template.id, "storeListingVersionId": slv_id}
    )
    assert row is not None
    assert row.scheduleCron == "0 8 * * *"


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
        "voice_preferences": "Clear and confident.",
        "boundaries": "Never invent customer evidence.",
        "preloads": [],
    }
    refreshed_template = await seed._upsert_template(entry)
    assert await seed._backfill_hired_copies(refreshed_template) == 1

    refreshed = await experts_db.get_expert(test_user.id, hired.expert.id)
    assert refreshed is not None
    assert refreshed.avatar_url == "/experts/maria.svg"
    assert refreshed.tagline == "Refreshed tagline"
    assert refreshed.bio == "Maria is a senior marketing strategist."
    assert refreshed.skills == ["Content strategy", "SEO writing"]
    # A user's rename of their own hire survives the refresh.
    assert refreshed.name == "My Maria"


# ─── Pods ──────────────────────────────────────────────────────────────


def _pod_name(prefix: str = "Pod") -> str:
    """Unique per call: pods are unique per (user, name) and the fixed test
    user's pods persist across tests and reruns."""
    return f"{prefix} {uuid.uuid4().hex[:8]}"


@pytest.mark.asyncio(loop_scope="session")
async def test_create_pod_and_assign_membership(server: SpinTestServer, test_user):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)

    name = _pod_name("Growth")
    pod = await experts_db.create_pod(test_user.id, name)
    assigned = await experts_db.assign_pod(test_user.id, hired.expert.id, pod.id)
    assert assigned.pod_id == pod.id

    pods = await experts_db.list_pods(test_user.id)
    matching = [p for p in pods if p.id == pod.id]
    assert len(matching) == 1
    assert matching[0].name == name
    # Membership is read off the expert list, not the pod payload.
    experts = await experts_db.list_experts(test_user.id)
    assert {e.id for e in experts if e.pod_id == pod.id} == {hired.expert.id}


@pytest.mark.asyncio(loop_scope="session")
async def test_create_pod_duplicate_name_raises(server: SpinTestServer, test_user):
    name = _pod_name("Growth")
    await experts_db.create_pod(test_user.id, name)
    with pytest.raises(experts_db.ExpertPodNameTakenError):
        await experts_db.create_pod(test_user.id, name)


@pytest.mark.asyncio(loop_scope="session")
async def test_create_pod_rejects_past_the_per_user_cap(
    server: SpinTestServer, test_user, monkeypatch: pytest.MonkeyPatch
):
    """The cap counts only the caller's own pods, so it is user-scoped."""
    # The fixed test user's pods persist across tests, so pin the cap one above
    # whatever is already there rather than assuming an empty slate.
    existing = len(await experts_db.list_pods(test_user.id))
    monkeypatch.setattr(experts_db, "MAX_PODS_PER_USER", existing + 1)

    await experts_db.create_pod(test_user.id, _pod_name("Capped"))
    with pytest.raises(experts_db.ExpertPodLimitReachedError):
        await experts_db.create_pod(test_user.id, _pod_name("Overflow"))


@pytest.mark.asyncio(loop_scope="session")
async def test_pod_names_unique_per_user_not_globally(
    server: SpinTestServer, test_user, other_user
):
    name = _pod_name("Shared")
    mine = await experts_db.create_pod(test_user.id, name)
    theirs = await experts_db.create_pod(other_user.id, name)
    assert mine.id != theirs.id


@pytest.mark.asyncio(loop_scope="session")
async def test_list_pods_is_user_scoped(server: SpinTestServer, test_user, other_user):
    mine = await experts_db.create_pod(test_user.id, _pod_name("Mine"))
    theirs = await experts_db.create_pod(other_user.id, _pod_name("Theirs"))

    pod_ids = {p.id for p in await experts_db.list_pods(test_user.id)}
    assert mine.id in pod_ids
    assert theirs.id not in pod_ids


@pytest.mark.asyncio(loop_scope="session")
async def test_list_pods_uses_creation_order(server: SpinTestServer, test_user):
    now = datetime.now(timezone.utc)
    later = await prisma.models.ExpertPod.prisma().create(
        data={
            "userId": test_user.id,
            "name": _pod_name("Later"),
            "createdAt": now + timedelta(days=1),
        }
    )
    earlier = await prisma.models.ExpertPod.prisma().create(
        data={
            "userId": test_user.id,
            "name": _pod_name("Earlier"),
            "createdAt": now,
        }
    )

    ordered_ids = [
        pod.id
        for pod in await experts_db.list_pods(test_user.id)
        if pod.id in {earlier.id, later.id}
    ]
    assert ordered_ids == [earlier.id, later.id]


@pytest.mark.asyncio(loop_scope="session")
async def test_assign_pod_rejects_other_users_pod(
    server: SpinTestServer, test_user, other_user
):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    foreign_pod = await experts_db.create_pod(other_user.id, _pod_name("Theirs"))

    with pytest.raises(experts_db.ExpertPodNotFoundError):
        await experts_db.assign_pod(test_user.id, hired.expert.id, foreign_pod.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_assign_pod_rejects_other_users_expert(
    server: SpinTestServer, test_user, other_user
):
    template = await _seed_template(name="Maria", preload_listings=[])
    their_hire = await experts_db.hire_expert(other_user.id, template.id, None)
    pod = await experts_db.create_pod(test_user.id, _pod_name("Growth"))

    with pytest.raises(experts_db.ExpertNotFoundError):
        await experts_db.assign_pod(test_user.id, their_hire.expert.id, pod.id)

    theirs = await experts_db.get_expert(other_user.id, their_hire.expert.id)
    assert theirs is not None
    assert theirs.pod_id is None


@pytest.mark.asyncio(loop_scope="session")
async def test_assign_pod_rejects_template_expert(server: SpinTestServer, test_user):
    template = await _seed_template(name="Maria", preload_listings=[])
    pod = await experts_db.create_pod(test_user.id, _pod_name("Growth"))

    with pytest.raises(experts_db.ExpertNotFoundError):
        await experts_db.assign_pod(test_user.id, template.id, pod.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_assign_pod_rejects_archived_expert(server: SpinTestServer, test_user):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    await experts_db.archive_expert(test_user.id, hired.expert.id)
    pod = await experts_db.create_pod(test_user.id, _pod_name("Growth"))

    with pytest.raises(experts_db.ExpertNotFoundError):
        await experts_db.assign_pod(test_user.id, hired.expert.id, pod.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_assign_pod_unknown_expert_raises(server: SpinTestServer, test_user):
    pod = await experts_db.create_pod(test_user.id, _pod_name("Growth"))
    with pytest.raises(experts_db.ExpertNotFoundError):
        await experts_db.assign_pod(test_user.id, "does-not-exist", pod.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_assign_pod_none_detaches(server: SpinTestServer, test_user):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    pod = await experts_db.create_pod(test_user.id, _pod_name("Growth"))
    await experts_db.assign_pod(test_user.id, hired.expert.id, pod.id)

    detached = await experts_db.assign_pod(test_user.id, hired.expert.id, None)
    assert detached.pod_id is None


@pytest.mark.asyncio(loop_scope="session")
async def test_assign_pod_moves_between_pods(server: SpinTestServer, test_user):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    first_pod = await experts_db.create_pod(test_user.id, _pod_name("First"))
    second_pod = await experts_db.create_pod(test_user.id, _pod_name("Second"))
    await experts_db.assign_pod(test_user.id, hired.expert.id, first_pod.id)

    moved = await experts_db.assign_pod(test_user.id, hired.expert.id, second_pod.id)

    assert moved.pod_id == second_pod.id


@pytest.mark.asyncio(loop_scope="session")
async def test_assign_pod_deleted_concurrently_raises_not_found(
    server: SpinTestServer, test_user
):
    """A pod deleted between the ownership check and the update must surface
    as not-found, not a foreign-key 500."""
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    pod = await experts_db.create_pod(test_user.id, _pod_name("Growth"))

    real_find_first = prisma.models.ExpertPod.prisma().find_first

    async def find_then_delete(*args, **kwargs):
        row = await real_find_first(*args, **kwargs)
        await prisma.models.ExpertPod.prisma().delete(where={"id": pod.id})
        return row

    with (
        patch.object(
            prisma.models.ExpertPod.prisma().__class__,
            "find_first",
            side_effect=find_then_delete,
        ),
        pytest.raises(experts_db.ExpertPodNotFoundError),
    ):
        await experts_db.assign_pod(test_user.id, hired.expert.id, pod.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_deleting_pod_detaches_its_experts(server: SpinTestServer, test_user):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    pod = await experts_db.create_pod(test_user.id, _pod_name("Growth"))
    await experts_db.assign_pod(test_user.id, hired.expert.id, pod.id)

    await prisma.models.ExpertPod.prisma().delete(where={"id": pod.id})

    refreshed = await experts_db.get_expert(test_user.id, hired.expert.id)
    assert refreshed is not None
    assert refreshed.pod_id is None


# ─── Work surface: run composition (pure, no DB) ────────────────────────


def _run_execution(**overrides) -> SimpleNamespace:
    values = {
        "id": "exec-1",
        "agentGraphId": "graph-1",
        "executionStatus": "COMPLETED",
        "startedAt": None,
        "endedAt": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _run_workflow(name: str = "SEO Blog Writer") -> SimpleNamespace:
    return SimpleNamespace(
        libraryAgentId="library-agent-1",
        StoreListingVersion=SimpleNamespace(name=name),
    )


def test_to_expert_run_uses_workflow_name_and_deep_link():
    run = experts_db._to_expert_run(
        _run_execution(), _run_workflow(), "table", "result", needs_review=True
    )
    assert run.execution_id == "exec-1"
    assert run.agent_name == "SEO Blog Writer"
    assert run.output_type == "table"
    assert run.output_key == "result"
    assert run.needs_review is True
    assert run.link == (
        "/library/agents/library-agent-1?activeTab=runs&activeItem=exec-1"
    )


def test_to_expert_run_falls_back_when_workflow_unresolved():
    run = experts_db._to_expert_run(
        _run_execution(), None, "unknown", None, needs_review=False
    )
    assert run.agent_name == "Agent task"
    assert run.library_agent_id is None
    assert run.output_key is None
    assert run.link is None


def _output_node_exec(
    name: str,
    value,
    *,
    execution_data: dict | None = None,
    added_minutes: int = 0,
    queued_minutes: int | None = None,
    stats: dict | None = None,
) -> SimpleNamespace:
    base = datetime(2026, 8, 14, 9, 0, tzinfo=timezone.utc)
    return SimpleNamespace(
        stats=stats,
        executionData=execution_data,
        queuedTime=(
            base + timedelta(minutes=queued_minutes)
            if queued_minutes is not None
            else None
        ),
        addedTime=base + timedelta(minutes=added_minutes),
        Input=(
            [
                SimpleNamespace(name="name", data=name),
                SimpleNamespace(name="value", data=value),
            ]
            if execution_data is None
            else []
        ),
    )


def test_outputs_from_node_execs_builds_pin_map_from_input_rows():
    outputs = experts_db._outputs_from_node_execs(
        [
            _output_node_exec("status", "ok", added_minutes=0),
            _output_node_exec("results", [{"metric": "signups"}], added_minutes=1),
        ]
    )
    assert outputs == {"status": ["ok"], "results": [[{"metric": "signups"}]]}


def test_outputs_from_node_execs_prefers_execution_data_and_orders_by_time():
    outputs = experts_db._outputs_from_node_execs(
        [
            _output_node_exec(
                "", None, execution_data={"name": "rows", "value": 2}, added_minutes=5
            ),
            _output_node_exec(
                "", None, execution_data={"name": "rows", "value": 1}, added_minutes=1
            ),
        ]
    )
    assert outputs == {"rows": [1, 2]}


def test_outputs_from_node_execs_orders_queued_rows_before_unqueued_rows():
    outputs = experts_db._outputs_from_node_execs(
        [
            _output_node_exec("rows", 2, added_minutes=0, queued_minutes=5),
            _output_node_exec("rows", 1, added_minutes=1),
        ]
    )
    assert outputs == {"rows": [2, 1]}


def test_outputs_from_node_execs_skips_rows_without_name():
    outputs = experts_db._outputs_from_node_execs(
        [_output_node_exec("", None, execution_data={"other": "x"})]
    )
    assert outputs == {}


def test_outputs_from_node_execs_prefers_moderation_cleared_inputs():
    outputs = experts_db._outputs_from_node_execs(
        [
            _output_node_exec(
                "",
                None,
                execution_data={"name": "stale", "value": "stale"},
                stats={"cleared_inputs": {"name": ["report"], "value": ["cleared"]}},
            )
        ]
    )
    assert outputs == {"report": ["cleared"]}


def test_outputs_from_node_execs_uses_last_cleared_input_message():
    outputs = experts_db._outputs_from_node_execs(
        [
            _output_node_exec(
                "",
                None,
                stats={"cleared_inputs": {"name": ["rows"], "value": ["a", "b"]}},
            )
        ]
    )
    assert outputs == {"rows": ["b"]}


def test_outputs_from_node_execs_falls_back_when_stats_are_corrupt():
    outputs = experts_db._outputs_from_node_execs(
        [_output_node_exec("status", "ok", stats={"cleared_inputs": "not-a-dict"})]
    )
    assert outputs == {"status": ["ok"]}


@pytest.mark.asyncio
async def test_list_expert_runs_scopes_queries_and_matches_pending_review():
    expert_client = SimpleNamespace(
        find_first=AsyncMock(return_value=SimpleNamespace(Workflows=[]))
    )
    execution_client = SimpleNamespace(
        find_many=AsyncMock(
            return_value=[
                _run_execution(id="exec-1"),
                _run_execution(id="exec-2", executionStatus="REVIEW"),
            ]
        )
    )
    review_client = SimpleNamespace(
        find_many=AsyncMock(return_value=[SimpleNamespace(graphExecId="exec-2")])
    )

    with (
        patch.object(prisma.models.Expert, "prisma", return_value=expert_client),
        patch.object(
            prisma.models.AgentGraphExecution,
            "prisma",
            return_value=execution_client,
        ),
        patch.object(
            prisma.models.PendingHumanReview,
            "prisma",
            return_value=review_client,
        ),
        patch.object(
            experts_db,
            "_classify_run_outputs",
            new=AsyncMock(
                return_value={
                    "exec-1": ("table", "rows"),
                    "exec-2": ("unknown", None),
                }
            ),
        ),
    ):
        runs = await experts_db.list_expert_runs("owner-1", "expert-1")

    expert_where = expert_client.find_first.await_args.kwargs["where"]
    assert expert_where == {
        "id": "expert-1",
        "ownerUserId": "owner-1",
        "isTemplate": False,
        "isArchived": False,
        "visibility": prisma.enums.ResourceVisibility.PRIVATE,
    }
    execution_where = execution_client.find_many.await_args.kwargs["where"]
    assert execution_where == {
        "userId": "owner-1",
        "expertId": "expert-1",
        "isDeleted": False,
    }
    review_where = review_client.find_many.await_args.kwargs["where"]
    assert review_where["userId"] == "owner-1"
    assert set(review_where["graphExecId"]["in"]) == {"exec-1", "exec-2"}
    assert [run.status for run in runs] == ["completed", "review"]
    assert [run.needs_review for run in runs] == [False, True]


@pytest.mark.asyncio
async def test_list_expert_runs_rejects_missing_or_foreign_expert():
    expert_client = SimpleNamespace(find_first=AsyncMock(return_value=None))

    with (
        patch.object(
            prisma.models.Expert,
            "prisma",
            return_value=expert_client,
        ),
        pytest.raises(experts_db.ExpertNotFoundError),
    ):
        await experts_db.list_expert_runs("owner-1", "foreign-expert")

    assert expert_client.find_first.await_args.kwargs["where"]["ownerUserId"] == (
        "owner-1"
    )


@pytest.mark.asyncio
async def test_classify_run_outputs_degrades_only_corrupt_execution():
    node_client = SimpleNamespace(
        find_many=AsyncMock(
            return_value=[
                SimpleNamespace(
                    agentGraphExecutionId="exec-bad",
                    **vars(_output_node_exec("broken", {"bad": True})),
                ),
                SimpleNamespace(
                    agentGraphExecutionId="exec-good",
                    **vars(_output_node_exec("report", "word " * 100)),
                ),
            ]
        )
    )

    with (
        patch.object(
            prisma.models.AgentNodeExecution,
            "prisma",
            return_value=node_client,
        ),
        patch.object(
            experts_db,
            "classify_run_output",
            side_effect=[ValueError("corrupt output"), ("doc", "report")],
        ),
    ):
        classified = await experts_db._classify_run_outputs(["exec-bad", "exec-good"])

    assert classified == {
        "exec-bad": ("unknown", None),
        "exec-good": ("doc", "report"),
    }


@pytest.mark.asyncio(loop_scope="session")
async def test_update_soul_fields_patches_subset_only(
    server: SpinTestServer, test_user
):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)

    updated = await experts_db.update_soul_fields(
        test_user.id,
        hired.expert.id,
        voice_preferences="Warm and concise.",
    )

    assert updated.voice_preferences == "Warm and concise."
    # Untouched fields (including the name) are preserved.
    assert updated.identity == hired.expert.identity
    assert updated.name == hired.expert.name
    assert updated.boundaries == hired.expert.boundaries


@pytest.mark.asyncio(loop_scope="session")
async def test_update_soul_fields_scopes_by_owner(
    server: SpinTestServer, test_user, other_user
):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)

    with pytest.raises(experts_db.ExpertNotFoundError):
        await experts_db.update_soul_fields(
            other_user.id, hired.expert.id, identity="Hijacked identity."
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_update_soul_fields_concurrent_disjoint_edits_both_persist(
    server: SpinTestServer, test_user
):
    """Each call writes only its own column, so neither edit is clobbered."""
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)

    await asyncio.gather(
        experts_db.update_soul_fields(
            test_user.id, hired.expert.id, voice_preferences="Warm and concise."
        ),
        experts_db.update_soul_fields(
            test_user.id, hired.expert.id, boundaries="Never email externally."
        ),
    )

    fetched = await experts_db.get_expert(test_user.id, hired.expert.id)
    assert fetched is not None
    assert fetched.voice_preferences == "Warm and concise."
    assert fetched.boundaries == "Never email externally."


@pytest.mark.asyncio(loop_scope="session")
async def test_update_soul_fields_rejects_empty_patch():
    with pytest.raises(ValueError, match="At least one Soul field"):
        await experts_db.update_soul_fields("user-1", "expert-1")


@pytest.mark.asyncio(loop_scope="session")
async def test_update_soul_fields_if_current_requires_expected_value():
    with pytest.raises(ValueError, match="Expected value required"):
        await experts_db.update_soul_fields_if_current(
            "user-1", "expert-1", identity="New identity."
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_update_soul_fields_if_current_is_atomic(
    server: SpinTestServer, test_user
):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)
    before = hired.expert.voice_preferences

    applied = await experts_db.update_soul_fields_if_current(
        test_user.id,
        hired.expert.id,
        voice_preferences="Warm and concise.",
        expected_voice_preferences=before,
    )
    stale_apply = await experts_db.update_soul_fields_if_current(
        test_user.id,
        hired.expert.id,
        voice_preferences="Stale overwrite.",
        expected_voice_preferences=before,
    )

    fetched = await experts_db.get_expert(test_user.id, hired.expert.id)
    assert applied is True
    assert stale_apply is False
    assert fetched is not None
    assert fetched.voice_preferences == "Warm and concise."


@pytest.mark.asyncio(loop_scope="session")
async def test_update_soul_fields_if_current_scopes_by_owner(
    server: SpinTestServer, test_user, other_user
):
    template = await _seed_template(name="Maria", preload_listings=[])
    hired = await experts_db.hire_expert(test_user.id, template.id, None)

    applied = await experts_db.update_soul_fields_if_current(
        other_user.id,
        hired.expert.id,
        identity="Hijacked identity.",
        expected_identity=hired.expert.identity,
    )

    fetched = await experts_db.get_expert(test_user.id, hired.expert.id)
    assert applied is False
    assert fetched is not None
    assert fetched.identity == hired.expert.identity


def test_expert_soul_fields_patch_rejects_blank_identity():
    with pytest.raises(pydantic.ValidationError):
        ExpertSoulFieldsPatch(identity="   ")


def test_expert_soul_fields_patch_enforces_length_caps():
    with pytest.raises(pydantic.ValidationError):
        ExpertSoulFieldsPatch(identity="x" * 10_001)
    with pytest.raises(pydantic.ValidationError):
        ExpertSoulFieldsPatch(voice_preferences="x" * 4_001)
    with pytest.raises(pydantic.ValidationError):
        ExpertSoulFieldsPatch(boundaries="x" * 4_001)


def test_expert_soul_fields_patch_strips_and_preserves_none():
    patch = ExpertSoulFieldsPatch(
        voice_preferences="   ", boundaries="  Keep it short.  "
    )
    assert patch.voice_preferences == ""
    assert patch.boundaries == "Keep it short."
    assert patch.identity is None
