"""The preview seed against a real Postgres (``server`` starts the test stack).

The seed has to publish what the store endpoints read, change nothing when it
runs again, and leave alone what someone in the preview changed since. The
fixture removes everything the seed wrote, before and after, because several
skill tests need the listing table to themselves.
"""

from unittest.mock import AsyncMock, patch

import prisma.enums
import prisma.models
import pytest

from backend.api.features.orgs.db import get_user_default_team
from backend.data.graph import create_graph, make_graph_model

from . import db as store_db
from . import model as store_model
from . import preview_seed, skill_db
from .preview_seed_content import ADMIN, CREATORS, LISTINGS, SKILLS

CREATOR_USERNAMES = [creator.username for creator in CREATORS]
# What the store_browse load test searches for.
STORE_SEARCHES = (
    "email",
    "research",
    "summarize news",
    "calendar",
    "social media posts",
    "lead generation",
)


@pytest.fixture(scope="module")
async def seeded(server):
    await _remove_seed()
    try:
        # Approval embeds a listing when a provider is configured; stay offline.
        with patch.object(store_db, "ensure_embedding", AsyncMock(return_value=False)):
            first = await preview_seed.seed_marketplace()
            second = await preview_seed.seed_marketplace()
        yield first, second
    finally:
        await _remove_seed()


async def _remove_seed() -> None:
    graph_ids = [preview_seed.seed_graph_id(listing) for listing in LISTINGS]
    user_ids = [preview_seed.seed_user_id(user) for user in preview_seed.all_users()]
    # Listings first: their versions hold the graphs and their reviews the users.
    await prisma.models.StoreListing.prisma().delete_many(
        where={"agentGraphId": {"in": graph_ids}}
    )
    await prisma.models.SkillListing.prisma().delete_many(
        where={"slug": {"in": [skill.slug for skill in SKILLS]}}
    )
    await prisma.models.Organization.prisma().delete_many(
        where={"bootstrapUserId": {"in": user_ids}}
    )
    await prisma.models.User.prisma().delete_many(where={"id": {"in": user_ids}})


async def test_the_first_run_publishes_everything(seeded):
    first, _ = seeded
    assert first.failed == {}
    assert set(first.listings.values()) == {"created"}
    assert set(first.skills.values()) == {"created"}
    assert first.reviews == sum(len(listing.reviews) for listing in LISTINGS)


async def test_a_second_run_changes_nothing(seeded):
    _, second = seeded
    assert second.failed == {}
    assert set(second.listings.values()) == {"existing"}
    assert set(second.skills.values()) == {"existing"}
    assert second.reviews == 0
    graph_ids = [preview_seed.seed_graph_id(listing) for listing in LISTINGS]
    assert await prisma.models.StoreListingVersion.prisma().count(
        where={"agentGraphId": {"in": graph_ids}}
    ) == len(LISTINGS)
    assert await prisma.models.StoreListingReview.prisma().count(
        where={"StoreListingVersion": {"is": {"agentGraphId": {"in": graph_ids}}}}
    ) == sum(len(listing.reviews) for listing in LISTINGS)


async def test_the_store_lists_every_listing(seeded):
    agents = await store_db.get_store_agents(creators=CREATOR_USERNAMES, page_size=100)
    assert {agent.slug for agent in agents.agents} == {
        listing.slug for listing in LISTINGS
    }
    featured = await store_db.get_store_agents(
        featured=True, creators=CREATOR_USERNAMES, page_size=100
    )
    assert {agent.slug for agent in featured.agents} == {
        listing.slug for listing in LISTINGS if listing.featured
    }


async def test_a_listing_opens_by_creator_and_slug(seeded):
    listing = next(listing for listing in LISTINGS if listing.reviews)
    creator = next(c for c in CREATORS if c.key == listing.creator)

    details = await store_db.get_store_agent_details(creator.username, listing.slug)

    assert details.agent_name == listing.name
    assert details.categories == list(listing.categories)
    assert details.rating == pytest.approx(sum(listing.reviews) / len(listing.reviews))
    graph = await store_db.get_agent(details.store_listing_version_id)
    assert graph.id == preview_seed.seed_graph_id(listing)


async def test_sorting_by_rating_puts_the_best_reviewed_listing_first(seeded):
    rated = await store_db.get_store_agents(
        creators=CREATOR_USERNAMES,
        sorted_by=store_db.StoreAgentsSortOptions.RATING,
        page_size=1,
    )
    best = max(
        (listing for listing in LISTINGS if listing.reviews),
        key=lambda listing: sum(listing.reviews) / len(listing.reviews),
    )
    assert rated.agents[0].slug == best.slug


@pytest.mark.parametrize("query", STORE_SEARCHES)
async def test_a_store_search_finds_a_seeded_listing(seeded, query):
    found = await store_db.get_store_agents(
        search_query=query, creators=CREATOR_USERNAMES
    )
    assert found.agents, f"nothing seeded matches '{query}'"


async def test_the_creators_list_shows_every_creator(seeded):
    creators = await store_db.get_store_creators(search_query="Sample", page_size=100)
    seeded_creators = {
        creator.username: creator.num_agents
        for creator in creators.creators
        if creator.username in CREATOR_USERNAMES
    }
    assert seeded_creators == {
        creator.username: sum(listing.creator == creator.key for listing in LISTINGS)
        for creator in CREATORS
    }


async def test_skills_list_and_open_by_slug(seeded):
    skills = await skill_db.get_marketplace_skills(page_size=100)
    assert {skill.slug for skill in SKILLS} <= {skill.slug for skill in skills.skills}

    skill = await skill_db.get_marketplace_skill(SKILLS[0].slug)
    creator = next(c for c in CREATORS if c.key == SKILLS[0].creator)
    assert skill.creator == creator.username
    assert skill.body.startswith("# ")


async def test_a_listing_rejected_since_stays_rejected(seeded):
    listing = LISTINGS[1]
    row = await prisma.models.StoreListing.prisma().find_unique_or_raise(
        where={"agentGraphId": preview_seed.seed_graph_id(listing)}
    )
    assert row.activeVersionId is not None
    await store_db.review_store_submission(
        store_listing_version_id=row.activeVersionId,
        is_approved=False,
        external_comments="",
        internal_comments="",
        reviewer_id=preview_seed.seed_user_id(ADMIN),
        notify_creator=False,
    )

    with patch.object(store_db, "ensure_embedding", AsyncMock(return_value=False)):
        summary = await preview_seed.seed_marketplace()

    assert summary.listings[listing.slug] == "existing"
    after = await prisma.models.StoreListing.prisma().find_unique_or_raise(
        where={"id": row.id}
    )
    assert after.hasApprovedVersion is False

    # Put it back, so the module's other tests see the marketplace as seeded.
    with patch.object(store_db, "ensure_embedding", AsyncMock(return_value=False)):
        await store_db.review_store_submission(
            store_listing_version_id=row.activeVersionId,
            is_approved=True,
            external_comments="",
            internal_comments="",
            reviewer_id=preview_seed.seed_user_id(ADMIN),
            notify_creator=False,
        )
    await _assert_live(listing)


def _owner_id(listing) -> str:
    return preview_seed.seed_user_id(
        next(c for c in CREATORS if c.key == listing.creator)
    )


async def _resubmit(
    listing, changes_summary: str = preview_seed.SEED_NOTE, graph_version: int = 1
) -> store_model.StoreSubmission:
    """Drop the listing and submit it again: by default as a seed run that
    stopped right after submitting would have left it."""
    graph_id = preview_seed.seed_graph_id(listing)
    owner_id = _owner_id(listing)
    await prisma.models.StoreListing.prisma().delete_many(
        where={"agentGraphId": graph_id}
    )
    org_id, _ = await get_user_default_team(owner_id)
    return await store_db.create_store_submission(
        user_id=owner_id,
        graph_id=graph_id,
        graph_version=graph_version,
        slug=listing.slug,
        name=listing.name,
        sub_heading=listing.sub_heading,
        categories=list(listing.categories),
        changes_summary=changes_summary,
        organization_id=org_id,
    )


async def _assert_live(listing) -> None:
    row = await prisma.models.StoreListing.prisma().find_unique_or_raise(
        where={"agentGraphId": preview_seed.seed_graph_id(listing)},
        include={"ActiveVersion": True},
    )
    assert row.hasApprovedVersion
    assert row.ActiveVersion is not None
    assert row.ActiveVersion.submissionStatus == prisma.enums.SubmissionStatus.APPROVED
    creator = next(c for c in CREATORS if c.key == listing.creator)
    details = await store_db.get_store_agent_details(creator.username, listing.slug)
    assert details.agent_name == listing.name


async def test_a_submission_an_interrupted_run_left_pending_is_approved(seeded):
    listing = LISTINGS[2]
    await _resubmit(listing)

    with patch.object(store_db, "ensure_embedding", AsyncMock(return_value=False)):
        summary = await preview_seed.seed_marketplace()

    assert summary.listings[listing.slug] == "approved"
    await _assert_live(listing)


async def test_an_approval_interrupted_after_pointing_the_listing_is_finished(seeded):
    listing = LISTINGS[3]
    submission = await _resubmit(listing)
    # What review_store_submission has committed when it stops before it marks
    # the version approved.
    await prisma.models.StoreListing.prisma().update(
        where={"id": submission.listing_id},
        data={
            "hasApprovedVersion": True,
            "ActiveVersion": {"connect": {"id": submission.listing_version_id}},
        },
    )

    with patch.object(store_db, "ensure_embedding", AsyncMock(return_value=False)):
        summary = await preview_seed.seed_marketplace()

    assert summary.listings[listing.slug] == "approved"
    await _assert_live(listing)


async def test_someone_elses_pending_submission_waits_for_a_real_review(seeded):
    listing = LISTINGS[4]
    # An admin acting as the synthetic creator, say.
    submission = await _resubmit(listing, changes_summary="Tweaked the template")

    with patch.object(store_db, "ensure_embedding", AsyncMock(return_value=False)):
        summary = await preview_seed.seed_marketplace()

    assert summary.listings[listing.slug] == "existing"
    version = await prisma.models.StoreListingVersion.prisma().find_unique_or_raise(
        where={"id": submission.listing_version_id}
    )
    assert version.submissionStatus == prisma.enums.SubmissionStatus.PENDING

    # Put it back, so the module's other tests see the marketplace as seeded.
    await prisma.models.StoreListing.prisma().delete_many(
        where={"agentGraphId": preview_seed.seed_graph_id(listing)}
    )
    with patch.object(store_db, "ensure_embedding", AsyncMock(return_value=False)):
        await preview_seed.seed_marketplace()
    await _assert_live(listing)


async def _save_an_edited_version(listing) -> int:
    """Save the listing's workflow again, as its creator editing it in the
    builder would: a new version of the same graph."""
    owner_id = _owner_id(listing)
    org_id, team_id = await get_user_default_team(owner_id)
    graph = preview_seed.build_graph(listing)
    graph.version = 2
    edited = make_graph_model(graph, owner_id)
    edited.reassign_ids(user_id=owner_id, reassign_graph_id=False)
    created = await create_graph(
        edited, owner_id, organization_id=org_id, team_id=team_id
    )
    return created.version


async def test_a_submission_of_an_edited_workflow_waits_even_with_the_seeds_note(
    seeded,
):
    listing = LISTINGS[5]
    graph_id = preview_seed.seed_graph_id(listing)
    edited_version = await _save_an_edited_version(listing)
    # The publish form keeps the previous submission's note.
    submission = await _resubmit(listing, graph_version=edited_version)

    with patch.object(store_db, "ensure_embedding", AsyncMock(return_value=False)):
        summary = await preview_seed.seed_marketplace()

    assert summary.listings[listing.slug] == "existing"
    version = await prisma.models.StoreListingVersion.prisma().find_unique_or_raise(
        where={"id": submission.listing_version_id}
    )
    assert version.submissionStatus == prisma.enums.SubmissionStatus.PENDING

    # Put it back, so the module's other tests see the marketplace as seeded.
    await prisma.models.StoreListing.prisma().delete_many(
        where={"agentGraphId": graph_id}
    )
    await prisma.models.AgentGraph.prisma().delete(
        where={"graphVersionId": {"id": graph_id, "version": edited_version}}
    )
    with patch.object(store_db, "ensure_embedding", AsyncMock(return_value=False)):
        await preview_seed.seed_marketplace()
    await _assert_live(listing)
