"""StoreCreator stats against a real Postgres.

The creator card used to count every live listing with any approved version,
including listings whose active version is hidden or not approved, so it said
more workflows than the creator's page lists. ``db_test.py`` mocks Prisma, so
this needs the database (``server`` starts the test stack).
"""

from uuid import uuid4

import prisma.enums
import prisma.models
import pytest

from backend.api.features.store import db as store_db
from backend.data.db import execute_raw_with_schema

APPROVED = prisma.enums.SubmissionStatus.APPROVED
PENDING = prisma.enums.SubmissionStatus.PENDING


async def _refresh() -> None:
    await execute_raw_with_schema(
        "SELECT {schema_prefix}refresh_store_materialized_views()"
    )


async def _seed_listing(
    graph_id: str,
    owner: str,
    versions: list[tuple[prisma.enums.SubmissionStatus, bool, str]],
    active: int | None,
) -> list[str]:
    """One listing with a version per (status, isAvailable, category).

    ``active`` is the index of the version the listing points at, or None for
    a listing with no active version. Returns the version ids.
    """
    await prisma.models.AgentGraph.prisma().create(
        data={
            "id": graph_id,
            "version": 1,
            "name": graph_id,
            "description": "test",
            "userId": owner,
            "isActive": True,
        }
    )
    listing = await prisma.models.StoreListing.prisma().create(
        data={
            "slug": graph_id,
            "agentGraphId": graph_id,
            "owningUserId": owner,
            "hasApprovedVersion": True,
        }
    )
    version_ids: list[str] = []
    for number, (status, available, category) in enumerate(versions, start=1):
        version = await prisma.models.StoreListingVersion.prisma().create(
            data={
                "storeListingId": listing.id,
                "version": number,
                "agentGraphId": graph_id,
                "agentGraphVersion": 1,
                "name": graph_id,
                "subHeading": "",
                "description": "",
                "imageUrls": [],
                "categories": [category],
                "submissionStatus": status,
                "isAvailable": available,
            }
        )
        version_ids.append(version.id)
    if active is not None:
        await prisma.models.StoreListing.prisma().update(
            where={"id": listing.id}, data={"activeVersionId": version_ids[active]}
        )
    return version_ids


@pytest.mark.asyncio(loop_scope="session")
async def test_creator_stats_cover_only_the_listings_the_store_shows(server):
    tag = uuid4().hex[:12]
    owner, runner = f"owner-{tag}", f"runner-{tag}"
    for user_id in (owner, runner):
        await prisma.models.User.prisma().create(
            data={"id": user_id, "email": f"{user_id}@example.invalid"}
        )
    await prisma.models.Profile.prisma().create(
        data={
            "userId": owner,
            "name": owner,
            "username": owner,
            "avatarUrl": "",
            "description": "",
            "links": [],
        }
    )
    visible = f"graph-visible-{tag}"
    unavailable = f"graph-unavailable-{tag}"
    pending_active = f"graph-pending-{tag}"
    no_active = f"graph-noactive-{tag}"
    graph_ids = [visible, unavailable, pending_active, no_active]
    try:
        # The visible listing's earlier approved version carries a category
        # the listed version dropped; the card shows the listed one only.
        visible_versions = await _seed_listing(
            visible,
            owner,
            [(APPROVED, True, "dropped"), (APPROVED, True, "shown")],
            active=1,
        )
        unavailable_versions = await _seed_listing(
            unavailable, owner, [(APPROVED, False, "hidden")], active=0
        )
        await _seed_listing(
            pending_active,
            owner,
            [(APPROVED, True, "hidden"), (PENDING, True, "hidden")],
            active=1,
        )
        await _seed_listing(no_active, owner, [(APPROVED, True, "hidden")], active=None)

        # Reviews on both approved versions of the visible listing count, as
        # in StoreAgent.rating; the hidden listing's review does not.
        for version_id, score in (
            (visible_versions[0], 4),
            (visible_versions[1], 5),
            (unavailable_versions[0], 1),
        ):
            await prisma.models.StoreListingReview.prisma().create(
                data={
                    "storeListingVersionId": version_id,
                    "reviewByUserId": runner,
                    "score": score,
                }
            )
        for graph_id, runs in ((visible, 2), (unavailable, 7)):
            for _ in range(runs):
                await prisma.models.AgentGraphExecution.prisma().create(
                    data={
                        "agentGraphId": graph_id,
                        "agentGraphVersion": 1,
                        "userId": runner,
                    }
                )
        await _refresh()

        creator = await store_db.get_store_creator(owner)
        listed = await store_db.get_store_agents(creators=[owner])

        assert listed.pagination.total_items == 1
        assert [a.agent_graph_id for a in listed.agents] == [visible]
        assert creator.num_agents == listed.pagination.total_items
        assert creator.agent_runs == 2
        assert creator.agent_rating == 4.5
        assert creator.top_categories == ["shown"]
    finally:
        await prisma.models.AgentGraphExecution.prisma().delete_many(
            where={"agentGraphId": {"in": graph_ids}}
        )
        # Deleting a listing cascades to its versions and their reviews.
        await prisma.models.StoreListing.prisma().delete_many(
            where={"agentGraphId": {"in": graph_ids}}
        )
        await prisma.models.AgentGraph.prisma().delete_many(
            where={"id": {"in": graph_ids}}
        )
        await prisma.models.User.prisma().delete_many(
            where={"id": {"in": [owner, runner]}}
        )
        await _refresh()
