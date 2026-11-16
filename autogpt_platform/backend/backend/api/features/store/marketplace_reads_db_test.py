"""Public marketplace reads against a real Postgres.

The creator views are SQL and the listing-version filter is a Prisma query,
so ``db_test.py``'s mocks cannot run either; ``server`` starts the test stack.
"""

from uuid import uuid4

import prisma.enums
import prisma.models
import pytest

from backend.api.features.store import db as store_db
from backend.api.features.store import exceptions as store_exceptions
from backend.util.exceptions import NotFoundError

APPROVED = prisma.enums.SubmissionStatus.APPROVED
PENDING = prisma.enums.SubmissionStatus.PENDING
REJECTED = prisma.enums.SubmissionStatus.REJECTED


@pytest.mark.asyncio(loop_scope="session")
async def test_a_creator_page_answers_once_a_workflow_is_listed(server):
    tag = uuid4().hex[:12]
    owner = await _seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        with pytest.raises(store_exceptions.CreatorNotFoundError):
            await store_db.get_store_creator(owner)
        own = await store_db.get_store_creator(owner, include_unlisted=True)
        assert own.num_agents == 0

        await _seed_listing(graph_id, owner, [APPROVED])

        page = await store_db.get_store_creator(owner)
        assert page.num_agents == 1
    finally:
        await _clean_up(owner, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
async def test_a_creator_without_an_avatar_is_listed(server):
    tag = uuid4().hex[:12]
    owner = await _seed_creator(tag, avatar_url=None)
    graph_id = f"graph-{tag}"
    try:
        await _seed_listing(graph_id, owner, [APPROVED])

        listed = await store_db.get_store_creators(search_query=tag)
        page = await store_db.get_store_creator(owner)

        assert [(c.username, c.avatar_url) for c in listed.creators] == [(owner, None)]
        assert page.avatar_url is None
    finally:
        await _clean_up(owner, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
async def test_a_listing_graph_is_served_only_for_a_listed_version(server):
    tag = uuid4().hex[:12]
    owner = await _seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        approved, pending, rejected = await _seed_listing(
            graph_id, owner, [APPROVED, PENDING, REJECTED]
        )

        graph = await store_db.get_available_graph(approved)
        assert graph.id == graph_id
        for version_id in (pending, rejected):
            with pytest.raises(NotFoundError):
                await store_db.get_available_graph(version_id)

        await prisma.models.StoreListing.prisma().update_many(
            where={"agentGraphId": graph_id}, data={"isDeleted": True}
        )
        with pytest.raises(NotFoundError):
            await store_db.get_available_graph(approved)
    finally:
        await _clean_up(owner, [graph_id])


async def _seed_creator(tag: str, avatar_url: str | None) -> str:
    """A user and their marketplace profile; returns the username."""
    username = f"creator-{tag}"
    await prisma.models.User.prisma().create(
        data={"id": username, "email": f"{username}@example.invalid"}
    )
    await prisma.models.Profile.prisma().create(
        data={
            "userId": username,
            "name": username,
            "username": username,
            "avatarUrl": avatar_url,
            "description": "",
            "links": [],
        }
    )
    return username


async def _seed_listing(
    graph_id: str, owner: str, statuses: list[prisma.enums.SubmissionStatus]
) -> list[str]:
    """One listing with a version per status, the first one active."""
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
            "hasApprovedVersion": APPROVED in statuses,
        }
    )
    version_ids: list[str] = []
    for number, status in enumerate(statuses, start=1):
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
                "categories": [],
                "submissionStatus": status,
            }
        )
        version_ids.append(version.id)
    await prisma.models.StoreListing.prisma().update(
        where={"id": listing.id}, data={"activeVersionId": version_ids[0]}
    )
    return version_ids


async def _clean_up(owner: str, graph_ids: list[str]) -> None:
    # Deleting a listing cascades to its versions.
    await prisma.models.StoreListing.prisma().delete_many(
        where={"agentGraphId": {"in": graph_ids}}
    )
    await prisma.models.AgentGraph.prisma().delete_many(where={"id": {"in": graph_ids}})
    await prisma.models.User.prisma().delete_many(where={"id": owner})
