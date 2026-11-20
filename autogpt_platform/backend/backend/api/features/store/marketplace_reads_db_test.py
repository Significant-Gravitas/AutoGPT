"""Public Marketplace reads and admin listing search against PostgreSQL."""

from uuid import uuid4

import prisma.enums
import prisma.models
import pytest

from backend.api.features.store import db as store_db
from backend.api.features.store import exceptions as store_exceptions
from backend.api.features.store.marketplace_db_test_helpers import (
    APPROVED,
    PENDING,
    REJECTED,
    clean_up,
    seed_creator,
    seed_listing,
)
from backend.util.exceptions import NotFoundError


@pytest.mark.asyncio(loop_scope="session")
async def test_a_creator_page_answers_once_a_workflow_is_listed(server):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        with pytest.raises(store_exceptions.CreatorNotFoundError):
            await store_db.get_store_creator(owner)
        own = await store_db.get_store_creator(owner, include_unlisted=True)
        assert own.num_agents == 0

        await seed_listing(graph_id, owner, [APPROVED])

        page = await store_db.get_store_creator(owner)
        assert page.num_agents == 1
    finally:
        await clean_up(owner, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
async def test_a_creator_without_an_avatar_is_listed(server):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url=None)
    graph_id = f"graph-{tag}"
    try:
        await seed_listing(graph_id, owner, [APPROVED])

        listed = await store_db.get_store_creators(search_query=tag)
        page = await store_db.get_store_creator(owner)

        assert [(c.username, c.avatar_url) for c in listed.creators] == [(owner, None)]
        assert page.avatar_url is None
    finally:
        await clean_up(owner, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
async def test_a_listing_graph_is_served_only_for_a_listed_version(server):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        approved, pending, rejected = await seed_listing(
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
        await clean_up(owner, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
async def test_the_admin_listing_search_finds_owners_by_email_literally(server):
    """An admin search matches the owner's email in the listings query itself,
    with `_` as the character: `x_y` finds its owner's listing and not that of
    an `xzy` lookalike. The slugs don't hold the term, so only the email can
    match."""
    tag = uuid4().hex[:12]
    owner = await seed_creator(f"x_y{tag}", avatar_url="")
    lookalike = await seed_creator(f"xzy{tag}", avatar_url="")
    graph_ids = [f"graph-{uuid4().hex[:12]}", f"graph-{uuid4().hex[:12]}"]
    try:
        await seed_listing(graph_ids[0], owner, [PENDING])
        await seed_listing(graph_ids[1], lookalike, [PENDING])

        found = await store_db.get_admin_listings_with_versions(
            search_query=f"x_y{tag}"
        )

        assert [listing.graph_id for listing in found.listings] == [graph_ids[0]]
        assert found.pagination.total_items == 1
    finally:
        await clean_up(owner, graph_ids[:1])
        await clean_up(lookalike, graph_ids[1:])
