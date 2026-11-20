"""Public Marketplace creator and listing reads against PostgreSQL."""

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
