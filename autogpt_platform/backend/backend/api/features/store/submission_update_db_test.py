"""Marketplace listing updates against PostgreSQL."""

from unittest.mock import AsyncMock
from uuid import uuid4

import prisma.models
import pytest

from backend.api.features.store import db as store_db
from backend.api.features.store.marketplace_db_test_helpers import (
    PENDING,
    approve,
    clean_up,
    seed_creator,
)
from backend.data.graph import Graph, create_graph

LIVE_DESCRIPTION = "Finds the decision-makers at any company."
NEWER_DESCRIPTION = "Now finds their email too."


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    ("sent", "kept"),
    [
        ("", LIVE_DESCRIPTION),
        (" \n ", LIVE_DESCRIPTION),
        (NEWER_DESCRIPTION, NEWER_DESCRIPTION),
    ],
    ids=["empty", "whitespace", "replaced"],
)
async def test_an_update_keeps_the_listing_description_unless_it_sends_one(
    server, monkeypatch, sent: str, kept: str
):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    monkeypatch.setattr(store_db, "ensure_embedding", AsyncMock())
    try:
        await create_graph(Graph(id=graph_id, name=graph_id, description="test"), owner)
        listed = await _submit(owner, graph_id, LIVE_DESCRIPTION)
        await approve(listed.listing_version_id, owner)

        update = await _submit(owner, graph_id, sent)
        stored = await prisma.models.StoreListingVersion.prisma().find_unique_or_raise(
            where={"id": update.listing_version_id}
        )
        assert (update.listing_version, stored.description, update.description) == (
            2,
            kept,
            kept,
        )

        await approve(update.listing_version_id, owner)
        live = await store_db.get_store_agent_details(owner, graph_id)
        assert (live.active_version_id, live.description) == (
            update.listing_version_id,
            kept,
        )
    finally:
        await clean_up(owner, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    ("listed", "pending", "wiped", "kept"),
    [
        (LIVE_DESCRIPTION, None, "", LIVE_DESCRIPTION),
        (LIVE_DESCRIPTION, None, " \n ", LIVE_DESCRIPTION),
        (LIVE_DESCRIPTION, NEWER_DESCRIPTION, "", NEWER_DESCRIPTION),
        ("", None, "", ""),
    ],
    ids=[
        "from-approved-version",
        "past-whitespace",
        "from-superseded-update",
        "none-to-keep",
    ],
)
async def test_a_blank_update_skips_the_versions_an_earlier_one_wiped(
    server, monkeypatch, listed: str, pending: str | None, wiped: str, kept: str
):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    monkeypatch.setattr(store_db, "ensure_embedding", AsyncMock())
    try:
        await create_graph(Graph(id=graph_id, name=graph_id, description="test"), owner)
        first = await _submit(owner, graph_id, listed)
        await approve(first.listing_version_id, owner)
        if pending:
            await _submit(owner, graph_id, pending)
        blank = await _store_blank_update(graph_id, wiped)
        await approve(blank.id, owner)
        blanked = await store_db.get_store_agent_details(owner, graph_id)
        assert (blanked.active_version_id, blanked.description) == (blank.id, wiped)

        update = await _submit(owner, graph_id, "")
        stored = await prisma.models.StoreListingVersion.prisma().find_unique_or_raise(
            where={"id": update.listing_version_id}
        )
        assert (update.listing_version, stored.description) == (blank.version + 1, kept)

        await approve(update.listing_version_id, owner)
        live = await store_db.get_store_agent_details(owner, graph_id)
        assert (live.active_version_id, live.description) == (
            update.listing_version_id,
            kept,
        )
    finally:
        await clean_up(owner, [graph_id])


async def _submit(owner: str, graph_id: str, description: str):
    return await store_db.create_store_submission(
        owner,
        graph_id,
        1,
        graph_id,
        graph_id,
        "Find decision-makers fast",
        description=description,
    )


async def _store_blank_update(
    graph_id: str, description: str
) -> prisma.models.StoreListingVersion:
    """Store a blank update the way ``create_store_submission`` did before
    it fell back to an earlier description."""
    versions = prisma.models.StoreListingVersion.prisma()
    latest = await versions.find_first_or_raise(
        where={"agentGraphId": graph_id}, order={"version": "desc"}
    )
    await versions.update_many(
        where={
            "storeListingId": latest.storeListingId,
            "submissionStatus": PENDING,
            "isDeleted": False,
        },
        data={"isDeleted": True},
    )
    return await versions.create(
        data={
            "storeListingId": latest.storeListingId,
            "version": latest.version + 1,
            "agentGraphId": graph_id,
            "agentGraphVersion": 1,
            "name": graph_id,
            "subHeading": latest.subHeading,
            "description": description,
            "imageUrls": [],
            "categories": [],
            "submissionStatus": PENDING,
        }
    )
