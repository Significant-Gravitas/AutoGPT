"""The live version of a creator's listing, against PostgreSQL."""

from unittest.mock import AsyncMock
from uuid import uuid4

import prisma.models
import pytest

from backend.api.features.store import db as store_db
from backend.api.features.store.live_submission import get_live_submission
from backend.api.features.store.marketplace_db_test_helpers import (
    APPROVED,
    PENDING,
    approve,
    clean_up,
    reject,
    seed_creator,
    seed_listing,
    submit,
)
from backend.blocks.agent import AgentExecutorBlock
from backend.data.graph import Graph, create_graph
from backend.util.exceptions import NotFoundError
from backend.util.json import SafeJson


@pytest.fixture(autouse=True)
def _no_embeddings(monkeypatch):
    monkeypatch.setattr(store_db, "ensure_embedding", AsyncMock())


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("history", ["unlisted", "pending", "rejected", "withdrawn"])
async def test_a_workflow_with_nothing_live_has_no_live_submission(
    server, history: str
):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        await create_graph(Graph(id=graph_id, name=graph_id, description="test"), owner)
        if history != "unlisted":
            first = await submit(owner, graph_id)
            if history == "rejected":
                await reject(first.listing_version_id, owner)
            elif history == "withdrawn":
                await approve(first.listing_version_id, owner)
                await reject(first.listing_version_id, owner)

        assert await get_live_submission(owner, graph_id) is None
    finally:
        await clean_up(owner, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "hidden_by", ["listing-deleted", "no-approved-version", "active-not-approved"]
)
async def test_a_listing_the_marketplace_hides_has_no_live_submission(
    server, hidden_by: str
):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        [active] = await seed_listing(graph_id, owner, [APPROVED])
        if hidden_by == "active-not-approved":
            await prisma.models.StoreListingVersion.prisma().update(
                where={"id": active}, data={"submissionStatus": PENDING}
            )
        else:
            await prisma.models.StoreListing.prisma().update(
                where={"agentGraphId": graph_id},
                data=(
                    {"isDeleted": True}
                    if hidden_by == "listing-deleted"
                    else {"hasApprovedVersion": False}
                ),
            )
        with pytest.raises(NotFoundError):
            await store_db.get_store_agent_details(owner, graph_id)

        assert await get_live_submission(owner, graph_id) is None
    finally:
        await clean_up(owner, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
async def test_the_live_submission_is_the_version_the_marketplace_shows(server):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        await create_graph(Graph(id=graph_id, name=graph_id, description="test"), owner)
        listed = await submit(owner, graph_id)
        await approve(listed.listing_version_id, owner)
        update = await submit(owner, graph_id)

        live = await get_live_submission(owner, graph_id)
        shown = await store_db.get_store_agent_details(owner, graph_id)
        assert live is not None
        assert (live.listing_version_id, live.listing_version, live.status) == (
            shown.active_version_id,
            1,
            APPROVED,
        )
        assert (live.graph_id, live.slug, live.name) == (graph_id, graph_id, graph_id)

        await approve(update.listing_version_id, owner)
        live = await get_live_submission(owner, graph_id)
        assert live is not None
        assert live.listing_version_id == update.listing_version_id
    finally:
        await clean_up(owner, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
async def test_the_live_submission_carries_the_live_versions_schedule(server):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        await create_graph(Graph(id=graph_id, name=graph_id, description="test"), owner)
        listed = await submit(owner, graph_id, schedule="0 9 * * 1")
        await approve(listed.listing_version_id, owner)
        # Approval copied the schedule onto the workflow; give the workflow its
        # own, so only a read of the listing's live version passes.
        await prisma.models.AgentGraph.prisma().update(
            where={"graphVersionId": {"id": graph_id, "version": 1}},
            data={"recommendedScheduleCron": "0 7 * * *"},
        )
        update = await submit(owner, graph_id, schedule="0 8 * * *")

        live = await get_live_submission(owner, graph_id)
        assert live is not None
        assert live.recommended_schedule_cron == "0 9 * * 1"

        await approve(update.listing_version_id, owner)
        live = await get_live_submission(owner, graph_id)
        assert live is not None
        assert live.recommended_schedule_cron == "0 8 * * *"
    finally:
        await clean_up(owner, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
async def test_a_newer_hidden_sub_agent_version_is_not_the_live_one(server):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    graph_id, parent_id = f"graph-{tag}", f"parent-{tag}"
    try:
        await create_graph(Graph(id=graph_id, name=graph_id, description="test"), owner)
        listed = await submit(owner, graph_id)
        await approve(listed.listing_version_id, owner)
        hidden = await _approve_as_sub_agent(owner, graph_id, parent_id)
        assert (hidden.version, hidden.submissionStatus, hidden.isAvailable) == (
            2,
            APPROVED,
            False,
        )

        live = await get_live_submission(owner, graph_id)
        assert live is not None
        assert (live.listing_version_id, live.listing_version) == (
            listed.listing_version_id,
            1,
        )
    finally:
        await clean_up(owner, [graph_id, parent_id])


@pytest.mark.asyncio(loop_scope="session")
async def test_rejecting_the_live_version_returns_the_listings_new_live_one(server):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    graph_id, parent_id = f"graph-{tag}", f"parent-{tag}"
    try:
        await create_graph(Graph(id=graph_id, name=graph_id, description="test"), owner)
        first = await submit(owner, graph_id)
        await approve(first.listing_version_id, owner)
        second = await submit(owner, graph_id)
        await approve(second.listing_version_id, owner)
        hidden = await _approve_as_sub_agent(owner, graph_id, parent_id)
        assert (hidden.version, hidden.isAvailable) == (3, False)

        await reject(second.listing_version_id, owner)

        live = await get_live_submission(owner, graph_id)
        shown = await store_db.get_store_agent_details(owner, graph_id)
        assert live is not None
        assert (live.listing_version_id, live.listing_version) == (
            first.listing_version_id,
            1,
        )
        assert shown.active_version_id == live.listing_version_id
    finally:
        await clean_up(owner, [graph_id, parent_id])


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    ("filed_in_org", "caller", "caller_org", "sees_it"),
    [
        (False, "owner", None, True),
        (False, "stranger", None, False),
        (False, "owner", "other", True),
        (False, "stranger", "listing", False),
        (True, "stranger", "listing", True),
        (True, "stranger", "other", False),
        (True, "owner", "other", False),
    ],
    ids=[
        "own",
        "another-creators",
        "own-tenant-less-in-an-org",
        "another-creators-tenant-less",
        "own-orgs",
        "another-orgs",
        "own-but-another-orgs",
    ],
)
async def test_only_the_callers_own_or_their_orgs_listing_is_returned(
    server, filed_in_org: bool, caller: str, caller_org: str | None, sees_it: bool
):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    orgs: dict[str, prisma.models.Organization] = {}
    try:
        for name in ("listing", "other"):
            orgs[name] = await prisma.models.Organization.prisma().create(
                data={"name": f"{name} {tag}", "slug": f"{name}-{tag}"}
            )
        [live_version] = await seed_listing(graph_id, owner, [APPROVED])
        if filed_in_org:
            await prisma.models.StoreListing.prisma().update(
                where={"agentGraphId": graph_id},
                data={"OwningOrg": {"connect": {"id": orgs["listing"].id}}},
            )

        live = await get_live_submission(
            owner if caller == "owner" else f"stranger-{tag}",
            graph_id,
            organization_id=orgs[caller_org].id if caller_org else None,
        )

        assert (live.listing_version_id if live else None) == (
            live_version if sees_it else None
        )
    finally:
        await clean_up(owner, [graph_id])
        await prisma.models.Organization.prisma().delete_many(
            where={"id": {"in": [org.id for org in orgs.values()]}}
        )


async def _approve_as_sub_agent(
    owner: str, graph_id: str, parent_id: str
) -> prisma.models.StoreListingVersion:
    """Approve a parent workflow that runs ``graph_id`` v2, which files an
    approved but hidden version on ``graph_id``'s listing; return it."""
    await create_graph(
        Graph(id=graph_id, version=2, name=graph_id, description="test"), owner
    )
    await create_graph(Graph(id=parent_id, name=parent_id, description="test"), owner)
    await prisma.models.AgentNode.prisma().create(
        data={
            "agentGraphId": parent_id,
            "agentGraphVersion": 1,
            "agentBlockId": AgentExecutorBlock().id,
            "constantInput": SafeJson({"graph_id": graph_id, "graph_version": 2}),
        }
    )
    parent = await submit(owner, parent_id)
    await approve(parent.listing_version_id, owner)
    return await prisma.models.StoreListingVersion.prisma().find_first_or_raise(
        where={"agentGraphId": graph_id}, order={"version": "desc"}
    )
