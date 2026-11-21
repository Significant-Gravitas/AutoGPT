"""Moderation of workflows with Marketplace dependencies against PostgreSQL."""

import asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

import prisma.enums
import prisma.models
import pytest

from backend.api.features.store import db as store_db
from backend.api.features.store.marketplace_db_test_helpers import (
    APPROVED,
    PENDING,
    REJECTED,
    approve,
    clean_up,
    overlap_reviews,
    reject,
    seed_creator,
    seed_listing,
    wait_for_blocked_statement,
)
from backend.blocks.agent import AgentExecutorBlock
from backend.data.graph import is_graph_published_in_marketplace
from backend.util.exceptions import ConflictError, NotFoundError
from backend.util.json import SafeJson


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_approvals_of_nested_workflows(server, monkeypatch):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    parent, leaf, outer = (
        f"{prefix}-{tag}" for prefix in ("a-parent", "b-leaf", "c-outer")
    )
    try:
        parent_id, leaf_id = await _seed_parent_with_child(owner, parent, leaf, PENDING)
        [outer_id] = await seed_listing(outer, owner, [PENDING])
        await _add_dependency(outer, parent)
        outer_has_parent, parent_has_leaf = asyncio.Event(), asyncio.Event()
        approve_child = store_db._approve_sub_agent

        async def overlap(tx, graph, main_name, main_version, main_user):
            if main_name == outer and graph.id == leaf:
                await outer_has_parent.wait()
                await wait_for_blocked_statement(tx, until=parent_has_leaf)
            await approve_child(tx, graph, main_name, main_version, main_user)
            if main_name == outer and graph.id == parent:
                outer_has_parent.set()
            if main_name == parent and graph.id == leaf:
                parent_has_leaf.set()

        monkeypatch.setattr(store_db, "_approve_sub_agent", overlap)
        monkeypatch.setattr(store_db, "ensure_embedding", AsyncMock())
        async with asyncio.timeout(15), asyncio.TaskGroup() as reviews:
            reviews.create_task(approve(outer_id, owner))
            await outer_has_parent.wait()
            reviews.create_task(approve(parent_id, owner))

        for graph, live in [(outer, outer_id), (parent, parent_id), (leaf, None)]:
            listing = await prisma.models.StoreListing.prisma().find_unique_or_raise(
                where={"agentGraphId": graph}
            )
            assert listing.hasApprovedVersion and listing.activeVersionId == live
        version = await prisma.models.StoreListingVersion.prisma().find_unique_or_raise(
            where={"id": leaf_id}
        )
        assert version.submissionStatus == APPROVED
    finally:
        await clean_up(owner, [outer, parent, leaf])


@pytest.mark.asyncio(loop_scope="session")
async def test_rejection_serializes_with_approval_of_a_parent_workflow(
    server, monkeypatch
):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    parent_graph, child_graph = f"parent-{tag}", f"child-{tag}"
    try:
        parent, child = await _seed_parent_with_child(
            owner, parent_graph, child_graph, PENDING
        )
        monkeypatch.setattr(store_db, "ensure_embedding", AsyncMock())
        waiting_statement = await overlap_reviews(
            monkeypatch,
            lambda: reject(child, owner),
            lambda: approve(parent, owner),
        )
        assert (
            waiting_statement and '"StoreListing"' in waiting_statement
        ), waiting_statement
        child_version = (
            await prisma.models.StoreListingVersion.prisma().find_unique_or_raise(
                where={"id": child}, include={"StoreListing": True}
            )
        )
        assert child_version.submissionStatus == APPROVED
        assert child_version.StoreListing is not None
        assert child_version.StoreListing.activeVersionId is None
        assert child_version.StoreListing.hasApprovedVersion
        assert (
            await store_db.get_store_agent_details(owner, parent_graph)
        ).active_version_id == parent
    finally:
        await clean_up(owner, [parent_graph, child_graph])


@pytest.mark.asyncio(loop_scope="session")
async def test_rejection_of_an_approved_child_stands_when_parent_approval_overlaps(
    server, monkeypatch
):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    parent_graph, child_graph = f"parent-{tag}", f"child-{tag}"
    try:
        parent, child = await _seed_parent_with_child(
            owner, parent_graph, child_graph, APPROVED
        )
        parent_checked_child = asyncio.Event()
        withdrawal_committed = asyncio.Event()

        async def reject_child():
            await reject(child, owner)
            withdrawal_committed.set()

        async def finish_parent_after_withdrawal():
            parent_checked_child.set()
            await withdrawal_committed.wait()

        _after_child_listing_read(
            monkeypatch, child_graph, finish_parent_after_withdrawal
        )
        monkeypatch.setattr(store_db, "ensure_embedding", AsyncMock())
        await overlap_reviews(
            monkeypatch,
            reject_child,
            lambda: approve(parent, owner),
            second_read=parent_checked_child,
        )

        child_version = (
            await prisma.models.StoreListingVersion.prisma().find_unique_or_raise(
                where={"id": child}
            )
        )
        assert child_version.submissionStatus == REJECTED
        assert not await is_graph_published_in_marketplace(child_graph, 1)
        with pytest.raises(NotFoundError):
            await store_db.get_store_agent_details(owner, child_graph)
        assert (
            await store_db.get_store_agent_details(owner, parent_graph)
        ).active_version_id == parent
    finally:
        await clean_up(owner, [parent_graph, child_graph])


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "explicit_approval", [False, True], ids=["parent-only", "also-approved-directly"]
)
async def test_parent_approval_handles_a_child_withdrawn_before_its_first_read(
    server, monkeypatch, explicit_approval: bool
):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    parent_graph, child_graph = f"parent-{tag}", f"child-{tag}"
    try:
        parent, child = await _seed_parent_with_child(
            owner, parent_graph, child_graph, APPROVED
        )
        get_sub_graphs = store_db.get_sub_graphs

        async def withdraw_before_child_read(graph):
            if graph.id == parent_graph:
                await reject(child, owner)
            return await get_sub_graphs(graph)

        monkeypatch.setattr(store_db, "get_sub_graphs", withdraw_before_child_read)
        monkeypatch.setattr(store_db, "ensure_embedding", AsyncMock())
        if explicit_approval:
            _after_child_listing_read(
                monkeypatch, child_graph, lambda: approve(child, owner)
            )
        await approve(parent, owner)

        assert await is_graph_published_in_marketplace(child_graph, 1)
        listing = await prisma.models.StoreListing.prisma().find_unique_or_raise(
            where={"agentGraphId": child_graph}
        )
        assert listing.hasApprovedVersion
        assert listing.activeVersionId == (child if explicit_approval else None)
        assert (
            await store_db.get_store_agent_details(owner, parent_graph)
        ).active_version_id == parent
    finally:
        await clean_up(owner, [parent_graph, child_graph])


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "delete_superseded", [False, True], ids=["superseded", "deleted"]
)
async def test_parent_review_rolls_back_when_the_child_submission_is_replaced(
    server, monkeypatch, delete_superseded: bool
):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    parent_graph, child_graph = f"parent-{tag}", f"child-{tag}"
    try:
        parent, child = await _seed_parent_with_child(
            owner, parent_graph, child_graph, PENDING
        )
        listings = prisma.models.StoreListing.prisma()
        versions = prisma.models.StoreListingVersion.prisma()

        async def replace_child():
            await store_db.create_store_submission(
                owner, child_graph, 1, child_graph, child_graph, "Replacement"
            )
            if delete_superseded:
                await store_db.delete_store_submission_or_raise(owner, child)

        monkeypatch.setattr(store_db, "ensure_embedding", AsyncMock())
        with monkeypatch.context() as overlap:
            _after_child_listing_read(overlap, child_graph, replace_child)
            with pytest.raises(ConflictError, match="changed during review"):
                await approve(parent, owner)

        listing = await listings.find_unique_or_raise(
            where={"agentGraphId": child_graph}
        )
        assert not listing.hasApprovedVersion
        assert (
            await versions.find_unique_or_raise(where={"id": parent})
        ).submissionStatus == PENDING
        [replacement] = await versions.find_many(
            where={"storeListingId": listing.id, "isDeleted": False}
        )
        assert replacement.id != child and replacement.submissionStatus == PENDING

        await approve(parent, owner)
        assert (
            await versions.find_unique_or_raise(where={"id": replacement.id})
        ).submissionStatus == APPROVED
        assert (
            await store_db.get_store_agent_details(owner, parent_graph)
        ).active_version_id == parent
    finally:
        await clean_up(owner, [parent_graph, child_graph])


def _after_child_listing_read(monkeypatch, child_graph: str, action):
    listing_actions = type(prisma.models.StoreListing.prisma())
    find_first = listing_actions.find_first
    fired = False

    async def read_then_act(actions, **kwargs):
        nonlocal fired
        listing = await find_first(actions, **kwargs)
        if (
            not fired
            and listing
            and listing.agentGraphId == child_graph
            and listing.Versions
        ):
            fired = True
            async with asyncio.timeout(10):
                await action()
        return listing

    monkeypatch.setattr(listing_actions, "find_first", read_then_act)


async def _seed_parent_with_child(
    owner: str,
    parent_graph: str,
    child_graph: str,
    child_status: prisma.enums.SubmissionStatus,
) -> tuple[str, str]:
    [parent] = await seed_listing(parent_graph, owner, [PENDING])
    [child] = await seed_listing(child_graph, owner, [child_status])
    await _add_dependency(parent_graph, child_graph)
    return parent, child


async def _add_dependency(parent_graph: str, child_graph: str):
    await prisma.models.AgentNode.prisma().create(
        data={
            "agentGraphId": parent_graph,
            "agentGraphVersion": 1,
            "agentBlockId": AgentExecutorBlock().id,
            "constantInput": SafeJson({"graph_id": child_graph, "graph_version": 1}),
        }
    )
