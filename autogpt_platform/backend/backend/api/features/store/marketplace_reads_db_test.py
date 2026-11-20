"""Public marketplace reads against a real Postgres.

The creator views are SQL and the listing-version filter is a Prisma query,
so ``db_test.py``'s mocks cannot run either; ``server`` starts the test stack.
"""

import asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

import prisma.enums
import prisma.models
import pytest
from fakeredis import FakeAsyncRedis, FakeRedis, FakeServer

from backend.api.features.admin import store_admin_routes
from backend.api.features.store import cache as store_cache
from backend.api.features.store import db as store_db
from backend.api.features.store import exceptions as store_exceptions
from backend.api.features.store import submission_media
from backend.api.features.store.model import ReviewSubmissionRequest
from backend.blocks.agent import AgentExecutorBlock
from backend.data.graph import is_graph_published_in_marketplace
from backend.util import cache as cache_util
from backend.util.exceptions import ConflictError, DatabaseError, NotFoundError
from backend.util.json import SafeJson

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


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    ("available", "deleted"), [(False, False), (True, True)], ids=["hidden", "deleted"]
)
async def test_rejection_serves_the_newest_remaining_public_version(
    server, available: bool, deleted: bool
):
    tag = uuid4().hex[:12]
    owner = await _seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        _, older, fallback, unlisted, live = await _seed_listing(
            graph_id, owner, [APPROVED] * 5
        )
        await prisma.models.StoreListing.prisma().update(
            where={"agentGraphId": graph_id},
            data={"ActiveVersion": {"connect": {"id": live}}},
        )
        await prisma.models.StoreListingVersion.prisma().update(
            where={"id": unlisted},
            data={"isAvailable": available, "isDeleted": deleted},
        )

        await _reject(older, owner)
        assert (
            await store_db.get_store_agent_details(owner, graph_id)
        ).active_version_id == live

        await _reject(live, owner)
        page = await store_db.get_store_agent_details(owner, graph_id)
        assert page.active_version_id == fallback
    finally:
        await _clean_up(owner, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    ("listed", "remaining_available"),
    [(True, False), (False, False), (False, True)],
    ids=["published", "hidden-only", "unlisted-with-available-version"],
)
async def test_rejection_can_leave_an_approved_version_without_publishing_it(
    server, listed: bool, remaining_available: bool
):
    tag = uuid4().hex[:12]
    owner = await _seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        first, remaining = await _seed_listing(graph_id, owner, [APPROVED, APPROVED])
        listings = prisma.models.StoreListing.prisma()
        versions = prisma.models.StoreListingVersion.prisma()
        await versions.update(where={"id": first}, data={"isAvailable": listed})
        await versions.update(
            where={"id": remaining}, data={"isAvailable": remaining_available}
        )
        if not listed:
            await listings.update(
                where={"agentGraphId": graph_id},
                data={"ActiveVersion": {"disconnect": True}},
            )

        for version_id, approval_remains in [(first, True), (remaining, False)]:
            await _reject(version_id, owner)
            listing = await listings.find_unique_or_raise(
                where={"agentGraphId": graph_id}
            )
            assert listing.activeVersionId is None
            assert listing.hasApprovedVersion is approval_remains
            with pytest.raises(NotFoundError):
                await store_db.get_store_agent_details(owner, graph_id)
    finally:
        await _clean_up(owner, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
async def test_rejection_commits_the_verdict_and_public_pointer_together(
    server, monkeypatch
):
    tag = uuid4().hex[:12]
    owner = await _seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        live, survivor, newest = await _seed_listing(graph_id, owner, [APPROVED] * 3)
        with pytest.raises(DatabaseError):
            await _reject(live, f"missing-reviewer-{tag}")
        page = await store_db.get_store_agent_details(owner, graph_id)
        assert page.active_version_id == live

        await _overlap_reviews(
            monkeypatch, lambda: _reject(newest, owner), lambda: _reject(live, owner)
        )
        page = await store_db.get_store_agent_details(owner, graph_id)
        assert page.active_version_id == survivor
    finally:
        await _clean_up(owner, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
async def test_rejection_and_replacement_of_the_same_pending_submission(
    server, monkeypatch
):
    tag = uuid4().hex[:12]
    owner = await _seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        live, pending = await _seed_listing(graph_id, owner, [APPROVED, PENDING])
        versions = prisma.models.StoreListingVersion.prisma()
        create_version = type(versions).create
        ready_to_insert = asyncio.Event()

        async def insert_after_rejection_waits(actions, **kwargs):
            ready_to_insert.set()
            await _wait_for_blocked_statement(actions._client)
            return await create_version(actions, **kwargs)

        monkeypatch.setattr(type(versions), "create", insert_after_rejection_waits)
        async with asyncio.timeout(15), asyncio.TaskGroup() as requests:
            replacing = requests.create_task(
                store_db.create_store_submission(
                    owner, graph_id, 1, graph_id, graph_id, "Replacement"
                )
            )
            await ready_to_insert.wait()
            requests.create_task(_reject(pending, owner))

        old = await versions.find_unique_or_raise(where={"id": pending})
        new = await versions.find_unique_or_raise(
            where={"id": replacing.result().listing_version_id}
        )
        assert old.isDeleted and old.submissionStatus == REJECTED
        assert not new.isDeleted and new.submissionStatus == PENDING
        assert (
            await store_db.get_store_agent_details(owner, graph_id)
        ).active_version_id == live
    finally:
        await _clean_up(owner, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "delete_replaced", [False, True], ids=["superseded", "removed"]
)
async def test_approval_refuses_a_submission_replaced_during_review(
    server, monkeypatch, delete_replaced: bool
):
    tag = uuid4().hex[:12]
    owner = await _seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        live, pending = await _seed_listing(graph_id, owner, [APPROVED, PENDING])
        get_sub_graphs = store_db.get_sub_graphs

        async def replace_after_review_read(graph):
            await _submit(owner, graph_id)
            if delete_replaced:
                await store_db.delete_store_submission_or_raise(owner, pending)
            return await get_sub_graphs(graph)

        embedding = AsyncMock()
        monkeypatch.setattr(store_db, "ensure_embedding", embedding)
        with monkeypatch.context() as overlap:
            overlap.setattr(store_db, "get_sub_graphs", replace_after_review_read)
            with pytest.raises(ConflictError):
                await _approve(pending, owner)

        versions = prisma.models.StoreListingVersion.prisma()
        old = await versions.find_unique(where={"id": pending})
        if delete_replaced:
            assert old is None
        else:
            assert old and old.isDeleted and old.submissionStatus == PENDING
            embedding.reset_mock()
            with pytest.raises(ConflictError):
                await _approve(pending, owner)
            embedding.assert_not_awaited()
        assert (
            await store_db.get_store_agent_details(owner, graph_id)
        ).active_version_id == live
        replacement = await versions.find_first_or_raise(
            where={
                "agentGraphId": graph_id,
                "isDeleted": False,
                "submissionStatus": PENDING,
            }
        )
        await _approve(replacement.id, owner)
        assert (
            await store_db.get_store_agent_details(owner, graph_id)
        ).active_version_id == replacement.id
    finally:
        await _clean_up(owner, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
async def test_rejection_refreshes_a_listing_cached_after_an_overlapping_approval(
    server, monkeypatch
):
    tag = uuid4().hex[:12]
    owner = await _seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        live, pending = await _seed_listing(graph_id, owner, [APPROVED, PENDING])
        request = ReviewSubmissionRequest(
            store_listing_version_id=pending, is_approved=False, comments=""
        )
        review = store_db.review_store_submission
        read_details = AsyncMock(wraps=store_db.get_store_agent_details)
        monkeypatch.setattr(store_db, "get_store_agent_details", read_details)

        async def review_with_overlap(**kwargs):
            if not kwargs["is_approved"]:
                await store_admin_routes.review_submission(
                    pending,
                    request.model_copy(update={"is_approved": True}),
                    user_id=owner,
                )
                for _ in range(2):
                    page = await store_cache.get_cached_agent_details(owner, graph_id)
                    assert page.active_version_id == pending
                read_details.assert_awaited_once()
            return await review(**kwargs, notify_creator=False)

        monkeypatch.setattr(store_db, "review_store_submission", review_with_overlap)
        monkeypatch.setattr(store_db, "ensure_embedding", AsyncMock())
        redis_server = FakeServer()
        with FakeRedis(server=redis_server) as redis:
            monkeypatch.setattr(cache_util, "_get_redis", lambda: redis)
            async with FakeAsyncRedis(server=redis_server) as async_redis:
                monkeypatch.setattr(
                    cache_util, "_get_async_redis", AsyncMock(return_value=async_redis)
                )
                await store_admin_routes.review_submission(
                    pending, request, user_id=owner
                )
                assert (
                    await store_db.get_store_agent_details(owner, graph_id)
                ).active_version_id == live
                assert (
                    await store_cache.get_cached_agent_details(owner, graph_id)
                ).active_version_id == live
    finally:
        await _clean_up(owner, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
async def test_creator_can_resubmit_after_deleting_the_latest_replacement(server):
    tag = uuid4().hex[:12]
    owner = await _seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        [live] = await _seed_listing(graph_id, owner, [APPROVED])
        superseded = await _submit(owner, graph_id)
        latest = await _submit(owner, graph_id)
        await store_db.delete_store_submission_or_raise(
            owner, latest.listing_version_id
        )

        new = await _submit(owner, graph_id)
        assert (
            superseded.listing_version,
            latest.listing_version,
            new.listing_version,
        ) == (2, 3, 3)
        assert new.listing_version_id != latest.listing_version_id
        assert (
            await store_db.get_store_agent_details(owner, graph_id)
        ).active_version_id == live
    finally:
        await _clean_up(owner, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
async def test_rejection_serializes_with_approval_of_a_parent_workflow(
    server, monkeypatch
):
    tag = uuid4().hex[:12]
    owner = await _seed_creator(tag, avatar_url="")
    parent_graph, child_graph = f"parent-{tag}", f"child-{tag}"
    try:
        parent, child = await _seed_parent_with_child(
            owner, parent_graph, child_graph, PENDING
        )
        monkeypatch.setattr(store_db, "ensure_embedding", AsyncMock())
        waiting_statement = await _overlap_reviews(
            monkeypatch,
            lambda: _reject(child, owner),
            lambda: _approve(parent, owner),
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
        await _clean_up(owner, [parent_graph, child_graph])


@pytest.mark.asyncio(loop_scope="session")
async def test_rejection_of_an_approved_child_stands_when_parent_approval_overlaps(
    server, monkeypatch
):
    tag = uuid4().hex[:12]
    owner = await _seed_creator(tag, avatar_url="")
    parent_graph, child_graph = f"parent-{tag}", f"child-{tag}"
    try:
        parent, child = await _seed_parent_with_child(
            owner, parent_graph, child_graph, APPROVED
        )
        parent_checked_child = asyncio.Event()
        withdrawal_committed = asyncio.Event()

        async def reject_child():
            await _reject(child, owner)
            withdrawal_committed.set()

        async def finish_parent_after_withdrawal(**kwargs):
            parent_checked_child.set()
            await withdrawal_committed.wait()

        monkeypatch.setattr(
            store_db, "ensure_embedding", finish_parent_after_withdrawal
        )
        await _overlap_reviews(
            monkeypatch,
            reject_child,
            lambda: _approve(parent, owner),
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
        await _clean_up(owner, [parent_graph, child_graph])


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "explicit_approval", [False, True], ids=["parent-only", "also-approved-directly"]
)
async def test_parent_approval_handles_a_child_withdrawn_before_its_first_read(
    server, monkeypatch, explicit_approval: bool
):
    tag = uuid4().hex[:12]
    owner = await _seed_creator(tag, avatar_url="")
    parent_graph, child_graph = f"parent-{tag}", f"child-{tag}"
    try:
        parent, child = await _seed_parent_with_child(
            owner, parent_graph, child_graph, APPROVED
        )
        get_sub_graphs = store_db.get_sub_graphs

        async def withdraw_before_child_read(graph):
            if graph.id == parent_graph:
                await _reject(child, owner)
            return await get_sub_graphs(graph)

        monkeypatch.setattr(store_db, "get_sub_graphs", withdraw_before_child_read)
        monkeypatch.setattr(store_db, "ensure_embedding", AsyncMock())
        if explicit_approval:
            _after_child_listing_read(
                monkeypatch, child_graph, lambda: _approve(child, owner)
            )
        await _approve(parent, owner)

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
        await _clean_up(owner, [parent_graph, child_graph])


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "delete_superseded", [False, True], ids=["superseded", "deleted"]
)
async def test_parent_review_rolls_back_when_the_child_submission_is_replaced(
    server, monkeypatch, delete_superseded: bool
):
    tag = uuid4().hex[:12]
    owner = await _seed_creator(tag, avatar_url="")
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
                await _approve(parent, owner)

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

        await _approve(parent, owner)
        assert (
            await versions.find_unique_or_raise(where={"id": replacement.id})
        ).submissionStatus == APPROVED
        assert (
            await store_db.get_store_agent_details(owner, parent_graph)
        ).active_version_id == parent
    finally:
        await _clean_up(owner, [parent_graph, child_graph])


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    ("change", "expected_status", "expected_private_file"),
    [
        ("withdraw-before-copy", REJECTED, "cover.png"),
        ("withdraw-during-copy", REJECTED, "cover.png"),
        ("edit-during-copy", APPROVED, "replacement.png"),
        ("none", APPROVED, None),
    ],
)
async def test_rejection_checks_approval_when_republishing_fallback_media(
    server, monkeypatch, change: str, expected_status, expected_private_file: str | None
):
    tag = uuid4().hex[:12]
    owner = await _seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        live, fallback = await _seed_listing(graph_id, owner, [APPROVED, APPROVED])
        versions = prisma.models.StoreListingVersion.prisma()
        image = submission_media.url(owner, "images", "cover.png")
        published = "https://example.test/published/cover.png"
        await versions.update(where={"id": fallback}, data={"imageUrls": [image]})

        async def prepare_avatar(_owner):
            if change == "withdraw-before-copy":
                await _reject(fallback, owner)

        async def copy_images(urls, owners):
            assert image in urls and owners == {owner}
            if change == "withdraw-during-copy":
                await _reject(fallback, owner)
            elif change == "edit-during-copy":
                await versions.update(
                    where={"id": fallback},
                    data={
                        "imageUrls": [
                            submission_media.url(owner, "images", "replacement.png")
                        ]
                    },
                )
            return {image: published}

        copy = AsyncMock(side_effect=copy_images)
        monkeypatch.setattr(store_db, "publish_creator_avatar", prepare_avatar)
        monkeypatch.setattr(store_db.public_media, "publishing_enabled", lambda: True)
        monkeypatch.setattr(store_db.public_media, "publish_urls", copy)
        await _reject(live, owner)

        assert copy.await_count == (0 if change == "withdraw-before-copy" else 1)
        row = await versions.find_unique_or_raise(where={"id": fallback})
        assert row.submissionStatus == expected_status
        assert row.imageUrls == [
            (
                submission_media.url(owner, "images", expected_private_file)
                if expected_private_file is not None
                else published
            )
        ]
        if expected_status == REJECTED:
            listing = await prisma.models.StoreListing.prisma().find_unique_or_raise(
                where={"agentGraphId": graph_id}
            )
            assert listing.activeVersionId is None
            with pytest.raises(NotFoundError):
                await store_db.get_store_agent_details(owner, graph_id)
    finally:
        await _clean_up(owner, [graph_id])


async def _overlap_reviews(
    monkeypatch, first, second, *, second_read: asyncio.Event | None = None
):
    """Release the withdrawal once the other review blocks or reads the child."""
    withdraw = store_db._withdraw_store_version
    first_locked = asyncio.Event()
    waiting_statement = None

    async def hold_first_withdrawal(tx, version):
        nonlocal waiting_statement
        fallback = await withdraw(tx, version)
        if first_locked.is_set():
            assert waiting_statement, "The second review never waited on the first"
            return fallback
        first_locked.set()
        waiting_statement = await _wait_for_blocked_statement(tx, until=second_read)
        return fallback

    monkeypatch.setattr(store_db, "_withdraw_store_version", hold_first_withdrawal)
    async with asyncio.timeout(10), asyncio.TaskGroup() as reviews:
        reviews.create_task(first())
        await first_locked.wait()
        reviews.create_task(second())
    if second_read is None:
        assert waiting_statement
    return waiting_statement


async def _wait_for_blocked_statement(tx, *, until: asyncio.Event | None = None):
    while until is None or not until.is_set():
        await tx.execute_raw("SELECT pg_stat_clear_snapshot()")
        rows = await tx.query_raw(
            "SELECT query FROM pg_stat_activity WHERE datname = current_database() "
            "AND pg_backend_pid() = ANY(pg_blocking_pids(pid))"
        )
        if rows:
            return rows[0]["query"]
        await asyncio.sleep(0.02)
    return None


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


async def _submit(owner: str, graph_id: str):
    return await store_db.create_store_submission(
        owner, graph_id, 1, graph_id, graph_id, "Replacement"
    )


async def _approve(version_id: str, reviewer_id: str):
    await store_db.review_store_submission(
        version_id, True, "", "", reviewer_id, notify_creator=False
    )


async def _reject(version_id: str, reviewer_id: str):
    await store_db.review_store_submission(
        version_id, False, "", "", reviewer_id, notify_creator=False
    )


async def _seed_parent_with_child(
    owner: str,
    parent_graph: str,
    child_graph: str,
    child_status: prisma.enums.SubmissionStatus,
) -> tuple[str, str]:
    [parent] = await _seed_listing(parent_graph, owner, [PENDING])
    [child] = await _seed_listing(child_graph, owner, [child_status])
    await prisma.models.AgentNode.prisma().create(
        data={
            "agentGraphId": parent_graph,
            "agentGraphVersion": 1,
            "agentBlockId": AgentExecutorBlock().id,
            "constantInput": SafeJson({"graph_id": child_graph, "graph_version": 1}),
        }
    )
    return parent, child


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
    """Seed a listing and activate its first version if it is approved."""
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
    if statuses and statuses[0] == APPROVED:
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
