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

from backend.api.features.store import db as store_db
from backend.api.features.store import exceptions as store_exceptions
from backend.api.features.store import submission_media
from backend.blocks.agent import AgentExecutorBlock
from backend.util.exceptions import DatabaseError, NotFoundError
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
async def test_rejection_serializes_with_approval_of_a_parent_workflow(
    server, monkeypatch
):
    tag = uuid4().hex[:12]
    owner = await _seed_creator(tag, avatar_url="")
    parent_graph, child_graph = f"parent-{tag}", f"child-{tag}"
    try:
        [parent] = await _seed_listing(parent_graph, owner, [PENDING])
        [child] = await _seed_listing(child_graph, owner, [PENDING])
        await prisma.models.AgentNode.prisma().create(
            data={
                "agentGraphId": parent_graph,
                "agentGraphVersion": 1,
                "agentBlockId": AgentExecutorBlock().id,
                "constantInput": SafeJson(
                    {"graph_id": child_graph, "graph_version": 1}
                ),
            }
        )
        monkeypatch.setattr(store_db, "ensure_embedding", AsyncMock())
        waiting_statement = await _overlap_reviews(
            monkeypatch,
            lambda: _reject(child, owner),
            lambda: store_db.review_store_submission(
                parent, True, "", "", owner, notify_creator=False
            ),
        )
        assert '"StoreListing"' in waiting_statement, waiting_statement
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
@pytest.mark.parametrize("withdraw_at", ["before-copy", "in-flight", "never"])
async def test_rejection_checks_approval_when_republishing_fallback_media(
    server, monkeypatch, withdraw_at: str
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
            if withdraw_at == "before-copy":
                await _reject(fallback, owner)

        async def copy_images(urls, owners):
            assert image in urls and owners == {owner}
            if withdraw_at == "in-flight":
                await _reject(fallback, owner)
            return {image: published}

        copy = AsyncMock(side_effect=copy_images)
        monkeypatch.setattr(store_db, "publish_creator_avatar", prepare_avatar)
        monkeypatch.setattr(store_db.public_media, "publishing_enabled", lambda: True)
        monkeypatch.setattr(store_db.public_media, "publish_urls", copy)
        await _reject(live, owner)

        assert copy.await_count == (0 if withdraw_at == "before-copy" else 1)
        row = await versions.find_unique_or_raise(where={"id": fallback})
        assert row.submissionStatus == (
            APPROVED if withdraw_at == "never" else REJECTED
        )
        assert row.imageUrls == [published if withdraw_at == "never" else image]
    finally:
        await _clean_up(owner, [graph_id])


async def _overlap_reviews(monkeypatch, first, second):
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
        while True:
            await tx.execute_raw("SELECT pg_stat_clear_snapshot()")
            rows = await tx.query_raw(
                "SELECT query FROM pg_stat_activity WHERE datname = current_database() "
                "AND pg_backend_pid() = ANY(pg_blocking_pids(pid))"
            )
            if rows:
                waiting_statement = rows[0]["query"]
                return fallback
            await asyncio.sleep(0.02)

    monkeypatch.setattr(store_db, "_withdraw_store_version", hold_first_withdrawal)
    async with asyncio.timeout(10), asyncio.TaskGroup() as reviews:
        reviews.create_task(first())
        await first_locked.wait()
        reviews.create_task(second())
    assert waiting_statement
    return waiting_statement


async def _reject(version_id: str, reviewer_id: str):
    await store_db.review_store_submission(
        version_id, False, "", "", reviewer_id, notify_creator=False
    )


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
