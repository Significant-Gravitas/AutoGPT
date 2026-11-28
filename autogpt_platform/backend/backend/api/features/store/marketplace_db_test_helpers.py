"""Shared fixtures and operations for PostgreSQL Marketplace tests."""

import asyncio

import prisma.enums
import prisma.models

from backend.api.features.store import db as store_db
from backend.data.graph import Graph, create_graph

APPROVED = prisma.enums.SubmissionStatus.APPROVED
PENDING = prisma.enums.SubmissionStatus.PENDING
REJECTED = prisma.enums.SubmissionStatus.REJECTED


async def overlap_reviews(
    monkeypatch, first, second, *, second_read: asyncio.Event | None = None
):
    """Release the withdrawal once the other review blocks or reads the child."""
    withdraw = store_db.withdraw_store_version
    first_locked = asyncio.Event()
    waiting_statement = None

    async def hold_first_withdrawal(tx, version):
        nonlocal waiting_statement
        fallback = await withdraw(tx, version)
        if first_locked.is_set():
            assert waiting_statement, "The second review never waited on the first"
            return fallback
        first_locked.set()
        waiting_statement = await wait_for_blocked_statement(tx, until=second_read)
        return fallback

    monkeypatch.setattr(store_db, "withdraw_store_version", hold_first_withdrawal)
    async with asyncio.timeout(10), asyncio.TaskGroup() as reviews:
        reviews.create_task(first())
        await first_locked.wait()
        reviews.create_task(second())
    if second_read is None:
        assert waiting_statement
    return waiting_statement


async def wait_for_blocked_statement(tx, *, until: asyncio.Event | None = None):
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


async def submit(owner: str, graph_id: str, schedule: str | None = None):
    return await store_db.create_store_submission(
        owner,
        graph_id,
        1,
        graph_id,
        graph_id,
        "Replacement",
        recommended_schedule_cron=schedule,
    )


async def approve(version_id: str, reviewer_id: str):
    await store_db.review_store_submission(
        version_id, True, "", "", reviewer_id, notify_creator=False
    )


async def reject(version_id: str, reviewer_id: str):
    await store_db.review_store_submission(
        version_id, False, "", "", reviewer_id, notify_creator=False
    )


async def seed_creator(tag: str, avatar_url: str | None) -> str:
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


async def seed_listing(
    graph_id: str, owner: str, statuses: list[prisma.enums.SubmissionStatus]
) -> list[str]:
    """Seed a listing and activate its first version if it is approved."""
    await create_graph(Graph(id=graph_id, name=graph_id, description="test"), owner)
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


async def clean_up(owner: str, graph_ids: list[str]) -> None:
    # Deleting a listing cascades to its versions.
    await prisma.models.StoreListing.prisma().delete_many(
        where={"agentGraphId": {"in": graph_ids}}
    )
    await prisma.models.AgentGraph.prisma().delete_many(where={"id": {"in": graph_ids}})
    await prisma.models.User.prisma().delete_many(where={"id": owner})
