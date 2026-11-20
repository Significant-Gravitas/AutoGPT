"""Direct Marketplace moderation and fallback selection against PostgreSQL."""

import asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

import prisma.enums
import prisma.models
import pytest

from backend.api.features.library.db import (
    is_store_listing_version_available_for_install,
)
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
    submit,
    wait_for_blocked_statement,
)
from backend.data.db import transaction
from backend.util.exceptions import ConflictError, DatabaseError, NotFoundError


@pytest.mark.asyncio(loop_scope="session")
async def test_live_version_can_be_installed_while_the_next_one_is_embedding(
    server, monkeypatch
):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        live, pending = await seed_listing(graph_id, owner, [APPROVED, PENDING])
        install_checks = []

        async def install_during_embedding(**kwargs):
            async with transaction() as tx:
                install_checks.append(
                    await is_store_listing_version_available_for_install(
                        live, tx=tx, lock_rows=True
                    )
                )

        monkeypatch.setattr(store_db, "ensure_embedding", install_during_embedding)
        async with asyncio.timeout(15):
            await approve(pending, owner)
        assert install_checks == [True]
        page = await store_db.get_store_agent_details(owner, graph_id)
        assert page.active_version_id == pending
    finally:
        await clean_up(owner, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
async def test_rejection_of_a_submission_deleted_during_review(server, monkeypatch):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        live, pending = await seed_listing(graph_id, owner, [APPROVED, PENDING])
        versions = prisma.models.StoreListingVersion.prisma()
        read = type(versions).find_unique

        async def delete_after_read(actions, **kwargs):
            version = await read(actions, **kwargs)
            if (
                version
                and version.id == pending
                and "AgentGraph" in kwargs.get("include", {})
            ):
                await store_db.delete_store_submission_or_raise(owner, pending)
            return version

        with monkeypatch.context() as overlap:
            overlap.setattr(type(versions), "find_unique", delete_after_read)
            with pytest.raises(ConflictError, match="Refresh"):
                await reject(pending, owner)

        with pytest.raises(NotFoundError):
            await reject(pending, owner)
        assert await versions.find_unique(where={"id": pending}) is None
        page = await store_db.get_store_agent_details(owner, graph_id)
        assert page.active_version_id == live
    finally:
        await clean_up(owner, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    ("available", "deleted"), [(False, False), (True, True)], ids=["hidden", "deleted"]
)
async def test_rejection_serves_the_newest_remaining_public_version(
    server, available: bool, deleted: bool
):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        _, older, fallback, unlisted, live = await seed_listing(
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

        await reject(older, owner)
        assert (
            await store_db.get_store_agent_details(owner, graph_id)
        ).active_version_id == live

        await reject(live, owner)
        page = await store_db.get_store_agent_details(owner, graph_id)
        assert page.active_version_id == fallback
    finally:
        await clean_up(owner, [graph_id])


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
    owner = await seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        first, remaining = await seed_listing(graph_id, owner, [APPROVED, APPROVED])
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
            await reject(version_id, owner)
            listing = await listings.find_unique_or_raise(
                where={"agentGraphId": graph_id}
            )
            assert listing.activeVersionId is None
            assert listing.hasApprovedVersion is approval_remains
            with pytest.raises(NotFoundError):
                await store_db.get_store_agent_details(owner, graph_id)
    finally:
        await clean_up(owner, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
async def test_rejection_commits_the_verdict_and_public_pointer_together(
    server, monkeypatch
):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        live, survivor, newest = await seed_listing(graph_id, owner, [APPROVED] * 3)
        with pytest.raises(DatabaseError):
            await reject(live, f"missing-reviewer-{tag}")
        page = await store_db.get_store_agent_details(owner, graph_id)
        assert page.active_version_id == live

        await overlap_reviews(
            monkeypatch, lambda: reject(newest, owner), lambda: reject(live, owner)
        )
        page = await store_db.get_store_agent_details(owner, graph_id)
        assert page.active_version_id == survivor
    finally:
        await clean_up(owner, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
async def test_rejection_and_replacement_of_the_same_pending_submission(
    server, monkeypatch
):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        live, pending = await seed_listing(graph_id, owner, [APPROVED, PENDING])
        versions = prisma.models.StoreListingVersion.prisma()
        create_version = type(versions).create
        ready_to_insert = asyncio.Event()

        async def insert_after_rejection_waits(actions, **kwargs):
            ready_to_insert.set()
            await wait_for_blocked_statement(actions._client)
            return await create_version(actions, **kwargs)

        monkeypatch.setattr(type(versions), "create", insert_after_rejection_waits)
        async with asyncio.timeout(15), asyncio.TaskGroup() as requests:
            replacing = requests.create_task(
                store_db.create_store_submission(
                    owner, graph_id, 1, graph_id, graph_id, "Replacement"
                )
            )
            await ready_to_insert.wait()
            requests.create_task(reject(pending, owner))

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
        await clean_up(owner, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "delete_replaced", [False, True], ids=["superseded", "removed"]
)
async def test_approval_refuses_a_submission_replaced_during_review(
    server, monkeypatch, delete_replaced: bool
):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        live, pending = await seed_listing(graph_id, owner, [APPROVED, PENDING])
        get_sub_graphs = store_db.get_sub_graphs

        async def replace_after_review_read(graph):
            await submit(owner, graph_id)
            if delete_replaced:
                await store_db.delete_store_submission_or_raise(owner, pending)
            return await get_sub_graphs(graph)

        embedding = AsyncMock()
        monkeypatch.setattr(store_db, "ensure_embedding", embedding)
        with monkeypatch.context() as overlap:
            overlap.setattr(store_db, "get_sub_graphs", replace_after_review_read)
            with pytest.raises(ConflictError):
                await approve(pending, owner)

        versions = prisma.models.StoreListingVersion.prisma()
        old = await versions.find_unique(where={"id": pending})
        if delete_replaced:
            assert old is None
        else:
            assert old and old.isDeleted and old.submissionStatus == PENDING
            embedding.reset_mock()
            with pytest.raises(ConflictError):
                await approve(pending, owner)
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
        await approve(replacement.id, owner)
        assert (
            await store_db.get_store_agent_details(owner, graph_id)
        ).active_version_id == replacement.id
    finally:
        await clean_up(owner, [graph_id])
