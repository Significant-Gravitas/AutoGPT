"""Marketplace cache and media publication after moderation."""

from unittest.mock import AsyncMock
from uuid import uuid4

import prisma.enums
import prisma.models
import pytest
from fakeredis import FakeAsyncRedis, FakeRedis, FakeServer

from backend.api.features.admin import store_admin_routes
from backend.api.features.store import cache as store_cache
from backend.api.features.store import db as store_db
from backend.api.features.store import submission_media
from backend.api.features.store.marketplace_db_test_helpers import (
    APPROVED,
    PENDING,
    REJECTED,
    clean_up,
    reject,
    seed_creator,
    seed_listing,
)
from backend.api.features.store.model import ReviewSubmissionRequest
from backend.util import cache as cache_util
from backend.util.exceptions import NotFoundError


@pytest.mark.asyncio(loop_scope="session")
async def test_rejection_refreshes_a_listing_cached_after_an_overlapping_approval(
    server, monkeypatch
):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        live, pending = await seed_listing(graph_id, owner, [APPROVED, PENDING])
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
        await clean_up(owner, [graph_id])


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
    owner = await seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        live, fallback = await seed_listing(graph_id, owner, [APPROVED, APPROVED])
        versions = prisma.models.StoreListingVersion.prisma()
        image = submission_media.url(owner, "images", "cover.png")
        published = "https://example.test/published/cover.png"
        await versions.update(where={"id": fallback}, data={"imageUrls": [image]})

        async def prepare_avatar(_owner):
            if change == "withdraw-before-copy":
                await reject(fallback, owner)

        async def copy_images(urls, owners):
            assert image in urls and owners == {owner}
            if change == "withdraw-during-copy":
                await reject(fallback, owner)
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
        await reject(live, owner)

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
        await clean_up(owner, [graph_id])
