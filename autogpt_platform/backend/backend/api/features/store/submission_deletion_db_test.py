"""Marketplace submission deletion against PostgreSQL."""

import asyncio
from uuid import uuid4

import prisma.models
import pytest

from backend.api.features.store import db as store_db
from backend.api.features.store import exceptions as store_exceptions
from backend.api.features.store.marketplace_db_test_helpers import (
    APPROVED,
    PENDING,
    clean_up,
    seed_creator,
    seed_listing,
    submit,
    wait_for_blocked_statement,
)


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "creator_first", [False, True], ids=["delete-first", "create-first"]
)
async def test_deleting_the_last_submission_preserves_a_concurrent_replacement(
    server, monkeypatch, creator_first: bool
):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        [original] = await seed_listing(graph_id, owner, [PENDING])
        versions = prisma.models.StoreListingVersion.prisma()
        version_actions = type(versions)
        operation = version_actions.create if creator_first else version_actions.count
        first_ready, second_done = asyncio.Event(), asyncio.Event()

        async def overlap(actions, **kwargs):
            result = await operation(actions, **kwargs)
            first_ready.set()
            await wait_for_blocked_statement(actions._client, until=second_done)
            return result

        async def run_second(action, *args):
            await first_ready.wait()
            try:
                return await action(*args)
            finally:
                second_done.set()

        monkeypatch.setattr(version_actions, operation.__name__, overlap)
        delete = store_db.delete_store_submission_or_raise
        async with asyncio.timeout(15), asyncio.TaskGroup() as requests:
            if creator_first:
                replacement = requests.create_task(submit(owner, graph_id))
                requests.create_task(run_second(delete, owner, original))
            else:
                requests.create_task(delete(owner, original))
                replacement = requests.create_task(run_second(submit, owner, graph_id))

        kept = await versions.find_unique_or_raise(
            where={"id": replacement.result().listing_version_id}
        )
        assert kept.submissionStatus == PENDING and not kept.isDeleted
        assert await versions.find_unique(where={"id": original}) is None
    finally:
        await clean_up(owner, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
async def test_creator_deletion_preserves_approved_submissions(server):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        [live] = await seed_listing(graph_id, owner, [APPROVED])
        with pytest.raises(store_exceptions.SubmissionNotFoundError):
            await store_db.delete_store_submission_or_raise(owner, str(uuid4()))
        with pytest.raises(store_exceptions.SubmissionNotFoundError):
            await store_db.delete_store_submission_or_raise("other-user", live)
        with pytest.raises(store_exceptions.InvalidOperationError):
            await store_db.delete_store_submission_or_raise(owner, live)

        assert (
            await store_db.get_store_agent_details(owner, graph_id)
        ).active_version_id == live
    finally:
        await clean_up(owner, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
async def test_creator_can_resubmit_after_deleting_the_latest_replacement(server):
    tag = uuid4().hex[:12]
    owner = await seed_creator(tag, avatar_url="")
    graph_id = f"graph-{tag}"
    try:
        [live] = await seed_listing(graph_id, owner, [APPROVED])
        superseded = await submit(owner, graph_id)
        latest = await submit(owner, graph_id)
        await store_db.delete_store_submission_or_raise(
            owner, latest.listing_version_id
        )

        new = await submit(owner, graph_id)
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
        await clean_up(owner, [graph_id])
