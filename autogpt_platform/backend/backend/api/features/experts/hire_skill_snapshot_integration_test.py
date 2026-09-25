"""Real PostgreSQL capture/locking tests, isolated from application services."""

import asyncio
import hashlib
import os
from unittest.mock import AsyncMock
from urllib.parse import urlparse
from uuid import uuid4

import prisma
import prisma.models
import pytest
import pytest_asyncio

from backend.api.features.experts import experts_db
from backend.api.features.experts.errors import ExpertTemplateNotFoundError
from backend.api.features.experts.hire_skill_snapshot import (
    capture_skill_snapshot,
    read_skill_snapshot,
)
from backend.api.features.store import skill_db
from backend.copilot.tools.skills import ParsedSkill, StoredSkill

pytestmark = pytest.mark.catalogue_isolated


@pytest.fixture(scope="session")
def graph_cleanup():
    """This file uses only its own database connection and uniquely named rows."""
    yield


@pytest_asyncio.fixture
async def db(monkeypatch):
    url = os.environ.get("CATALOGUE_TEST_DATABASE_URL")
    if not url:
        pytest.skip("CATALOGUE_TEST_DATABASE_URL must name a dedicated local database")
    if urlparse(url).hostname not in {"localhost", "127.0.0.1", "::1"}:
        pytest.fail("Only a dedicated local database is supported")
    client = prisma.Prisma(datasource={"url": url})
    await client.connect()
    monkeypatch.setattr(experts_db, "transaction", client.tx)
    monkeypatch.setattr(experts_db, "is_feature_enabled", AsyncMock(return_value=True))
    monkeypatch.setattr(prisma.models.Expert, "prisma", lambda: client.expert)
    monkeypatch.setattr(
        prisma.models.SkillListing, "prisma", lambda: client.skilllisting
    )
    monkeypatch.setattr(
        prisma.models.SkillListingVersion, "prisma", lambda: client.skilllistingversion
    )
    monkeypatch.setattr(
        prisma.models.SkillListingFile, "prisma", lambda: client.skilllistingfile
    )
    token = uuid4().hex
    user = await client.user.create(
        data={"id": str(uuid4()), "email": f"{token}@example.invalid"}
    )
    template = await client.expert.create(
        data={
            "name": f"test-{token}",
            "role": "Test",
            "identity": "Test template",
            "isTemplate": True,
        }
    )
    listing = await client.skilllisting.create(
        data={"slug": f"test-{token}", "hasApprovedVersion": True}
    )
    raw = f"---\nname: {listing.slug}\ndescription: Original package\nmetadata:\n  preserved: true\n---\n\nOriginal body.\n"
    version = await client.skilllistingversion.create(
        data={
            "skillListingId": listing.id,
            "name": "Original package",
            "description": "Original package",
            "body": "Original body.",
            "categories": [],
            "submissionStatus": "APPROVED",
            "skillMarkdown": raw,
        }
    )
    await client.skilllisting.update(
        where={"id": listing.id}, data={"activeVersionId": version.id}
    )
    await client.skilllistingfile.create(
        data={
            "skillListingVersionId": version.id,
            "relativePath": "scripts/run.py",
            "content": prisma.Base64.encode(b"original script\n"),
            "mimeType": "text/x-python",
            "sizeBytes": 16,
            "sha256": hashlib.sha256(b"original script\n").hexdigest(),
            "isExecutable": True,
        }
    )
    await client.expertskilllisting.create(
        data={"expertId": template.id, "skillListingId": listing.id, "position": 0}
    )
    try:
        yield client, user, template, listing, version, raw
    finally:
        await client.expert.delete_many(where={"sourceTemplateId": template.id})
        await client.expert.delete(where={"id": template.id})
        await client.skilllistingversion.delete_many(
            where={"skillListingId": listing.id}
        )
        await client.skilllisting.delete(where={"id": listing.id})
        await client.user.delete(where={"id": user.id})
        await client.disconnect()


@pytest.mark.asyncio
async def test_created_hire_persists_versions_and_retry_keeps_retired_original_package(
    db, monkeypatch
):
    client, user, template, listing, version, raw = db
    hire, state = await experts_db._reserve_hired_expert(
        user.id,
        template.id,
        {
            "ownerUserId": user.id,
            "sourceTemplateId": template.id,
            "name": template.name,
            "role": "Test",
            "identity": "Test hire",
        },
    )
    assert state == "created"
    snapshot = read_skill_snapshot(hire.skillInstallSnapshot)
    assert [(p.slug, p.version_id) for p in snapshot.packages] == [
        (listing.slug, version.id)
    ]

    replacement = await client.skilllistingversion.create(
        data={
            "skillListingId": listing.id,
            "version": 2,
            "name": "New package",
            "description": "Changed",
            "body": "Changed body",
            "categories": [],
            "submissionStatus": "APPROVED",
        }
    )
    await client.skilllisting.update(
        where={"id": listing.id},
        data={"activeVersionId": replacement.id, "isDeleted": True},
    )
    await client.expertskilllisting.delete_many(where={"expertId": template.id})
    store = AsyncMock(
        return_value=[
            StoredSkill(
                ParsedSkill(listing.slug, "Original package", "Original body."), False
            )
        ]
    )
    monkeypatch.setattr(skill_db, "store_user_skills", store)
    outcome = await skill_db.install_pinned_marketplace_skills(
        user.id, hire.id, [listing.slug]
    )

    assert outcome[0].name == listing.slug
    [write] = store.await_args.args[1]
    assert write.skill_markdown == raw
    assert write.files[0].content == b"original script\n"
    assert write.files[0].is_executable
    again, state = await experts_db._reserve_hired_expert(user.id, template.id, {})
    assert (
        state == "existing" and again.skillInstallSnapshot == hire.skillInstallSnapshot
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("skills_enabled", [True, False])
async def test_withdrawal_after_initial_template_read_refuses_hire_creation(
    db, monkeypatch, skills_enabled
):
    client, user, template, _, _, _ = db
    monkeypatch.setattr(
        experts_db, "is_feature_enabled", AsyncMock(return_value=skills_enabled)
    )
    stale_create_data = {
        "ownerUserId": user.id,
        "sourceTemplateId": template.id,
        "name": template.name,
        "role": template.role,
        "identity": template.identity,
    }
    async with client.tx() as tx:
        await tx.query_raw(
            "SELECT id FROM \"CatalogueState\" WHERE id = 'marketplace' FOR UPDATE"
        )
        await tx.expert.update(where={"id": template.id}, data={"isArchived": True})

    with pytest.raises(ExpertTemplateNotFoundError):
        await experts_db._reserve_hired_expert(user.id, template.id, stale_create_data)
    assert await client.expert.count(where={"ownerUserId": user.id}) == 0


@pytest.mark.asyncio
async def test_hire_shared_lock_blocks_activation_until_snapshot_transaction_finishes(
    db,
):
    client, _, template, _, version, _ = db
    acquired = asyncio.Event()

    async def activate():
        async with client.tx() as tx:
            await tx.query_raw(
                "SELECT id FROM \"CatalogueState\" WHERE id = 'marketplace' FOR UPDATE"
            )
            acquired.set()

    task = None
    try:
        async with client.tx() as tx:
            snapshot = await capture_skill_snapshot(tx, template.id)
            assert snapshot.packages[0].version_id == version.id
            task = asyncio.create_task(activate())
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(acquired.wait(), timeout=0.2)
        await asyncio.wait_for(task, timeout=5)
        assert acquired.is_set()
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
