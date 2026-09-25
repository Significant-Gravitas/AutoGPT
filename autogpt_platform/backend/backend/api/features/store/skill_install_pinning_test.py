from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import prisma.enums
import prisma.models
import pytest

from backend.api.features.experts import experts_db
from backend.api.features.experts.hire_skill_snapshot import (
    LegacySkillSnapshotError,
    PinnedSkill,
    SkillInstallSnapshot,
)
from backend.api.features.store import skill_db
from backend.copilot.tools.skills import ParsedSkill, SkillFile, StoredSkill
from backend.util.exceptions import NotFoundError


def pinned_snapshot():
    return SkillInstallSnapshot(
        release_id="release-original",
        packages=[
            PinnedSkill(slug="research", version_id="original", title="Research")
        ],
    )


@pytest.mark.asyncio
async def test_hire_retry_uses_original_version_and_files_after_listing_changes(
    monkeypatch,
):
    listing = prisma.models.SkillListing.model_construct(
        id="listing", slug="research", activeVersionId="replacement", isDeleted=True
    )
    version = prisma.models.SkillListingVersion.model_construct(
        id="original",
        version=1,
        name="Research",
        description="Research topic",
        body="Original body",
        triggers=[],
        requiredProviders=[],
        scannedSha256=[],
        sourceRepo=None,
        sourceUrl=None,
        license=None,
        skillMarkdown="exact markdown",
        submissionStatus=prisma.enums.SubmissionStatus.APPROVED,
        isDeleted=False,
        isAvailable=True,
        SkillListing=listing,
    )
    expert_lookup = AsyncMock(
        return_value=SimpleNamespace(
            skillInstallSnapshot=pinned_snapshot().model_dump()
        )
    )
    version_lookup = AsyncMock(return_value=[version])
    monkeypatch.setattr(
        prisma.models.Expert,
        "prisma",
        lambda: SimpleNamespace(find_first=expert_lookup),
    )
    monkeypatch.setattr(
        prisma.models.SkillListingVersion,
        "prisma",
        lambda: SimpleNamespace(find_many=version_lookup),
    )
    files = AsyncMock(
        return_value={
            "original": [
                SkillFile(
                    relative_path="scripts/run.py",
                    content=b"original",
                    is_executable=True,
                )
            ]
        }
    )
    store = AsyncMock(
        return_value=[StoredSkill(ParsedSkill("research", "Research", "Body"), False)]
    )
    monkeypatch.setattr(skill_db, "_read_versions_files", files)
    monkeypatch.setattr(skill_db, "store_user_skills", store)
    monkeypatch.setattr(skill_db, "_record_scanned", AsyncMock())

    results = await skill_db.install_pinned_marketplace_skills(
        "owner", "hire", ["research"]
    )

    assert results[0].name == "research"
    files.assert_awaited_once_with(["original"])
    [write] = store.await_args.args[1]
    assert write.body == "Original body"
    assert write.skill_markdown == "exact markdown"
    assert write.files[0].content == b"original"
    assert write.files[0].is_executable is True
    assert expert_lookup.await_args.kwargs["where"]["ownerUserId"] == "owner"


@pytest.mark.asyncio
async def test_pinned_install_cannot_read_another_owners_snapshot(monkeypatch):
    monkeypatch.setattr(
        prisma.models.Expert,
        "prisma",
        lambda: SimpleNamespace(find_first=AsyncMock(return_value=None)),
    )
    read_files = AsyncMock()
    monkeypatch.setattr(skill_db, "_read_versions_files", read_files)
    with pytest.raises(NotFoundError):
        await skill_db.install_pinned_marketplace_skills(
            "stranger", "hire", ["research"]
        )
    read_files.assert_not_awaited()


@pytest.mark.asyncio
async def test_retry_rejects_packages_added_after_original_hire(monkeypatch):
    monkeypatch.setattr(
        prisma.models.Expert,
        "prisma",
        lambda: SimpleNamespace(
            find_first=AsyncMock(
                return_value=SimpleNamespace(
                    skillInstallSnapshot=pinned_snapshot().model_dump()
                )
            )
        ),
    )
    store = AsyncMock()
    monkeypatch.setattr(skill_db, "store_user_skills", store)
    with pytest.raises(ValueError, match="original hire"):
        await skill_db.install_pinned_marketplace_skills(
            "owner", "hire", ["new-package"]
        )
    store.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_hire_fails_before_any_setup_writes(monkeypatch):
    template = SimpleNamespace(id="template")
    legacy = SimpleNamespace(
        id="hire",
        ownerUserId="owner",
        sourceTemplateId="template",
        skillInstallSnapshot=None,
    )
    lookup = AsyncMock(side_effect=[template, legacy])
    monkeypatch.setattr(
        prisma.models.Expert, "prisma", lambda: SimpleNamespace(find_unique=lookup)
    )
    preloads = AsyncMock()
    skills = AsyncMock()
    routines = AsyncMock()
    monkeypatch.setattr(experts_db, "_install_preloads", preloads)
    monkeypatch.setattr(experts_db, "_install_bundled_skills", skills)
    monkeypatch.setattr(experts_db, "install_routines", routines)

    with pytest.raises(LegacySkillSnapshotError):
        await experts_db._install_hire_contents("owner", "hire", "template")

    preloads.assert_not_awaited()
    skills.assert_not_awaited()
    routines.assert_not_awaited()


@pytest.mark.asyncio
async def test_existing_hire_is_not_assigned_a_new_snapshot(monkeypatch):
    existing = SimpleNamespace(
        id="hire", visibility=prisma.enums.ResourceVisibility.PRIVATE, isArchived=False
    )
    tx = SimpleNamespace(
        execute_raw=AsyncMock(),
        expert=SimpleNamespace(
            find_first=AsyncMock(return_value=existing),
            update=AsyncMock(),
            create=AsyncMock(),
        ),
    )

    @asynccontextmanager
    async def transaction():
        yield tx

    capture = AsyncMock()
    monkeypatch.setattr(experts_db, "transaction", transaction)
    monkeypatch.setattr(experts_db, "capture_skill_snapshot", capture)

    result, state = await experts_db._reserve_hired_expert("owner", "template", {})

    assert result is existing and state == "existing"
    capture.assert_not_awaited()
    tx.expert.update.assert_not_awaited()
    tx.expert.create.assert_not_awaited()
