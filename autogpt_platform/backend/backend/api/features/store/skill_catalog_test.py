"""The publisher: one catalog commit in, marketplace rows out, with versions
that are immutable and content-addressed so re-publishing is a no-op and
rolling back is publishing an older commit."""

import uuid
from pathlib import Path

import prisma.enums
import prisma.models
import pytest

from backend.api.features.store import skill_catalog, skill_model
from backend.api.features.store.skill_catalog import (
    CatalogConflictError,
    publish_catalog,
)
from backend.api.features.store.skill_catalog_fixture import (
    expert_yaml,
    skill_md,
    write_catalog,
)
from backend.api.features.store.skill_catalog_release import load_release
from backend.api.features.store.skill_db_test import _make_listing
from backend.api.features.store.skill_submission_db import snapshot_version_files
from backend.util.test import SpinTestServer

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest.fixture(autouse=True)
async def clean_marketplace(server: SpinTestServer):
    """These tests own the platform listings and templates they create; the
    marketplace tests in skill_db_test.py need the table empty, so leave it
    that way."""
    yield
    await prisma.models.ExpertSkillListing.prisma().delete_many()
    await prisma.models.Expert.prisma().delete_many(
        where={"isTemplate": True, "templateKey": {"startswith": "cat-"}}
    )
    await prisma.models.SkillListingVersion.prisma().delete_many()
    await prisma.models.SkillListing.prisma().delete_many()
    await prisma.models.SkillCatalogRelease.prisma().delete_many()


def _key() -> str:
    return f"cat-{uuid.uuid4().hex[:8]}"


async def _publish(root: Path, *, revision: str = "a" * 40, **kwargs):
    return await publish_catalog(
        load_release(root), repository="test/catalog", revision=revision, **kwargs
    )


async def _listing(slug: str) -> prisma.models.SkillListing:
    listing = await prisma.models.SkillListing.prisma().find_unique(
        where={"slug": slug}, include={"ActiveVersion": {"include": {"Files": True}}}
    )
    assert listing is not None
    return listing


async def _versions(slug: str) -> list[prisma.models.SkillListingVersion]:
    return await prisma.models.SkillListingVersion.prisma().find_many(
        where={"SkillListing": {"is": {"slug": slug}}}, order={"version": "asc"}
    )


async def test_first_publish_creates_listings_versions_files_and_templates(
    tmp_path: Path,
):
    key = _key()
    write_catalog(
        tmp_path,
        {"cold-email": {"references/a.md": "# a\n"}, "warm-intro": {}},
        experts={key: expert_yaml(key, ["warm-intro", "cold-email"])},
    )

    summary = await _publish(tmp_path)

    assert sorted(summary.created) == ["cold-email", "warm-intro"]
    assert (summary.updated, summary.unchanged, summary.retired) == ([], 0, [])
    assert summary.experts == {"created": [key], "updated": [], "retired": []}
    cold = await _listing("cold-email")
    assert cold.hasApprovedVersion and not cold.isDeleted
    assert cold.owningUserId is None and cold.owningOrgId is None
    version = cold.ActiveVersion
    assert version is not None
    assert version.skillMarkdown == skill_md("cold-email")
    assert version.packageSha256 == load_release(tmp_path).packages[0].package_sha256
    assert version.submissionStatus == prisma.enums.SubmissionStatus.APPROVED
    assert [(f.relativePath, f.content.decode()) for f in version.Files or []] == [
        ("references/a.md", b"# a\n")
    ]
    template = await prisma.models.Expert.prisma().find_unique(
        where={"templateKey": key}, include={"BundledSkills": True}
    )
    assert template is not None and template.isTemplate and not template.isArchived
    bundled = sorted(template.BundledSkills or [], key=lambda b: b.position)
    assert [b.skillListingId for b in bundled] == [
        (await _listing("warm-intro")).id,
        cold.id,
    ]
    release = await prisma.models.SkillCatalogRelease.prisma().find_unique(
        where={"id": summary.release_id or ""}
    )
    assert release is not None
    assert (release.revision, release.releaseKey) == ("a" * 40, "test-release")


async def test_republishing_the_same_commit_changes_nothing(tmp_path: Path):
    write_catalog(tmp_path, {"cold-email": {"references/a.md": "# a\n"}})
    first = await _publish(tmp_path)

    second = await _publish(tmp_path)

    assert first.created == ["cold-email"]
    assert (second.created, second.updated, second.unchanged) == ([], [], 1)
    assert len(await _versions("cold-email")) == 1


async def test_a_changed_package_gets_a_new_version_and_the_old_one_stays(
    tmp_path: Path,
):
    v1 = write_catalog(tmp_path / "v1", {"cold-email": {"references/a.md": "# a\n"}})
    await _publish(v1, revision="1" * 40)
    v2 = write_catalog(
        tmp_path / "v2",
        {
            "cold-email": {
                "references/a.md": "# a, revised\n",
                "references/b.md": "# b\n",
            }
        },
    )

    summary = await _publish(v2, revision="2" * 40)

    assert summary.updated == ["cold-email"]
    versions = await _versions("cold-email")
    assert [v.version for v in versions] == [1, 2]
    listing = await _listing("cold-email")
    assert listing.activeVersionId == versions[1].id
    assert listing.ActiveVersion is not None
    assert sorted(f.relativePath for f in listing.ActiveVersion.Files or []) == [
        "references/a.md",
        "references/b.md",
    ]
    # Version 1 is untouched, hash and files included.
    stale = await prisma.models.SkillListingVersion.prisma().find_unique(
        where={"id": versions[0].id}, include={"Files": True}
    )
    assert stale is not None
    assert [f.content.decode() for f in stale.Files or []] == [b"# a\n"]


async def test_rolling_back_is_publishing_the_older_commit(tmp_path: Path):
    v1 = write_catalog(tmp_path / "v1", {"cold-email": {"references/a.md": "# a\n"}})
    v2 = write_catalog(tmp_path / "v2", {"cold-email": {"references/a.md": "# a2\n"}})
    await _publish(v1, revision="1" * 40)
    await _publish(v2, revision="2" * 40)

    summary = await _publish(v1, revision="1" * 40)

    assert summary.updated == ["cold-email"]
    versions = await _versions("cold-email")
    assert [v.version for v in versions] == [1, 2]  # no third version minted
    assert (await _listing("cold-email")).activeVersionId == versions[0].id


async def test_a_retirement_delists_without_deleting_and_can_be_undone(
    tmp_path: Path,
):
    v1 = write_catalog(tmp_path / "v1", {"cold-email": {}, "warm-intro": {}})
    await _publish(v1, revision="1" * 40)
    v2 = write_catalog(tmp_path / "v2", {"warm-intro": {}}, retirements=["cold-email"])

    summary = await _publish(v2, revision="2" * 40)

    assert summary.retired == ["cold-email"]
    retired = await _listing("cold-email")
    assert retired.isDeleted
    assert retired.ActiveVersion is not None and not retired.ActiveVersion.isAvailable
    assert len(await _versions("cold-email")) == 1

    back = await _publish(v1, revision="1" * 40)

    assert back.updated == ["cold-email"]
    revived = await _listing("cold-email")
    assert not revived.isDeleted
    assert revived.ActiveVersion is not None and revived.ActiveVersion.isAvailable
    assert len(await _versions("cold-email")) == 1


async def test_a_slug_a_user_owns_aborts_the_whole_publish(tmp_path: Path):
    user = await prisma.models.User.prisma().create(
        data={"id": str(uuid.uuid4()), "email": f"{uuid.uuid4().hex}@example.com"}
    )
    await prisma.models.Profile.prisma().create(
        data={
            "userId": user.id,
            "username": f"owner-{uuid.uuid4().hex[:8]}",
            "name": "Listing Owner",
            "description": "",
            "links": [],
        }
    )
    await prisma.models.SkillListing.prisma().create(
        data={"slug": "cold-email", "owningUserId": user.id}
    )
    write_catalog(tmp_path, {"cold-email": {}, "warm-intro": {}})

    with pytest.raises(CatalogConflictError, match="cold-email"):
        await _publish(tmp_path)

    assert (
        await prisma.models.SkillListing.prisma().find_unique(
            where={"slug": "warm-intro"}
        )
        is None
    )
    assert await prisma.models.SkillCatalogRelease.prisma().count() == 0
    await prisma.models.SkillListing.prisma().delete_many(where={"slug": "cold-email"})
    await prisma.models.Profile.prisma().delete_many(where={"userId": user.id})
    await prisma.models.User.prisma().delete(where={"id": user.id})


async def test_a_dry_run_reports_the_plan_and_writes_nothing(tmp_path: Path):
    key = _key()
    write_catalog(
        tmp_path, {"cold-email": {}}, experts={key: expert_yaml(key, ["cold-email"])}
    )

    summary = await _publish(tmp_path, dry_run=True)

    assert summary.dry_run
    assert summary.created == ["cold-email"]
    assert summary.experts["created"] == [key]
    assert await prisma.models.SkillListing.prisma().count() == 0
    assert await prisma.models.Expert.prisma().count(where={"templateKey": key}) == 0
    assert await prisma.models.SkillCatalogRelease.prisma().count() == 0


async def test_a_legacy_version_is_hashed_so_existing_copies_can_be_matched(
    tmp_path: Path,
):
    legacy = await _make_listing("cold-email", body="# Cold email\n\nOld body.\n")
    assert legacy.ActiveVersion is not None
    assert legacy.ActiveVersion.packageSha256 is None
    write_catalog(tmp_path, {"warm-intro": {}})

    summary = await _publish(tmp_path)

    assert summary.backfilled == 1
    assert summary.orphaned == ["cold-email"]  # not in the catalog, not retired: kept
    stamped = await prisma.models.SkillListingVersion.prisma().find_unique(
        where={"id": legacy.ActiveVersion.id}, include={"Files": True}
    )
    assert stamped is not None
    assert stamped.packageSha256 == skill_model.legacy_package_sha256(
        stamped, "cold-email"
    )
    assert not (await _listing("cold-email")).isDeleted


async def test_a_template_keeps_its_row_across_a_rename_and_retires_by_key(
    tmp_path: Path,
):
    key = _key()
    v1 = write_catalog(
        tmp_path / "v1",
        {"cold-email": {}},
        experts={key: expert_yaml(key, ["cold-email"], name="Max")},
    )
    await _publish(v1, revision="1" * 40)
    before = await prisma.models.Expert.prisma().find_unique(where={"templateKey": key})
    assert before is not None and before.name == "Max"
    v2 = write_catalog(
        tmp_path / "v2",
        {"cold-email": {}},
        experts={key: expert_yaml(key, ["cold-email"], name="Maximilian")},
    )

    renamed = await _publish(v2, revision="2" * 40)

    assert renamed.experts == {"created": [], "updated": [key], "retired": []}
    after = await prisma.models.Expert.prisma().find_unique(where={"templateKey": key})
    assert after is not None and after.id == before.id and after.name == "Maximilian"

    v3 = write_catalog(tmp_path / "v3", {"cold-email": {}}, retired_experts=[key])
    retired = await _publish(v3, revision="3" * 40)

    assert retired.experts["retired"] == [key]
    gone = await prisma.models.Expert.prisma().find_unique(where={"templateKey": key})
    assert gone is not None and gone.isArchived and gone.id == before.id


async def test_a_second_publish_finds_a_template_seeded_by_name_before_keys(
    tmp_path: Path,
):
    """Templates that predate ``templateKey`` are adopted by display name once
    and keyed from then on, rather than duplicated."""
    key = _key()
    unkeyed = await prisma.models.Expert.prisma().create(
        data={"name": "Priya Legacy", "role": "r", "identity": "i", "isTemplate": True}
    )
    write_catalog(
        tmp_path,
        {"cold-email": {}},
        experts={key: expert_yaml(key, ["cold-email"], name="Priya Legacy")},
    )

    summary = await _publish(tmp_path)

    assert summary.experts["updated"] == [key]
    adopted = await prisma.models.Expert.prisma().find_unique(where={"id": unkeyed.id})
    assert adopted is not None and adopted.templateKey == key
    assert (
        await prisma.models.Expert.prisma().count(where={"name": "Priya Legacy"}) == 1
    )
    await prisma.models.Expert.prisma().delete_many(where={"id": unkeyed.id})


async def test_publishing_skills_only_leaves_templates_alone(tmp_path: Path):
    key = _key()
    write_catalog(
        tmp_path, {"cold-email": {}}, experts={key: expert_yaml(key, ["cold-email"])}
    )

    summary = await _publish(tmp_path, seed_experts=False)

    assert summary.created == ["cold-email"]
    assert summary.experts == {}
    assert await prisma.models.Expert.prisma().count(where={"templateKey": key}) == 0


async def test_a_failed_publish_leaves_the_marketplace_as_it_was(
    tmp_path: Path, monkeypatch
):
    """One transaction: a package write that fails part-way rolls back every
    listing, so the shelf never shows half a release."""
    v1 = write_catalog(
        tmp_path / "v1", {"cold-email": {"references/a.md": "# a\n"}, "warm-intro": {}}
    )
    await _publish(v1, revision="1" * 40)
    v2 = write_catalog(
        tmp_path / "v2",
        {"cold-email": {"references/a.md": "# a2\n"}, "warm-intro": {"b.md": "b\n"}},
    )

    async def fail_on_warm_intro(version_id, files, tx):
        if any(f.relative_path == "b.md" for f in files):
            raise RuntimeError("the package write failed")
        await snapshot_version_files(version_id, files, tx)

    monkeypatch.setattr(skill_catalog, "snapshot_version_files", fail_on_warm_intro)
    with pytest.raises(RuntimeError, match="package write failed"):
        await _publish(v2, revision="2" * 40)

    for slug in ("cold-email", "warm-intro"):
        assert len(await _versions(slug)) == 1
    cold = await _listing("cold-email")
    assert cold.ActiveVersion is not None
    assert [f.content.decode() for f in cold.ActiveVersion.Files or []] == [b"# a\n"]
    assert await prisma.models.SkillCatalogRelease.prisma().count() == 1


async def test_publish_serialises_on_the_advisory_lock(tmp_path: Path, monkeypatch):
    """The lock key is what two deploys collide on; the statement must be the
    one Postgres serialises transactions on."""
    write_catalog(tmp_path, {"cold-email": {}})
    seen: list[str] = []
    original = skill_catalog._backfill_legacy_hashes

    async def spy(tx, *, dry_run):
        rows = await tx.query_raw(
            "SELECT objid FROM pg_locks WHERE locktype = 'advisory' AND pid = pg_backend_pid()"
        )
        seen.extend(str(row["objid"]) for row in rows)
        return await original(tx, dry_run=dry_run)

    monkeypatch.setattr(skill_catalog, "_backfill_legacy_hashes", spy)

    await _publish(tmp_path)

    assert seen, "publish ran without holding an advisory lock"
