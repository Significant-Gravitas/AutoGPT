import datetime
import hashlib
from unittest.mock import AsyncMock, MagicMock

import prisma
import prisma.actions
import prisma.enums
import prisma.models
import pytest

from backend.copilot.tools import skills as skills_module
from backend.copilot.tools.skills import ParsedSkill, SkillPackageError, StoredSkill
from backend.copilot.tools.skills_test import _FakeWorkspaceManager, _patch_skills_path
from backend.util.exceptions import NotFoundError
from backend.util.test import SpinTestServer

from . import skill_db, skill_seed


async def _make_listing(
    slug: str,
    *,
    approved: bool = True,
    has_approved_version: bool | None = None,
    updated_at: datetime.datetime | None = None,
    available: bool = True,
    categories: list[str] | None = None,
    body: str = "# body\n",
) -> prisma.models.SkillListing:
    listing = await prisma.models.SkillListing.prisma().create(
        data={
            "slug": slug,
            "hasApprovedVersion": (
                approved if has_approved_version is None else has_approved_version
            ),
        }
    )
    version = await prisma.models.SkillListingVersion.prisma().create(
        data={
            "skillListingId": listing.id,
            "name": slug.replace("-", " ").title(),
            "description": f"{slug} description",
            "body": body,
            "triggers": ["t1"],
            "categories": categories or ["content"],
            "requiredProviders": ["google"],
            "isAvailable": available,
            "submissionStatus": (
                prisma.enums.SubmissionStatus.APPROVED
                if approved
                else prisma.enums.SubmissionStatus.PENDING
            ),
        }
    )
    if updated_at is not None:
        await prisma.models.SkillListingVersion.prisma().update(
            where={"id": version.id}, data={"updatedAt": updated_at}
        )
    updated = await prisma.models.SkillListing.prisma().update(
        where={"id": listing.id},
        data={"activeVersionId": version.id},
        include={"ActiveVersion": True},
    )
    assert updated is not None
    return updated


@pytest.fixture(autouse=True)
async def clean_skill_listings(server: SpinTestServer):
    # These tests assert on the WHOLE marketplace, so they need the listing
    # table to themselves — take it only when empty, never by emptying it.
    if await prisma.models.SkillListing.prisma().count():
        pytest.fail(
            "this database already holds skill listings; run this file against a "
            "throwaway Postgres, not the one every worktree here shares"
        )
    yield
    await prisma.models.SkillListingVersion.prisma().delete_many()
    await prisma.models.SkillListing.prisma().delete_many()


async def test_browse_puts_the_most_recently_updated_first():
    # Created oldest-first, so insertion order alone would fail this.
    utc = datetime.timezone.utc
    await _make_listing(
        "older-one", updated_at=datetime.datetime(2026, 1, 1, tzinfo=utc)
    )
    await _make_listing(
        "newer-one", updated_at=datetime.datetime(2026, 6, 1, tzinfo=utc)
    )

    result = await skill_db.get_marketplace_skills()

    assert [s.slug for s in result.skills] == ["newer-one", "older-one"]
    assert result.pagination.total_items == 2


async def test_browse_hides_unapproved_and_unavailable_listings():
    await _make_listing("live-one")
    await _make_listing("pending-one", approved=False)
    await _make_listing("withdrawn-one", available=False)

    result = await skill_db.get_marketplace_skills()

    assert [s.slug for s in result.skills] == ["live-one"]


async def test_browse_hides_a_listing_whose_live_version_was_rejected():
    """`hasApprovedVersion` means some version was approved once, not that the
    active one still is — a version rejected on re-review must leave the shelf."""
    await _make_listing("re-rejected-one", approved=False, has_approved_version=True)

    result = await skill_db.get_marketplace_skills()

    assert [s.slug for s in result.skills] == []


async def test_browse_filters_by_category():
    await _make_listing("sales-one", categories=["sales"])
    await _make_listing("content-one", categories=["content"])

    result = await skill_db.get_marketplace_skills(category="sales")

    assert [s.slug for s in result.skills] == ["sales-one"]


async def test_browse_matches_a_legacy_category_alias():
    """`writing` folds onto `content`, so a filter finds it before the backfill."""
    await _make_listing("legacy-one", categories=["writing"])

    result = await skill_db.get_marketplace_skills(category="content")

    assert [s.slug for s in result.skills] == ["legacy-one"]


async def test_detail_carries_the_body_and_providers():
    await _make_listing("detail-one", body="# how to\nsteps\n")

    detail = await skill_db.get_marketplace_skill("detail-one")

    assert detail.body == "# how to\nsteps\n"
    assert detail.required_providers == ["google"]
    assert detail.triggers == ["t1"]


# Two ways a listing can be off the shelf: never approved (the listing flag is
# false) and approved once but rejected on re-review (the flag stays true and
# only the live version's status says so).
OFF_SHELF = [
    pytest.param(False, id="never-approved"),
    pytest.param(True, id="live-version-rejected"),
]


@pytest.mark.parametrize("has_approved_version", OFF_SHELF)
async def test_detail_of_an_off_shelf_listing_is_not_found(has_approved_version: bool):
    await _make_listing(
        "off-shelf-one", approved=False, has_approved_version=has_approved_version
    )

    with pytest.raises(NotFoundError):
        await skill_db.get_marketplace_skill("off-shelf-one")


async def test_live_skills_holds_only_listings_on_the_shelf():
    live_one = await _make_listing("live-one")
    draft = await _make_listing("draft-one", approved=False, has_approved_version=True)
    never = await _make_listing("never-approved", approved=False)
    unavailable = await _make_listing("unavailable-one", available=False)
    deleted = await _make_listing("deleted-one")
    await prisma.models.SkillListing.prisma().update(
        where={"id": deleted.id}, data={"isDeleted": True}
    )
    withdrawn = await _make_listing("withdrawn-one")
    assert withdrawn.activeVersionId is not None
    await prisma.models.SkillListingVersion.prisma().update(
        where={"id": withdrawn.activeVersionId}, data={"isDeleted": True}
    )

    live = await skill_db.get_live_skills(
        [
            live_one.id,
            draft.id,
            never.id,
            unavailable.id,
            deleted.id,
            withdrawn.id,
            "missing-one",
        ]
    )

    assert list(live) == [live_one.id]
    assert live[live_one.id].name == "Live One"


@pytest.mark.parametrize("expert_id", [None, "expert-1"])
async def test_install_stores_under_the_listing_slug_and_counts(mocker, expert_id):
    listing = await _make_listing(
        f"install-one-{expert_id or 'library'}", body="# do this\n"
    )
    stored = _patch_store(mocker)

    result = await skill_db.install_marketplace_skill(
        "user-1", listing.slug, expert_id=expert_id
    )

    stored.assert_awaited_once()
    assert _written(stored).name == listing.slug
    assert stored.await_args.kwargs["expert_id"] == expert_id
    assert _written(stored).body == "# do this\n"
    # An install fills the platform's budget on that owner, not the owner's.
    assert stored.await_args.kwargs["origin"] == "marketplace"
    assert result.name == listing.slug
    assert result.required_providers == ["google"]

    refreshed = await prisma.models.SkillListing.prisma().find_unique(
        where={"id": listing.id}
    )
    assert refreshed is not None
    assert refreshed.installCount == 1


@pytest.mark.parametrize(
    "expert_id, folder",
    [(None, "/skills"), ("expert-a", "/experts/expert-a/skills")],
    ids=["library", "expert"],
)
async def test_install_writes_into_the_target_owners_folder_and_no_other(
    mocker, expert_id, folder
):
    """The test above proves the id reaches ``store_user_skill``; this one runs
    the real write, so a folder that ignored it would show up here."""
    listing = await _make_listing(f"folder-one-{expert_id or 'library'}")
    workspace = _FakeWorkspaceManager()
    experts = MagicMock()
    experts.add_expert_skill_names = AsyncMock()
    mocker.patch("backend.copilot.tools.skills.experts_db", return_value=experts)

    with _patch_skills_path(workspace):
        await skill_db.install_marketplace_skill(
            "user-1", listing.slug, expert_id=expert_id
        )
        # A second install of the same skill overwrites its copy rather than
        # leaving a second one behind.
        await skill_db.install_marketplace_skill(
            "user-1", listing.slug, expert_id=expert_id
        )

    assert list(workspace.files) == [f"{folder}/{listing.slug}/SKILL.md"]
    refreshed = await prisma.models.SkillListing.prisma().find_unique(
        where={"id": listing.id}
    )
    assert refreshed is not None
    assert refreshed.installCount == 1
    if expert_id is None:
        experts.add_expert_skill_names.assert_not_awaited()
    else:
        experts.add_expert_skill_names.assert_awaited_with(
            "user-1", expert_id, [listing.slug]
        )


async def test_a_batch_install_looks_up_listings_and_the_owners_skills_once(mocker):
    listings = [await _make_listing(f"batch-one-{i}") for i in range(3)]
    workspace = _FakeWorkspaceManager()
    experts = MagicMock()
    experts.add_expert_skill_names = AsyncMock()
    mocker.patch("backend.copilot.tools.skills.experts_db", return_value=experts)
    listed = mocker.spy(skills_module, "_list_user_skills_from_workspace")
    queried = mocker.spy(prisma.actions.SkillListingActions, "find_many")

    with _patch_skills_path(workspace):
        locked = mocker.spy(skills_module, "AsyncClusterLock")
        outcomes = await skill_db.install_marketplace_skills(
            "user-1",
            [listing.slug for listing in listings] + ["missing"],
            expert_id="expert-a",
        )

    assert [type(o).__name__ for o in outcomes] == ["InstalledSkill"] * 3 + [
        "NotFoundError"
    ]
    # Kills: a batch that stores (lists, locks, looks up) once per skill.
    assert (listed.call_count, locked.call_count, queried.call_count) == (1, 1, 1)
    # Kills: recording each name on the expert's row in its own write.
    experts.add_expert_skill_names.assert_awaited_once_with(
        "user-1", "expert-a", [listing.slug for listing in listings]
    )
    assert sorted(workspace.files) == sorted(
        f"/experts/expert-a/skills/{listing.slug}/SKILL.md" for listing in listings
    )
    counts = await prisma.models.SkillListing.prisma().find_many(
        where={"id": {"in": [listing.id for listing in listings]}}
    )
    assert [c.installCount for c in counts] == [1, 1, 1]


async def test_a_failed_name_write_fails_only_the_skills_it_could_not_record(mocker):
    listings = [await _make_listing(f"names-one-{i}") for i in range(3)]
    lost = listings[1].slug
    workspace = _FakeWorkspaceManager()
    experts = MagicMock()
    experts.add_expert_skill_names = AsyncMock(side_effect=RuntimeError("row busy"))

    async def add_one(user_id, expert_id, name):
        if name == lost:
            raise RuntimeError("row busy")

    experts.add_expert_skill_name = AsyncMock(side_effect=add_one)
    mocker.patch("backend.copilot.tools.skills.experts_db", return_value=experts)

    with _patch_skills_path(workspace):
        outcomes = await skill_db.install_marketplace_skills(
            "user-1", [listing.slug for listing in listings], expert_id="expert-a"
        )

    # Kills: failing every skill in the batch when the one-write record fails.
    assert [type(o).__name__ for o in outcomes] == [
        "InstalledSkill",
        "RuntimeError",
        "InstalledSkill",
    ]
    assert [c.args[2] for c in experts.add_expert_skill_name.await_args_list] == [
        listing.slug for listing in listings
    ]


async def test_an_install_skips_the_scan_only_for_bytes_an_earlier_install_scanned(
    mocker,
):
    listing = await _make_listing("scan-once-one", body="# v1\n")
    assert listing.activeVersionId is not None
    workspace = _FakeWorkspaceManager()
    experts = MagicMock()
    experts.add_expert_skill_names = AsyncMock()
    mocker.patch("backend.copilot.tools.skills.experts_db", return_value=experts)

    def install_and_read(expert_id: str) -> tuple[str, frozenset[str]]:
        path = f"/experts/{expert_id}/skills/{listing.slug}/SKILL.md"
        written = hashlib.sha256(workspace.files[path]).hexdigest()
        return written, workspace.scanned[path]

    with _patch_skills_path(workspace):
        await skill_db.install_marketplace_skill("u", listing.slug, expert_id="a")
        first, first_skip = install_and_read("a")
        await skill_db.install_marketplace_skill("u", listing.slug, expert_id="b")
        second, second_skip = install_and_read("b")
        await prisma.models.SkillListingVersion.prisma().update(
            where={"id": listing.activeVersionId}, data={"body": "# v2\n"}
        )
        await skill_db.install_marketplace_skill("u", listing.slug, expert_id="c")
        changed, changed_skip = install_and_read("c")

    version = await prisma.models.SkillListingVersion.prisma().find_unique(
        where={"id": listing.activeVersionId}
    )
    assert version is not None
    # Kills: recording nothing, or recording anything but the bytes written.
    assert first not in first_skip
    assert second == first and second in second_skip
    assert changed != first and changed not in changed_skip
    assert set(version.scannedSha256) == {first, changed}


async def test_a_failed_scan_record_does_not_fail_the_install(mocker):
    listing = await _make_listing("scan-once-record-fails")
    workspace = _FakeWorkspaceManager()
    experts = MagicMock()
    experts.add_expert_skill_names = AsyncMock()
    mocker.patch("backend.copilot.tools.skills.experts_db", return_value=experts)
    mocker.patch.object(
        prisma.actions.SkillListingVersionActions,
        "update",
        AsyncMock(side_effect=RuntimeError("db down")),
    )

    with _patch_skills_path(workspace):
        result = await skill_db.install_marketplace_skill(
            "u", listing.slug, expert_id="a"
        )

    # Kills: letting the scan cache's write fail the install it follows.
    assert result.name == listing.slug
    assert f"/experts/a/skills/{listing.slug}/SKILL.md" in workspace.files


async def test_install_of_a_listing_with_no_files_passes_an_empty_package(mocker):
    """`files=None` means "leave the folder alone", so a single-file listing
    installed over a package would leave the old package's files in place."""
    listing = await _make_listing("no-files-one")
    stored = _patch_store(mocker)

    await skill_db.install_marketplace_skill("user-1", listing.slug)

    assert _written(stored).files == []


async def test_install_carries_the_versions_files_and_their_executable_bits(mocker):
    listing = await _make_listing("with-files-one")
    assert listing.activeVersionId is not None
    await prisma.models.SkillListingFile.prisma().create_many(
        data=[
            {
                "skillListingVersionId": listing.activeVersionId,
                "relativePath": "scripts/run.py",
                "sizeBytes": 5,
                "sha256": "x",
                "isExecutable": True,
                "content": prisma.Base64.encode(b"print"),
            },
            {
                "skillListingVersionId": listing.activeVersionId,
                "relativePath": "references/a.md",
                "sizeBytes": 3,
                "sha256": "y",
                "content": prisma.Base64.encode(b"ref"),
            },
        ]
    )
    stored = _patch_store(mocker)

    await skill_db.install_marketplace_skill("user-1", listing.slug)

    files = _written(stored).files
    assert [(f.relative_path, f.content, f.is_executable) for f in files] == [
        ("references/a.md", b"ref", False),
        ("scripts/run.py", b"print", True),
    ]


async def test_detail_lists_the_package_files_without_their_bytes(mocker):
    listing = await _make_listing("detail-files-one")
    assert listing.activeVersionId is not None
    await prisma.models.SkillListingFile.prisma().create(
        data={
            "skillListingVersionId": listing.activeVersionId,
            "relativePath": "scripts/run.py",
            "sizeBytes": 5,
            "sha256": "x",
            "content": prisma.Base64.encode(b"print"),
        }
    )

    detail = await skill_db.get_marketplace_skill("detail-files-one")

    assert [(f.path, f.size_bytes) for f in detail.files] == [("scripts/run.py", 5)]


async def test_reinstalling_stores_again_but_does_not_count_again(mocker):
    listing = await _make_listing("install-twice", body="# do this\n")
    stored = _patch_store(mocker, is_new=False)

    await skill_db.install_marketplace_skill("user-1", "install-twice")

    stored.assert_awaited_once()
    refreshed = await prisma.models.SkillListing.prisma().find_unique(
        where={"id": listing.id}
    )
    assert refreshed is not None
    assert refreshed.installCount == 0


@pytest.mark.parametrize("has_approved_version", OFF_SHELF)
async def test_install_of_an_off_shelf_listing_never_reaches_the_library(
    mocker, has_approved_version: bool
):
    await _make_listing(
        "off-shelf-one", approved=False, has_approved_version=has_approved_version
    )
    stored = _patch_store(mocker)

    with pytest.raises(NotFoundError):
        await skill_db.install_marketplace_skill("user-1", "off-shelf-one")

    stored.assert_not_awaited()


CATALOG_YML = (
    "skills:\n"
    "  - slug: brand-voice-guide\n"
    "    categories: [content]\n"
    "    required_providers: []\n"
    "    source: platform\n"
    "  - slug: cold-email\n"
    "    categories: [sales]\n"
    "    required_providers: [google]\n"
    "    source: acme/marketing-skills/skills/cold-email\n"
    "    license: MIT\n"
)
COLD_EMAIL_MD = (
    "---\n"
    "name: cold-email\n"
    "description: Write cold emails.\n"
    "license: MIT\n"
    "metadata:\n"
    "  source: acme/marketing-skills\n"
    "  source_url: https://github.com/acme/marketing-skills/tree/abc/skills/cold-email\n"
    "---\n\n# Cold email\n\nSee references/frameworks.md.\n"
)


def _write_catalog(
    root, frameworks: str = "# Frameworks\n", brand_body: str = "# Voice\n"
):
    (root / "catalog.yml").write_text(CATALOG_YML, encoding="utf-8")
    brand = root / "skills" / "brand-voice-guide"
    brand.mkdir(parents=True, exist_ok=True)
    (brand / "SKILL.md").write_text(
        "---\nname: brand-voice-guide\ndescription: Keep one voice.\n---\n\n"
        + brand_body,
        encoding="utf-8",
    )
    cold = root / "skills" / "cold-email" / "references"
    cold.mkdir(parents=True, exist_ok=True)
    (cold.parent / "SKILL.md").write_text(COLD_EMAIL_MD, encoding="utf-8")
    (cold / "frameworks.md").write_text(frameworks, encoding="utf-8")
    return root


async def test_seed_is_idempotent_and_rewrites_the_live_package_in_place(tmp_path):
    catalog = _write_catalog(tmp_path)
    first = await skill_seed.seed_catalog_skills(catalog)
    _write_catalog(tmp_path, frameworks="# Frameworks v2\n")
    second = await skill_seed.seed_catalog_skills(catalog)

    assert first == second
    assert await prisma.models.SkillListingVersion.prisma().count() == 2
    files = await prisma.models.SkillListingFile.prisma().find_many()
    assert [(f.relativePath, f.content.decode()) for f in files] == [
        ("references/frameworks.md", b"# Frameworks v2\n")
    ]


async def test_seed_delists_a_retired_starter_slug(tmp_path, monkeypatch):
    """A slug that shipped and was then renamed keeps its listing row, so the
    seed hides it: delisted and its version unavailable. A listing that is
    merely absent from this catalog is not retired and stays live, as does
    the listing still in it."""
    catalog = _write_catalog(tmp_path)
    stale = catalog / "skills" / "stale-but-live"
    stale.mkdir(parents=True)
    (stale / "SKILL.md").write_text(
        "---\nname: stale-but-live\ndescription: Still live.\n---\n\n# Stale\n",
        encoding="utf-8",
    )
    (catalog / "catalog.yml").write_text(
        (catalog / "catalog.yml").read_text(encoding="utf-8")
        + "  - slug: stale-but-live\n"
        "    categories: [content]\n"
        "    required_providers: []\n"
        "    source: platform\n",
        encoding="utf-8",
    )
    await skill_seed.seed_catalog_skills(catalog)
    (catalog / "catalog.yml").write_text(
        "skills:\n"
        "  - slug: brand-voice-guide\n"
        "    categories: [content]\n"
        "    required_providers: []\n"
        "    source: platform\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(skill_seed, "RETIRED_STARTER_SLUGS", ["cold-email"])

    await skill_seed.seed_catalog_skills(catalog)

    async def listing(slug: str) -> prisma.models.SkillListing:
        row = await prisma.models.SkillListing.prisma().find_unique(
            where={"slug": slug}, include={"ActiveVersion": True}
        )
        assert row is not None and row.ActiveVersion is not None, slug
        return row

    retired = await listing("cold-email")
    assert retired.isDeleted and not retired.ActiveVersion.isAvailable
    for slug in ("brand-voice-guide", "stale-but-live"):
        live = await listing(slug)
        assert not live.isDeleted and live.ActiveVersion.isAvailable, slug


async def test_seed_keeps_the_old_package_when_file_replacement_fails(mocker, tmp_path):
    catalog = _write_catalog(tmp_path)
    await skill_seed.seed_catalog_skills(catalog)
    listing = await prisma.models.SkillListing.prisma().find_unique(
        where={"slug": "cold-email"}, include={"ActiveVersion": True}
    )
    assert listing is not None and listing.ActiveVersion is not None
    version_id = listing.ActiveVersion.id
    original_snapshot = skill_seed.snapshot_version_files

    async def fail_after_delete(skill_listing_version_id, files, tx):
        if skill_listing_version_id == version_id:
            await prisma.models.SkillListingFile.prisma(tx).delete_many(
                where={"skillListingVersionId": skill_listing_version_id}
            )
            raise RuntimeError("file write failed")
        await original_snapshot(skill_listing_version_id, files, tx)

    mocker.patch.object(
        skill_seed, "snapshot_version_files", side_effect=fail_after_delete
    )
    _write_catalog(tmp_path, frameworks="# Frameworks v2\n")

    with pytest.raises(RuntimeError, match="file write failed"):
        await skill_seed.seed_catalog_skills(catalog)

    files = await prisma.models.SkillListingFile.prisma().find_many(
        where={"skillListingVersionId": version_id}
    )
    assert [(file.relativePath, file.content.decode()) for file in files] == [
        ("references/frameworks.md", b"# Frameworks\n")
    ]


async def test_seed_rolls_back_every_listing_when_one_file_write_fails(
    mocker, tmp_path
):
    catalog = _write_catalog(tmp_path)
    await skill_seed.seed_catalog_skills(catalog)
    cold_email = await prisma.models.SkillListing.prisma().find_unique(
        where={"slug": "cold-email"}, include={"ActiveVersion": True}
    )
    assert cold_email is not None and cold_email.ActiveVersion is not None
    failing_version_id = cold_email.ActiveVersion.id
    original_snapshot = skill_seed.snapshot_version_files

    async def fail_on_cold_email(skill_listing_version_id, files, tx):
        if skill_listing_version_id == failing_version_id:
            raise RuntimeError("file write failed")
        await original_snapshot(skill_listing_version_id, files, tx)

    mocker.patch.object(
        skill_seed, "snapshot_version_files", side_effect=fail_on_cold_email
    )
    _write_catalog(tmp_path, frameworks="# Frameworks v2\n", brand_body="# Voice v2\n")

    with pytest.raises(RuntimeError, match="file write failed"):
        await skill_seed.seed_catalog_skills(catalog)

    brand_voice = await prisma.models.SkillListing.prisma().find_unique(
        where={"slug": "brand-voice-guide"}, include={"ActiveVersion": True}
    )
    assert brand_voice is not None and brand_voice.ActiveVersion is not None
    assert brand_voice.ActiveVersion.body == "# Voice\n"


async def test_seed_rejects_a_slug_owned_by_a_creator(tmp_path, setup_test_user):
    await prisma.models.Profile.prisma().upsert(
        where={"userId": setup_test_user},
        data={
            "create": {
                "userId": setup_test_user,
                "username": "catalog-collision-owner",
                "name": "Catalog Collision Owner",
                "description": "",
                "links": [],
            },
            "update": {},
        },
    )
    owned = await prisma.models.SkillListing.prisma().create(
        data={"slug": "cold-email", "owningUserId": setup_test_user}
    )

    with pytest.raises(ValueError, match="conflicts with an owned listing"):
        await skill_seed.seed_catalog_skills(_write_catalog(tmp_path))

    unchanged = await prisma.models.SkillListing.prisma().find_unique(
        where={"id": owned.id}
    )
    assert unchanged is not None
    assert unchanged.owningUserId == setup_test_user
    assert unchanged.hasApprovedVersion is False
    assert (
        await prisma.models.SkillListingVersion.prisma().count(
            where={"skillListingId": owned.id}
        )
        == 0
    )


async def test_seeded_skills_carry_their_attribution(tmp_path):
    await skill_seed.seed_catalog_skills(_write_catalog(tmp_path))

    vendored = await skill_db.get_marketplace_skill("cold-email")
    own = await skill_db.get_marketplace_skill("brand-voice-guide")

    assert (vendored.source_repo, vendored.license) == ("acme/marketing-skills", "MIT")
    assert vendored.source_url is not None and vendored.source_url.endswith(
        "/skills/cold-email"
    )
    assert (own.source_repo, own.source_url, own.license) == (None, None, None)


async def test_seeded_skills_install_with_their_package_files(mocker, tmp_path):
    await skill_seed.seed_catalog_skills(_write_catalog(tmp_path))
    stored = _patch_store(mocker)

    result = await skill_db.install_marketplace_skill("user-1", "cold-email")

    assert result.name == "cold-email"
    assert _written(stored).name == "cold-email"
    assert _written(stored).extra == {
        "license": "MIT",
        "source": "acme/marketing-skills",
        "source_url": (
            "https://github.com/acme/marketing-skills/tree/abc/skills/cold-email"
        ),
    }
    assert [(f.relative_path, f.content) for f in _written(stored).files] == [
        ("references/frameworks.md", b"# Frameworks\n")
    ]


async def test_a_bad_catalog_entry_writes_nothing(tmp_path):
    catalog = _write_catalog(tmp_path)
    (catalog / "skills" / "cold-email" / ".env").write_text("x=1", encoding="utf-8")

    with pytest.raises(SkillPackageError):
        await skill_seed.seed_catalog_skills(catalog)

    assert await prisma.models.SkillListing.prisma().count() == 0


def _patch_store(mocker, *, is_new: bool = True) -> AsyncMock:
    """Stand in for the locked write, reporting every skill stored."""

    async def store(user_id, writes, **kwargs):
        return [
            StoredSkill(
                ParsedSkill(name=w.name, description=w.description, body=w.body),
                is_new,
            )
            for w in writes
        ]

    return mocker.patch.object(
        skill_db, "store_user_skills", AsyncMock(side_effect=store)
    )


def _written(stored: AsyncMock):
    [write] = stored.await_args.args[1]
    return write
