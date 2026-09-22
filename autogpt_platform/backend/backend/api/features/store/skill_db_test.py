import datetime
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import prisma
import prisma.enums
import prisma.models
import pytest

from backend.copilot.tools.skills import SkillPackageError, read_user_skill_package
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
    files: dict[str, str] | None = None,
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
    for relative_path, content in sorted((files or {}).items()):
        raw = content.encode()
        await prisma.models.SkillListingFile.prisma().create(
            data={
                "skillListingVersionId": version.id,
                "relativePath": relative_path,
                "sizeBytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "content": prisma.Base64.encode(raw),
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


async def test_live_listings_are_the_same_rows_with_their_active_version():
    live_one = await _make_listing("live-one")
    never = await _make_listing("never-approved", approved=False)

    listings = await skill_db.get_live_listings([live_one.id, never.id])

    assert list(listings) == [live_one.id]
    assert listings[live_one.id].ActiveVersion is not None
    assert listings[live_one.id].ActiveVersion.body == "# body\n"


async def test_an_installable_skill_is_exactly_what_an_install_stores():
    """An export of a template carries the listing without installing it, so
    what it renders has to be byte-for-byte what a hire would have written."""
    listing = await _make_listing("brand-voice-guide", body="# Voice\n\nBe kind.\n")

    with _patch_skills_path(_FakeWorkspaceManager()):
        await skill_db.install_marketplace_skill(
            "user-1", listing.slug, expert_id="expert-a"
        )
        stored = await read_user_skill_package(
            "user-1", listing.slug, expert_id="expert-a"
        )
    parsed, package = await skill_db.installable_skill(listing)

    assert stored is not None
    assert package.skill_md == stored.skill_md
    assert package.files == stored.files == []
    assert parsed.name == "brand-voice-guide"
    assert parsed.description == "brand-voice-guide description"
    assert parsed.version == "1"


async def test_an_installable_skill_carries_the_published_files_too():
    """A listing can publish siblings beside its root. Reading only the root
    would export a bundled skill stripped of the scripts and references a hire
    of the very same listing gets."""
    listing = await _make_listing(
        "web-scraper",
        files={"scripts/run.py": "print('hi')\n", "references/API.md": "# API\n"},
    )

    with _patch_skills_path(_FakeWorkspaceManager()):
        await skill_db.install_marketplace_skill(
            "user-1", listing.slug, expert_id="expert-a"
        )
        stored = await read_user_skill_package(
            "user-1", listing.slug, expert_id="expert-a"
        )
    _, package = await skill_db.installable_skill(listing)

    assert stored is not None
    assert [f.relative_path for f in package.files] == [
        "references/API.md",
        "scripts/run.py",
    ]

    # Same siblings, whatever order each side lists them in.
    def by_path(files):
        return sorted(files, key=lambda f: f.relative_path)

    assert by_path(package.files) == by_path(stored.files)


async def test_an_installable_skill_fails_where_the_install_would():
    listing = await _make_listing("no-body", body="   ")

    with pytest.raises(ValueError, match="body is required"):
        await skill_db.installable_skill(listing)


@pytest.mark.parametrize("expert_id", [None, "expert-1"])
async def test_install_stores_under_the_listing_slug_and_counts(mocker, expert_id):
    listing = await _make_listing(
        f"install-one-{expert_id or 'library'}", body="# do this\n"
    )
    stored = mocker.patch.object(skill_db, "store_user_skill")
    listed = mocker.patch.object(skill_db, "list_user_skills", return_value=[])

    result = await skill_db.install_marketplace_skill(
        "user-1", listing.slug, expert_id=expert_id
    )

    # The count checks the folder being written: the expert's or the library.
    listed.assert_awaited_once_with("user-1", expert_id, heal_missing=False)
    stored.assert_awaited_once()
    assert stored.await_args.kwargs["name"] == listing.slug
    assert stored.await_args.kwargs["expert_id"] == expert_id
    assert stored.await_args.kwargs["body"] == "# do this\n"
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
    experts.add_expert_skill_name = AsyncMock()
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
        experts.add_expert_skill_name.assert_not_awaited()
    else:
        experts.add_expert_skill_name.assert_awaited_with(
            "user-1", expert_id, listing.slug
        )


async def test_install_of_a_listing_with_no_files_passes_an_empty_package(mocker):
    """`files=None` means "leave the folder alone", so a single-file listing
    installed over a package would leave the old package's files in place."""
    listing = await _make_listing("no-files-one")
    stored = mocker.patch.object(skill_db, "store_user_skill")
    mocker.patch.object(skill_db, "list_user_skills", return_value=[])

    await skill_db.install_marketplace_skill("user-1", listing.slug)

    assert stored.await_args.kwargs["files"] == []


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
    stored = mocker.patch.object(skill_db, "store_user_skill")
    mocker.patch.object(skill_db, "list_user_skills", return_value=[])

    await skill_db.install_marketplace_skill("user-1", listing.slug)

    files = stored.await_args.kwargs["files"]
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
    stored = mocker.patch.object(skill_db, "store_user_skill")
    mocker.patch.object(
        skill_db,
        "list_user_skills",
        return_value=[SimpleNamespace(name="install-twice")],
    )

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
    stored = mocker.patch.object(skill_db, "store_user_skill")

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
    stored = mocker.patch.object(skill_db, "store_user_skill")
    mocker.patch.object(skill_db, "list_user_skills", return_value=[])

    result = await skill_db.install_marketplace_skill("user-1", "cold-email")

    assert result.name == "cold-email"
    assert stored.await_args.kwargs["name"] == "cold-email"
    assert stored.await_args.kwargs["extra"] == {
        "license": "MIT",
        "source": "acme/marketing-skills",
        "source_url": (
            "https://github.com/acme/marketing-skills/tree/abc/skills/cold-email"
        ),
    }
    assert [
        (f.relative_path, f.content) for f in stored.await_args.kwargs["files"]
    ] == [("references/frameworks.md", b"# Frameworks\n")]


async def test_a_bad_catalog_entry_writes_nothing(tmp_path):
    catalog = _write_catalog(tmp_path)
    (catalog / "skills" / "cold-email" / ".env").write_text("x=1", encoding="utf-8")

    with pytest.raises(SkillPackageError):
        await skill_seed.seed_catalog_skills(catalog)

    assert await prisma.models.SkillListing.prisma().count() == 0
