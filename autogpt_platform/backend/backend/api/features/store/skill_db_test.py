import datetime

import prisma.enums
import prisma.models
import pytest

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
    await prisma.models.SkillListingVersion.prisma().delete_many()
    await prisma.models.SkillListing.prisma().delete_many()
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


async def test_install_stores_under_the_listing_slug_and_counts(mocker):
    listing = await _make_listing("install-one", body="# do this\n")
    stored = mocker.patch.object(skill_db, "store_user_skill")

    result = await skill_db.install_marketplace_skill("user-1", "install-one")

    stored.assert_awaited_once()
    assert stored.await_args.kwargs["name"] == "install-one"
    assert stored.await_args.kwargs["body"] == "# do this\n"
    assert result.name == "install-one"
    assert result.required_providers == ["google"]

    refreshed = await prisma.models.SkillListing.prisma().find_unique(
        where={"id": listing.id}
    )
    assert refreshed is not None
    assert refreshed.installCount == 1


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


async def test_seed_is_idempotent_and_keeps_one_version_per_listing():
    first = await skill_seed.seed_starter_skills()
    second = await skill_seed.seed_starter_skills()

    assert first == second
    versions = await prisma.models.SkillListingVersion.prisma().count()
    assert versions == len(skill_seed.STARTER_SKILLS)


async def test_seeded_skills_are_installable_under_their_own_slug(mocker):
    await skill_seed.seed_starter_skills()
    stored = mocker.patch.object(skill_db, "store_user_skill")

    result = await skill_db.install_marketplace_skill("user-1", "brand-voice-guide")

    assert result.name == "brand-voice-guide"
    assert stored.await_args.kwargs["name"] == "brand-voice-guide"


def test_seed_rejects_a_file_whose_frontmatter_name_is_not_the_slug(
    monkeypatch, tmp_path
):
    """The frontmatter name becomes the installed skill's name, so a mismatch
    would install a starter skill under a name the marketplace never shows."""
    (tmp_path / "mismatched.md").write_text(
        '---\nname: "something-else"\ndescription: "d"\n---\n\nbody\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(skill_seed, "_CONTENT_DIR", tmp_path)

    with pytest.raises(ValueError, match="must match"):
        skill_seed._load("mismatched")
