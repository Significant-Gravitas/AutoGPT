"""Database access for marketplace skill listings.

Browse is rooted at :class:`SkillListingVersion` rather than the listing because
every filter and the ordering read version columns; ``ActiveFor`` constrains the
row to the listing's live version. Install count is displayed but is not a sort
key — a catalogue this size gains nothing from it, and it lives on the listing
because installs accumulate across versions. It counts distinct installers: a
re-install overwrites the user's copy and is not counted again.
"""

import prisma.enums
import prisma.models
import prisma.types

from backend.copilot.tools.skills import list_user_skills, store_user_skill
from backend.util.exceptions import NotFoundError
from backend.util.models import Pagination

from . import skill_model
from .categories import category_filter_values

_LISTING_INCLUDE: prisma.types.SkillListingInclude = {
    "ActiveVersion": True,
    "CreatorProfile": True,
}


async def get_marketplace_skills(
    *,
    category: str | None = None,
    search_query: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> skill_model.MarketplaceSkillsResponse:
    where = _live_version_where(category, search_query)
    total = await prisma.models.SkillListingVersion.prisma().count(where=where)
    versions = await prisma.models.SkillListingVersion.prisma().find_many(
        where=where,
        include={"ActiveFor": {"include": _LISTING_INCLUDE}},
        # Seeded versions share a timestamp, and an undefined order between
        # pages then drops or repeats a row; `id` is the uuid primary key.
        order=[{"updatedAt": "desc"}, {"id": "desc"}],
        skip=(page - 1) * page_size,
        take=page_size,
    )
    listings = [v.ActiveFor for v in versions if v.ActiveFor is not None]
    return skill_model.MarketplaceSkillsResponse(
        skills=[skill_model.MarketplaceSkill.from_db(listing) for listing in listings],
        pagination=Pagination(
            total_items=total,
            total_pages=(total + page_size - 1) // page_size,
            current_page=page,
            page_size=page_size,
        ),
    )


async def get_marketplace_skill(slug: str) -> skill_model.MarketplaceSkillDetails:
    listing = await _find_live_listing(slug)
    return skill_model.MarketplaceSkillDetails.from_db(listing)


async def install_marketplace_skill(
    user_id: str, slug: str
) -> skill_model.InstalledSkill:
    """Copy a listing's SKILL.md into the caller's own skill library.

    The listing's slug becomes the installed skill's name, so an install is
    idempotent and a re-install picks up a newer approved version.
    """
    listing = await _find_live_listing(slug)
    active = skill_model.active_version(listing)
    # A re-install overwrites the user's existing copy, so counting it again
    # would report installs rather than installers.
    is_new = all(s.name != listing.slug for s in await list_user_skills(user_id))
    # #14414 makes this the expert-owned install's call site, adding both the
    # `expert_id` argument here and the keyword on `store_user_skill`.
    await store_user_skill(
        user_id,
        name=listing.slug,
        description=active.description,
        body=active.body,
        triggers=list(active.triggers),
        version=str(active.version),
    )
    if is_new:
        await prisma.models.SkillListing.prisma().update(
            where={"id": listing.id}, data={"installCount": {"increment": 1}}
        )
    return skill_model.InstalledSkill(
        name=listing.slug, required_providers=list(active.requiredProviders)
    )


async def _find_live_listing(slug: str) -> prisma.models.SkillListing:
    listing = await prisma.models.SkillListing.prisma().find_first(
        where={"slug": slug, "isDeleted": False, "hasApprovedVersion": True},
        include=_LISTING_INCLUDE,
    )
    if listing is None or listing.ActiveVersion is None:
        raise NotFoundError(f"Skill '{slug}' not found")
    active = listing.ActiveVersion
    if (
        active.submissionStatus != prisma.enums.SubmissionStatus.APPROVED
        or not active.isAvailable
        or active.isDeleted
    ):
        raise NotFoundError(f"Skill '{slug}' not found")
    return listing


def _live_version_where(
    category: str | None, search_query: str | None
) -> prisma.types.SkillListingVersionWhereInput:
    where: prisma.types.SkillListingVersionWhereInput = {
        "submissionStatus": prisma.enums.SubmissionStatus.APPROVED,
        "isAvailable": True,
        "isDeleted": False,
        "ActiveFor": {"is": {"isDeleted": False, "hasApprovedVersion": True}},
    }
    if category_values := category_filter_values(category):
        where["categories"] = {"has_some": category_values}
    if search_query and search_query.strip():
        needle = search_query.strip()
        where["OR"] = [
            {"name": {"contains": needle, "mode": "insensitive"}},
            {"description": {"contains": needle, "mode": "insensitive"}},
        ]
    return where
