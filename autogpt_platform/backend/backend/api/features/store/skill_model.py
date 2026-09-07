"""API models for marketplace skill listings.

A skill listing publishes a ``SKILL.md`` — instructions and examples the
copilot reads — rather than a runnable graph. The listing's ``slug`` is both
its marketplace URL segment and the name the skill takes once installed, so
the two can never drift; ``name`` is the human title shown on the card.
"""

import datetime

import prisma.models
import pydantic

from backend.util.models import Pagination


class MarketplaceSkill(pydantic.BaseModel):
    slug: str
    name: str
    description: str
    categories: list[str]
    required_providers: list[str]
    is_verified: bool
    install_count: int
    creator: str | None = None
    creator_avatar: str | None = None

    @classmethod
    def from_db(cls, listing: prisma.models.SkillListing) -> "MarketplaceSkill":
        version = active_version(listing)
        profile = listing.CreatorProfile
        return cls(
            slug=listing.slug,
            name=version.name,
            description=version.description,
            categories=list(version.categories),
            required_providers=list(version.requiredProviders),
            is_verified=version.isVerified,
            install_count=listing.installCount,
            creator=profile.username if profile else None,
            creator_avatar=profile.avatarUrl if profile else None,
        )


class MarketplaceSkillsResponse(pydantic.BaseModel):
    skills: list[MarketplaceSkill]
    pagination: Pagination


class MarketplaceSkillDetails(MarketplaceSkill):
    skill_listing_version_id: str
    body: str
    triggers: list[str]
    updated_at: datetime.datetime

    @classmethod
    def from_db(cls, listing: prisma.models.SkillListing) -> "MarketplaceSkillDetails":
        version = active_version(listing)
        summary = MarketplaceSkill.from_db(listing)
        return cls(
            **summary.model_dump(),
            skill_listing_version_id=version.id,
            body=version.body,
            triggers=list(version.triggers),
            updated_at=version.updatedAt,
        )


class InstalledSkill(pydantic.BaseModel):
    name: str = pydantic.Field(
        description="Name the skill was installed under, for a link into the library."
    )
    required_providers: list[str] = pydantic.Field(
        description=(
            "Integrations the skill's instructions assume. Not a precondition — "
            "the install has already succeeded."
        )
    )


def active_version(
    listing: prisma.models.SkillListing,
) -> prisma.models.SkillListingVersion:
    if listing.ActiveVersion is None:
        raise ValueError(f"Skill listing '{listing.slug}' has no active version")
    return listing.ActiveVersion
