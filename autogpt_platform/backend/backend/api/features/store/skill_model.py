"""API models for marketplace skill listings.

A skill listing publishes a ``SKILL.md`` — instructions and examples the
copilot reads — rather than a runnable graph. The listing's ``slug`` is both
its marketplace URL segment and the name the skill takes once installed, so
the two can never drift; ``name`` is the human title shown on the card.
"""

import datetime

import prisma.enums
import prisma.models
import pydantic

from backend.util.models import Pagination

from .categories import validate_canonical_categories


class MarketplaceSkill(pydantic.BaseModel):
    slug: str
    name: str
    description: str
    categories: list[str]
    required_providers: list[str]
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


class SkillSubmissionRequest(pydantic.BaseModel):
    """Publish one of the caller's own library skills as a marketplace listing."""

    skill_name: str = pydantic.Field(
        description="Slug of the caller's library skill to publish."
    )
    categories: list[str]
    required_providers: list[str] = pydantic.Field(
        default_factory=list,
        description="Integrations the skill's instructions assume are connected.",
    )
    changes_summary: str | None = None

    @pydantic.field_validator("categories")
    @classmethod
    def _canonical_categories(cls, value: list[str]) -> list[str]:
        return validate_canonical_categories(value)


class SkillSubmission(pydantic.BaseModel):
    skill_listing_version_id: str
    slug: str
    name: str
    description: str
    categories: list[str]
    required_providers: list[str]
    version: int
    status: prisma.enums.SubmissionStatus
    review_comments: str | None = None
    is_live: bool = pydantic.Field(
        description="Whether this version is the one the marketplace serves."
    )

    @classmethod
    def from_db(
        cls,
        version: prisma.models.SkillListingVersion,
        listing: prisma.models.SkillListing,
    ) -> "SkillSubmission":
        return cls(
            skill_listing_version_id=version.id,
            slug=listing.slug,
            name=version.name,
            description=version.description,
            categories=list(version.categories),
            required_providers=list(version.requiredProviders),
            version=version.version,
            status=version.submissionStatus,
            review_comments=version.reviewComments,
            is_live=listing.activeVersionId == version.id,
        )
