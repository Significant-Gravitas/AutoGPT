"""API models for marketplace skill listings.

A skill listing publishes a ``SKILL.md`` — instructions and examples the
copilot reads — and the files beside it, rather than a runnable graph. The
listing's ``slug`` is both its marketplace URL segment and the name the skill
takes once installed, so the two can never drift. ``name`` is that slug again,
never a display string: the title on the card is :func:`skill_title`.
"""

import datetime
import re

import prisma.enums
import prisma.models
import pydantic

from backend.util.models import Pagination

from .categories import validate_canonical_categories


class MarketplaceSkill(pydantic.BaseModel):
    slug: str
    name: str
    title: str
    description: str
    categories: list[str]
    required_providers: list[str]
    install_count: int
    creator: str | None = None
    creator_avatar: str | None = None
    source_repo: str | None = pydantic.Field(
        default=None,
        description="GitHub repo a vendored skill was taken from, as owner/name.",
    )
    source_url: str | None = None
    license: str | None = None

    @classmethod
    def from_db(cls, listing: prisma.models.SkillListing) -> "MarketplaceSkill":
        version = active_version(listing)
        profile = listing.CreatorProfile
        return cls(
            slug=listing.slug,
            name=version.name,
            title=skill_title(version.name, version.body),
            description=version.description,
            categories=list(version.categories),
            required_providers=list(version.requiredProviders),
            install_count=listing.installCount,
            creator=profile.username if profile else None,
            creator_avatar=profile.avatarUrl if profile else None,
            source_repo=version.sourceRepo,
            source_url=version.sourceUrl,
            license=version.license,
        )


class MarketplaceSkillsResponse(pydantic.BaseModel):
    skills: list[MarketplaceSkill]
    pagination: Pagination


class SkillPackageFile(pydantic.BaseModel):
    """One file beside the published ``SKILL.md``, by its path relative to the
    skill folder. The contents are not inlined — a package runs to 20 MiB."""

    path: str
    size_bytes: int
    # Null for an extension `mimetypes` cannot name — a `Makefile`, a `.toml`
    # — which the viewer still serves, deciding on the bytes instead.
    mime_type: str | None = None
    is_executable: bool = False


class MarketplaceSkillDetails(MarketplaceSkill):
    skill_listing_version_id: str
    body: str
    triggers: list[str]
    updated_at: datetime.datetime
    files: list[SkillPackageFile] = []

    @classmethod
    def from_db(
        cls,
        listing: prisma.models.SkillListing,
        files: list[SkillPackageFile] | None = None,
    ) -> "MarketplaceSkillDetails":
        version = active_version(listing)
        summary = MarketplaceSkill.from_db(listing)
        return cls(
            **summary.model_dump(),
            skill_listing_version_id=version.id,
            body=version.body,
            triggers=list(version.triggers),
            updated_at=version.updatedAt,
            files=files or [],
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


# A body's title is its first heading, so anything before one rules it out.
# ATX rules: horizontal space after the `#`, and a trailing run of `#` is a
# closing marker rather than part of the title.
_TITLE_HEADING_RE = re.compile(
    r"\s*#[ \t]+(\S(?:.*?\S)?)(?:[ \t]+#+)?[ \t]*(?:\r?\n|$)"
)


def skill_title(name: str, body: str) -> str:
    """The listing's display title: the author's own H1.

    A skill's ``name`` is a slug, so deriving a title from it destroys the
    author's casing — "seo-content-brief" reads back as "Seo content brief".
    The humanised slug is only the fallback for a body that opens with prose.
    """
    heading = _TITLE_HEADING_RE.match(body)
    if heading:
        return heading.group(1)
    words = re.sub(r"[-_]+", " ", name).strip()
    return words[:1].upper() + words[1:] if words else name


def active_version(
    listing: prisma.models.SkillListing,
) -> prisma.models.SkillListingVersion:
    if listing.ActiveVersion is None:
        raise ValueError(f"Skill listing '{listing.slug}' has no active version")
    return listing.ActiveVersion


class SkillReviewRequest(pydantic.BaseModel):
    """An admin's verdict on one submission; the version comes from the path."""

    is_approved: bool
    comments: str
    internal_comments: str | None = None


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
    files: list[SkillPackageFile] = pydantic.Field(
        default_factory=list,
        description=(
            "Files beside the submitted SKILL.md, so a reviewer sees what a "
            "package ships before approving it."
        ),
    )

    @classmethod
    def from_db(
        cls,
        version: prisma.models.SkillListingVersion,
        listing: prisma.models.SkillListing,
        files: list[SkillPackageFile] | None = None,
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
            files=files or [],
        )
