"""API models for marketplace skill listings.

A skill listing publishes a ``SKILL.md`` — instructions and examples the
copilot reads — and the files beside it, rather than a runnable graph. The
listing's ``slug`` is both its marketplace URL segment and the name the skill
takes once installed, so the two can never drift. ``name`` is that slug again,
never a display string: the title on the card is :func:`skill_title`.
"""

import base64
import datetime
import re

import prisma.enums
import prisma.models
import pydantic

from backend.copilot.tools.skills import ParsedSkill, SkillFile, render_skill_markdown
from backend.data.skill_package import SKILL_MD, file_sha256, package_tree_sha256
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


class ActiveSkillVersion(pydantic.BaseModel):
    """What the marketplace serves for a slug right now: the version an
    installed copy's baseline is compared against to know whether it is
    behind, and the hash that says whether the copy was edited."""

    listing_id: str
    version_id: str
    package_sha256: str | None = None
    # Delisted, or its live version withdrawn: copies stay, nothing updates.
    retired: bool = False


class SkillVersionPackage(pydantic.BaseModel):
    """One version's whole package as an install writes it: the SKILL.md
    text verbatim plus every file beside it. What a fast-forward writes,
    and what a merge uses as the update and as the baseline."""

    version_id: str
    listing_id: str
    slug: str
    version: int
    package_sha256: str | None
    skill_markdown: str
    files: list[SkillFile]
    required_providers: list[str] = []
    # SHA-256s an earlier install already scanned clean; skips the rescan.
    scanned_checksums: list[str] = []

    # The copilot executor has no database connection, so a package reaches
    # it over the DB manager RPC as JSON, and JSON has no bytes: the default
    # encoding tries UTF-8 and a font, a PNG or a spreadsheet in a package
    # fails it. Base64 carries any file, and only over the wire.
    @pydantic.field_serializer("files", when_used="json")
    def _files_over_the_wire(self, files: list[SkillFile]) -> list[dict[str, object]]:
        return [
            {
                "relative_path": entry.relative_path,
                "content_b64": base64.b64encode(entry.content).decode("ascii"),
                "is_executable": entry.is_executable,
            }
            for entry in files
        ]

    @pydantic.field_validator("files", mode="before")
    @classmethod
    def _files_from_the_wire(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        return [
            (
                SkillFile(
                    relative_path=str(item["relative_path"]),
                    content=base64.b64decode(item["content_b64"]),
                    is_executable=bool(item.get("is_executable", False)),
                )
                if isinstance(item, dict) and "content_b64" in item
                else item
            )
            for item in value
        ]


def legacy_skill_markdown(version: prisma.models.SkillListingVersion, slug: str) -> str:
    """Exactly what an install wrote for a version with no ``skillMarkdown``:
    the row's fields rendered through the same path the install used, under
    the listing slug. Byte-stable on purpose: the hash backfill and every
    later install of such a version depend on producing the same text."""
    return render_skill_markdown(
        ParsedSkill(
            name=slug,
            description=version.description.strip(),
            body=version.body.strip(),
            triggers=tuple(t.strip() for t in version.triggers if t.strip()),
            version=str(version.version),
            extra={
                key: value
                for key, value in (
                    ("license", version.license),
                    ("source", version.sourceRepo),
                    ("source_url", version.sourceUrl),
                )
                if value is not None
            },
        )
    )


def legacy_package_sha256(version: prisma.models.SkillListingVersion, slug: str) -> str:
    """The content hash of a pre-publisher version: its rendered SKILL.md plus
    its files, by the catalog's tree formula."""
    return package_sha256_of(
        legacy_skill_markdown(version, slug),
        [
            (row.relativePath, row.sha256, row.isExecutable)
            for row in version.Files or []
        ],
    )


def package_sha256_of(skill_markdown: str, files: list[tuple[str, str, bool]]) -> str:
    """The content hash of a package as installed: the SKILL.md text plus
    ``(path, sha256, executable)`` per sibling. Creator-published versions
    carry no stored hash, so their identity is computed from this whenever a
    copy needs comparing."""
    hashed = [(SKILL_MD, file_sha256(skill_markdown.encode("utf-8")), False)]
    return package_tree_sha256(hashed + files)


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
