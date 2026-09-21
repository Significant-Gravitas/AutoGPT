"""What an exported expert is, as a file.

The manifest is the contract between an export and every future import, so it
is a closed format: ``extra="forbid"`` everywhere means a key the format has no
name for is refused rather than carried. That is what makes the ticket's hard
rule enforceable — memory, conversations and workspace files cannot be exported
because there is nowhere in this schema to put them, and cannot be imported
because an archive that invents a place for them fails validation.

Skills ride as :class:`SkillPackage` values, unchanged from the skill package
format, so an expert's skills and a standalone skill stay one thing.
"""

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from backend.api.features.experts.models import (
    _EXPERT_SOUL_TEXT_MAX_LENGTH,
    EXPERT_AVATAR_URL_MAX_LENGTH,
    EXPERT_COLOR_MAX_LENGTH,
    EXPERT_DAY_ONE_MAX_ITEMS,
    EXPERT_IDENTITY_MAX_LENGTH,
    EXPERT_NAME_MAX_LENGTH,
    EXPERT_TAGLINE_MAX_LENGTH,
    ExpertDayOneItem,
    VoiceSample,
    validate_avatar_url,
)
from backend.copilot.tools.skills import (
    MAX_PACKAGE_BYTES,
    SkillPackage,
    SkillPackageError,
    validate_package,
)
from backend.data.graph import Graph

# An expert's roster is bounded by these long before a package is; they exist so
# a hand-written manifest cannot ask the importer to create thousands of rows.
MAX_PACKAGE_SKILLS = 50
MAX_PACKAGE_WORKFLOWS = 50
MAX_AVATAR_BYTES = 2 * 1024 * 1024
# The embedded graphs dominate the manifest, and a workflow carries one even
# when it also carries a marketplace reference: 50 of a large graph fits here,
# and this is still a fifth of the zip cap.
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
AVATAR_EXTENSIONS = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}
AVATAR_PATHS = {
    f"avatar.{extension}": mime for extension, mime in AVATAR_EXTENSIONS.items()
}

_SLUG_MAX_LENGTH = 100
_WORKFLOW_NAME_MAX_LENGTH = 500
_WORKFLOW_TEXT_MAX_LENGTH = 5_000


class ExpertPackageError(ValueError):
    """A package that breaks a cap or the format.

    ``over_limit`` separates a size refusal — 413 at the REST edge — from a
    malformed one, which is 400, exactly as ``SkillPackageError`` does.
    """

    def __init__(self, message: str, *, over_limit: bool = False):
        super().__init__(message)
        self.over_limit = over_limit


class PackagedIdentity(BaseModel):
    """Who the expert is on the outside — the roster card, before anyone opens
    a chat."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=EXPERT_NAME_MAX_LENGTH)
    role: str = Field(default="", max_length=EXPERT_NAME_MAX_LENGTH)
    tagline: str | None = Field(default=None, max_length=EXPERT_TAGLINE_MAX_LENGTH)
    bio: str | None = Field(default=None, max_length=EXPERT_IDENTITY_MAX_LENGTH)
    color: str = Field(default="", max_length=EXPERT_COLOR_MAX_LENGTH)
    categories: list[str] = Field(default_factory=list, max_length=20)


class PackagedSoul(BaseModel):
    """Everything the expert is told about itself. Lengths mirror
    ``ExpertSoulUpdate`` so a manifest that validates here still validates when
    the importer writes it."""

    model_config = ConfigDict(extra="forbid")

    identity: str = Field(default="", max_length=EXPERT_IDENTITY_MAX_LENGTH)
    voice_preferences: str = Field(default="", max_length=_EXPERT_SOUL_TEXT_MAX_LENGTH)
    boundaries: str = Field(default="", max_length=_EXPERT_SOUL_TEXT_MAX_LENGTH)
    voice_samples: list[VoiceSample] = Field(default_factory=list, max_length=8)


class PackagedSkill(BaseModel):
    """A skill's card in the manifest; its files live under ``skills/<slug>/``
    in the archive, and the reader refuses a manifest and a tree that disagree."""

    model_config = ConfigDict(extra="forbid")

    slug: str = Field(min_length=1, max_length=_SLUG_MAX_LENGTH)
    name: str = Field(min_length=1, max_length=_SLUG_MAX_LENGTH)
    description: str = Field(default="", max_length=_WORKFLOW_TEXT_MAX_LENGTH)


class PackagedWorkflow(BaseModel):
    """One of the expert's agents, carried twice over: as a marketplace
    reference and as the graph itself.

    The reference is preferred on import because it keeps the new owner on the
    published agent (updates, credits, attribution); the embedded graph is the
    fallback for a listing the importer cannot see.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="", max_length=_WORKFLOW_NAME_MAX_LENGTH)
    description: str = Field(default="", max_length=_WORKFLOW_TEXT_MAX_LENGTH)
    store_listing_slug: str | None = Field(default=None, max_length=_SLUG_MAX_LENGTH)
    store_listing_version_id: str | None = None
    creator_username: str | None = Field(default=None, max_length=_SLUG_MAX_LENGTH)
    graph: Graph | None = None
    schedule_cron: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def check_a_source_is_present(self):
        if self.graph is None and not (
            self.store_listing_version_id or self.store_listing_slug
        ):
            raise ValueError(
                "workflow needs a store listing reference or an embedded graph"
            )
        return self


class PackagedAvatar(BaseModel):
    """Where the expert's picture comes from: a member of this archive, or a
    URL the importer may keep as-is. Never both, so there is nothing to
    reconcile on import."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["file", "url"]
    path: str | None = None
    url: str | None = Field(default=None, max_length=EXPERT_AVATAR_URL_MAX_LENGTH)

    @model_validator(mode="after")
    def check_exactly_one_source(self):
        if (self.kind == "file") != (self.path is not None):
            raise ValueError("a 'file' avatar carries a path and nothing else")
        if (self.kind == "url") != (self.url is not None):
            raise ValueError("a 'url' avatar carries a url and nothing else")
        if self.path is not None and self.path not in AVATAR_PATHS:
            raise ValueError(
                f"avatar file must be one of {sorted(AVATAR_PATHS)}, not '{self.path[:40]}'"
            )
        if self.url is not None:
            validate_avatar_url(self.url)
        return self


class ExpertManifest(BaseModel):
    """``expert.json``: the whole expert minus its bytes.

    Closed on purpose — see the module docstring. ``format_version`` is a
    literal so a future format is refused by an old reader rather than
    half-understood.
    """

    model_config = ConfigDict(extra="forbid")

    format_version: Literal[1] = 1
    identity: PackagedIdentity
    soul: PackagedSoul = Field(default_factory=PackagedSoul)
    # A template's "What {name} sets up on day one" — written by its creator
    # rather than derived from the workflows, so nothing else can restore it.
    day_one: list[ExpertDayOneItem] = Field(
        default_factory=list, max_length=EXPERT_DAY_ONE_MAX_ITEMS
    )
    skills: list[PackagedSkill] = Field(
        default_factory=list, max_length=MAX_PACKAGE_SKILLS
    )
    workflows: list[PackagedWorkflow] = Field(
        default_factory=list, max_length=MAX_PACKAGE_WORKFLOWS
    )
    avatar: PackagedAvatar | None = None
    # The expert's tool settings, copied verbatim by hire and by import alike.
    # Opaque on purpose: its shape belongs to the copilot, and a schema here
    # would refuse a package written by a newer one.
    tool_profile: JsonValue | None = None

    @model_validator(mode="after")
    def check_skill_slugs_are_unique(self):
        slugs = [skill.slug for skill in self.skills]
        duplicate = next((s for s in slugs if slugs.count(s) > 1), None)
        if duplicate:
            raise ValueError(f"skill '{duplicate}' is listed twice")
        return self


class ExpertPackage(BaseModel):
    """A manifest plus the bytes it refers to: every skill's files, and the
    avatar when it travels inside the archive rather than as a URL."""

    manifest: ExpertManifest
    skills: dict[str, SkillPackage] = Field(default_factory=dict)
    avatar_bytes: bytes | None = None
    avatar_mime: str | None = None

    @property
    def slug(self) -> str:
        """The filename stem a download is offered under."""
        return expert_slug(self.manifest.identity.name)

    @property
    def size_bytes(self) -> int:
        return (
            len(manifest_json(self.manifest))
            + sum(skill.size_bytes for skill in self.skills.values())
            + len(self.avatar_bytes or b"")
        )


def validate_expert_package(package: ExpertPackage) -> None:
    """Refuse a package that the ``.expert.zip`` reader would refuse.

    The exporter runs it before a file is written and the zip writer runs it
    again, so a download is never an archive that fails on re-import: the
    manifest cap, each skill's own caps, the avatar cap, and the combined
    uncompressed size the reader bounds the whole tree by.
    """
    size = len(manifest_json(package.manifest))
    if size > MAX_MANIFEST_BYTES:
        raise ExpertPackageError(
            f"expert.json would be {size} bytes; the limit is {MAX_MANIFEST_BYTES}",
            over_limit=True,
        )
    if len(package.skills) > MAX_PACKAGE_SKILLS:
        raise ExpertPackageError(
            f"package carries {len(package.skills)} skills; the limit is "
            f"{MAX_PACKAGE_SKILLS}",
            over_limit=True,
        )
    for slug, skill in sorted(package.skills.items()):
        try:
            validate_package(skill)
        except SkillPackageError as exc:
            raise ExpertPackageError(
                f"skill '{slug[:120]}': {exc}", over_limit=exc.over_limit
            )
    if (
        package.avatar_bytes is not None
        and len(package.avatar_bytes) > MAX_AVATAR_BYTES
    ):
        raise ExpertPackageError(
            f"avatar is {len(package.avatar_bytes)} bytes; the limit is "
            f"{MAX_AVATAR_BYTES}",
            over_limit=True,
        )
    if package.size_bytes > MAX_PACKAGE_BYTES:
        raise ExpertPackageError(
            f"package unpacks to {package.size_bytes} bytes; the limit is "
            f"{MAX_PACKAGE_BYTES}",
            over_limit=True,
        )


def manifest_json(manifest: ExpertManifest) -> bytes:
    """The manifest as it is written into an archive.

    Sorted and indented rather than pydantic's field order, so the file stays
    diffable and two exports of an unchanged expert are byte-identical.
    """
    return json.dumps(
        manifest.model_dump(mode="json"), sort_keys=True, indent=2
    ).encode("utf-8")


_SLUG_SEPARATORS = re.compile(r"[^a-z0-9]+")


def expert_slug(name: str) -> str:
    """A filename stem from an expert's name: ascii, lowercase, no separators a
    Content-Disposition header or a filesystem would have to escape."""
    slug = _SLUG_SEPARATORS.sub("-", name.strip().lower()).strip("-")
    return slug[:60].strip("-") or "expert"
