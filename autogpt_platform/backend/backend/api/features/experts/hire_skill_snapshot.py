"""Pin marketplace packages at hire creation, before background setup starts."""

import prisma
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from typing_extensions import Self

from backend.api.features.experts.errors import ExpertTemplateNotFoundError
from backend.data.db import query_raw_with_schema


class LegacySkillSnapshotError(ValueError):
    """The old hire has no trustworthy record of its originally requested bundle."""


class PinnedSkill(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    slug: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    title: str


class SkillInstallSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    release_id: str | None = None
    packages: list[PinnedSkill]

    @model_validator(mode="after")
    def unique_packages(self) -> Self:
        if len({p.slug for p in self.packages}) != len(self.packages):
            raise ValueError("Skill install snapshot contains duplicate package names")
        return self


async def capture_skill_snapshot(
    tx: prisma.Prisma, template_id: str, *, enabled: bool = True
) -> SkillInstallSnapshot:
    """Hold the publisher's shared lock until the containing hire transaction ends."""
    release_id = await lock_catalogue_read(tx)
    template = await tx.expert.find_first(
        where={
            "id": template_id,
            "isTemplate": True,
            "isArchived": False,
            "ownerUserId": None,
            "organizationId": None,
            "teamId": None,
        }
    )
    if template is None:
        raise ExpertTemplateNotFoundError(template_id)
    rows = (
        await query_raw_with_schema(
            "SELECT l.slug, v.id AS version_id, v.name AS title "
            'FROM {schema_prefix}"ExpertSkillListing" a '
            'JOIN {schema_prefix}"SkillListing" l ON l.id = a."skillListingId" '
            'JOIN {schema_prefix}"SkillListingVersion" v '
            'ON v.id = l."activeVersionId" '
            'WHERE a."expertId" = $1 AND NOT l."isDeleted" '
            'AND l."hasApprovedVersion" AND NOT v."isDeleted" '
            'AND v."isAvailable" AND v."submissionStatus" = \'APPROVED\' '
            "ORDER BY a.position, l.id",
            template_id,
            client=tx,
        )
        if enabled
        else []
    )
    return SkillInstallSnapshot(
        release_id=release_id,
        packages=[PinnedSkill.model_validate(row) for row in rows],
    )


async def lock_catalogue_read(tx: prisma.Prisma) -> str | None:
    states = await query_raw_with_schema(
        'SELECT "activeReleaseId" FROM {schema_prefix}"CatalogueState" '
        "WHERE id = 'marketplace' FOR SHARE",
        client=tx,
    )
    if len(states) != 1:
        raise RuntimeError("Catalogue state is missing; cannot safely pin a hire")
    return states[0]["activeReleaseId"]


def read_skill_snapshot(value: JsonValue | None) -> SkillInstallSnapshot:
    if value is None:
        raise LegacySkillSnapshotError(
            "This hire predates versioned skill setup. Its original skill bundle "
            "must be recovered explicitly before retrying; no current template "
            "skills have been installed."
        )
    return SkillInstallSnapshot.model_validate(value)
