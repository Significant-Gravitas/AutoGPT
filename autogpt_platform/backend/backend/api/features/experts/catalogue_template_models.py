"""Typed requests and canonical persona definitions for scoped provisioning."""

from typing import Literal

from pydantic import model_validator
from typing_extensions import Self

from backend.api.features.experts.avatar_catalog import resolve_avatar_url
from backend.api.features.experts.models import encode_day_one, encode_voice_preferences
from backend.api.features.experts.roster_types import RosterEntry
from backend.api.features.experts.seed import ROSTER
from backend.api.features.store.catalog_release_model import Digest, Slug, StrictModel
from backend.api.features.store.categories import validate_canonical_categories


class TemplateAdoption(StrictModel):
    schema_version: Literal[1] = 1
    experts: dict[Slug, str | None]

    @model_validator(mode="after")
    def unique_records(self) -> Self:
        ids = [value for value in self.experts.values() if value is not None]
        if len(ids) != len(set(ids)) or any(not value for value in ids):
            raise ValueError("template IDs must be nonempty and unique")
        return self


class TemplatePreview(StrictModel):
    preview_sha256: Digest
    state_sha256: Digest
    definitions_sha256: Digest
    adoption_sha256: Digest
    create_keys: list[str]
    preserve_keys: list[str]
    preload_versions: dict[str, str]


class ProvisionedTemplates(StrictModel):
    experts: dict[str, str]
    activate_experts: list[str]


class TemplateFields(StrictModel):
    name: str
    role: str
    jobTitle: str
    tagline: str
    avatarUrl: str | None
    identity: str
    bio: str
    voicePreferences: str
    boundaries: str
    categories: list[str]
    dayOne: list[dict[str, str]]


class TemplateRoutine(StrictModel):
    key: str
    title: str
    prompt: str
    crons: list[str]
    asks: list[str]
    sessionMode: Literal["FRESH", "PINNED", "THREAD"]


class TemplatePreload(StrictModel):
    slug: str
    cron: str | None


class TemplateDefinition(StrictModel):
    fields: TemplateFields
    routines: list[TemplateRoutine]
    preloads: list[TemplatePreload]

    @model_validator(mode="after")
    def unique_children(self) -> Self:
        if len({routine.key for routine in self.routines}) != len(self.routines):
            raise ValueError("duplicate template routine key")
        if len({preload.slug for preload in self.preloads}) != len(self.preloads):
            raise ValueError("duplicate template preload")
        return self


class TemplateRecord(StrictModel):
    id: str
    name: str
    isTemplate: bool
    ownerUserId: str | None
    organizationId: str | None
    teamId: str | None
    fingerprint: str


class PreloadVersion(StrictModel):
    slug: str
    listing_id: str
    version_id: str
    fingerprint: str


class CatalogueCoordinationState(StrictModel):
    activeReleaseId: str | None
    generation: int


def template_definitions() -> dict[str, TemplateDefinition]:
    # Importing ROSTER reads definitions, never invoking legacy seed helpers.
    definitions = {entry["name"].lower(): _definition(entry) for entry in ROSTER}
    if len(definitions) != len(ROSTER):
        raise ValueError("duplicate roster key")
    return definitions


def _definition(entry: RosterEntry) -> TemplateDefinition:
    return TemplateDefinition(
        fields=TemplateFields(
            name=entry["name"],
            role=entry["role"],
            jobTitle=entry["job_title"],
            tagline=entry["tagline"],
            avatarUrl=resolve_avatar_url(entry["avatar_url"]),
            identity=entry["identity"],
            bio=entry["bio"],
            voicePreferences=encode_voice_preferences(
                entry["voice_preferences"], entry.get("voice_samples") or []
            ),
            boundaries=entry["boundaries"],
            categories=validate_canonical_categories(entry["categories"]),
            dayOne=encode_day_one(entry["day_one"]),
        ),
        routines=[
            TemplateRoutine.model_validate(
                {
                    "key": routine["key"],
                    "title": routine["title"],
                    "prompt": routine["prompt"],
                    "crons": routine["crons"],
                    "asks": routine["asks"],
                    "sessionMode": routine["session_mode"],
                }
            )
            for routine in entry["routines"]
        ],
        preloads=[
            TemplatePreload.model_validate(preload) for preload in entry["preloads"]
        ],
    )
