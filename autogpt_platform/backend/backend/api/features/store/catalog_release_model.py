"""Strict, portable catalogue release and environment adoption contracts."""

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from typing_extensions import Self

Slug = Annotated[str, Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PackageFile(StrictModel):
    path: str
    sha256: Digest
    executable: bool


class PackageManifest(StrictModel):
    slug: Slug
    tree_sha256: Digest
    files: list[PackageFile]

    @model_validator(mode="after")
    def validate_tree(self) -> Self:
        paths = [file.path for file in self.files]
        if paths != sorted(set(paths)) or "SKILL.md" not in paths:
            raise ValueError(
                "package files must be unique, sorted and include SKILL.md"
            )
        if any(file.path == "SKILL.md" and file.executable for file in self.files):
            raise ValueError("executable SKILL.md is not supported")
        if any(
            path.startswith("/")
            or "\\" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
            for path in paths
        ):
            raise ValueError("unsafe package path")
        if digest([file.model_dump() for file in self.files]) != self.tree_sha256:
            raise ValueError("package tree hash mismatch")
        return self


class ExpertManifest(StrictModel):
    key: Slug
    skills: list[Slug]

    @model_validator(mode="after")
    def unique_skills(self) -> Self:
        if len(self.skills) != len(set(self.skills)):
            raise ValueError("duplicate expert skill assignment")
        return self


class ReleaseManifest(StrictModel):
    schema_version: Literal[1]
    release_key: str = Field(min_length=1, max_length=200)
    provenance: dict[str, JsonValue] = Field(default_factory=dict)
    catalog_sha256: Digest
    packages: list[PackageManifest]
    experts: list[ExpertManifest]
    retirements: list[Slug] = Field(default_factory=list)
    system_packages: list[JsonValue] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        packages = [package.slug for package in self.packages]
        experts = [expert.key for expert in self.experts]
        for names in (packages, experts, self.retirements):
            if len(names) != len(set(names)):
                raise ValueError("duplicate release entry")
        if set(packages) & set(self.retirements):
            raise ValueError("an active package cannot also be retired")
        for expert in self.experts:
            if set(expert.skills) & set(self.retirements):
                raise ValueError("an assigned skill cannot be retired")
            if set(expert.skills) - set(packages):
                raise ValueError(f"expert {expert.key} references missing packages")
        if self.system_packages:
            raise ValueError("system packages require a supported publisher capability")
        return self


class Adoption(StrictModel):
    skills: dict[Slug, str | None]
    experts: dict[Slug, str]
    activate_experts: list[Slug] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        if (
            len(self.activate_experts) != len(set(self.activate_experts))
            or set(self.activate_experts) - self.experts.keys()
        ):
            raise ValueError("activation keys must uniquely identify adopted experts")
        for ids in (list(self.skills.values()), list(self.experts.values())):
            present = [record_id for record_id in ids if record_id is not None]
            if any(not record_id for record_id in present):
                raise ValueError("record IDs must not be empty")
            if len(present) != len(set(present)):
                raise ValueError("duplicate adoption record ID")
        return self


class PublishedSkill(StrictModel):
    listing_id: str
    version_id: str | None
    retired: bool = False
    has_approved_version: bool = True


class PublishedExpert(StrictModel):
    expert_id: str
    skills: list[str]
    is_archived: bool = False


class ReleaseSnapshot(StrictModel):
    skills: dict[str, PublishedSkill]
    experts: dict[str, PublishedExpert]


class Preview(StrictModel):
    database_target: str
    rollback_release_id: str
    release_id: str
    revision: str
    previous_release_id: str | None
    generation: int
    state_sha256: str
    adoption_sha256: str
    create_skills: list[str]
    update_skills: list[str]
    retire_skills: list[str]
    expert_keys: list[str]
    activate_experts: list[str]
