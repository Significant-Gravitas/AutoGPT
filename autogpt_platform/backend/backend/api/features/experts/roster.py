"""The expert roster as the skills catalog ships it: one YAML file per expert.

``Significant-Gravitas/skills-catalog`` holds ``experts/<key>.yml`` beside the
skill packages. Each file is the whole template: persona, voice, day-one rows,
preload workflows, routine proposals and the ordered skills it bundles. The
catalog's ``release.json`` binds every file's hash into the release, and
``store.skill_catalog`` publishes the roster together with the skills so a
template never names a package the marketplace does not have.

This module only reads and validates. What a roster entry *does* to the
database is ``experts.seed``.
"""

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.api.features.experts.models import ExpertDayOneItem, VoiceSample
from backend.api.features.experts.roster_types import (
    PreloadSeed,
    RosterEntry,
    RoutineSeed,
)
from backend.api.features.store.categories import validate_canonical_categories

EXPERTS_DIR = "experts"
EXPERT_FILE_SUFFIX = ".yml"

_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class RosterError(ValueError):
    """An expert file is missing, malformed or inconsistent with its name."""


class PreloadSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: str = Field(min_length=1)
    # Five-field cron; None installs the workflow without a schedule.
    cron: str | None = None


class RoutineSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(pattern=_KEY_RE.pattern)
    title: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    crons: list[str] = []
    asks: list[str] = []
    session_mode: Literal["THREAD", "FRESH"] = "THREAD"


class ExpertSpec(BaseModel):
    """One ``experts/<key>.yml``. Field names match ``RosterEntry`` so the
    two never drift; ``skills`` is the ordered bundle."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(pattern=_KEY_RE.pattern)
    name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    job_title: str = Field(min_length=1)
    tagline: str = Field(min_length=1)
    avatar_url: str | None = None
    categories: list[str] = Field(min_length=1)
    bio: str = Field(min_length=1)
    identity: str = Field(min_length=1)
    voice_preferences: str = ""
    voice_samples: list[VoiceSample] = []
    boundaries: str = ""
    day_one: list[ExpertDayOneItem] = []
    preloads: list[PreloadSpec] = []
    routines: list[RoutineSpec] = []
    skills: list[str] = []

    @field_validator("categories")
    @classmethod
    def canonical_categories(cls, value: list[str]) -> list[str]:
        return validate_canonical_categories(value)

    @field_validator("skills")
    @classmethod
    def unique_skills(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("a skill is bundled twice")
        for slug in value:
            if not _KEY_RE.fullmatch(slug):
                raise ValueError(f"'{slug}' is not a valid skill slug")
        return value

    @model_validator(mode="after")
    def unique_children(self) -> "ExpertSpec":
        routine_keys = [routine.key for routine in self.routines]
        if len(set(routine_keys)) != len(routine_keys):
            raise ValueError("a routine key is used twice")
        preload_slugs = [preload.slug for preload in self.preloads]
        if len(set(preload_slugs)) != len(preload_slugs):
            raise ValueError("a preload slug is listed twice")
        return self


def parse_expert(text: str, *, expected_key: str | None = None) -> RosterEntry:
    """Validate one expert file's text into the roster entry the seed takes."""
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RosterError(f"not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise RosterError("an expert file must contain a mapping")
    try:
        spec = ExpertSpec.model_validate(raw)
    except ValueError as exc:
        raise RosterError(str(exc)) from exc
    if expected_key is not None and spec.key != expected_key:
        raise RosterError(
            f"key is '{spec.key}' but the file is named for '{expected_key}'"
        )
    return roster_entry(spec)


def roster_entry(spec: ExpertSpec) -> RosterEntry:
    return RosterEntry(
        key=spec.key,
        name=spec.name,
        role=spec.role,
        job_title=spec.job_title,
        tagline=spec.tagline,
        avatar_url=spec.avatar_url,
        bio=spec.bio,
        bundled_skills=list(spec.skills),
        categories=list(spec.categories),
        identity=spec.identity,
        voice_preferences=spec.voice_preferences,
        voice_samples=list(spec.voice_samples),
        boundaries=spec.boundaries,
        day_one=list(spec.day_one),
        preloads=[PreloadSeed(slug=p.slug, cron=p.cron) for p in spec.preloads],
        routines=[
            RoutineSeed(
                key=r.key,
                title=r.title,
                prompt=r.prompt,
                crons=list(r.crons),
                asks=list(r.asks),
                session_mode=r.session_mode,
            )
            for r in spec.routines
        ],
    )


def expert_file(root: Path, key: str) -> Path:
    return root / EXPERTS_DIR / f"{key}{EXPERT_FILE_SUFFIX}"


def load_roster(root: Path) -> list[RosterEntry]:
    """Every ``experts/*.yml`` under *root*, sorted by key.

    A file whose ``key`` disagrees with its name fails, because the release
    manifest and the template lookup both go by the file name.
    """
    directory = root / EXPERTS_DIR
    if not directory.is_dir():
        raise RosterError(f"{EXPERTS_DIR}/ is missing from the catalog")
    roster: list[RosterEntry] = []
    for path in sorted(directory.glob(f"*{EXPERT_FILE_SUFFIX}")):
        key = path.name[: -len(EXPERT_FILE_SUFFIX)]
        try:
            roster.append(
                parse_expert(path.read_text(encoding="utf-8"), expected_key=key)
            )
        except RosterError as exc:
            raise RosterError(f"{EXPERTS_DIR}/{path.name}: {exc}") from exc
    names = [entry["name"].lower() for entry in roster]
    if len(set(names)) != len(names):
        raise RosterError("two experts share a display name")
    return roster
