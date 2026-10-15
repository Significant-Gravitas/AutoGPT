"""The ``.expert.zip`` file: an expert template taken off the platform.

The file is the marketplace packing list and nothing else: ``expert.json``
(persona, soul, routines, and an index of what else is in the zip), each
bundled skill's whole package under ``skills/<slug>/``, and each workflow as
``workflows/<nn>-<slug>.json`` in the format ``download_agent_file`` serves.

Memory, conversations and workspace files are kept out by construction, not by
a filter: every manifest model refuses a key it does not declare, and
:func:`validate_expert_package` refuses a member the manifest does not name.
"""

import json
import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from backend.api.features.experts.models import (
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

EXPERT_PACKAGE_FORMAT_VERSION = 1
MANIFEST_PATH = "expert.json"
SKILLS_DIR = "skills"
WORKFLOWS_DIR = "workflows"
PACKAGE_SUFFIX = ".expert.zip"

_SLUG_MAX_CHARS = 60
_SKILL_SLUG_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")
_WORKFLOW_PATH_RE = re.compile(rf"^{WORKFLOWS_DIR}/\d{{2,}}-[a-z0-9-]+\.json$")


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PackagedVoiceSample(VoiceSample):
    model_config = ConfigDict(extra="forbid")


class PackagedDayOneItem(ExpertDayOneItem):
    model_config = ConfigDict(extra="forbid")


class PackagedSkill(_Closed):
    """A bundled skill's card; its files are the folder at ``path``."""

    slug: str
    title: str
    description: str
    path: str


class PackagedWorkflow(_Closed):
    """A workflow's index entry; its graph is the file at ``path``."""

    path: str
    name: str | None
    description: str | None
    store_listing_version_id: str
    schedule_cron: str | None
    # Credential references are provider names only, never ids or values.
    integration_providers: list[str]


class PackagedRoutine(_Closed):
    key: str | None
    title: str
    prompt: str
    crons: list[str]
    asks: list[str]
    session_mode: str = "THREAD"


class ExpertManifest(_Closed):
    """``expert.json``. Protected soul rules are platform constants and owner,
    budget, schedule ids and credential ids belong to a hire, so none of
    them is here."""

    format_version: Literal[1] = EXPERT_PACKAGE_FORMAT_VERSION
    name: str
    role: str
    job_title: str | None
    tagline: str | None
    bio: str | None
    categories: list[str]
    color: str
    # A URL, not the image: template avatars ship under /public or are our
    # own https uploads, and embedding one would mean fetching it server-side.
    avatar_url: str | None
    identity: str
    voice_preferences: str
    voice_samples: list[PackagedVoiceSample]
    boundaries: str
    day_one: list[PackagedDayOneItem]
    skills: list[PackagedSkill]
    workflows: list[PackagedWorkflow]
    routines: list[PackagedRoutine]

    @field_validator("avatar_url")
    @classmethod
    def check_avatar_url(cls, value: str | None) -> str | None:
        return validate_avatar_url(value)


class ExpertPackage(BaseModel):
    """The manifest plus the bytes it names; what the zip is written from."""

    manifest: ExpertManifest
    skills: dict[str, SkillPackage]
    # Manifest path -> the workflow file's bytes.
    workflows: dict[str, bytes]


class ExpertPackageError(ValueError):
    """A package that breaks the format or the size cap. ``over_limit`` is the
    413 case, as on :class:`SkillPackageError`."""

    def __init__(self, message: str, *, over_limit: bool = False):
        super().__init__(message)
        self.over_limit = over_limit


def validate_expert_package(package: ExpertPackage) -> None:
    """Refuse a package whose members and manifest disagree, or that is over
    :data:`MAX_PACKAGE_BYTES` uncompressed, before a byte is written."""
    manifest = package.manifest
    _check_skills(manifest.skills, package.skills)
    _check_workflows(manifest.workflows, package.workflows)
    total = (
        len(manifest_json(manifest))
        + sum(skill.size_bytes for skill in package.skills.values())
        + sum(len(content) for content in package.workflows.values())
    )
    if total > MAX_PACKAGE_BYTES:
        raise ExpertPackageError(
            f"expert package is {total} bytes; the limit is {MAX_PACKAGE_BYTES}",
            over_limit=True,
        )


def _check_skills(cards: list[PackagedSkill], skills: dict[str, SkillPackage]) -> None:
    slugs = [card.slug for card in cards]
    if len(set(slugs)) != len(slugs) or set(slugs) != set(skills):
        raise ExpertPackageError("manifest skills and packaged skills differ")
    for card in cards:
        if not _SKILL_SLUG_RE.match(card.slug):
            raise ExpertPackageError(f"skill slug '{card.slug[:64]}' is not a folder")
        if card.path != skill_path(card.slug):
            raise ExpertPackageError(f"skill '{card.slug}' has path '{card.path}'")
        try:
            validate_package(skills[card.slug])
        except SkillPackageError as exc:
            raise ExpertPackageError(
                f"skill '{card.slug}': {exc}", over_limit=exc.over_limit
            ) from exc


def _check_workflows(entries: list[PackagedWorkflow], files: dict[str, bytes]) -> None:
    paths = [entry.path for entry in entries]
    if len(set(paths)) != len(paths) or set(paths) != set(files):
        raise ExpertPackageError("manifest workflows and packaged workflows differ")
    for path in paths:
        if not _WORKFLOW_PATH_RE.match(path):
            raise ExpertPackageError(f"workflow path '{path[:120]}' is not allowed")


def manifest_json(manifest: ExpertManifest) -> bytes:
    """Sorted and indented, so two downloads of one template are identical
    and a diff of two files reads."""
    return (
        json.dumps(
            manifest.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def skill_path(slug: str) -> str:
    return f"{SKILLS_DIR}/{slug}/"


def workflow_path(index: int, name: str | None) -> str:
    """The position keeps two workflows of one name apart and the files in
    roster order."""
    return f"{WORKFLOWS_DIR}/{index + 1:02d}-{_slug(name or '') or 'workflow'}.json"


def package_filename(name: str) -> str:
    return f"{_slug(name) or 'expert'}{PACKAGE_SUFFIX}"


def _slug(text: str) -> str:
    ascii_text = (
        unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    )
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower().replace("'", ""))
    return slug[:_SLUG_MAX_CHARS].strip("-")
