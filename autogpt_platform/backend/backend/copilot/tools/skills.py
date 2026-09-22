"""Self-distilled skill registry for CoPilot.

Skills follow the Anthropic Agent Skills protocol — each skill is a
folder under ``workspace://skills/{slug}/`` containing a ``SKILL.md``
file with YAML frontmatter (``name``, ``description``, optional
``triggers`` / ``version``) plus a markdown body. Optional sibling
``references/``, ``scripts/``, and ``assets/`` files live in the same
folder and are reachable via :tool:`read_workspace_file`.

The model discovers skills via the ``<available_skills>`` block that
:mod:`backend.copilot.service` injects into the first user message
(see ``inject_user_context``). It loads body + sibling list with
``read_skill`` and writes new skills via ``store_skill`` whenever a
procedure is worth re-using.

Default seeded skills (the agent-building guide and the MCP-tool guide)
ship as on-disk markdown under ``copilot/sdk/`` and are surfaced through
the same registry so the model uses one mechanism for both built-in
and user-distilled knowledge.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import posixpath
import re
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import yaml
from pydantic import BaseModel

from backend.api.features.store.exceptions import VirusDetectedError, VirusScanError
from backend.copilot.model import ChatSession
from backend.copilot.service import SKILLS_UPDATE_TAG, strip_server_injected_tags
from backend.data.db_accessors import experts_db, workspace_db
from backend.data.redis_client import get_redis_async
from backend.data.skill_capacity import MAX_SKILLS_PER_EXPERT
from backend.data.skill_capacity import SKILL_ORIGIN_LABELS as _ORIGIN_LABELS
from backend.data.skill_capacity import SKILL_ORIGIN_MARKETPLACE
from backend.data.skill_capacity import SKILL_ORIGIN_METADATA_KEY as _META_SKILL_ORIGIN
from backend.data.skill_capacity import SKILL_ORIGIN_USER
from backend.data.skill_capacity import SKILL_ORIGINS as _SKILL_ORIGINS
from backend.data.skill_capacity import SkillLimitError, SkillOwnedError
from backend.data.skill_capacity import normalize_skill_origin as _normalize_origin
from backend.data.workspace_scope import (
    EXPERT_SKILL_SCOPE_DENIED,
    WorkspaceAccessDeniedError,
    WorkspaceScope,
    expert_skills_folder,
)
from backend.executor.cluster_lock import AsyncClusterLock
from backend.util.exceptions import ConflictError
from backend.util.feature_flag import Flag, is_feature_enabled
from backend.util.workspace import WorkspaceManager

from .base import BaseTool
from .models import ErrorResponse, ResponseType, ToolResponseBase
from .workdir import (
    read_workdir_bytes,
    remove_from_workdir,
    save_to_workdir,
    set_executable,
    workdir_root,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Limits — keep the per-turn <available_skills> index small enough that it
# does not strain Anthropic prompt caches and does not crowd out the user's
# turn budget.  A typical user skill line lands around 150-200 chars
# (~50 tok), so 150 entries ≈ 7.5k tokens.  Filling every description and
# trigger to the per-field caps below is roughly 66k tokens under the same
# estimate; actual token cost varies by content and tokenizer.
# The cap is per owner folder and per origin: what the owner saves and what
# the platform installs (a hire's bundle, a marketplace install) each get
# MAX_SKILLS_PER_EXPERT, so a template's size never eats the owner's own room.
# Built-in seeded skills are tiny so first-touch users see well under
# 200 tokens of overhead.
# ---------------------------------------------------------------------------
MAX_NAME_CHARS = 64
MAX_DESCRIPTION_CHARS = 1024
# Loaded only on activation, so it costs nothing per turn; 50k clears
# skill-creator's 33 KB SKILL.md, the package authors are told to copy.
MAX_BODY_CHARS = 50_000
# Triggers appear inline in the per-turn ``<available_skills>`` index,
# so an unbounded list (or one huge trigger) would balloon the prefix
# the model parses every turn.  Cap both the count and the per-entry
# length so a misbehaving caller cannot blow the token budget.
MAX_TRIGGERS = 10
MAX_TRIGGER_CHARS = 64
# Files a skill may carry beside its SKILL.md.  The largest package in the
# public ``anthropics/skills`` set is 83 files, so 100 clears the ecosystem
# with room to spare while keeping one activation's copy into the sandbox
# bounded.  Enumeration fetches one more than this so a folder that exceeds
# it is reported rather than silently truncated.
MAX_PACKAGE_FILES = 100
# Largest public package file is 237 KB and package 5.4 MB, so these leave
# room for a font while keeping a hire-time install bounded.
MAX_PACKAGE_FILE_BYTES = 2 * 1024 * 1024
MAX_PACKAGE_BYTES = 20 * 1024 * 1024
# The spec keeps references one level deep; 8 stops a path of directories.
MAX_PACKAGE_PATH_DEPTH = 8
# An E2B write is a 200 ms round trip and a workspace read a blob fetch, so
# 60 of either in series is seconds of a turn.  Bounded, not unlimited.
_COPY_CONCURRENCY = 16
# Attempts at reading a package whose tree keeps moving under the read.
_PACKAGE_READ_ATTEMPTS = 3
# Passes delete_user_skill will make over a folder, each one page deep.
# Bounded so a file that cannot be deleted can never spin the loop.
_DELETE_PASSES = 20
SKILL_FOLDER = "/skills"


def skill_folder(expert_id: str | None) -> str:
    """Owner folder. Personal Otto's skills live under ``/skills``; each
    expert's own skills under ``/experts/<id>/skills``. Ownership is the
    folder — there is no separate assignment record."""
    return SKILL_FOLDER if expert_id is None else expert_skills_folder(expert_id)


# Redis-cached index TTL.  Skill content changes only on store/delete so a
# 60s TTL with explicit invalidation gives near-zero index latency on warm
# turns without unbounded staleness for cross-instance edits.
SKILLS_INDEX_CACHE_TTL_S = 60
# Versioned: an entry cached before origins were recorded would count every
# installed skill as the owner's for a TTL after deploy.
SKILLS_INDEX_CACHE_KEY = "copilot:skills_index:v2:{user_id}"

# A skill name on an expert's row that resolves to no folder in Otto's
# library — a marketplace attachment, or a skill deleted after assignment —
# can never be copied, so the heal below remembers it and stops re-scanning.
# The TTL bounds how long a name that becomes copyable waits for its backfill.
SKILLS_HEAL_BACKOFF_TTL_S = 600
SKILLS_HEAL_BACKOFF_KEY = "copilot:skills_heal_backoff:{user_id}:expert:{expert_id}"

# WorkspaceFile.metadata keys for the skill index fast path — avoids the
# storage read when the metadata was written at store time (anything
# pre-dating this change falls back to read+parse, see
# ``_list_user_skills_from_workspace``).
_META_KIND = "kind"
_META_KIND_VALUE = "copilot_skill"
_META_DESCRIPTION = "description"
_META_TRIGGERS = "triggers"
_META_VERSION = "version"
# The workspace has no mode bits, so a script's executable bit survives
# store → copy → sandbox as this flag.
_META_EXECUTABLE = "executable"
# Where a skill in an owner's folder came from.  Kept in the row's metadata
# (server-written; the frontmatter is the author's to edit) so the per-owner
# cap counts what the owner saved apart from what the platform installed.
# The built-in defaults: never stored, never counted, so not a storable origin.
SKILL_ORIGIN_PLATFORM = "platform"

# Skill names are slug-like — lowercase letters, digits, dashes, underscores.
# Must start and end with [a-z0-9] (no trailing/leading punctuation) so the
# folder name is clean and the on-screen index never has dangling dashes.
# Length cap matches MAX_NAME_CHARS via the {0,62} interior + 1 anchor at
# each end.
_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")


# ---------------------------------------------------------------------------
# Default skills — migrated from the legacy ``get_agent_building_guide``
# tool so users get a uniform discovery surface.  These
# are *read-only* — store_skill / delete_skill refuse to touch them.  Body
# is loaded from disk lazily so adding more defaults is a drop-in.
# ---------------------------------------------------------------------------
_SDK_DIR = Path(__file__).parent.parent / "sdk"


@dataclass(frozen=True)
class _DefaultSkill:
    name: str
    description: str
    body_path: Path
    triggers: tuple[str, ...] = ()


DEFAULT_SKILLS: tuple[_DefaultSkill, ...] = (
    _DefaultSkill(
        name="agent_building_guide",
        description=(
            "Agent JSON building protocol — block IDs, link semantics, "
            "AgentExecutorBlock + MCPToolBlock usage, and the iterative "
            "create → dry-run → fix loop."
        ),
        body_path=_SDK_DIR / "agent_generation_guide.md",
        triggers=(
            "create_agent",
            "edit_agent",
            "validate_agent_graph",
            "fix_agent_graph",
        ),
    ),
)

_DEFAULT_SKILLS_BY_NAME: dict[str, _DefaultSkill] = {s.name: s for s in DEFAULT_SKILLS}


# ---------------------------------------------------------------------------
# Frontmatter + on-the-wire format helpers
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)


# Frontmatter outside the core skill fields. Dropping it rewrites the author's
# SKILL.md on every store, so it rides through parse and render.
_CARRIED_FRONTMATTER_KEYS = (
    "license",
    "compatibility",
    "allowed-tools",
    "metadata",
    "source",
    "source_url",
)


@dataclass(frozen=True)
class ParsedSkill:
    """A SKILL.md decoded into its frontmatter metadata + markdown body."""

    name: str
    description: str
    body: str
    triggers: tuple[str, ...] = ()
    version: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    # Recorded on the row at store time, never parsed from the file.  ``None``
    # is a row stored before origins were recorded: it counts against the
    # owner's budget (see :func:`budget_origin`) but is nobody's to defend,
    # so a platform install may claim it.
    origin: str | None = None


def budget_origin(skill: ParsedSkill) -> str:
    """The budget a stored skill fills: its recorded origin, or the owner's
    for a row that has none."""
    return skill.origin or SKILL_ORIGIN_USER


def parse_skill_markdown(text: str, fallback_name: str = "") -> ParsedSkill | None:
    """Parse a ``SKILL.md`` blob (YAML frontmatter + markdown body).

    Returns ``None`` if the file does not match the canonical shape —
    callers treat that as "this isn't a real skill, skip it" so a stray
    file in ``skills/`` cannot break the index.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return None
    raw_meta, body = match.group(1), match.group(2)
    try:
        meta = yaml.safe_load(raw_meta) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(meta, dict):
        return None
    name = str(meta.get("name") or fallback_name).strip()
    description = str(meta.get("description") or "").strip()
    if not name or not description:
        return None
    raw_triggers = meta.get("triggers") or ()
    if isinstance(raw_triggers, str):
        raw_triggers = [t.strip() for t in raw_triggers.split(",") if t.strip()]
    elif not isinstance(raw_triggers, list):
        raw_triggers = []
    triggers = tuple(str(t).strip() for t in raw_triggers if str(t).strip())
    version = meta.get("version")
    return ParsedSkill(
        name=name,
        description=description,
        body=body.lstrip("\n"),
        triggers=triggers,
        version=str(version) if version is not None else None,
        extra={k: meta[k] for k in _CARRIED_FRONTMATTER_KEYS if k in meta},
    )


def render_skill_markdown(skill: ParsedSkill) -> str:
    """Render a :class:`ParsedSkill` to canonical ``SKILL.md`` text.

    Kept symmetric with :func:`parse_skill_markdown` so a round-trip via
    ``store_skill`` → workspace → ``read_skill`` produces the same
    metadata the model wrote.
    """
    meta: dict[str, Any] = {"name": skill.name, "description": skill.description}
    for key in _CARRIED_FRONTMATTER_KEYS:
        if key in skill.extra:
            meta[key] = skill.extra[key]
    if skill.triggers:
        meta["triggers"] = list(skill.triggers)
    if skill.version:
        meta["version"] = skill.version
    frontmatter = yaml.safe_dump(meta, sort_keys=False).strip()
    return f"---\n{frontmatter}\n---\n\n{skill.body.rstrip()}\n"


def _validate_name(name: str) -> str | None:
    if not _NAME_RE.match(name):
        return (
            "name must be a slug (lowercase a-z, 0-9, _ or -; "
            f"1-{MAX_NAME_CHARS} chars; must start with a letter or digit)"
        )
    if name in _DEFAULT_SKILLS_BY_NAME:
        return f"'{name}' is a built-in skill and cannot be overwritten"
    return None


def validate_skill_content(
    description: str, body: str, triggers: Iterable[str]
) -> None:
    trigger_list = list(triggers)
    if not description:
        raise ValueError("description is required")
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise ValueError(
            f"description is {len(description)}/{MAX_DESCRIPTION_CHARS} chars "
            f"— trim {len(description) - MAX_DESCRIPTION_CHARS} "
            "(it appears in every turn's skills index)"
        )
    if not body:
        raise ValueError("body is required")
    if len(body) > MAX_BODY_CHARS:
        raise ValueError(f"body must be ≤{MAX_BODY_CHARS} chars")
    if len(trigger_list) > MAX_TRIGGERS:
        raise ValueError(
            f"triggers must be ≤{MAX_TRIGGERS} entries "
            "(they are inlined in <available_skills> every turn)"
        )
    oversized_trigger = next(
        (trigger for trigger in trigger_list if len(trigger) > MAX_TRIGGER_CHARS),
        None,
    )
    if oversized_trigger is not None:
        raise ValueError(
            f"trigger '{oversized_trigger[:32]}…' exceeds {MAX_TRIGGER_CHARS} chars"
        )


# ---------------------------------------------------------------------------
# Workspace registry — read/write/delete user skills as folders under
# ``/skills/{slug}/SKILL.md``.  We construct a *session-less*
# :class:`WorkspaceManager` so the skill folder is shared across every
# session for the user (the default per-session prefix would silo each
# skill to one chat).
# ---------------------------------------------------------------------------


async def _get_user_skill_manager(
    user_id: str, scope: WorkspaceScope | None = None
) -> WorkspaceManager:
    workspace = await workspace_db().get_or_create_workspace(user_id)
    return WorkspaceManager(user_id, workspace.id, session_id=None, scope=scope)


async def resolve_skill_scope(
    user_id: str | None, expert_id: str | None
) -> WorkspaceScope | None:
    """Workspace grants for an expert session's skill folder.

    ``None`` means unrestricted — personal Otto, REST callers, and
    anonymous default-skill reads. The scope comes from the persisted expert
    attribution on the session, never from a tool argument.
    """
    if not user_id or expert_id is None:
        return None
    return await workspace_db().resolve_expert_workspace_scope(user_id, expert_id)


class SkillOwner(BaseModel):
    """Whose skill folder a tool call operates on, plus the workspace scope
    that call must run under."""

    expert_id: str | None
    scope: WorkspaceScope | None


async def resolve_skill_owner(
    user_id: str, session: ChatSession, requested_expert_id: str | None
) -> SkillOwner | str:
    """Decide the skill owner for a tool call, or return a denial message.

    An expert session always operates on its own folder; naming any other
    owner is refused. Personal Otto operates on its own folder by
    default and may name one of the owner's active experts to manage that
    expert's skills.
    """
    if session.expert_id is not None:
        if requested_expert_id and requested_expert_id != session.expert_id:
            return EXPERT_SKILL_SCOPE_DENIED
        return SkillOwner(
            expert_id=session.expert_id,
            scope=await resolve_skill_scope(user_id, session.expert_id),
        )
    if not requested_expert_id:
        return SkillOwner(expert_id=None, scope=None)
    expert = await experts_db().get_expert(
        user_id, requested_expert_id, include_workflows=False
    )
    if expert is None:
        return f"Expert '{requested_expert_id}' was not found on this account."
    return SkillOwner(expert_id=expert.id, scope=None)


# Best-effort package-write coordination. Root publication enforces capacity
# in PostgreSQL independently of this lease or the cached skill index.
_SKILL_WRITE_LOCK_KEY_PREFIX = "copilot:skill_write:"
_SKILL_WRITE_LOCK_TTL_SECONDS = 30


def _skill_md_path(name: str, expert_id: str | None = None) -> str:
    return f"{skill_folder(expert_id)}/{name}/SKILL.md"


def _load_default_body(skill: _DefaultSkill) -> str:
    """Read a default skill's body from disk (cached at module level
    via :func:`functools.lru_cache` would re-read on test reloads, so
    we hit the disk each call — these files are small)."""
    return skill.body_path.read_text(encoding="utf-8")


def get_default_skill_with_body(name: str) -> ParsedSkill | None:
    """Return a built-in default skill — name, description, triggers,
    and body — or ``None`` if *name* is not a registered default.

    Public counterpart to :func:`read_user_skill_with_body` so the REST
    layer can resolve a skill slug (default or user-distilled) without
    reaching into private helpers.  Raises ``OSError`` only when the
    body file is unreadable, which the REST layer translates into a
    500 response with a sanitised detail.
    """
    default = _DEFAULT_SKILLS_BY_NAME.get(name)
    if default is None:
        return None
    body = _load_default_body(default)
    return ParsedSkill(
        name=default.name,
        description=default.description,
        body=body,
        triggers=default.triggers,
        origin=SKILL_ORIGIN_PLATFORM,
    )


# ---------------------------------------------------------------------------
# The package boundary — a skill as its whole directory.  Everything that
# writes a package (``store_user_skill``, the copy to an expert, and the
# upload/install paths above this layer) goes through these types, so the
# caps and the path rules are stated once.
# ---------------------------------------------------------------------------

_PACKAGE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")
_ROOT_SKILL_MD = "SKILL.md"


class SkillPackageError(ValueError):
    """A package that breaks a cap or a path rule.

    ``over_limit`` separates a size or count refusal — 413 at the REST edge —
    from a malformed one, which is 400.  A ``ValueError`` subclass so the
    handlers that already map validation failures keep working.
    """

    def __init__(self, message: str, *, over_limit: bool = False):
        super().__init__(message)
        self.over_limit = over_limit


class SkillFile(BaseModel):
    """One file beside a package's ``SKILL.md``, by its path relative to the
    skill folder (``scripts/with_server.py``, ``references/REFERENCE.md``)."""

    relative_path: str
    # pydantic encodes a ``str`` here as utf-8, so a caller may pass either.
    content: bytes
    is_executable: bool = False

    @property
    def size_bytes(self) -> int:
        """Derived, never stored: a size that could disagree with the content
        would make every cap below lie."""
        return len(self.content)


class SkillPackage(BaseModel):
    """A whole skill: its ``SKILL.md`` text plus every sibling file."""

    skill_md: str
    files: list[SkillFile] = []

    @property
    def size_bytes(self) -> int:
        return len(self.skill_md.encode("utf-8")) + sum(
            f.size_bytes for f in self.files
        )


def validate_package(package: SkillPackage) -> None:
    """Refuse a package that breaks a cap or carries an unsafe path.

    Runs over the whole package so a caller can write nothing on failure;
    every message names the field and the number it broke.
    """
    if len(package.files) > MAX_PACKAGE_FILES:
        raise SkillPackageError(
            f"package has {len(package.files)} files; the limit is "
            f"{MAX_PACKAGE_FILES}",
            over_limit=True,
        )
    seen: set[str] = set()
    for entry in package.files:
        path_error = _package_path_error(entry.relative_path)
        if path_error:
            raise SkillPackageError(path_error)
        if entry.relative_path in seen:
            raise SkillPackageError(
                f"file '{entry.relative_path}' appears twice in the package"
            )
        seen.add(entry.relative_path)
        if entry.size_bytes > MAX_PACKAGE_FILE_BYTES:
            raise SkillPackageError(
                f"file '{entry.relative_path}' is {entry.size_bytes} bytes; "
                f"the limit is {MAX_PACKAGE_FILE_BYTES}",
                over_limit=True,
            )
    if package.size_bytes > MAX_PACKAGE_BYTES:
        raise SkillPackageError(
            f"package is {package.size_bytes} bytes; the limit is "
            f"{MAX_PACKAGE_BYTES}",
            over_limit=True,
        )


def _package_path_error(path: str) -> str | None:
    """Why *path* may not be written into a skill folder, or ``None``.

    Rejects what a zip member or an API caller can smuggle in: an escape out
    of the folder, a hidden file, a backslash or NUL a later consumer would
    read as a separator or a terminator.
    """
    named = f"file path '{path[:120]}'"
    if not path or path != path.strip():
        return f"{named} is empty or padded with whitespace"
    if path.startswith("/"):
        return f"{named} must be relative to the skill folder"
    if "\\" in path or "\x00" in path:
        return f"{named} may not contain a backslash or a NUL byte"
    if posixpath.normpath(path) != path:
        return f"{named} is not normalised (no '.', '..' or repeated '/')"
    if path == _ROOT_SKILL_MD:
        return f"{named} is the package root; pass it as skill_md, not a file"
    segments = path.split("/")
    if len(segments) > MAX_PACKAGE_PATH_DEPTH:
        return (
            f"{named} is {len(segments)} segments deep; the limit is "
            f"{MAX_PACKAGE_PATH_DEPTH}"
        )
    for segment in segments:
        if not _PACKAGE_SEGMENT_RE.match(segment):
            return (
                f"{named} has an unusable segment '{segment[:40]}' — use "
                "letters, digits, '.', '_' or '-', not starting with '.'"
            )
    return None


class SkillNotFoundError(Exception):
    """Raised by :func:`delete_user_skill` when the skill is missing."""


class BuiltInSkillError(Exception):
    """Raised by :func:`delete_user_skill` for default seeded skills."""


async def delete_user_skill(
    user_id: str,
    name: str,
    *,
    expert_id: str | None = None,
    scope: WorkspaceScope | None = None,
) -> str:
    """Delete a user-distilled skill folder by slug from *expert_id*'s folder
    (personal Otto's when ``None``).

    The whole tree goes, however many files it holds — a package written
    before the cap existed still has to be removable.  Returns the normalised
    slug on success so callers can echo it back.
    Raises :class:`BuiltInSkillError` for default skills,
    :class:`SkillNotFoundError` if the skill does not exist, and
    ``ValueError`` if ``name`` is blank.  Sibling-file cleanup is
    best-effort — a transient delete failure on a non-SKILL.md file logs
    but does not abort the overall delete (the SKILL.md removal is what
    makes the skill disappear from ``<available_skills>``).
    """
    slug = name.strip().lower()
    if not slug:
        raise ValueError("name is required")
    if slug in _DEFAULT_SKILLS_BY_NAME:
        raise BuiltInSkillError(f"'{slug}' is a built-in skill and cannot be deleted")

    manager = await _get_user_skill_manager(user_id, scope)
    info = await manager.get_file_info_by_path(_skill_md_path(slug, expert_id))
    if info is None:
        if expert_id is not None:
            # A stale name from an earlier failed delete: drop it so the row
            # never lists a skill the expert cannot read.
            await experts_db().remove_expert_skill_name(user_id, expert_id, slug)
        raise SkillNotFoundError(f"Skill '{slug}' not found")

    # Audit-log the delete BEFORE the workspace mutation runs so the
    # log row is present even if the delete itself raises mid-cleanup.
    # Decode best-effort: a malformed body is fine for the audit trail —
    # the slug + truncated description is what an operator needs to
    # correlate later support requests.
    description_for_log = ""
    try:
        raw = await manager.read_file(_skill_md_path(slug, expert_id))
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = ""
        if text:
            parsed = parse_skill_markdown(text, fallback_name=slug)
            if parsed is not None:
                description_for_log = parsed.description
    except Exception:
        # Audit-log enrichment is best-effort; failure must not block the
        # delete the user requested.
        pass
    logger.info(
        "[skills] user %s… deleting skill %s: %s",
        user_id[:8],
        slug,
        description_for_log[:60],
    )

    await manager.delete_file(info.id)
    # One page is capped at MAX_PACKAGE_FILES, so a larger folder needs more
    # than one pass; anything left behind keeps consuming the user's quota and
    # is inherited by the next skill stored under this slug.  A pass that
    # deletes nothing new ends the loop, so a file that cannot be deleted stops
    # it rather than spinning it.
    attempted: set[str] = set()
    while True:
        try:
            siblings = await _list_package_files(manager, skill_folder(expert_id), slug)
        except Exception:
            break
        fresh = [s for s in siblings if s.file_id not in attempted]
        if not fresh:
            break
        for sibling in fresh:
            attempted.add(sibling.file_id)
            try:
                await manager.delete_file(sibling.file_id)
            except Exception:
                logger.warning(
                    "[skills] failed to delete sibling %s",
                    sibling.path,
                    exc_info=True,
                )
    await invalidate_skills_index_cache(user_id, expert_id)
    if expert_id is not None:
        await experts_db().remove_expert_skill_name(user_id, expert_id, slug)
    return slug


async def store_user_skill(
    user_id: str,
    *,
    name: str,
    description: str,
    body: str,
    triggers: list[str] | None = None,
    version: str | None = None,
    extra: Mapping[str, Any] | None = None,
    files: list[SkillFile] | None = None,
    expert_id: str | None = None,
    scope: WorkspaceScope | None = None,
    origin: str = SKILL_ORIGIN_USER,
) -> ParsedSkill:
    """Validate + persist a user-distilled skill, returning the stored skill.

    The skill lands in *expert_id*'s folder (personal Otto's when
    ``None``) and becomes that owner's skill. Shared by the ``store_skill``
    copilot tool and the REST ``POST /skills`` upload endpoint so both honour
    the same validation, per-owner cap, and write-lock semantics.  Raises :class:`ValueError` for any validation
    failure, :class:`SkillLimitError` when the owner's cap for skills of
    *origin* is reached, and propagates ``VirusDetectedError`` /
    ``VirusScanError`` (and any other workspace write error) to the caller.

    *origin* says who put the skill there — the owner
    (``SKILL_ORIGIN_USER``, the default) or the platform
    (``SKILL_ORIGIN_MARKETPLACE``: a hire's bundle, a marketplace install).
    Each origin has a cap of its own, so a template's bundle never takes a
    slot from the skills the owner saves to that expert.

    *files* is the whole package: it replaces the folder's contents, so a
    file the caller leaves out is deleted.  ``None`` — every single-file
    caller — leaves the existing siblings alone, which is what keeps the
    model's own ``store_skill`` from wiping a package it only rewrote the
    body of.
    """
    name = name.strip().lower()
    # Strip any server-injected XML tags (``<available_skills>``,
    # ``<env_context>``, etc.) from the persisted fields *before* storage —
    # when the skill is later loaded that text lands in conversation history
    # and could otherwise appear alongside the real server-injected versions.
    description = strip_server_injected_tags(description.strip())
    body = strip_server_injected_tags(body.strip())
    triggers = [
        strip_server_injected_tags(t.strip())
        for t in (triggers or [])
        if str(t).strip()
    ]
    triggers = [t for t in triggers if t]

    name_err = _validate_name(name)
    if name_err:
        raise ValueError(name_err)
    if origin not in _SKILL_ORIGINS:
        raise ValueError(f"origin must be one of {', '.join(sorted(_SKILL_ORIGINS))}")
    validate_skill_content(description, body, triggers)

    parsed = ParsedSkill(
        name=name,
        description=description,
        body=body,
        triggers=tuple(triggers),
        version=version,
        extra=dict(extra or {}),
        origin=origin,
    )
    rendered = render_skill_markdown(parsed)
    if files is not None:
        # Whole-package validation before the first write, so a package that
        # breaks a cap leaves the stored skill exactly as it was.
        validate_package(SkillPackage(skill_md=rendered, files=files))

    # Coordinate ordinary package writes; the database owns the hard cap.
    lock: AsyncClusterLock | None = None
    lock_held = False
    try:
        lock = AsyncClusterLock(
            redis=await get_redis_async(),
            key=f"{_SKILL_WRITE_LOCK_KEY_PREFIX}{user_id}:{expert_id or 'autopilot'}",
            owner_id=uuid.uuid4().hex,
            timeout=_SKILL_WRITE_LOCK_TTL_SECONDS,
        )
        for _ in range(10):
            if (await lock.try_acquire()) == lock.owner_id:
                lock_held = True
                break
            await asyncio.sleep(0.1)
    except Exception:
        logger.warning(
            "[skills] failed to acquire write lock for user %s — "
            "falling back to unlocked best-effort write",
            user_id,
            exc_info=True,
        )
    try:
        manager = await _get_user_skill_manager(user_id, scope)
        existing = await _list_user_skills_from_workspace(user_id, expert_id, scope)
        same_origin = {s.name for s in existing if budget_origin(s) == origin}
        if origin == SKILL_ORIGIN_MARKETPLACE and any(
            s.name == name and s.origin == SKILL_ORIGIN_USER for s in existing
        ):
            raise SkillOwnedError(
                f"'{name}' is one of the owner's own skills; rename or delete "
                "it before installing a skill by that name."
            )
        at_cap = len(same_origin) >= MAX_SKILLS_PER_EXPERT
        is_new = name not in same_origin
        if at_cap and is_new:
            raise SkillLimitError(
                f"Skill limit reached ({MAX_SKILLS_PER_EXPERT} {_ORIGIN_LABELS[origin]} "
                "skills). Delete an unused skill first."
            )

        metadata: dict[str, Any] = {
            _META_KIND: _META_KIND_VALUE,
            _META_DESCRIPTION: description,
            _META_TRIGGERS: list(triggers),
            _META_SKILL_ORIGIN: origin,
        }
        if version:
            metadata[_META_VERSION] = version
        folder = skill_folder(expert_id)
        stale = (
            await _list_package_files(manager, folder, name, cap=None)
            if files is not None
            else []
        )
        # The root is what indexes the skill, so it goes last: a new skill
        # that fails part-way is never indexed.  An upsert cannot be made
        # atomic here — the old bytes are gone once overwritten.  Serial
        # because ``write_file``'s quota check is read-then-write.
        existing_paths = {f.path for f in stale}
        written: set[str] = set()
        try:
            for entry in files or []:
                path = f"{folder}/{name}/{entry.relative_path}"
                written.add(path)
                await manager.write_file(
                    content=entry.content,
                    filename=entry.relative_path.rsplit("/", 1)[-1],
                    path=path,
                    mime_type=None,
                    overwrite=True,
                    metadata=(
                        {_META_EXECUTABLE: True} if entry.is_executable else None
                    ),
                )
            await manager.write_file(
                content=rendered.encode("utf-8"),
                filename="SKILL.md",
                path=_skill_md_path(name, expert_id),
                mime_type="text/markdown",
                overwrite=True,
                metadata=metadata,
            )
        except Exception:
            # Not a rollback: a file already here keeps the new bytes, so an
            # upsert can fail mixed. Undo only what this call created — deleting
            # the rest would turn a failed write into a lost file.
            await _delete_paths(manager, written - existing_paths)
            raise
        await _delete_paths(
            manager, {f.path for f in stale if f.path not in written}, stale
        )
        await invalidate_skills_index_cache(user_id, expert_id)
        if expert_id is not None:
            await experts_db().add_expert_skill_name(user_id, expert_id, name)
        return parsed
    finally:
        if lock is not None and lock_held:
            try:
                await lock.release()
            except Exception:
                logger.warning(
                    "[skills] failed to release write lock for user %s",
                    user_id,
                    exc_info=True,
                )


async def _delete_paths(
    manager: WorkspaceManager,
    paths: set[str],
    known: list[SkillFileInfo] | None = None,
) -> None:
    """Delete workspace files by path, best-effort: a file that will not go
    is logged, never raised, because every caller here has already done the
    thing the user asked for."""
    ids = {f.path: f.file_id for f in known or []}
    for path in paths:
        try:
            file_id = ids.get(path)
            if file_id is None:
                info = await manager.get_file_info_by_path(path)
                if info is None:
                    continue
                file_id = info.id
            await manager.delete_file(file_id)
        except Exception:
            logger.warning("[skills] failed to delete %s", path, exc_info=True)


async def _parse_skill_from_workspace(
    manager: WorkspaceManager, file_path: str
) -> ParsedSkill | None:
    """Read + parse a single SKILL.md.  Returns ``None`` on read failure,
    decode failure, or malformed frontmatter — callers treat that as
    "skip this file", matching the old serial loop's contract.
    """
    slug = file_path.rsplit("/", 2)[-2] if "/" in file_path else ""
    try:
        raw = await manager.read_file(file_path)
    except Exception:
        logger.warning("[skills] failed to read %s", file_path, exc_info=True)
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("[skills] non-UTF-8 contents at %s — skipping", file_path)
        return None
    return parse_skill_markdown(text, fallback_name=slug)


def _index_entry_from_metadata(slug: str, meta: dict) -> ParsedSkill | None:
    """Build a body-less :class:`ParsedSkill` from ``WorkspaceFile.metadata``
    written at store time.  Returns ``None`` when the metadata is missing
    a description (older SKILL.md predating the metadata-cache change) so
    the caller falls back to reading the file.
    """
    if meta.get(_META_KIND) != _META_KIND_VALUE:
        return None
    description = meta.get(_META_DESCRIPTION)
    if not isinstance(description, str) or not description:
        return None
    raw_triggers = meta.get(_META_TRIGGERS) or ()
    if not isinstance(raw_triggers, (list, tuple)):
        raw_triggers = ()
    triggers = tuple(str(t) for t in raw_triggers if isinstance(t, str) and t)
    version = meta.get(_META_VERSION)
    return ParsedSkill(
        name=slug,
        description=description,
        body="",
        triggers=triggers,
        version=str(version) if version else None,
        origin=_skill_origin(meta),
    )


def _skill_origin(meta: Mapping[str, Any]) -> str | None:
    """The origin recorded on a row, or ``None`` when none was."""
    return _normalize_origin(meta.get(_META_SKILL_ORIGIN))


async def _list_user_skills_from_workspace(
    user_id: str,
    expert_id: str | None = None,
    scope: WorkspaceScope | None = None,
) -> list[ParsedSkill]:
    """Workspace-side listing — no caching.  Body fields are always empty
    because the index never needs them; :func:`read_user_skill_with_body`
    is the path for retrieving full content.

    Uses ``WorkspaceFile.metadata`` (written at store time) for the fast
    path and falls back to a parallelised read of the SKILL.md body for
    any file missing the metadata (older skills, or skills written by a
    deployment before the metadata-cache change shipped).

    The SKILL.md filter is applied in the query, not after it: the listing
    is ordered newest-first and capped, so filtering client-side let one
    package's files push older skills out of the index entirely.
    """
    manager = await _get_user_skill_manager(user_id, scope)
    folder = skill_folder(expert_id)

    skills: list[ParsedSkill] = []
    needs_read: list[tuple[Any, dict[str, Any]]] = []
    for f, slug in await _list_skill_roots(manager, folder):
        meta = f.metadata if isinstance(f.metadata, dict) else {}
        entry = _index_entry_from_metadata(slug, meta)
        if entry is not None:
            skills.append(entry)
        else:
            needs_read.append((f, meta))

    if needs_read:
        parsed = await asyncio.gather(
            *(_parse_skill_from_workspace(manager, f.path) for f, _ in needs_read),
        )
        for (_, meta), p in zip(needs_read, parsed):
            if p is None:
                continue
            # Index never needs the body — drop it so the cache payload
            # stays small (defaults are already body-less, fast-path
            # entries are body-less, keep the contract uniform).  The
            # origin is the row's, whatever the file says.
            skills.append(
                ParsedSkill(
                    name=p.name,
                    description=p.description,
                    body="",
                    triggers=p.triggers,
                    version=p.version,
                    origin=_skill_origin(meta),
                )
            )

    skills.sort(key=lambda s: s.name)
    return skills


def _skills_cache_key(user_id: str, expert_id: str | None = None) -> str:
    base = SKILLS_INDEX_CACHE_KEY.format(user_id=user_id)
    return base if expert_id is None else f"{base}:expert:{expert_id}"


async def _read_skills_cache(
    user_id: str, expert_id: str | None = None
) -> list[ParsedSkill] | None:
    try:
        redis = await get_redis_async()
        raw = await redis.get(_skills_cache_key(user_id, expert_id))
    except Exception:
        # Cache is best-effort — a redis blip must not break the turn.
        return None
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        if not isinstance(payload, list):
            return None
        return [
            ParsedSkill(
                name=str(item["name"]),
                description=str(item["description"]),
                body="",
                triggers=tuple(str(t) for t in item.get("triggers", [])),
                version=item.get("version"),
                origin=_normalize_origin(item.get("origin")),
            )
            for item in payload
            if isinstance(item, dict) and "name" in item and "description" in item
        ]
    except Exception:
        # Malformed cache entry → ignore and rebuild.
        return None


async def _write_skills_cache(
    user_id: str, skills: list[ParsedSkill], expert_id: str | None = None
) -> None:
    try:
        redis = await get_redis_async()
        payload = json.dumps(
            [
                {
                    "name": s.name,
                    "description": s.description,
                    "triggers": list(s.triggers),
                    "version": s.version,
                    "origin": s.origin,
                }
                for s in skills
            ]
        )
        await redis.set(
            _skills_cache_key(user_id, expert_id),
            payload,
            ex=SKILLS_INDEX_CACHE_TTL_S,
        )
    except Exception:
        # Cache write is best-effort.
        pass


async def invalidate_skills_index_cache(
    user_id: str, expert_id: str | None = None
) -> None:
    """Drop the cached per-user skill index so the next turn rebuilds it.
    Called by ``store_skill`` / ``delete_user_skill`` so an edit shows up
    immediately rather than after the 60s TTL.

    Also drops the expert's heal backoff: the folder just changed, so a name
    that resolved to nothing before may resolve now.
    """
    keys = [_skills_cache_key(user_id, expert_id)]
    if expert_id is not None:
        keys.append(_heal_backoff_key(user_id, expert_id))
    try:
        redis = await get_redis_async()
        await redis.delete(*keys)
    except Exception:
        # Cache invalidate is best-effort — at worst the user sees stale
        # state for up to ``SKILLS_INDEX_CACHE_TTL_S`` seconds.
        pass


async def list_user_skills(
    user_id: str,
    expert_id: str | None = None,
    scope: WorkspaceScope | None = None,
    *,
    heal_missing: bool = True,
) -> list[ParsedSkill]:
    """Return the skills owned by *expert_id* (personal Otto when ``None``).

    Two-level fast path: a 60s Redis cache covers warm turns, the
    ``WorkspaceFile.metadata`` index covers cold turns without any storage
    reads, and a parallelised body parse covers any skills written before
    the metadata-cache change shipped.  Skips files whose contents do not
    parse as valid SKILL.md so a stray file in ``/skills/`` cannot break
    the index.
    """
    cached = await _read_skills_cache(user_id, expert_id)
    if cached is not None:
        return cached
    skills = await _list_user_skills_from_workspace(user_id, expert_id, scope)
    if (
        heal_missing
        and expert_id is not None
        and await _copy_assigned_skills_not_yet_owned(user_id, expert_id, skills)
    ):
        skills = await _list_user_skills_from_workspace(user_id, expert_id, scope)
    await _write_skills_cache(user_id, skills, expert_id)
    return skills


async def _copy_assigned_skills_not_yet_owned(
    user_id: str, expert_id: str, owned: list[ParsedSkill]
) -> bool:
    """Give the expert a copy of every skill its row lists but its folder
    lacks, and report whether anything landed.

    Assignments made before skills were owned per expert point at Otto's
    library and have no copy anywhere, so without this an existing expert
    silently drops to the built-in defaults. Runs only on a cache miss, and
    only while something is actually missing. A name that resolves to nothing
    is left on the row: a marketplace attachment has no folder to copy by
    design, and a storage blip must not delete an assignment — but it is
    remembered, so the next cold turn skips a scan that cannot succeed.
    """
    expert = await experts_db().get_expert(user_id, expert_id, include_workflows=False)
    if expert is None:
        return False
    have = {skill_name_key(s.name) for s in owned}
    missing = [
        name
        for name in expert.skills or []
        if name.strip()
        and skill_name_key(name) not in have
        and name.strip().lower() not in _DEFAULT_SKILLS_BY_NAME
    ]
    if not missing or await _heal_backoff_covers(user_id, expert_id, missing):
        return False
    folders = await find_user_skill_slugs(user_id, missing)
    copied = False
    unresolved: list[str] = []
    for name in missing:
        folder = folders.get(name.strip().lower())
        if folder is None:
            unresolved.append(name.strip().lower())
            continue
        try:
            if await copy_skill_to_expert(user_id, expert_id, folder):
                copied = True
        except Exception:
            logger.exception(
                "[skills] failed to copy assigned skill %r to expert #%s",
                name,
                expert_id,
            )
    # After the copies, so a copy's own cache invalidation cannot drop it.
    await _set_heal_backoff(user_id, expert_id, unresolved)
    return copied


def skill_name_key(name: str) -> str:
    """Compare key for skill names: the row may carry a display name ("Deep
    Research") for the skill whose folder is ``deep-research``."""
    return re.sub(r"[\s_-]+", "-", name.strip().lower())


def _heal_backoff_key(user_id: str, expert_id: str) -> str:
    return SKILLS_HEAL_BACKOFF_KEY.format(user_id=user_id, expert_id=expert_id)


async def _heal_backoff_covers(
    user_id: str, expert_id: str, missing: list[str]
) -> bool:
    """True when every still-missing name was already proved uncopyable, so
    the scan can be skipped. A name outside the marker retries immediately —
    including one whose copy failed transiently."""
    try:
        redis = await get_redis_async()
        raw = await redis.get(_heal_backoff_key(user_id, expert_id))
        known = set(json.loads(raw)) if raw else set()
    except Exception:
        # Losing the marker costs one repeat scan, never correctness.
        return False
    return all(name.strip().lower() in known for name in missing)


async def _set_heal_backoff(user_id: str, expert_id: str, names: list[str]) -> None:
    """Record the names that resolved to nothing, or clear the marker when
    none did."""
    key = _heal_backoff_key(user_id, expert_id)
    try:
        redis = await get_redis_async()
        if not names:
            await redis.delete(key)
            return
        await redis.set(
            key, json.dumps(sorted(set(names))), ex=SKILLS_HEAL_BACKOFF_TTL_S
        )
    except Exception:
        pass


async def read_user_skill_with_body(
    user_id: str,
    name: str,
    *,
    expert_id: str | None = None,
    scope: WorkspaceScope | None = None,
) -> ParsedSkill | None:
    """Return a single skill owned by *expert_id* (personal Otto when
    ``None``) with its body populated.

    Used by the ``read_skill`` MCP tool and the REST GET ``/skills/{name}``
    endpoint that powers the library UI's expand-to-view dialog.  Returns
    ``None`` when no SKILL.md exists at the slug — callers translate that
    into a 404 / structured error response.
    """
    slug = name.strip().lower()
    if not slug:
        return None
    manager = await _get_user_skill_manager(user_id, scope)
    return await _parse_skill_from_workspace(manager, _skill_md_path(slug, expert_id))


async def read_user_skill_package(
    user_id: str,
    name: str,
    *,
    expert_id: str | None = None,
    scope: WorkspaceScope | None = None,
) -> SkillPackage | None:
    """The whole stored skill — the ``SKILL.md`` exactly as stored plus every
    sibling — or ``None`` when the slug has no ``SKILL.md``.

    What the zip download hands out, so a downloaded package re-uploads to a
    byte-identical tree. :func:`read_user_skill_with_body` is the root alone.

    Only a missing skill answers ``None``: a storage failure or an undecodable
    file raises, because a download that quietly omits part of the tree is worse
    than one that fails.

    The read is bracketed by a fingerprint of the tree and retried when it
    moves, because ``store_user_skill`` writes the siblings before the root: a
    concurrent store caught mid-write would otherwise hand back one version's
    ``SKILL.md`` with another's files. Exhausting the attempts raises rather
    than serving a package that may be mixed.
    """
    slug = name.strip().lower()
    if not slug:
        return None
    manager = await _get_user_skill_manager(user_id, scope)
    folder = skill_folder(expert_id)
    root_path = _skill_md_path(slug, expert_id)
    for _ in range(_PACKAGE_READ_ATTEMPTS):
        before = await _package_fingerprint(manager, folder, slug, root_path)
        if before is None:
            return None
        try:
            raw = await manager.read_file(root_path)
        except FileNotFoundError:
            return None
        # Only a moved fingerprint retries; any other failure in here is this
        # caller's answer, not a concurrent write.
        files = await _read_package_files(manager, folder, slug, complete=True)
        if await _package_fingerprint(manager, folder, slug, root_path) == before:
            return SkillPackage(skill_md=raw.decode("utf-8"), files=files)
    raise ConflictError(
        f"Skill '{slug}' was being changed while it was read. Try again."
    )


async def _package_fingerprint(
    manager: WorkspaceManager, folder: str, slug: str, root_path: str
) -> tuple[str, tuple[tuple[str, str], ...]] | None:
    """Row ids of a package's ``SKILL.md`` and every sibling, ``None`` when the
    skill is not there.

    Sound only because ``write_file`` mints a fresh id per write and overwrites
    by deleting and recreating rather than updating in place, so no write to
    this tree can leave the ids untouched.
    """
    root = await manager.get_file_info_by_path(root_path)
    if root is None:
        return None
    files = await _list_package_files(manager, folder, slug, cap=None)
    return root.id, tuple(sorted((f.path, f.file_id) for f in files))


async def list_user_skill_files(
    user_id: str,
    name: str,
    *,
    expert_id: str | None = None,
    scope: WorkspaceScope | None = None,
) -> list[SkillFileInfo]:
    """Every file beside a stored skill's ``SKILL.md`` — ``references/``,
    ``scripts/``, ``assets/``, or anything the model stashed there — with the
    size and executable bit the library UI's file tree shows.

    Returns ``[]`` on any error: listing decorates the read it accompanies and
    must not fail it.
    """
    slug = name.strip().lower()
    if not slug:
        return []
    try:
        manager = await _get_user_skill_manager(user_id, scope)
        return await _list_package_files(manager, skill_folder(expert_id), slug)
    except Exception:
        logger.warning(
            "[skills] failed to list package files for %s", slug, exc_info=True
        )
        return []


async def find_user_skill_slug(user_id: str, name: str) -> str | None:
    """Folder slug of personal Otto's skill called *name*."""
    return (await find_user_skill_slugs(user_id, [name])).get(name.strip().lower())


async def find_user_skill_slugs(user_id: str, names: list[str]) -> dict[str, str]:
    """Folder slug per requested name, keyed by the lowercased name.

    Matches the folder first, then the frontmatter name — a skill written by
    hand may be listed under a name that differs from its folder. One listing
    covers the whole batch, and a folder carrying store-time metadata is
    matched without reading it, so only hand-written skills cost a fetch.
    """
    wanted = {n.strip().lower() for n in names if n.strip()}
    if not wanted:
        return {}
    manager = await _get_user_skill_manager(user_id)
    found: dict[str, str] = {}
    unnamed: list[Any] = []
    for f, slug in await _list_skill_roots(manager, SKILL_FOLDER):
        if slug.strip().lower() in wanted:
            found[slug.strip().lower()] = slug
            continue
        meta = f.metadata if isinstance(f.metadata, dict) else {}
        if meta.get(_META_KIND) != _META_KIND_VALUE:
            unnamed.append((f, slug))
    missing = wanted - set(found)
    if not missing or not unnamed:
        return found
    parsed = await asyncio.gather(
        *(_parse_skill_from_workspace(manager, f.path) for f, _ in unnamed)
    )
    for (_, slug), entry in zip(unnamed, parsed):
        if entry is None:
            continue
        name = entry.name.strip().lower()
        if name in missing:
            found.setdefault(name, slug)
    return found


async def copy_skill_to_expert(user_id: str, expert_id: str, name: str) -> str | None:
    """Give *expert_id* its own copy of one of personal Otto's skills.

    Returns the stored slug, or ``None`` when Otto has no such skill, and
    raises when the source package cannot be read whole — a partial copy would
    be permanent, since the next call sees the root and returns early.
    Idempotent: an expert that already owns the slug keeps its copy. The
    whole package goes through :func:`store_user_skill`, so the copy is
    validated, capped and written siblings-first exactly like any other.
    """
    slug = name.strip().lower()
    if not slug:
        return None
    manager = await _get_user_skill_manager(user_id)
    if await manager.get_file_info_by_path(_skill_md_path(slug, expert_id)):
        # Reconcile a copy whose row update failed earlier; idempotent.
        await experts_db().add_expert_skill_name(user_id, expert_id, slug)
        return slug
    source = await read_user_skill_with_body(user_id, slug)
    if source is None:
        return None
    # The origin lives on the row, not in the file the parse above read; a
    # bundled skill copied into an expert stays a bundled one there.
    root = await manager.get_file_info_by_path(_skill_md_path(slug))
    meta = root.metadata if root is not None and isinstance(root.metadata, dict) else {}
    stored = await store_user_skill(
        user_id,
        name=slug,
        description=source.description,
        body=source.body,
        triggers=list(source.triggers),
        version=source.version,
        extra=source.extra,
        files=await _read_package_files(manager, SKILL_FOLDER, slug),
        expert_id=expert_id,
        origin=_skill_origin(meta) or SKILL_ORIGIN_USER,
    )
    return stored.name


async def _read_package_files(
    manager: WorkspaceManager, folder: str, slug: str, *, complete: bool = False
) -> list[SkillFile]:
    """Load a stored package's siblings into memory, ready to be written
    somewhere else.  Reads run concurrently — each is a blob fetch, and a
    60-file package read one at a time is a hire the user waits through.

    A file that cannot be read raises, because the caller's copy is idempotent
    on the root alone: a package written without it would never be repaired.
    ``complete`` additionally refuses a tree that is over the files cap, which
    a download owes its caller; the copy path truncates instead, so a legacy
    oversized folder can still be hired rather than blocking the hire outright.
    """
    prefix = f"{folder}/{slug}/"
    infos = await _list_package_files(manager, folder, slug)
    if len(infos) > MAX_PACKAGE_FILES:
        # Written before the cap existed, or by hand.
        # Keep both branches: raising repairs a failed read, but not a documented
        # cap — so the download refuses and the copy truncates.
        if complete:
            raise SkillPackageError(
                f"stored package has more than {MAX_PACKAGE_FILES} files and "
                "cannot be served whole",
                over_limit=True,
            )
        # Validating it whole would fail and take the hire down, so truncate
        # as read_skill does.
        logger.warning(
            "[skills] package %s has more than %s files; copying the first %s",
            slug,
            MAX_PACKAGE_FILES,
            MAX_PACKAGE_FILES,
        )
        infos = infos[:MAX_PACKAGE_FILES]
    limit = asyncio.Semaphore(_COPY_CONCURRENCY)

    async def load(info: SkillFileInfo) -> SkillFile:
        async with limit:
            content = await manager.read_file(info.path)
        return SkillFile(
            relative_path=info.path[len(prefix) :],
            content=content,
            is_executable=info.is_executable,
        )

    # ``return_exceptions`` so one failure does not leave its siblings' reads
    # unawaited; the first is re-raised once they have all settled.
    loaded = await asyncio.gather(
        *(load(info) for info in infos), return_exceptions=True
    )
    failure = next((r for r in loaded if isinstance(r, BaseException)), None)
    if failure is not None:
        raise failure
    return [f for f in loaded if isinstance(f, SkillFile)]


def get_default_skills_for_index() -> list[ParsedSkill]:
    """Return the default seeded skills with **empty bodies** — suitable
    for rendering the per-turn ``<available_skills>`` index, which only
    uses ``name`` / ``description`` / ``triggers``.  Skips the per-turn
    disk read of the (~20KB) default-skill bodies.
    """
    return [
        ParsedSkill(
            name=default.name,
            description=default.description,
            body="",
            triggers=default.triggers,
            origin=SKILL_ORIGIN_PLATFORM,
        )
        for default in DEFAULT_SKILLS
    ]


def get_default_skills() -> list[ParsedSkill]:
    """Return the default seeded skills with bodies populated from disk.

    Used by callers that actually need the body text (e.g. :tool:`read_skill`
    on a default skill).  Index-only callers should use
    :func:`get_default_skills_for_index` instead so the per-turn skill
    index build does not incur a body-sized disk read for every default.
    """
    result: list[ParsedSkill] = []
    for default in DEFAULT_SKILLS:
        try:
            body = _load_default_body(default)
        except OSError:
            logger.warning(
                "[skills] default body missing for %s at %s",
                default.name,
                default.body_path,
            )
            continue
        result.append(
            ParsedSkill(
                name=default.name,
                description=default.description,
                body=body,
                triggers=default.triggers,
                origin=SKILL_ORIGIN_PLATFORM,
            )
        )
    return result


async def list_all_skills(
    user_id: str | None,
    expert_id: str | None = None,
    scope: WorkspaceScope | None = None,
) -> list[ParsedSkill]:
    """Default seeded skills first, then the owner's own skills.

    Defaults always lead so the model sees the built-in agent-building
    guide before any user customisation.  Index-only — default bodies
    are NOT loaded; :tool:`read_skill` re-reads them on demand via
    :func:`get_default_skills`.

    *expert_id* selects whose skills follow the defaults: an expert's own
    folder, or personal Otto's when ``None``. Neither ever sees the
    other's skills.
    """
    skills = get_default_skills_for_index()
    if user_id:
        skills.extend(await list_user_skills(user_id, expert_id, scope))
    return skills


def render_skills_index(skills: list[ParsedSkill]) -> str:
    """Render skills as a compact one-line-each index for the
    ``<available_skills>`` injection block.

    Format: ``- name: <slug> — <description> — triggers: t1, t2``
    The ``name:`` prefix anchors the slug visually so the model picks
    the right slug to pass into ``read_skill(name=...)``; the ``triggers:``
    suffix surfaces the matchable hints inline (no parenthetical) so a
    plain substring scan of the directive line covers both the slug and
    its triggers.
    """
    if not skills:
        return ""
    lines = []
    for s in skills:
        trigger_hint = f" — triggers: {', '.join(s.triggers)}" if s.triggers else ""
        lines.append(f"- name: {s.name} — {s.description}{trigger_hint}")
    return "\n".join(lines)


async def is_skills_feature_enabled(user_id: str | None) -> bool:
    """Per-user kill-switch for the skills feature (``COPILOT_SKILLS``
    LaunchDarkly flag).  Default-on; the flag exists only so we can
    disable the feature without a redeploy.  Anonymous calls
    (no ``user_id``) treat as enabled so unauthenticated paths don't
    surprise-break.
    """
    if not user_id:
        return True
    return await is_feature_enabled(Flag.COPILOT_SKILLS, user_id, default=True)


async def build_skills_context(
    user_id: str | None, expert_id: str | None = None
) -> str:
    """Build the body of the ``<available_skills>`` block injected into
    the first user message.  Returns ``""`` if there are no skills to
    show — :func:`inject_user_context` then omits the block entirely so
    sessions with no skill state don't pay the tag overhead.

    Also returns ``""`` when the ``COPILOT_SKILLS`` LD flag is off for
    this user, so the kill-switch fully suppresses the per-turn index
    cost (no list query, no Redis hit, nothing to cache).

    ``expert_id`` is the session's persisted expert attribution; the index
    then holds platform defaults plus that expert's own skills.
    """
    if not await is_skills_feature_enabled(user_id):
        return ""
    skills = await list_all_skills(user_id, expert_id)
    index = render_skills_index(skills)
    if not index:
        return ""
    return (
        "Skills are reusable procedures loaded with "
        '`run_capability(id="tool:read_skill", input={"name": ...})`. '
        "Match the user's request to a skill's triggers (substring or "
        "close paraphrase) and load the "
        "full body before acting; distill a new one with `tool:store_skill` "
        "after you complete a non-trivial procedure worth reusing.\n"
        f"{index}"
    )


# Non-greedy: history holds at most one ``<available_skills>`` block (the
# first-turn injection), but a greedy match across two blocks would swallow
# the user text between them.
_SKILLS_BLOCK_RE = re.compile(r"<available_skills>(.*?)</available_skills>", re.DOTALL)
# One index line per skill: ``- name: <slug> — <description> …``.
_SKILLS_INDEX_LINE_RE = re.compile(r"^- name:\s*(\S+)", re.MULTILINE)

# How many added/removed slugs to name inline before falling back to a
# count — the notice is a nudge to call ``list_skills``, not the index.
_MAX_UPDATE_NAMES = 10


def previously_seen_skill_slugs(contents: Iterable[str]) -> set[str]:
    """Slugs from every ``<available_skills>`` block in *contents*.

    Pure parser over already-persisted session text — what the model saw at
    session start. ``Iterable`` (not ``ChatMessage``) so callers pass plain
    message contents without importing the chat model here.
    """
    seen: set[str] = set()
    for content in contents:
        if not content:
            continue
        for block in _SKILLS_BLOCK_RE.findall(content):
            seen.update(_SKILLS_INDEX_LINE_RE.findall(block))
    return seen


async def build_skills_update_notice(
    user_id: str | None,
    expert_id: str | None = None,
    prior_contents: Iterable[str] = (),
) -> str:
    """Per-turn ``<skills_update>`` notice, or ``""`` when nothing drifted.

    Compares the registry now (``list_all_skills``: defaults plus the
    session owner's own skills) against the ``<available_skills>`` index
    baked into the session history at session start. Same set → ``""`` so
    steady-state turns pay nothing. Any add or removal renders a small
    notice naming the delta and pointing at ``list_skills`` — query-only
    context the engines prepend to the current turn's model input without
    persisting, mirroring the builder-context pattern.

    Never raises: a registry or flag lookup failure degrades to ``""`` so
    a skills hiccup can't block the turn.
    """
    if not user_id:
        return ""
    try:
        if not await is_skills_feature_enabled(user_id):
            return ""
        current = await list_all_skills(user_id, expert_id)
    except Exception:
        logger.exception("[skills] failed to diff skills for update notice")
        return ""
    current_slugs = {s.name for s in current}
    seen = previously_seen_skill_slugs(prior_contents)
    added = sorted(slug for slug in current_slugs if slug not in seen)
    removed = sorted(slug for slug in seen if slug not in current_slugs)
    if not added and not removed:
        return ""

    def _names(slugs: list[str]) -> str:
        if len(slugs) > _MAX_UPDATE_NAMES:
            head = ", ".join(slugs[:_MAX_UPDATE_NAMES])
            return f"{head}, and {len(slugs) - _MAX_UPDATE_NAMES} more"
        return ", ".join(slugs)

    lines = [
        "Your available skills changed since this conversation started, "
        "so the <available_skills> index in the first message is stale."
    ]
    if added:
        lines.append(f"New skills: {_names(added)}.")
    if removed:
        lines.append(f"Removed skills: {_names(removed)}.")
    lines.append(
        "Call `tool:list_skills` to see the current list, then "
        "`tool:read_skill` to load a new skill's body before using it."
    )
    return (
        f"<{SKILLS_UPDATE_TAG}>\n" + "\n".join(lines) + f"\n</{SKILLS_UPDATE_TAG}>\n\n"
    )


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


_EXPERT_ID_PARAM = {
    "type": "string",
    "description": "Manage this expert's skills (Otto only).",
}


class StoreSkillResponse(ToolResponseBase):
    type: ResponseType = ResponseType.SKILL_STORED
    name: str
    description: str
    triggers: list[str] = []
    expert_id: str | None = None


class SkillFileInfo(BaseModel):
    """One file of a skill package, by its workspace path."""

    path: str
    file_id: str
    size_bytes: int = 0
    is_executable: bool = False
    # The workspace hashes every write with the same sha256 the manifest
    # records, so a match means the copy is current without reading the blob.
    checksum: str | None = None


class ReadSkillResponse(ToolResponseBase):
    type: ResponseType = ResponseType.SKILL_LOADED
    name: str
    description: str
    body: str
    triggers: list[str] = []
    sibling_files: list[str] = []
    files: list[SkillFileInfo] = []
    # Where the package was copied; ``None`` for a single-file skill or when
    # the copy failed, in which case ``files`` still names the workspace paths.
    package_dir: str | None = None
    is_default: bool = False
    expert_id: str | None = None


class DeleteSkillResponse(ToolResponseBase):
    type: ResponseType = ResponseType.SKILL_DELETED
    name: str
    expert_id: str | None = None


class ListSkillsResponse(ToolResponseBase):
    type: ResponseType = ResponseType.SKILL_LIST
    skills: list[dict[str, Any]]
    expert_id: str | None = None


class StoreSkillTool(BaseTool):
    """Persist a self-distilled procedure as a reusable skill.

    Call this after completing a non-trivial multi-step task that is
    likely to recur (e.g. a stable integration pattern, a debugging
    recipe).  Skills follow the canonical SKILL.md frontmatter format
    (name, description, optional triggers) plus a markdown body.
    """

    @property
    def name(self) -> str:
        return "store_skill"

    @property
    def description(self) -> str:
        return (
            "Save a reusable procedure as a skill. Surfaces in "
            "<available_skills> next turn; loads via tool:read_skill."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Slug id (a-z, 0-9, _, -).",
                },
                "description": {
                    "type": "string",
                    "description": f"One-line hook (≤{MAX_DESCRIPTION_CHARS} chars).",
                },
                "body": {
                    "type": "string",
                    "description": (
                        "Markdown body. Sections: Why / Trigger / Steps / Notes. "
                        "Must be non-empty after trim — empty bodies are rejected."
                    ),
                },
                "triggers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tool/keyword triggers.",
                },
                "expert_id": _EXPERT_ID_PARAM,
            },
            "required": ["name", "description", "body"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        name: str = "",
        description: str = "",
        body: str = "",
        triggers: list[str] | None = None,
        expert_id: str | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )
        owner = await resolve_skill_owner(user_id, session, expert_id)
        if isinstance(owner, str):
            return ErrorResponse(
                message=owner, error="access_denied", session_id=session_id
            )
        if not await is_skills_feature_enabled(user_id):
            return ErrorResponse(
                message=(
                    "Skill registry is currently disabled for this user. "
                    "Re-enable the ``copilot-skills`` LaunchDarkly flag to "
                    "use ``store_skill`` / ``read_skill`` / ``delete_skill`` "
                    "/ ``list_skills``."
                ),
                error="feature_disabled",
                session_id=session_id,
            )

        try:
            parsed = await store_user_skill(
                user_id,
                name=name,
                description=description,
                body=body,
                triggers=triggers,
                expert_id=owner.expert_id,
                scope=owner.scope,
            )
        except (VirusDetectedError, VirusScanError) as exc:
            logger.warning("[skills] virus scan failed for %s: %s", name, exc)
            return ErrorResponse(
                message="Skill content rejected by virus scan",
                error=str(exc),
                session_id=session_id,
            )
        except (ValueError, SkillLimitError, SkillOwnedError, ConflictError) as exc:
            return ErrorResponse(message=str(exc), session_id=session_id)
        except Exception as exc:
            logger.exception("[skills] failed to store skill %s", name)
            return ErrorResponse(
                message=f"Failed to store skill: {exc}",
                error=str(exc),
                session_id=session_id,
            )

        return StoreSkillResponse(
            name=parsed.name,
            description=parsed.description,
            triggers=list(parsed.triggers),
            expert_id=owner.expert_id,
            message=(
                f"Skill '{parsed.name}' stored for "
                f"{_owner_label(owner.expert_id)}. It will appear in "
                "<available_skills> on the next turn."
            ),
            session_id=session_id,
        )


def _owner_label(expert_id: str | None) -> str:
    return "personal Otto" if expert_id is None else f"expert {expert_id}"


class ReadSkillTool(BaseTool):
    """Load a skill by name: its body, its package listing, and a copy of
    that package in the working directory the model's shell runs in."""

    @property
    def name(self) -> str:
        return "read_skill"

    @property
    def description(self) -> str:
        return (
            "Read a skill's body + sibling-file list by name, and copy any "
            "package files into the working directory. Call when a task "
            "matches an <available_skills> entry."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name."},
                "expert_id": _EXPERT_ID_PARAM,
            },
            "required": ["name"],
        }

    @property
    def requires_auth(self) -> bool:
        # Default skills are usable anonymously (mirrors the legacy
        # get_agent_building_guide tool which set requires_auth=False).
        return False

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        name: str = "",
        expert_id: str | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not await is_skills_feature_enabled(user_id):
            return ErrorResponse(
                message="Skill registry is currently disabled for this user.",
                error="feature_disabled",
                session_id=session_id,
            )
        name = name.strip().lower()
        if not name:
            return ErrorResponse(message="name is required", session_id=session_id)

        default = _DEFAULT_SKILLS_BY_NAME.get(name)
        if default is not None:
            try:
                body = _load_default_body(default)
            except OSError as exc:
                logger.exception("[skills] failed to read default skill %s", name)
                return ErrorResponse(
                    message=f"Failed to load default skill: {exc}",
                    session_id=session_id,
                )
            return ReadSkillResponse(
                name=default.name,
                description=default.description,
                body=body,
                triggers=list(default.triggers),
                sibling_files=[],
                is_default=True,
                message=f"Loaded default skill '{name}'.",
                session_id=session_id,
            )

        if not user_id:
            return ErrorResponse(
                message="Authentication required to read user skills",
                session_id=session_id,
            )

        owner = await resolve_skill_owner(user_id, session, expert_id)
        if isinstance(owner, str):
            return ErrorResponse(
                message=owner, error="access_denied", session_id=session_id
            )

        try:
            manager = await _get_user_skill_manager(user_id, owner.scope)
            raw = await manager.read_file(_skill_md_path(name, owner.expert_id))
        except WorkspaceAccessDeniedError:
            return ErrorResponse(
                message=EXPERT_SKILL_SCOPE_DENIED,
                error="access_denied",
                session_id=session_id,
            )
        except FileNotFoundError:
            return ErrorResponse(
                message=(
                    f"Skill '{name}' not found. "
                    "Check <available_skills> for valid names."
                ),
                session_id=session_id,
            )
        except Exception as exc:
            logger.exception("[skills] failed to read user skill %s", name)
            return ErrorResponse(
                message=f"Failed to read skill: {exc}",
                error=str(exc),
                session_id=session_id,
            )

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return ErrorResponse(
                message=(
                    f"Skill '{name}' is malformed (non-UTF-8 contents). "
                    "Re-create it with tool:store_skill."
                ),
                session_id=session_id,
            )
        parsed = parse_skill_markdown(text, fallback_name=name)
        if parsed is None:
            return ErrorResponse(
                message=(
                    f"Skill '{name}' is malformed (missing/invalid "
                    "frontmatter). Re-create it with tool:store_skill."
                ),
                session_id=session_id,
            )

        # List the package files (references/, scripts/, assets/, ...) so
        # the model knows what else lives in the bundle.
        folder = skill_folder(owner.expert_id)
        listed = True
        try:
            package_files = await _list_package_files(manager, folder, name)
        except Exception:
            logger.warning(
                "[skills] failed to list package files for %s", name, exc_info=True
            )
            package_files = []
            listed = False

        notes: list[str] = []
        # A listing that failed is not an empty package: treating it as one
        # would prune every file the last activation wrote.
        complete = listed and len(package_files) <= MAX_PACKAGE_FILES
        if listed and not complete:
            package_files = package_files[:MAX_PACKAGE_FILES]
            notes.append(
                f"Only the first {MAX_PACKAGE_FILES} package files are listed."
            )

        # Runs even for a skill with no files: a package deleted and re-stored
        # under the same slug leaves its old files in the working directory.
        package_dir, warning = await _sync_skill_package(
            manager,
            package_files,
            folder=folder,
            slug=name,
            session_id=session_id,
            complete=complete,
        )
        if warning:
            notes.append(warning)
        if package_dir:
            notes.append(
                f"Package files are at {package_dir}; relative paths in the "
                "body resolve there. Run scripts with bash_exec from that "
                "directory."
            )

        return ReadSkillResponse(
            name=parsed.name,
            description=parsed.description,
            body=parsed.body,
            triggers=list(parsed.triggers),
            sibling_files=[f.path for f in package_files],
            files=package_files,
            package_dir=package_dir,
            is_default=False,
            expert_id=owner.expert_id,
            message=" ".join([f"Loaded skill '{name}'.", *notes]),
            session_id=session_id,
        )


class DeleteSkillTool(BaseTool):
    """Remove a user-created skill (cannot delete built-in defaults)."""

    @property
    def name(self) -> str:
        return "delete_skill"

    @property
    def description(self) -> str:
        return "Delete a user-created skill (defaults cannot be removed)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name."},
                "expert_id": _EXPERT_ID_PARAM,
            },
            "required": ["name"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        name: str = "",
        expert_id: str | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )
        owner = await resolve_skill_owner(user_id, session, expert_id)
        if isinstance(owner, str):
            return ErrorResponse(
                message=owner, error="access_denied", session_id=session_id
            )
        if not await is_skills_feature_enabled(user_id):
            return ErrorResponse(
                message="Skill registry is currently disabled for this user.",
                error="feature_disabled",
                session_id=session_id,
            )
        try:
            slug = await delete_user_skill(
                user_id, name, expert_id=owner.expert_id, scope=owner.scope
            )
        except WorkspaceAccessDeniedError:
            return ErrorResponse(
                message=EXPERT_SKILL_SCOPE_DENIED,
                error="access_denied",
                session_id=session_id,
            )
        except ValueError as exc:
            return ErrorResponse(message=str(exc), session_id=session_id)
        except BuiltInSkillError as exc:
            return ErrorResponse(message=str(exc), session_id=session_id)
        except SkillNotFoundError as exc:
            return ErrorResponse(message=str(exc), session_id=session_id)
        except ConflictError as exc:
            return ErrorResponse(message=str(exc), session_id=session_id)
        except Exception as exc:
            logger.exception("[skills] delete failed for %s", name)
            return ErrorResponse(
                message=f"Failed to delete skill: {exc}", session_id=session_id
            )

        return DeleteSkillResponse(
            name=slug,
            expert_id=owner.expert_id,
            message=f"Skill '{slug}' deleted for {_owner_label(owner.expert_id)}.",
            session_id=session_id,
        )


class ListSkillsTool(BaseTool):
    """Return the current skill index — the same content the model sees
    auto-injected in ``<available_skills>``.  Useful for the model to
    re-check after a ``store_skill`` / ``delete_skill`` call without
    waiting for the next turn's index refresh.
    """

    @property
    def name(self) -> str:
        return "list_skills"

    @property
    def description(self) -> str:
        return "List the skills available here (platform defaults + your own)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"expert_id": _EXPERT_ID_PARAM},
            "required": [],
        }

    @property
    def requires_auth(self) -> bool:
        return False

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        expert_id: str | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        if not await is_skills_feature_enabled(user_id):
            return ErrorResponse(
                message="Skill registry is currently disabled for this user.",
                error="feature_disabled",
                session_id=session.session_id,
            )
        owner_id: str | None = None
        owner_scope: WorkspaceScope | None = None
        if user_id:
            owner = await resolve_skill_owner(user_id, session, expert_id)
            if isinstance(owner, str):
                return ErrorResponse(
                    message=owner, error="access_denied", session_id=session.session_id
                )
            owner_id = owner.expert_id
            owner_scope = owner.scope
        skills = await list_all_skills(user_id, owner_id, owner_scope)
        payload = [
            {
                "name": s.name,
                "description": s.description,
                "triggers": list(s.triggers),
                "is_default": s.name in _DEFAULT_SKILLS_BY_NAME,
                "origin": budget_origin(s),
            }
            for s in skills
        ]
        return ListSkillsResponse(
            skills=payload,
            expert_id=owner_id,
            message=f"{len(payload)} skill(s) available for {_owner_label(owner_id)}.",
            session_id=session.session_id,
        )


# ---------------------------------------------------------------------------
# Package files — enumerating a skill's folder, and keeping the model's
# working directory in step with it.
# ---------------------------------------------------------------------------


# Rows a listing will scan before giving up. The SKILL.md name filter runs in
# the query but depth cannot, so a page of newest-first rows can be entirely
# nested SKILL.md files and yield no roots at all; the bound is the most a
# compliant folder can hold, every allowed skill carrying a full package.
# Roots one folder can hold: a full budget for every origin.
_MAX_ROOTS_PER_FOLDER = MAX_SKILLS_PER_EXPERT * len(_SKILL_ORIGINS)
_MAX_ROOT_SCAN = _MAX_ROOTS_PER_FOLDER * (MAX_PACKAGE_FILES + 1)


async def _list_skill_roots(
    manager: WorkspaceManager, folder: str
) -> list[tuple[Any, str]]:
    """``(file, slug)`` for every package root directly under *folder*.

    Pages until the roots run out rather than filtering one capped page: a
    package shipping its own example ``SKILL.md`` files would otherwise fill
    the page and hide older skills, which is the defect this listing exists
    to avoid.
    """
    page = _MAX_ROOTS_PER_FOLDER * 4  # over-fetch in case of strays
    roots: list[tuple[Any, str]] = []
    offset = 0
    while offset < _MAX_ROOT_SCAN and len(roots) <= _MAX_ROOTS_PER_FOLDER:
        rows = await manager.list_files(
            path=f"{folder}/",
            limit=page,
            offset=offset,
            include_all_sessions=True,
            name_contains="SKILL.md",
        )
        for row in rows:
            slug = _root_skill_slug(row.path, folder)
            if slug is not None:
                roots.append((row, slug))
        if len(rows) < page:
            break
        offset += page
    return roots


def _root_skill_slug(path: str, folder: str) -> str | None:
    """Folder slug when *path* is a package root, ``<folder>/<slug>/SKILL.md``.

    ``None`` for a SKILL.md nested deeper: a package may ship one as an
    example, and indexing it would invent a skill nobody stored.
    """
    prefix = f"{folder}/"
    suffix = "/SKILL.md"
    if not path.startswith(prefix) or not path.endswith(suffix):
        return None
    slug = path[len(prefix) : -len(suffix)]
    return slug if slug and "/" not in slug else None


async def _list_package_files(
    manager: WorkspaceManager,
    folder: str,
    slug: str,
    *,
    cap: int | None = MAX_PACKAGE_FILES,
) -> list[SkillFileInfo]:
    """Every file in a skill's folder except its own SKILL.md, stopping at
    ``cap`` + 1 so a caller can tell a full package from an oversized one.
    ``cap=None`` drains the folder in one read, which the orphan prune needs:
    a file it cannot see is one it would leave behind.

    Nested paths are kept: a package's ``scripts/`` and ``references/`` are
    what make it more than one file.
    """
    prefix = f"{folder}/{slug}/"
    root = f"{prefix}SKILL.md"
    page = (cap or MAX_PACKAGE_FILES) + 1
    files: list[SkillFileInfo] = []
    offset = 0
    while cap is None or len(files) <= cap:
        rows = await manager.list_files(
            path=prefix, limit=page, offset=offset, include_all_sessions=True
        )
        for row in rows:
            if row.path == root:
                continue
            meta = row.metadata if isinstance(row.metadata, dict) else {}
            files.append(
                SkillFileInfo(
                    path=row.path,
                    file_id=row.id,
                    size_bytes=row.size_bytes or 0,
                    is_executable=bool(meta.get(_META_EXECUTABLE)),
                    checksum=getattr(row, "checksum", None),
                )
            )
            if cap is not None and len(files) > cap:
                break
        if len(rows) < page:
            break
        offset += page
    return files


# A skill's body references its resources by relative path ("run
# scripts/extract.py"), so the files have to exist where the model's shell
# runs before any of that is actionable: the E2B box in production, the
# bubblewrap directory locally.  The manifest records what each file hashed
# to, so re-activating a skill in a later turn copies only what changed.
# Bookkeeping, not part of the package, so it is kept OUT of the directory the
# package is copied into: a skill shipping its own ``.package.json`` would
# otherwise be overwritten by it, and the digest would then match forever.
_MANIFEST_DIR = ".skill-packages"
_MANIFEST_SHA = "sha256"
_MANIFEST_EXEC = "executable"
# A package with no bits to give still gets ``scripts/`` marked, because that
# is where the spec puts its runnables.
_EXECUTABLE_PREFIX = "scripts/"


class _CopiedFile(NamedTuple):
    """One file settled in the working directory. ``target`` is its path there
    when this pass wrote it, and ``None`` when the manifest already matched."""

    relative: str
    digest: str
    executable: bool
    target: str | None


# One E2B ``files.write`` is an HTTP round trip and measured 200 ms, so a
# 60-file package copied serially would cost 12 s of the turn (0.35 s
# concurrent).  Bounded, because each copy also reads a blob.
_COPY_CONCURRENCY = 16


async def _sync_skill_package(
    manager: WorkspaceManager,
    files: list[SkillFileInfo],
    *,
    folder: str,
    slug: str,
    session_id: str,
    complete: bool,
) -> tuple[str | None, str | None]:
    """Make the turn's working directory match the skill's package.

    Returns ``(package_dir, warning)``, the directory being ``None`` when the
    skill carries no files.  A skill's body is worth having without its
    resources, so every failure here is a warning the model reads rather than
    an error that withholds the skill.

    *complete* says whether *files* is the whole package: a truncated listing
    cannot tell a removed file from an unlisted one, so it prunes nothing.
    """
    workdir = workdir_root(session_id)
    package_dir = f"{workdir}/skills/{slug}"
    manifest_path = f"{workdir}/{_MANIFEST_DIR}/{slug}.json"
    prefix = f"{folder}/{slug}/"
    manifest = await _read_package_manifest(manifest_path, session_id)
    limit = asyncio.Semaphore(_COPY_CONCURRENCY)

    async def copy(info: SkillFileInfo) -> _CopiedFile | None:
        """The file's manifest entry once it is in place, or ``None`` when it
        could not be put there."""
        relative = info.path[len(prefix) :] if info.path.startswith(prefix) else ""
        if not _is_safe_relative(relative):
            logger.warning("[skills] skipping odd package path %s", info.path)
            return None
        executable = info.is_executable or relative.startswith(_EXECUTABLE_PREFIX)
        settled = {_MANIFEST_SHA: info.checksum, _MANIFEST_EXEC: executable}
        # The row's checksum is recomputed on every write, so it describes the
        # bytes as stored; matching it means the copy on disk is current and the
        # blob does not have to be fetched to find that out.
        if info.checksum and manifest.get(relative) == settled:
            return _CopiedFile(relative, info.checksum, executable, None)
        async with limit:
            try:
                content = await manager.read_file(info.path)
            except Exception:
                logger.warning("[skills] failed to read %s", info.path, exc_info=True)
                return None
            digest = hashlib.sha256(content).hexdigest()
            # The mode is part of what was materialised: a file whose bit flips
            # without its bytes changing still has to be re-chmodded.
            if manifest.get(relative) == {
                _MANIFEST_SHA: digest,
                _MANIFEST_EXEC: executable,
            }:
                return _CopiedFile(relative, digest, executable, None)
            target = await save_to_workdir(
                f"{package_dir}/{relative}", content, session_id
            )
        if isinstance(target, ErrorResponse):
            logger.warning(
                "[skills] failed to materialise %s: %s", info.path, target.message
            )
            return None
        return _CopiedFile(relative, digest, executable, target)

    copied = [c for c in await asyncio.gather(*(copy(info) for info in files)) if c]
    written = {
        c.relative: {_MANIFEST_SHA: c.digest, _MANIFEST_EXEC: c.executable}
        for c in copied
    }

    # A later activation trusts this instead of re-doing the work, so it records
    # only what happened — committing a failure would make it permanent.
    settled = dict(written)

    # Only files this pass wrote carry a ``target``; a manifest hit needs no
    # chmod, because its recorded mode already matches.
    unset = await set_executable(
        [c.target for c in copied if c.target and c.executable], True, session_id
    ) + await set_executable(
        [c.target for c in copied if c.target and not c.executable], False, session_id
    )
    for c in copied:
        if c.target in unset:
            settled.pop(c.relative, None)
    # A file the skill no longer has must not stay where bash_exec can run it,
    # and delete_skill followed by store_skill on the same slug is exactly that
    # case.  Prune by what the package HOLDS, not by what was copied: a copy
    # that failed leaves a current file whose earlier copy is still wanted.
    removed: set[str] = set()
    if complete:
        # The root is not a sibling and is never written here, so it can only
        # reach the manifest by a hand edit; excluding it keeps the prune from
        # acting on a name that does not belong to it.
        current = {info.path[len(prefix) :] for info in files} | {"SKILL.md"}
        stale = sorted(set(manifest) - current)
        left = set(
            await remove_from_workdir(
                [f"{package_dir}/{relative}" for relative in stale], session_id
            )
        )
        removed = {r for r in stale if f"{package_dir}/{r}" not in left}

    # THE INVARIANT, and why this statement keeps collecting edits: the manifest
    # may only lose an entry for a file we KNOW is gone — one we removed, or one
    # a listing we know was complete did not contain. Every branch that touches
    # it narrows `removed` or `settled` for its own way of not knowing: a delete
    # that failed, a chmod that did not apply, a listing that was truncated.
    # Narrow further if you must; never widen by taking one side of the diff.
    next_manifest = {
        relative: entry
        for relative, entry in manifest.items()
        if relative not in settled and relative not in removed
    }
    next_manifest.update(settled)
    # A single-file skill must not leave an empty package directory behind, so
    # the manifest is written only when there is, or was, something to track.
    if next_manifest or manifest:
        await _write_package_manifest(manifest_path, next_manifest, session_id)

    if not files:
        return None, None
    missing = len(files) - len(written)
    if not written:
        return None, (
            f"Could not copy {missing} package file(s) into the working "
            "directory; read them with read_workspace_file at the workspace "
            "paths listed in files."
        )
    if missing:
        return package_dir, f"{missing} of {len(files)} package file(s) failed to copy."
    return package_dir, None


def _is_safe_relative(path: str) -> bool:
    """A package path must stay inside the package directory.

    ``normpath`` alone does not settle it: it collapses ``ok/../../out`` to
    ``../out``, but an already-normal ``../escape`` comes back unchanged and
    compares equal. The parent segment is therefore rejected on its own.
    """
    return (
        bool(path)
        and not path.startswith("/")
        and ".." not in path.split("/")
        and posixpath.normpath(path) == path
    )


async def _read_package_manifest(
    manifest_path: str, session_id: str
) -> dict[str, dict[str, Any]]:
    """What the last activation wrote, per path — empty on a first run or any
    unreadable manifest, because a re-copy is cheap and a stale skip is not.

    A bare digest string is the pre-executable-tracking format; reading it as
    non-executable is the safe direction, since the worst it costs is one
    redundant chmod.

    The manifest lives in the model's own working directory, so its keys are
    untrusted input to a later rm: only paths we would have written survive.
    """
    raw = await read_workdir_bytes(manifest_path, session_id)
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(loaded, dict):
        return {}
    entries: dict[str, dict[str, Any]] = {}
    for path, value in loaded.items():
        if not isinstance(path, str) or not _is_safe_relative(path):
            continue
        if isinstance(value, str):
            entries[path] = {_MANIFEST_SHA: value, _MANIFEST_EXEC: False}
        elif isinstance(value, dict) and isinstance(value.get(_MANIFEST_SHA), str):
            entries[path] = {
                _MANIFEST_SHA: value[_MANIFEST_SHA],
                _MANIFEST_EXEC: bool(value.get(_MANIFEST_EXEC)),
            }
    return entries


async def _write_package_manifest(
    manifest_path: str, hashes: dict[str, dict[str, Any]], session_id: str
) -> None:
    result = await save_to_workdir(
        manifest_path, json.dumps(hashes).encode(), session_id
    )
    if isinstance(result, ErrorResponse):
        logger.warning("[skills] failed to write package manifest: %s", result.message)
