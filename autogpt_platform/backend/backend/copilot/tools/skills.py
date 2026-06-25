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
import json
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from backend.api.features.store.exceptions import VirusDetectedError, VirusScanError
from backend.copilot.model import ChatSession
from backend.copilot.service import strip_server_injected_tags
from backend.data.db_accessors import workspace_db
from backend.data.redis_client import get_redis_async
from backend.executor.cluster_lock import AsyncClusterLock
from backend.util.feature_flag import Flag, is_feature_enabled
from backend.util.workspace import WorkspaceManager

from .base import BaseTool
from .models import ErrorResponse, ResponseType, ToolResponseBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Limits — keep the per-turn <available_skills> index small enough that it
# does not strain Anthropic prompt caches and does not crowd out the user's
# turn budget.  A typical user skill line lands around 150-200 chars
# (~50 tok), so 50 entries ≈ 2.5k tokens.  The hard worst case — every
# description and every trigger filled to the per-field caps below — is
# ~12k tokens, used as the upper bound when sizing the prefix cache.
# Built-in seeded skills are tiny so first-touch users see well under
# 200 tokens of overhead.
# ---------------------------------------------------------------------------
MAX_USER_SKILLS = 50
MAX_NAME_CHARS = 64
MAX_DESCRIPTION_CHARS = 200
MAX_BODY_CHARS = 20_000
# Triggers appear inline in the per-turn ``<available_skills>`` index,
# so an unbounded list (or one huge trigger) would balloon the prefix
# the model parses every turn.  Cap both the count and the per-entry
# length so a misbehaving caller cannot blow the token budget.
MAX_TRIGGERS = 10
MAX_TRIGGER_CHARS = 64
SKILL_FOLDER = "/skills"

# Redis-cached index TTL.  Skill content changes only on store/delete so a
# 60s TTL with explicit invalidation gives near-zero index latency on warm
# turns without unbounded staleness for cross-instance edits.
SKILLS_INDEX_CACHE_TTL_S = 60
SKILLS_INDEX_CACHE_KEY = "copilot:skills_index:{user_id}"

# WorkspaceFile.metadata keys for the skill index fast path — avoids the
# storage read when the metadata was written at store time (anything
# pre-dating this change falls back to read+parse, see
# ``_list_user_skills_from_workspace``).
_META_KIND = "kind"
_META_KIND_VALUE = "copilot_skill"
_META_DESCRIPTION = "description"
_META_TRIGGERS = "triggers"
_META_VERSION = "version"

# Skill names are slug-like — lowercase letters, digits, dashes, underscores.
# Must start and end with [a-z0-9] (no trailing/leading punctuation) so the
# folder name is clean and the on-screen index never has dangling dashes.
# Length cap matches MAX_NAME_CHARS via the {0,62} interior + 1 anchor at
# each end.
_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")


# ---------------------------------------------------------------------------
# Default skills — migrated from the legacy ``get_agent_building_guide`` /
# ``get_mcp_guide`` tools so users get a uniform discovery surface.  These
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
    _DefaultSkill(
        name="mcp_tool_guide",
        description=(
            "MCP server URLs and auth setup — load before calling "
            "run_mcp_tool when you need server URLs or auth details."
        ),
        body_path=_SDK_DIR / "mcp_tool_guide.md",
        triggers=("run_mcp_tool",),
    ),
)

_DEFAULT_SKILLS_BY_NAME: dict[str, _DefaultSkill] = {s.name: s for s in DEFAULT_SKILLS}


# ---------------------------------------------------------------------------
# Frontmatter + on-the-wire format helpers
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)


@dataclass(frozen=True)
class ParsedSkill:
    """A SKILL.md decoded into its frontmatter metadata + markdown body."""

    name: str
    description: str
    body: str
    triggers: tuple[str, ...] = ()
    version: str | None = None


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
    )


def render_skill_markdown(skill: ParsedSkill) -> str:
    """Render a :class:`ParsedSkill` to canonical ``SKILL.md`` text.

    Kept symmetric with :func:`parse_skill_markdown` so a round-trip via
    ``store_skill`` → workspace → ``read_skill`` produces the same
    metadata the model wrote.
    """
    meta: dict[str, Any] = {"name": skill.name, "description": skill.description}
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


# ---------------------------------------------------------------------------
# Workspace registry — read/write/delete user skills as folders under
# ``/skills/{slug}/SKILL.md``.  We construct a *session-less*
# :class:`WorkspaceManager` so the skill folder is shared across every
# session for the user (the default per-session prefix would silo each
# skill to one chat).
# ---------------------------------------------------------------------------


async def _get_user_skill_manager(user_id: str) -> WorkspaceManager:
    workspace = await workspace_db().get_or_create_workspace(user_id)
    return WorkspaceManager(user_id, workspace.id, session_id=None)


# Redis lock key for serialising store_skill writes per user. A per-user
# distributed lock turns the otherwise-racy "count existing skills, then
# write a new one" into an atomic critical section so two concurrent
# ``store_skill`` calls cannot both pass the MAX_USER_SKILLS check.
# Held only for the duration of the count + write; skill reads stay
# lock-free.
_SKILL_WRITE_LOCK_KEY_PREFIX = "copilot:skill_write:"
_SKILL_WRITE_LOCK_TTL_SECONDS = 30


def _skill_md_path(name: str) -> str:
    return f"{SKILL_FOLDER}/{name}/SKILL.md"


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
    )


class SkillNotFoundError(Exception):
    """Raised by :func:`delete_user_skill` when the skill is missing."""


class BuiltInSkillError(Exception):
    """Raised by :func:`delete_user_skill` for default seeded skills."""


class SkillLimitError(Exception):
    """Raised by :func:`store_user_skill` when the per-user cap is reached."""


async def delete_user_skill(user_id: str, name: str) -> str:
    """Delete a user-distilled skill folder by slug.

    Returns the normalised slug on success so callers can echo it back.
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

    manager = await _get_user_skill_manager(user_id)
    info = await manager.get_file_info_by_path(_skill_md_path(slug))
    if info is None:
        raise SkillNotFoundError(f"Skill '{slug}' not found")

    # Audit-log the delete BEFORE the workspace mutation runs so the
    # log row is present even if the delete itself raises mid-cleanup.
    # Decode best-effort: a malformed body is fine for the audit trail —
    # the slug + truncated description is what an operator needs to
    # correlate later support requests.
    description_for_log = ""
    try:
        raw = await manager.read_file(_skill_md_path(slug))
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

    try:
        siblings = await manager.list_files(
            path=f"{SKILL_FOLDER}/{slug}/",
            limit=50,
            include_all_sessions=True,
        )
    except Exception:
        siblings = []
    await manager.delete_file(info.id)
    for sibling in siblings:
        if sibling.id == info.id:
            continue
        try:
            await manager.delete_file(sibling.id)
        except Exception:
            logger.warning(
                "[skills] failed to delete sibling %s",
                sibling.path,
                exc_info=True,
            )
    await invalidate_skills_index_cache(user_id)
    return slug


async def store_user_skill(
    user_id: str,
    *,
    name: str,
    description: str,
    body: str,
    triggers: list[str] | None = None,
    version: str | None = None,
) -> ParsedSkill:
    """Validate + persist a user-distilled skill, returning the stored skill.

    Shared by the ``store_skill`` copilot tool and the REST ``POST /skills``
    upload endpoint so both honour the same validation, per-user cap, and
    write-lock semantics.  Raises :class:`ValueError` for any validation
    failure, :class:`SkillLimitError` when the per-user cap is reached, and
    propagates ``VirusDetectedError`` / ``VirusScanError`` (and any other
    workspace write error) to the caller.
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
    if len(triggers) > MAX_TRIGGERS:
        raise ValueError(
            f"triggers must be ≤{MAX_TRIGGERS} entries "
            "(they are inlined in <available_skills> every turn)"
        )
    oversized_trigger = next((t for t in triggers if len(t) > MAX_TRIGGER_CHARS), None)
    if oversized_trigger is not None:
        raise ValueError(
            f"trigger '{oversized_trigger[:32]}…' exceeds {MAX_TRIGGER_CHARS} chars"
        )

    # Serialise the count-then-write critical section per-user so two
    # concurrent writers cannot both pass the MAX_USER_SKILLS check.
    # ``AsyncClusterLock.try_acquire`` is non-blocking, so poll for up to
    # ~1s before falling back to the strict-cap unlocked path below — without
    # the wait, two near-simultaneous calls at MAX-1 both proceed unlocked,
    # both see N<MAX, and both write (cap overruns by 1).  Lock failure
    # (Redis unavailable) still falls back to the unlocked write but the
    # cap-enforcement branch below refuses any at-cap write in that case.
    lock: AsyncClusterLock | None = None
    lock_held = False
    try:
        lock = AsyncClusterLock(
            redis=await get_redis_async(),
            key=f"{_SKILL_WRITE_LOCK_KEY_PREFIX}{user_id}",
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
        manager = await _get_user_skill_manager(user_id)
        # Enforce the per-user cap *before* we write.  When the lock IS held
        # this is a true atomic check-then-write — an upsert at-cap is safe
        # because no new slot is consumed.  When the lock FAILED to acquire,
        # the check is no longer atomic, so refuse any write at-or-above the
        # cap defensively (the caller can retry; a Redis blip is rare).
        existing = await list_user_skills(user_id)
        existing_slugs = {s.name for s in existing}
        at_cap = len(existing_slugs) >= MAX_USER_SKILLS
        is_new = name not in existing_slugs
        if at_cap and (is_new or not lock_held):
            if not lock_held:
                logger.warning(
                    "[skills] refusing at-cap unlocked write for user %s "
                    "(is_new=%s) — concurrent write could otherwise overrun "
                    "the cap",
                    user_id,
                    is_new,
                )
            raise SkillLimitError(
                f"Skill limit reached ({MAX_USER_SKILLS}). "
                "Delete an unused skill first."
            )

        parsed = ParsedSkill(
            name=name,
            description=description,
            body=body,
            triggers=tuple(triggers),
            version=version,
        )
        rendered = render_skill_markdown(parsed)
        metadata: dict[str, Any] = {
            _META_KIND: _META_KIND_VALUE,
            _META_DESCRIPTION: description,
            _META_TRIGGERS: list(triggers),
        }
        if version:
            metadata[_META_VERSION] = version
        await manager.write_file(
            content=rendered.encode("utf-8"),
            filename="SKILL.md",
            path=_skill_md_path(name),
            mime_type="text/markdown",
            overwrite=True,
            metadata=metadata,
        )
        await invalidate_skills_index_cache(user_id)
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
    )


async def _list_user_skills_from_workspace(user_id: str) -> list[ParsedSkill]:
    """Workspace-side listing — no caching.  Body fields are always empty
    because the index never needs them; :func:`read_user_skill_with_body`
    is the path for retrieving full content.

    Uses ``WorkspaceFile.metadata`` (written at store time) for the fast
    path and falls back to a parallelised read of the SKILL.md body for
    any file missing the metadata (older skills, or skills written by a
    deployment before the metadata-cache change shipped).
    """
    manager = await _get_user_skill_manager(user_id)
    files = await manager.list_files(
        path=f"{SKILL_FOLDER}/",
        limit=MAX_USER_SKILLS * 4,  # over-fetch in case of strays
        include_all_sessions=True,
    )
    md_files = [f for f in files if f.path.endswith("/SKILL.md")]

    skills: list[ParsedSkill] = []
    needs_read: list[Any] = []
    for f in md_files:
        slug = f.path.rsplit("/", 2)[-2] if "/" in f.path else ""
        meta = f.metadata if isinstance(f.metadata, dict) else {}
        entry = _index_entry_from_metadata(slug, meta)
        if entry is not None:
            skills.append(entry)
        else:
            needs_read.append(f)

    if needs_read:
        parsed = await asyncio.gather(
            *(_parse_skill_from_workspace(manager, f.path) for f in needs_read),
        )
        for p in parsed:
            if p is None:
                continue
            # Index never needs the body — drop it so the cache payload
            # stays small (defaults are already body-less, fast-path
            # entries are body-less, keep the contract uniform).
            skills.append(
                ParsedSkill(
                    name=p.name,
                    description=p.description,
                    body="",
                    triggers=p.triggers,
                    version=p.version,
                )
            )

    skills.sort(key=lambda s: s.name)
    return skills


def _skills_cache_key(user_id: str) -> str:
    return SKILLS_INDEX_CACHE_KEY.format(user_id=user_id)


async def _read_skills_cache(user_id: str) -> list[ParsedSkill] | None:
    try:
        redis = await get_redis_async()
        raw = await redis.get(_skills_cache_key(user_id))
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
            )
            for item in payload
            if isinstance(item, dict) and "name" in item and "description" in item
        ]
    except Exception:
        # Malformed cache entry → ignore and rebuild.
        return None


async def _write_skills_cache(user_id: str, skills: list[ParsedSkill]) -> None:
    try:
        redis = await get_redis_async()
        payload = json.dumps(
            [
                {
                    "name": s.name,
                    "description": s.description,
                    "triggers": list(s.triggers),
                    "version": s.version,
                }
                for s in skills
            ]
        )
        await redis.set(
            _skills_cache_key(user_id),
            payload,
            ex=SKILLS_INDEX_CACHE_TTL_S,
        )
    except Exception:
        # Cache write is best-effort.
        pass


async def invalidate_skills_index_cache(user_id: str) -> None:
    """Drop the cached per-user skill index so the next turn rebuilds it.
    Called by ``store_skill`` / ``delete_user_skill`` so an edit shows up
    immediately rather than after the 60s TTL.
    """
    try:
        redis = await get_redis_async()
        await redis.delete(_skills_cache_key(user_id))
    except Exception:
        # Cache invalidate is best-effort — at worst the user sees stale
        # state for up to ``SKILLS_INDEX_CACHE_TTL_S`` seconds.
        pass


async def list_user_skills(user_id: str) -> list[ParsedSkill]:
    """Return all skills the user has stored in workspace.

    Two-level fast path: a 60s Redis cache covers warm turns, the
    ``WorkspaceFile.metadata`` index covers cold turns without any storage
    reads, and a parallelised body parse covers any skills written before
    the metadata-cache change shipped.  Skips files whose contents do not
    parse as valid SKILL.md so a stray file in ``/skills/`` cannot break
    the index.
    """
    cached = await _read_skills_cache(user_id)
    if cached is not None:
        return cached
    skills = await _list_user_skills_from_workspace(user_id)
    await _write_skills_cache(user_id, skills)
    return skills


async def read_user_skill_with_body(user_id: str, name: str) -> ParsedSkill | None:
    """Return a single user-stored skill with its body populated.

    Used by the ``read_skill`` MCP tool and the REST GET ``/skills/{name}``
    endpoint that powers the library UI's expand-to-view dialog.  Returns
    ``None`` when no SKILL.md exists at the slug — callers translate that
    into a 404 / structured error response.
    """
    slug = name.strip().lower()
    if not slug:
        return None
    manager = await _get_user_skill_manager(user_id)
    return await _parse_skill_from_workspace(manager, _skill_md_path(slug))


async def list_user_skill_sibling_paths(user_id: str, name: str) -> list[str]:
    """Return the workspace paths of files siblings to ``SKILL.md`` in a
    user-stored skill's folder (``references/``, ``scripts/``, ``assets/``,
    or anything the model stashed there at distillation time).

    Used by the REST GET ``/skills/{name}`` endpoint so the library UI's
    expand-to-view dialog can show the model what extra artefacts the
    skill bundle carries.  Returns ``[]`` on any error — sibling listing
    is best-effort and must not fail the parent request.
    """
    slug = name.strip().lower()
    if not slug:
        return []
    try:
        manager = await _get_user_skill_manager(user_id)
        files = await manager.list_files(
            path=f"{SKILL_FOLDER}/{slug}/",
            limit=50,
            include_all_sessions=True,
        )
        return [f.path for f in files if not f.path.endswith("/SKILL.md")]
    except Exception:
        logger.warning(
            "[skills] failed to list sibling files for %s", slug, exc_info=True
        )
        return []


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
            )
        )
    return result


async def list_all_skills(user_id: str | None) -> list[ParsedSkill]:
    """Default seeded skills first, then user-distilled skills.

    Defaults always lead so the model sees the built-in agent-building
    guide before any user customisation.  Index-only — default bodies
    are NOT loaded; :tool:`read_skill` re-reads them on demand via
    :func:`get_default_skills`.
    """
    skills = get_default_skills_for_index()
    if user_id:
        skills.extend(await list_user_skills(user_id))
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


async def build_skills_context(user_id: str | None) -> str:
    """Build the body of the ``<available_skills>`` block injected into
    the first user message.  Returns ``""`` if there are no skills to
    show — :func:`inject_user_context` then omits the block entirely so
    sessions with no skill state don't pay the tag overhead.

    Also returns ``""`` when the ``COPILOT_SKILLS`` LD flag is off for
    this user, so the kill-switch fully suppresses the per-turn index
    cost (no list query, no Redis hit, nothing to cache).
    """
    if not await is_skills_feature_enabled(user_id):
        return ""
    skills = await list_all_skills(user_id)
    index = render_skills_index(skills)
    if not index:
        return ""
    return (
        "Skills are reusable procedures available via `read_skill(name)`. "
        "Match the user's request to a skill's triggers (substring or "
        "close paraphrase) and call `read_skill(name=...)` to load the "
        "full body before acting; distill a new one with `store_skill` "
        "after you complete a non-trivial procedure worth reusing.\n"
        f"{index}"
    )


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


class StoreSkillResponse(ToolResponseBase):
    type: ResponseType = ResponseType.SKILL_STORED
    name: str
    description: str
    triggers: list[str] = []


class ReadSkillResponse(ToolResponseBase):
    type: ResponseType = ResponseType.SKILL_LOADED
    name: str
    description: str
    body: str
    triggers: list[str] = []
    sibling_files: list[str] = []
    is_default: bool = False


class DeleteSkillResponse(ToolResponseBase):
    type: ResponseType = ResponseType.SKILL_DELETED
    name: str


class ListSkillsResponse(ToolResponseBase):
    type: ResponseType = ResponseType.SKILL_LIST
    skills: list[dict[str, Any]]


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
            "<available_skills> next turn; loads via read_skill(name)."
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
                    "description": "One-line hook (≤200 chars).",
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
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
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
            )
        except (VirusDetectedError, VirusScanError) as exc:
            logger.warning("[skills] virus scan failed for %s: %s", name, exc)
            return ErrorResponse(
                message="Skill content rejected by virus scan",
                error=str(exc),
                session_id=session_id,
            )
        except (ValueError, SkillLimitError) as exc:
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
            message=(
                f"Skill '{parsed.name}' stored. It will appear in "
                "<available_skills> on the next turn."
            ),
            session_id=session_id,
        )


class ReadSkillTool(BaseTool):
    """Load a skill's body + sibling-file listing by name."""

    @property
    def name(self) -> str:
        return "read_skill"

    @property
    def description(self) -> str:
        return (
            "Read a skill's body + sibling-file list by name. Call when "
            "a task matches an <available_skills> entry."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name."},
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

        try:
            manager = await _get_user_skill_manager(user_id)
            raw = await manager.read_file(_skill_md_path(name))
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
                    "Re-create it with store_skill."
                ),
                session_id=session_id,
            )
        parsed = parse_skill_markdown(text, fallback_name=name)
        if parsed is None:
            return ErrorResponse(
                message=(
                    f"Skill '{name}' is malformed (missing/invalid "
                    "frontmatter). Re-create it with store_skill."
                ),
                session_id=session_id,
            )

        # List sibling files (references/, scripts/, assets/, etc.) so
        # the model knows what else lives in the bundle.
        try:
            siblings = await manager.list_files(
                path=f"{SKILL_FOLDER}/{name}/",
                limit=50,
                include_all_sessions=True,
            )
            sibling_paths = [
                f.path for f in siblings if not f.path.endswith("/SKILL.md")
            ]
        except Exception:
            logger.warning(
                "[skills] failed to list siblings for %s", name, exc_info=True
            )
            sibling_paths = []

        return ReadSkillResponse(
            name=parsed.name,
            description=parsed.description,
            body=parsed.body,
            triggers=list(parsed.triggers),
            sibling_files=sibling_paths,
            is_default=False,
            message=f"Loaded skill '{name}'.",
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
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )
        if not await is_skills_feature_enabled(user_id):
            return ErrorResponse(
                message="Skill registry is currently disabled for this user.",
                error="feature_disabled",
                session_id=session_id,
            )
        try:
            slug = await delete_user_skill(user_id, name)
        except ValueError as exc:
            return ErrorResponse(message=str(exc), session_id=session_id)
        except BuiltInSkillError as exc:
            return ErrorResponse(message=str(exc), session_id=session_id)
        except SkillNotFoundError as exc:
            return ErrorResponse(message=str(exc), session_id=session_id)
        except Exception as exc:
            logger.exception("[skills] delete failed for %s", name)
            return ErrorResponse(
                message=f"Failed to delete skill: {exc}", session_id=session_id
            )

        return DeleteSkillResponse(
            name=slug,
            message=f"Skill '{slug}' deleted.",
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
        return "List all skills (defaults + user-distilled)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    @property
    def requires_auth(self) -> bool:
        return False

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        if not await is_skills_feature_enabled(user_id):
            return ErrorResponse(
                message="Skill registry is currently disabled for this user.",
                error="feature_disabled",
                session_id=session.session_id,
            )
        skills = await list_all_skills(user_id)
        payload = [
            {
                "name": s.name,
                "description": s.description,
                "triggers": list(s.triggers),
                "is_default": s.name in _DEFAULT_SKILLS_BY_NAME,
            }
            for s in skills
        ]
        return ListSkillsResponse(
            skills=payload,
            message=f"{len(payload)} skill(s) available.",
            session_id=session.session_id,
        )
