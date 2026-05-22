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

import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from backend.api.features.store.exceptions import VirusDetectedError, VirusScanError
from backend.copilot.model import ChatSession
from backend.data.db_accessors import workspace_db
from backend.data.redis_client import get_redis_async
from backend.executor.cluster_lock import AsyncClusterLock
from backend.util.workspace import WorkspaceManager

from .base import BaseTool
from .models import ErrorResponse, ResponseType, ToolResponseBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Limits — keep the per-turn <available_skills> index small enough that it
# does not strain Anthropic prompt caches and does not crowd out the user's
# turn budget.  At ~80 chars per index line, 50 entries ≈ 1.2k tokens; the
# default seeded skills are tiny so users effectively see ~50 slots.
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


class SkillNotFoundError(Exception):
    """Raised by :func:`delete_user_skill` when the skill is missing."""


class BuiltInSkillError(Exception):
    """Raised by :func:`delete_user_skill` for default seeded skills."""


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
    return slug


async def list_user_skills(user_id: str) -> list[ParsedSkill]:
    """Return all skills the user has stored in workspace.

    Skips files whose contents do not parse as valid SKILL.md — a stray
    file should not break the index.
    """
    manager = await _get_user_skill_manager(user_id)
    files = await manager.list_files(
        path=f"{SKILL_FOLDER}/",
        limit=MAX_USER_SKILLS * 4,  # over-fetch in case of strays
        include_all_sessions=True,
    )
    skills: list[ParsedSkill] = []
    for f in files:
        # Only consider SKILL.md manifests — sibling files in the same
        # folder are accessed by the model via read_workspace_file.
        if not f.path.endswith("/SKILL.md"):
            continue
        try:
            raw = await manager.read_file(f.path)
        except Exception:
            logger.warning("[skills] failed to read %s", f.path, exc_info=True)
            continue
        slug = f.path.rsplit("/", 2)[-2] if "/" in f.path else ""
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            # A stray binary file should not break the per-turn index —
            # log and skip, same contract as the read-failure branch.
            logger.warning("[skills] non-UTF-8 contents at %s — skipping", f.path)
            continue
        parsed = parse_skill_markdown(text, fallback_name=slug)
        if parsed is not None:
            skills.append(parsed)
    skills.sort(key=lambda s: s.name)
    return skills


def get_default_skills() -> list[ParsedSkill]:
    """Return the default seeded skills as :class:`ParsedSkill` records."""
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
    guide before any user customisation.
    """
    skills = get_default_skills()
    if user_id:
        skills.extend(await list_user_skills(user_id))
    return skills


def render_skills_index(skills: list[ParsedSkill]) -> str:
    """Render skills as a compact one-line-each index for the
    ``<available_skills>`` injection block.

    The line format intentionally mirrors how Anthropic surfaces skills
    in Claude Code — ``name — description (triggers: …)`` — so the model
    treats user-distilled skills with the same affordance as built-ins.
    """
    if not skills:
        return ""
    lines = []
    for s in skills:
        trigger_hint = f" (triggers: {', '.join(s.triggers)})" if s.triggers else ""
        lines.append(f"- {s.name} — {s.description}{trigger_hint}")
    return "\n".join(lines)


async def build_skills_context(user_id: str | None) -> str:
    """Build the body of the ``<available_skills>`` block injected into
    the first user message.  Returns ``""`` if there are no skills to
    show — :func:`inject_user_context` then omits the block entirely so
    sessions with no skill state don't pay the tag overhead.
    """
    skills = await list_all_skills(user_id)
    index = render_skills_index(skills)
    if not index:
        return ""
    return (
        "Skills are reusable procedures available via `read_skill(name)`. "
        "Load one before acting on a task it covers; distill a new one "
        "with `store_skill` after you complete a non-trivial procedure "
        "worth reusing.\n"
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
                        "Markdown body. Sections: Why / Trigger / Steps / Notes."
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

        name = name.strip().lower()
        description = description.strip()
        body = body.strip()
        triggers = [t.strip() for t in (triggers or []) if str(t).strip()]

        name_err = _validate_name(name)
        if name_err:
            return ErrorResponse(message=name_err, session_id=session_id)
        if not description:
            return ErrorResponse(
                message="description is required", session_id=session_id
            )
        if len(description) > MAX_DESCRIPTION_CHARS:
            return ErrorResponse(
                message=(
                    f"description must be ≤{MAX_DESCRIPTION_CHARS} chars "
                    "(it appears in every turn's skills index)"
                ),
                session_id=session_id,
            )
        if not body:
            return ErrorResponse(message="body is required", session_id=session_id)
        if len(body) > MAX_BODY_CHARS:
            return ErrorResponse(
                message=f"body must be ≤{MAX_BODY_CHARS} chars",
                session_id=session_id,
            )
        if len(triggers) > MAX_TRIGGERS:
            return ErrorResponse(
                message=(
                    f"triggers must be ≤{MAX_TRIGGERS} entries "
                    "(they are inlined in <available_skills> every turn)"
                ),
                session_id=session_id,
            )
        oversized_trigger = next(
            (t for t in triggers if len(t) > MAX_TRIGGER_CHARS), None
        )
        if oversized_trigger is not None:
            return ErrorResponse(
                message=(
                    f"trigger '{oversized_trigger[:32]}…' exceeds "
                    f"{MAX_TRIGGER_CHARS} chars"
                ),
                session_id=session_id,
            )

        # Serialise the count-then-write critical section per-user so two
        # concurrent ``store_skill`` calls cannot both pass the MAX
        # check.  Lock failure (Redis unavailable, already held) falls
        # back to a best-effort write without the cap — better than
        # blocking the user's distillation entirely, and the cap is a
        # soft per-turn budget cue, not a hard quota.  Wrap acquisition
        # in its own try/except so any transient Redis issue stays
        # contained — the rest of the path must still run unlocked.
        lock: AsyncClusterLock | None = None
        lock_held = False
        try:
            lock = AsyncClusterLock(
                redis=await get_redis_async(),
                key=f"{_SKILL_WRITE_LOCK_KEY_PREFIX}{user_id}",
                owner_id=uuid.uuid4().hex,
                timeout=_SKILL_WRITE_LOCK_TTL_SECONDS,
            )
            lock_held = (await lock.try_acquire()) == lock.owner_id
        except Exception:
            logger.warning(
                "[skills] failed to acquire write lock for user %s — "
                "falling back to unlocked best-effort write",
                user_id,
                exc_info=True,
            )
        try:
            manager = await _get_user_skill_manager(user_id)
            # Enforce the per-user cap *before* we write — counting only
            # existing slugs other than the one being upserted so updating
            # an existing skill doesn't trip the limit.  When the lock is
            # held this is a true atomic check-then-write; otherwise it
            # is best-effort.
            existing = await list_user_skills(user_id)
            existing_slugs = {s.name for s in existing}
            if name not in existing_slugs and len(existing_slugs) >= MAX_USER_SKILLS:
                return ErrorResponse(
                    message=(
                        f"Skill limit reached ({MAX_USER_SKILLS}). "
                        "Delete an unused skill with delete_skill first."
                    ),
                    session_id=session_id,
                )

            parsed = ParsedSkill(
                name=name,
                description=description,
                body=body,
                triggers=tuple(triggers),
            )
            rendered = render_skill_markdown(parsed)
            await manager.write_file(
                content=rendered.encode("utf-8"),
                filename="SKILL.md",
                path=_skill_md_path(name),
                mime_type="text/markdown",
                overwrite=True,
            )
        except (VirusDetectedError, VirusScanError) as exc:
            logger.warning("[skills] virus scan failed for %s: %s", name, exc)
            return ErrorResponse(
                message="Skill content rejected by virus scan",
                error=str(exc),
                session_id=session_id,
            )
        except ValueError as exc:
            return ErrorResponse(message=str(exc), session_id=session_id)
        except Exception as exc:
            logger.exception("[skills] failed to store skill %s", name)
            return ErrorResponse(
                message=f"Failed to store skill: {exc}",
                error=str(exc),
                session_id=session_id,
            )
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

        return StoreSkillResponse(
            name=name,
            description=description,
            triggers=list(triggers),
            message=(
                f"Skill '{name}' stored. It will appear in "
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
