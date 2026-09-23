"""Bring installed catalog skills up to their listing's newest version.

A hire's skills are copies in its owner's workspace, so a catalog edit reaches
them only if something rewrites the copy. That happens here, when the copy's
index is rebuilt: a copy still exactly as installed takes the new version
whole; a copy its owner edited gets a three-way merge (``skill_merge``) that
keeps every part the owner changed and records where the two collided.

Runs where the workspace is (the copilot executor and the API) rather than in
the deploy job, which has neither the workspace storage nor its write lock.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# A module reference, not names: ``skills`` imports this module while it is
# still loading, and its names exist only once it has finished.
from backend.copilot.tools import skills as skill_store
from backend.copilot.tools.skill_merge import SkillContent, merge_skill
from backend.data.db_accessors import skill_listings_db
from backend.data.skill_capacity import SKILL_ORIGIN_MARKETPLACE
from backend.data.workspace_scope import WorkspaceScope

if TYPE_CHECKING:
    from backend.api.features.store.skill_model import SkillRelease, SkillUpdate

logger = logging.getLogger(__name__)


async def refresh_installed_skills(
    user_id: str,
    expert_id: str | None,
    skills: list[skill_store.ParsedSkill],
    scope: WorkspaceScope | None = None,
) -> bool:
    """Update every copy whose listing has moved on. Returns whether any did."""
    installed = {
        skill.name: int(skill.installed_version)
        for skill in skills
        if skill.installed_version and skill.installed_version.isdigit()
    }
    if not installed:
        return False
    try:
        updates = await skill_listings_db().get_skill_updates(installed)
    except Exception:
        # An update can wait for the next rebuild; the turn cannot.
        logger.warning("[skills] could not check for skill updates", exc_info=True)
        return False
    by_name = {skill.name: skill for skill in skills}
    refreshed = False
    for update in updates:
        try:
            refreshed |= await _apply(
                user_id, expert_id, scope, by_name[update.slug], update
            )
        except Exception:
            logger.exception(
                f"[skills] could not update '{update.slug}' to "
                f"v{update.latest.version} for expert #{expert_id}"
            )
    return refreshed


async def _apply(
    user_id: str,
    expert_id: str | None,
    scope: WorkspaceScope | None,
    installed: skill_store.ParsedSkill,
    update: SkillUpdate,
) -> bool:
    if update.base is None:
        logger.warning(
            f"[skills] '{update.slug}' v{installed.installed_version} is gone; "
            "cannot tell the owner's edits from it, leaving the copy alone"
        )
        return False
    package = await skill_store.read_user_skill_package(
        user_id, update.slug, expert_id=expert_id, scope=scope
    )
    if package is None:
        return False
    current = skill_store.parse_skill_markdown(package.skill_md, update.slug)
    if current is None:
        return False
    merged = merge_skill(
        _content(update.base),
        _normalized(
            current.description,
            current.body,
            list(current.triggers),
            {f.relative_path: f.content for f in package.files},
        ),
        _content(update.latest),
    )
    executable = {
        f.relative_path
        for f in [*package.files, *update.latest.files]
        if f.is_executable
    }
    await skill_store.store_user_skill(
        user_id,
        name=update.slug,
        description=merged.content.description,
        body=merged.content.body,
        triggers=merged.content.triggers,
        version=str(update.latest.version),
        extra={**current.extra, **update.latest.extra},
        files=[
            skill_store.SkillFile(
                relative_path=path,
                content=content,
                is_executable=path in executable,
            )
            for path, content in merged.content.files.items()
        ],
        expert_id=expert_id,
        scope=scope,
        # An edited copy may have become the owner's; it stays in their budget.
        origin=installed.origin or SKILL_ORIGIN_MARKETPLACE,
        installed_version=str(update.latest.version),
        update_conflicts=[c.model_dump() for c in merged.conflicts],
    )
    logger.info(
        f"[skills] updated '{update.slug}' to v{update.latest.version} for "
        f"expert #{expert_id} with {len(merged.conflicts)} conflict(s) kept as the owner's"
    )
    return True


def _content(release: SkillRelease) -> SkillContent:
    return _normalized(
        release.description,
        release.body,
        release.triggers,
        {f.relative_path: f.content for f in release.files},
    )


def _normalized(
    description: str, body: str, triggers: list[str], files: dict[str, bytes]
) -> SkillContent:
    """Trimmed the way ``store_user_skill`` trims, so a copy nobody touched
    compares equal to the version it was installed from."""
    return SkillContent(
        description=description.strip(),
        body=body.strip(),
        triggers=[t.strip() for t in triggers if t.strip()],
        files=files,
    )
