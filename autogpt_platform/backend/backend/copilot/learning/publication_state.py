from backend.data.db_accessors import skill_versions_db
from backend.data.skill_versions import SkillHeadRecord, SkillVersionRecord

from .publication_models import PublishOutcome

WITHDRAWN_REASON = (
    "a source of the current version is no longer available; edit the skill, "
    "or restore an earlier version if there is one, to resume"
)


async def require_settled_head(
    user_id: str, head: SkillHeadRecord | None
) -> PublishOutcome | None:
    if head is None or head.current_version_id is None:
        return None
    current = await skill_versions_db().get_version(user_id, head.current_version_id)
    if _withdrawn(current):
        return PublishOutcome(status="paused", reason=WITHDRAWN_REASON)
    if current is None or current.state != "ready":
        return PublishOutcome(
            status="write_failed",
            reason="the current skill publication must recover before another update",
        )
    return None


async def current_version_withdrawn(user_id: str, head: SkillHeadRecord) -> bool:
    """Revocation found no earlier version to fall back to, so the current
    version stays invalidated. Learning waits for the owner rather than build
    on evidence that was withdrawn."""
    if head.current_version_id is None:
        return False
    return _withdrawn(
        await skill_versions_db().get_version(user_id, head.current_version_id)
    )


def _withdrawn(current: SkillVersionRecord | None) -> bool:
    return current is not None and current.state == "invalidated"
