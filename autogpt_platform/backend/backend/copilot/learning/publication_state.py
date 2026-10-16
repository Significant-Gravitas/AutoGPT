from backend.data.db_accessors import skill_versions_db
from backend.data.skill_versions import SkillHeadRecord

from .publication_models import PublishOutcome


async def require_settled_head(
    user_id: str, head: SkillHeadRecord | None
) -> PublishOutcome | None:
    if head is None or head.current_version_id is None:
        return None
    current = await skill_versions_db().get_version(user_id, head.current_version_id)
    if current is not None and current.state == "invalidated":
        # Revocation found no earlier version to fall back to. Learning waits
        # for the owner rather than build on evidence that was withdrawn.
        return PublishOutcome(
            status="paused",
            reason="a source of the current version is no longer available; "
            "edit the skill or restore an earlier version to resume",
        )
    if current is None or current.state != "ready":
        return PublishOutcome(
            status="write_failed",
            reason="the current skill publication must recover before another update",
        )
    return None
