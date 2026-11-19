"""Retrieve only the current owner's approved false-positive occurrences."""

from backend.data.db_accessors import skill_versions_db
from backend.data.skill_content_approvals import SkillContentApproval
from backend.data.skill_learning import owner_key_for


async def current_content_approvals(
    user_id: str, expert_id: str | None, skill_name: str
) -> list[SkillContentApproval]:
    versions = skill_versions_db()
    head = await versions.get_head(user_id, owner_key_for(expert_id), skill_name)
    if head is None or not head.current_version_id:
        return []
    current = await versions.get_version(user_id, head.current_version_id)
    if current is None or current.state != "ready" or current.head_id != head.id:
        return []
    return [a for a in current.content_approvals if a.approved_by_user_id == user_id]
