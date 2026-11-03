"""Reuse a known review target without billing while it waits for its owner."""

from backend.data.db_accessors import skill_versions_db
from backend.data.skill_reviews import LearningReviewRecord

from .contract import LearningScope
from .publication_state import current_version_withdrawn


async def review_target_is_paused(
    review: LearningReviewRecord, scope: LearningScope
) -> bool:
    if (
        review.disposition != "paused"
        or review.skill_name is None
        or review.user_id != scope.user_id
        or review.owner_key != scope.owner_key
    ):
        return False
    head = await skill_versions_db().get_head(
        scope.user_id, scope.owner_key, review.skill_name
    )
    if head is None:
        return False
    return head.learning_paused_at is not None or await current_version_withdrawn(
        scope.user_id, head
    )
