"""Reuse a known review target without billing while its owner pauses learning."""

from backend.data.db_accessors import skill_versions_db
from backend.data.skill_reviews import LearningReviewRecord

from .contract import LearningScope


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
    return head is not None and head.learning_paused_at is not None
