"""Feature-flag gate for AI-generated activity summaries on executions.

Summaries are persisted regardless of the flag, so every route that returns
executions has to scrub them when `AI_ACTIVITY_STATUS` is off for the user.
"""

from typing import TypeVar

from backend.data import execution as execution_db
from backend.util.feature_flag import Flag, is_feature_enabled

ExecutionT = TypeVar("ExecutionT", bound=execution_db.GraphExecutionMeta)


async def hide_activity_summaries_if_disabled(
    executions: list[ExecutionT], user_id: str
) -> list[ExecutionT]:
    """Hide activity summaries and scores if AI_ACTIVITY_STATUS feature is disabled."""
    if await is_feature_enabled(Flag.AI_ACTIVITY_STATUS, user_id):
        return executions
    return [_without_activity_features(execution) for execution in executions]


async def hide_activity_summary_if_disabled(
    execution: ExecutionT, user_id: str
) -> ExecutionT:
    """Hide activity summary and score for a single execution if AI_ACTIVITY_STATUS feature is disabled."""
    if await is_feature_enabled(Flag.AI_ACTIVITY_STATUS, user_id):
        return execution
    return _without_activity_features(execution)


def _without_activity_features(execution: ExecutionT) -> ExecutionT:
    if not execution.stats:
        return execution
    return execution.model_copy(
        update={"stats": execution.stats.without_activity_features()}
    )
