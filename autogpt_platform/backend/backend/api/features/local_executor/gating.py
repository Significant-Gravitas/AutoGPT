"""Deployment and per-user gates for Local PC executor routes."""

from backend.copilot.config import ChatConfig
from backend.util.feature_flag import Flag, evaluate_feature_flag, is_feature_enabled


async def is_local_executor_enabled(user_id: str) -> bool:
    """Return whether this deployment and user may access Local PC routes."""
    return ChatConfig().use_local_pc_executor and await is_feature_enabled(
        Flag.LOCAL_PC_EXECUTOR, user_id, default=False
    )


async def local_executor_turned_off(user_id: str) -> bool:
    """Whether the deployment or this user's rollout flag has turned the
    Local PC executor off, for a connection opened while it was on. A flag
    read that fails returns its default, which says nothing, so it keeps the
    connection rather than drop every connected machine at once."""
    if not ChatConfig().use_local_pc_executor:
        return True
    enabled, authoritative = await evaluate_feature_flag(
        Flag.LOCAL_PC_EXECUTOR, user_id, default=False
    )
    return authoritative and not enabled


async def is_workflow_recording_enabled(user_id: str) -> bool:
    """Return whether workflow recording is enabled for this Local PC user."""
    return await is_local_executor_enabled(user_id) and await is_feature_enabled(
        Flag.WORKFLOW_RECORDING, user_id, default=False
    )
