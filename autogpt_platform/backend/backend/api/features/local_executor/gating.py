"""Per-user gates for the Local PC executor, its computer use and recording.

Each is a PostHog flag, default off, so one switch per feature decides who
gets it in every deployment: a deployment without PostHog has them off unless
it forces one on (``FORCE_FLAG_<NAME>=true``).
"""

from backend.util.feature_flag import Flag, evaluate_feature_flag, is_feature_enabled


async def is_local_executor_enabled(user_id: str) -> bool:
    """Whether this user may link a computer and use Local PC routes."""
    return await is_feature_enabled(Flag.LOCAL_PC_EXECUTOR, user_id, default=False)


async def local_executor_turned_off(user_id: str) -> bool:
    """Whether the rollout flag has turned the Local PC executor off for this
    user, for a connection opened while it was on. A flag read that fails
    returns its default, which says nothing, so it keeps the connection rather
    than drop every connected machine at once."""
    enabled, authoritative = await evaluate_feature_flag(
        Flag.LOCAL_PC_EXECUTOR, user_id, default=False
    )
    return authoritative and not enabled


async def is_computer_use_enabled(user_id: str) -> bool:
    """Whether this user may let Otto drive their Local PC's apps. The shim
    must still offer computer use, and the user consent for the machine."""
    return await is_local_executor_enabled(user_id) and await is_feature_enabled(
        Flag.LOCAL_PC_COMPUTER_USE, user_id, default=False
    )


async def is_workflow_recording_enabled(user_id: str) -> bool:
    """Return whether workflow recording is enabled for this Local PC user."""
    return await is_local_executor_enabled(user_id) and await is_feature_enabled(
        Flag.WORKFLOW_RECORDING, user_id, default=False
    )
