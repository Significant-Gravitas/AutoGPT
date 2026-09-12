"""Presentation and runtime gates for shared memory during the org rollout."""

from copy import deepcopy
from typing import Any

from backend.util.feature_flag import Flag, is_feature_enabled

SHARED_MEMORY_DISABLED = "Organization collaboration is not enabled for this account."


async def shared_memory_enabled(user_id: str | None) -> bool:
    if not user_id:
        return False
    return await is_feature_enabled(Flag.SHOW_ORG_SETTINGS, user_id, default=False)


def memory_tool_parameters(
    tool_name: str, parameters: dict[str, Any], *, shared_memory: bool
) -> dict[str, Any]:
    """Hide shared-tier arguments without changing the shared tool instances."""
    if shared_memory or tool_name not in {
        "memory_store",
        "memory_search",
        "memory_forget_search",
        "memory_forget_confirm",
    }:
        return parameters
    parameters = deepcopy(parameters)
    properties = parameters.get("properties", {})
    properties.pop("tier", None)
    properties.pop("team_id", None)
    return parameters
