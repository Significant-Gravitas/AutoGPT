"""The checks an MCP tool call passes before the tool runs.

A Copilot tool trusts its caller: in a chat, that is AutoPilot's own model,
working inside the chat's session. Over MCP the caller is anything holding a
v2 credential, so the call is held to the credential's permissions first,
the way each REST route is.
"""

import logging
from typing import Any, Sequence

from mcp.server.fastmcp.exceptions import ToolError

from backend.copilot.config import ChatConfig
from backend.copilot.rate_limit import (
    RateLimitExceeded,
    RateLimitUnavailable,
    check_rate_limit,
    get_global_rate_limits,
    is_user_paywalled,
)
from backend.copilot.tools.base import BaseTool

logger = logging.getLogger(__name__)

config = ChatConfig()


def input_schema(tool: BaseTool) -> dict[str, Any]:
    """The JSON Schema of the arguments an external caller may pass.

    The chat's own schema leaves ``required`` out for the sake of a model whose
    tool call was cut short; an MCP client has no such excuse, so it is kept.
    """
    parameters = tool.external_parameters
    schema: dict[str, Any] = {
        "type": "object",
        "properties": parameters.get("properties", {}),
        "additionalProperties": False,
    }
    if required := parameters.get("required"):
        schema["required"] = list(required)
    return schema


def check_arguments(tool: BaseTool, args: dict[str, Any]) -> None:
    """Refuse an argument the tool doesn't advertise to external callers.

    The tools take ``**kwargs``, so an unadvertised argument would reach code
    the advertised schema keeps external callers away from.
    """
    allowed = tool.external_parameters.get("properties", {})
    if unknown := sorted(set(args) - set(allowed)):
        raise ToolError(f"Unknown argument(s): {', '.join(unknown)}")


def missing_scopes(
    tool: BaseTool,
    required_scopes: Sequence[str],
    args: dict[str, Any],
    granted: Sequence[str],
) -> list[str]:
    """The permissions this call needs that the credential doesn't carry.

    The tool's own permissions, plus those the arguments add: a branch that
    writes what those don't cover (``BaseTool.external_permissions``).
    """
    needed = [*required_scopes, *(p.value for p in tool.external_permissions(args))]
    return [scope for scope in dict.fromkeys(needed) if scope not in granted]


async def check_spend_allowance(user_id: str) -> None:
    """The pre-flight a chat turn runs, for a tool that spends what a turn does.

    Metering after the call only slows the next one down; without this a key
    on an unpaid account, or past its usage cap, keeps spending.
    """
    try:
        paywalled = await is_user_paywalled(user_id)
    except Exception as exc:
        logger.warning(f"MCP paywall check failed for {user_id}: {exc}")
        raise ToolError("Couldn't check your plan; retry shortly") from exc
    if paywalled:
        raise ToolError("This tool needs an active subscription")

    try:
        daily_limit, weekly_limit, _ = await get_global_rate_limits(
            user_id,
            config.daily_cost_limit_microdollars,
            config.weekly_cost_limit_microdollars,
        )
        await check_rate_limit(
            user_id=user_id,
            daily_cost_limit=daily_limit,
            weekly_cost_limit=weekly_limit,
        )
    except RateLimitExceeded as exc:
        raise ToolError(str(exc)) from exc
    except RateLimitUnavailable as exc:
        # Fail closed, as the chat route does: an outage can't prove the
        # caller is under their cap.
        raise ToolError("Usage limits are unavailable; retry shortly") from exc
