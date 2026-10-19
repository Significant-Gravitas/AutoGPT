"""The checks an MCP tool call passes before the tool runs.

A Copilot tool trusts its caller: in a chat, that is AutoPilot's own model,
working inside the chat's session. Over MCP the caller is anything holding a
v2 credential, so the call is held to the credential's permissions first,
the way each REST route is.
"""

import logging
from typing import Any, Sequence

from fastapi import HTTPException
from mcp.server.fastmcp.exceptions import ToolError
from prisma.enums import APIKeyPermission
from pydantic import ValidationError

from backend.api.utils.rate_limit import RateLimiter
from backend.copilot.config import ChatConfig
from backend.copilot.rate_limit import (
    RateLimitExceeded,
    RateLimitUnavailable,
    check_rate_limit,
    get_global_rate_limits,
    is_user_paywalled,
)
from backend.copilot.tools.base import BaseTool, without_card_hints
from backend.copilot.tools.run_agent import RunAgentInput
from backend.data.credit import get_credit_model

from .rate_limit import file_upload_limiter, graph_exec_limiter, search_limiter

logger = logging.getLogger(__name__)

config = ChatConfig()


def input_schema(tool: BaseTool) -> dict[str, Any]:
    """The JSON Schema of the arguments an external caller may pass.

    The chat's own schema leaves ``required`` out for the sake of a model whose
    tool call was cut short; an MCP client has no such excuse, so it is kept.
    The approval card's hints are left out, as they are from what the chat's
    model is sent: they aren't JSON Schema, and a client may refuse them.
    """
    parameters = without_card_hints(tool.external_parameters)
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


# The REST routes these tools mirror cap callers tighter than the global limit
# (`rate_limit.py`); a call over MCP is held to the same caps, in the same
# windows, so neither surface is a way around the other's.
TOOL_LIMITERS: dict[str, RateLimiter] = {
    "run_agent": graph_exec_limiter,
    "write_workspace_file": file_upload_limiter,
    "find_agent": search_limiter,
    "find_library_agent": search_limiter,
    "search_docs": search_limiter,
}


async def check_tool_limits(
    tool: BaseTool, args: dict[str, Any], user_id: str, organization_id: str
) -> None:
    """The per-endpoint caps, and for a real run, the zero-balance refusal."""
    for limiter in _tool_limiters(tool, args):
        try:
            await limiter.check(user_id)
        except HTTPException as exc:
            raise ToolError(str(exc.detail)) from exc
    if tool.name == "run_agent" and _starts_a_paid_run(args):
        try:
            credit_model = await get_credit_model(user_id, organization_id)
            balance = await credit_model.get_credits(user_id)
        except Exception as exc:
            logger.warning(f"MCP balance check failed for {user_id}: {exc}")
            raise ToolError("Couldn't check your balance; retry shortly") from exc
        if balance <= 0:
            raise ToolError(
                "Insufficient balance to run the agent. Please top up your account."
            )


def _tool_limiters(tool: BaseTool, args: dict[str, Any]) -> list[RateLimiter]:
    """The tool's own cap, plus the upload cap when an argument has it write a
    workspace file, as `write_to` does: the write the upload route caps."""
    limiters = [TOOL_LIMITERS[tool.name]] if tool.name in TOOL_LIMITERS else []
    if APIKeyPermission.WRITE_FILES in tool.external_permissions(args):
        limiters.append(file_upload_limiter)
    return list(dict.fromkeys(limiters))


def _starts_a_paid_run(args: dict[str, Any]) -> bool:
    """A run now, not a dry run and not a schedule, as REST's run route refuses.

    Read with the tool's own model, so this check and the tool agree on what
    the arguments ask for: a blank `cron` is no schedule to either. Arguments
    the model refuses start nothing.
    """
    try:
        params = RunAgentInput(**args)
    except ValidationError:
        return True
    return not (params.dry_run or params.schedule_name or params.cron)


async def check_plan(user_id: str) -> None:
    """Every tool call needs an active plan, as every chat turn does."""
    try:
        paywalled = await is_user_paywalled(user_id)
    except Exception as exc:
        logger.warning(f"MCP paywall check failed for {user_id}: {exc}")
        raise ToolError("Couldn't check your plan; retry shortly") from exc
    if paywalled:
        raise ToolError("This tool needs an active subscription")


async def check_spend_allowance(user_id: str) -> None:
    """The usage caps a chat turn checks, for a tool that spends what a turn does.

    Metering after the call only slows the next one down; without this a key
    past its usage cap keeps spending.
    """
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
