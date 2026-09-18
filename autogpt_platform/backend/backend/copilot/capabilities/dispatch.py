"""Resolve a ``run_capability`` dispatch into the tool call it is.

Deferred tools are absent from the model's tool list and reached by id
(#14569), so every engine sees the dispatcher's name where the tool's should
be. Each engine resolves the call here and then runs the inner tool through
its normal path, so the announce, the history row, the permission gate, the
circuit breaker and the frontend events all name the tool that ran.
"""

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NamedTuple

from backend.copilot.capabilities.registry import configured_tool, get_registry
from backend.copilot.capabilities.resolve import resolve_entry

if TYPE_CHECKING:
    from backend.copilot.tools.base import BaseTool

logger = logging.getLogger(__name__)

DISPATCHER_TOOL_NAME = "run_capability"


class DispatchedToolCall(NamedTuple):
    tool: "BaseTool"
    name: str
    args: dict[str, Any]


def resolve_tool_dispatch(
    tool_name: str, args: Mapping[str, Any] | None
) -> DispatchedToolCall | None:
    """The platform tool a ``run_capability`` call runs, or None when the call
    is not one — another tool, a block or MCP id, or ``validate_only``.

    Never raises: it runs on the streaming path, where a bad id must degrade to
    the dispatcher's own "unknown capability" answer rather than break the turn.
    """
    if tool_name != DISPATCHER_TOOL_NAME or not isinstance(args, Mapping):
        return None
    if args.get("validate_only"):
        return None
    try:
        entry = resolve_entry(get_registry(), str(args.get("id") or ""))
        if entry is None or entry.kind != "tool" or not entry.implementations:
            return None
        name = entry.implementations[0].ref
        tool = configured_tool(name)
    except Exception:
        logger.warning("Could not resolve capability dispatch", exc_info=True)
        return None
    if tool is None:
        return None
    payload = args.get("input")
    if payload is not None and not isinstance(payload, Mapping):
        # Coercing it to {} would run the tool on its defaults; the dispatcher
        # owns the "input must be an object" answer, so leave the call to it.
        return None
    return DispatchedToolCall(tool, name, dict(payload or {}))
