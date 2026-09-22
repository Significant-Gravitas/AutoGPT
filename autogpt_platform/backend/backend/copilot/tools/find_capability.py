"""Search the capability registry: integrations, blocks, MCP servers,
platform tools and the session owner's skills behind one query."""

import asyncio
import logging
from typing import Any

from backend.copilot.capabilities.index import SearchHit
from backend.copilot.capabilities.models import CapabilityKindName
from backend.copilot.capabilities.resolve import load_connection_state
from backend.copilot.context import get_current_permissions
from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import (
    CapabilityListResponse,
    ErrorResponse,
    NoResultsResponse,
    ToolResponseBase,
)
from .session_registry import session_registry

logger = logging.getLogger(__name__)

CONTEXTS = ("direct", "graph")
KINDS = ("tool", "block", "mcp_server", "skill")
_KIND_ARG: dict[str, CapabilityKindName] = {
    "tool": "tool",
    "block": "block",
    "mcp_server": "mcp_server",
    "skill": "skill",
}


class FindCapabilityTool(BaseTool):
    """Ranked capability search with connection state."""

    @property
    def name(self) -> str:
        return "find_capability"

    @property
    def description(self) -> str:
        return (
            "Search everything the platform can do: integrations, blocks, MCP "
            "servers, platform tools and skills, by service name or action. "
            "Results are ranked and show whether the user has connected each "
            "one. Call this before saying something is not possible. Then "
            "describe_capability(id) to see inputs, and run_capability(id, "
            "input) to act."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Service, action or name, e.g. 'linear issue', "
                        "'send email', 'http request', 'sentry'."
                    ),
                },
                "context": {
                    "type": "string",
                    "enum": list(CONTEXTS),
                    "description": (
                        "'direct' (default) for things to run now; 'graph' when "
                        "choosing blocks for an agent graph (includes graph-only "
                        "blocks such as inputs and outputs)."
                    ),
                    "default": "direct",
                },
                "kind": {
                    "type": "string",
                    "enum": list(KINDS),
                    "description": "Restrict to one kind of capability.",
                },
            },
            "required": ["query"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        query: str = "",
        context: str = "direct",
        kind: str | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        query = (query or "").strip()
        session_id = session.session_id
        if not query:
            return ErrorResponse(
                message="Please provide a query", session_id=session_id
            )
        if context not in CONTEXTS:
            context = "direct"
        if kind is not None and kind not in KINDS:
            return ErrorResponse(
                message=f"kind must be one of {', '.join(KINDS)}",
                session_id=session_id,
            )
        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )

        connections, index = await asyncio.gather(
            load_connection_state(user_id), session_registry(user_id, session)
        )
        result = index.search(
            query,
            context="graph" if context == "graph" else "direct",
            kind=_KIND_ARG.get(kind or ""),
            connections=connections,
            permissions=get_current_permissions(),
        )
        if not result.hits and not result.fallback:
            return NoResultsResponse(
                message=f"No capability found for '{query}'",
                suggestions=[
                    "Try the service name alone, or a broader action ('email', 'sheet', 'http')",
                    "For a service with no result, web_search '<service> MCP server' and "
                    "call run_capability with the server URL as the id",
                ],
                session_id=session_id,
            )
        return CapabilityListResponse(
            message=_message(
                result.service,
                len(result.hits),
                len(result.fallback),
                skills=any(hit.entry.kind == "skill" for hit in result.hits),
            ),
            query=query,
            capabilities=[_listing(hit) for hit in result.hits],
            count=len(result.hits),
            fallback=[_listing(hit) for hit in result.fallback],
            service=result.service,
            session_id=session_id,
        )


def _listing(hit: SearchHit) -> dict[str, Any]:
    listing = hit.entry.listing()
    if hit.entry.connection.required:
        listing["connected"] = hit.connected
    return listing


def _message(
    service: str | None, hits: int, fallback: int, *, skills: bool = False
) -> str:
    parts = [f"Found {hits} capabilit{'y' if hits == 1 else 'ies'}"]
    if service:
        parts.append(f"for {service}")
    text = " ".join(parts) + "."
    if fallback:
        text += f" {fallback} generic fallback(s) listed separately."
    text += (
        " Call describe_capability(id) before first use, then "
        "run_capability(id, input). connected=false means the user must sign in "
        "first: run_capability returns the sign-in card."
    )
    if skills:
        text += (
            " A kind=skill result is a saved procedure: "
            "run_capability(id, input={}) loads its body and package files; "
            "read it before acting."
        )
    return text
