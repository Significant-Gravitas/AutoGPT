"""MCP calls and models for Google Maps Code Assist.

Maps Code Assist (https://developers.google.com/maps/ai/code-assist) is only
offered as a remote MCP server; there is no REST API behind it. The server is
stateless and its tools need no credentials, so each block sends one JSON-RPC
``tools/call`` request.
"""

import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from backend.util.json import loads
from backend.util.request import Requests, Response

MCP_URL = "https://mapscodeassist.googleapis.com/mcp"
CLIENT_SOURCE = "autogpt-platform"

_ESCAPE = re.compile(r"\\([nrt'\"\\])")
_UNESCAPED = {"n": "\n", "r": "\r", "t": "\t", "'": "'", '"': '"', "\\": "\\"}


class MapsPlatformDocPassage(BaseModel):
    """A passage from Google Maps Platform documentation or sample code."""

    text: str = Field(description="The passage: documentation text or code")
    url: Optional[str] = Field(
        default=None,
        description="Where the passage comes from. Google requires it to be cited",
    )
    relevance_score: Optional[float] = Field(
        default=None, description="How relevant the passage is to the query"
    )
    api_state: Optional[str] = Field(
        default=None,
        description=(
            "How current the API it covers is, as Google labels it, e.g. NEW, "
            "CURRENT or LEGACY"
        ),
    )


class MapsCodeAssistError(Exception):
    """Maps Code Assist failed or returned an error."""


async def call_maps_code_assist(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call one Maps Code Assist tool and return its structured result.

    The tools are read-only, so throttled and 5xx replies are safe to retry.
    """
    response = await Requests(
        trusted_origins=[MCP_URL],
        raise_for_status=False,
        retry_max_attempts=3,
    ).post(
        MCP_URL,
        headers={"Accept": "application/json, text/event-stream"},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        },
    )
    if not response.ok:
        busy = response.status == 429 or response.status >= 500
        raise MapsCodeAssistError(
            f"Google Maps Code Assist returned HTTP {response.status}."
            + (" Try again later." if busy else "")
        )
    reply = jsonrpc_reply(response)
    if error := reply.get("error"):
        detail = error.get("message") if isinstance(error, dict) else error
        raise MapsCodeAssistError(f"Google Maps Code Assist error: {detail}")
    result = reply.get("result")
    if not isinstance(result, dict):
        raise MapsCodeAssistError("Google Maps Code Assist returned no result.")
    if result.get("isError"):
        raise MapsCodeAssistError(
            f"Google Maps Code Assist couldn't answer: {content_text(result) or 'no details'}"
        )
    return structured_result(result)


def jsonrpc_reply(response: Response) -> dict[str, Any]:
    """The JSON-RPC reply, whether the server answered in JSON or as SSE."""
    body = response.text()
    if "text/event-stream" in response.headers.get("content-type", ""):
        events = [
            line[len("data:") :].strip()
            for line in body.splitlines()
            if line.startswith("data:")
        ]
        body = events[-1] if events else ""
    reply = loads(body, fallback=None) if body else None
    if not isinstance(reply, dict):
        raise MapsCodeAssistError("Google Maps Code Assist sent an unreadable reply.")
    return reply


def structured_result(result: dict[str, Any]) -> dict[str, Any]:
    """A tool's structured output; older replies carry it as JSON text only."""
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    parsed = loads(content_text(result), fallback=None)
    if not isinstance(parsed, dict):
        raise MapsCodeAssistError("Google Maps Code Assist sent an unreadable result.")
    return parsed


def content_text(result: dict[str, Any]) -> str:
    return "\n".join(
        item.get("text") or ""
        for item in result.get("content") or []
        if isinstance(item, dict) and item.get("type") == "text"
    )


def to_maps_passage(context: dict[str, Any]) -> MapsPlatformDocPassage:
    """Map a retrieved ``Context`` to a passage."""
    uri = context.get("documentationUri")
    return MapsPlatformDocPassage(
        text=clean_text(context.get("text") or ""),
        url=(uri if "://" in uri else f"https://{uri}") if uri else None,
        relevance_score=context.get("score"),
        api_state=context.get("apiState"),
    )


def clean_text(text: str) -> str:
    """Undo the extra escaping on documentation passages.

    Documentation passages (not GitHub ones) arrive with their line breaks and
    quotes still escaped, as a literal backslash-n, which breaks the code
    samples in them. A passage with real line breaks is left alone.
    """
    if "\n" in text or "\\n" not in text:
        return text
    return _ESCAPE.sub(lambda match: _UNESCAPED[match.group(1)], text)
