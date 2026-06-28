"""Shared schema and input resolution for tools that take a full agent graph.

The full agent graph is the largest argument the assistant ever emits inline,
so on the SDK/OpenRouter path it is frequently truncated or dropped before the
tool ever runs (OPEN-3188). Two mitigations live here:

1. ``AGENT_JSON_SCHEMA`` — a *structured* object schema (``nodes``/``links``/…)
   instead of a bare ``{"type": "object"}``. Constrained tool-argument decoders
   need something to follow; an unconstrained object commonly collapses to
   ``{}`` or is omitted entirely.
2. ``agent_json_ref`` — a string parameter letting the model point at the
   workspace ``agent.json`` file it already wrote (see the agent-generation
   guide) instead of re-emitting the whole graph inline.

``resolve_agent_json_input`` accepts either form and returns the parsed graph,
working on both the SDK path (where the file-ref wrapper has already expanded a
``@@agptfile:`` token to file text) and the baseline path (where it has not).
"""

import json
import os
from typing import Any

from backend.copilot.model import ChatSession
from backend.copilot.sdk.file_ref import (
    FILE_REF_PREFIX,
    parse_file_ref,
    read_file_bytes,
)

# Structured (not bare ``{"type": "object"}``) so constrained tool-arg decoders
# have keys to follow instead of collapsing the value to ``{}``. Nested props are
# kept type-only to stay within the schema char budget.
AGENT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "Full agent graph. For large/existing agents use agent_json_ref instead.",
    "properties": {
        "id": {"type": "string"},
        "version": {"type": "integer"},
        "is_active": {"type": "boolean"},
        "name": {"type": "string"},
        "description": {"type": "string"},
        "nodes": {"type": "array", "items": {"type": "object"}},
        "links": {"type": "array", "items": {"type": "object"}},
    },
    "additionalProperties": True,
}

AGENT_JSON_REF_SCHEMA: dict[str, Any] = {
    "type": "string",
    "description": (
        "Workspace file holding the full agent graph (e.g. "
        "'workspace:///agent.json'). Use instead of agent_json for large/existing "
        "agents so the graph is loaded server-side, not re-emitted inline."
    ),
}


async def resolve_agent_json_input(
    agent_json: Any,
    agent_json_ref: Any,
    user_id: str | None,
    session: ChatSession,
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve the agent graph from ``agent_json`` or ``agent_json_ref``.

    Returns ``(graph, None)`` on success or ``(None, error_message)`` when the
    input is missing or cannot be parsed. ``agent_json`` (inline) wins when both
    are supplied. A returned ``(None, None)`` means neither argument carried a
    usable value — the caller decides which "missing" message to surface.
    """
    inline = _coerce_to_graph(agent_json)
    if inline is not None:
        return inline, None

    if isinstance(agent_json_ref, str) and agent_json_ref.strip():
        return await _resolve_ref(agent_json_ref.strip(), user_id, session)

    return None, None


def _coerce_to_graph(value: Any) -> dict[str, Any] | None:
    """Return a graph dict from an inline object or stringified-JSON object."""
    if isinstance(value, dict) and value:
        return value
    if isinstance(value, str) and value.strip():
        parsed = _try_parse_json(value)
        if isinstance(parsed, dict) and parsed:
            return parsed
    return None


async def _resolve_ref(
    ref: str, user_id: str | None, session: ChatSession
) -> tuple[dict[str, Any] | None, str | None]:
    """Load and parse an ``agent_json_ref`` value into a graph dict.

    Handles three forms: already-expanded JSON text (SDK path), a bare
    ``@@agptfile:`` token (baseline path), and a plain ``workspace://`` URI or
    filename.
    """
    # SDK path: the file-ref wrapper already expanded the token to file text.
    inline = _coerce_to_graph(ref)
    if inline is not None:
        return inline, None

    uri = _ref_to_uri(ref)
    if uri is None:
        return None, (
            f"Could not interpret agent_json_ref '{os.path.basename(ref)[:80]}' "
            "as a workspace file reference. Pass the agent graph as agent_json, "
            "or a 'workspace:///agent.json' reference as agent_json_ref."
        )

    try:
        data = await read_file_bytes(uri, user_id, session)
    except ValueError as exc:
        return None, f"Could not read agent_json_ref file: {exc}"

    parsed = _try_parse_json(data.decode("utf-8", errors="replace"))
    if not isinstance(parsed, dict) or not parsed:
        return None, (
            f"agent_json_ref file did not contain a JSON object: {uri}. Ensure the "
            "file holds the full agent graph before referencing it."
        )
    return parsed, None


def _ref_to_uri(ref: str) -> str | None:
    """Normalise an ``agent_json_ref`` value to a URI ``read_file_bytes`` accepts."""
    if ref.startswith(FILE_REF_PREFIX):
        file_ref = parse_file_ref(ref)
        return file_ref.uri if file_ref is not None else None
    if ref.startswith("workspace://") or ref.startswith("/"):
        return ref
    # Treat a bare value (e.g. "agent.json") as a workspace virtual path.
    return f"workspace:///{ref.lstrip('/')}"


def _try_parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
