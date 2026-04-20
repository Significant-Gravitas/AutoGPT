"""Builder-session context block helpers.

When a copilot session is bound to a builder graph (via
``session.metadata.builder_graph_id``), every turn's user message is
prefixed with a trusted ``<builder_context>`` block that contains:

- The bound graph's id, name, description, and current version.
- A compact summary of the graph's nodes and links (live snapshot).
- The full agent-building guide text.

This lets the LLM act on the current agent without a per-turn
``get_agent_building_guide`` + ``get_agent_as_json`` round-trip.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.copilot.model import ChatSession
from backend.copilot.tools.agent_generator import get_agent_as_json
from backend.copilot.tools.get_agent_building_guide import _load_guide

logger = logging.getLogger(__name__)


BUILDER_CONTEXT_TAG = "builder_context"

# Caps — mirror the frontend ``serializeGraphForChat`` defaults so the
# server-side block stays within a practical token budget even for large
# graphs.
_MAX_NODES = 100
_MAX_LINKS = 200
_MAX_DESCRIPTION_CHARS = 500


def _sanitize_for_xml(value: Any) -> str:
    """Escape XML special characters so user-controlled strings cannot break
    out of the ``<builder_context>`` wrapper.

    Mirrors the escaping pattern used by the frontend ``sanitizeForXml``
    helper in ``BuilderChatPanel/helpers.ts``.
    """
    s = "" if value is None else str(value)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _node_display_name(node: dict[str, Any]) -> str:
    """Return a short, human-friendly label for a node.

    ``input_default`` often carries a ``name``/``title`` the user set in the
    builder (e.g. for AgentInputBlock / AgentOutputBlock / AgentExecutorBlock).
    When absent, fall back to the block id.
    """
    defaults = node.get("input_default") or {}
    metadata = node.get("metadata") or {}
    for key in ("name", "title", "label"):
        value = defaults.get(key) or metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    block_id = node.get("block_id") or ""
    return block_id or "unknown"


def _format_nodes(nodes: list[dict[str, Any]]) -> str:
    if not nodes:
        return '<nodes count="0"/>'
    visible = nodes[:_MAX_NODES]
    lines = []
    for node in visible:
        node_id = _sanitize_for_xml(node.get("id") or "")
        name = _sanitize_for_xml(_node_display_name(node))
        block_id = _sanitize_for_xml(node.get("block_id") or "")
        lines.append(f"- {node_id}: {name} ({block_id})")
    extra = len(nodes) - len(visible)
    if extra > 0:
        lines.append(f"({extra} more not shown)")
    body = "\n".join(lines)
    return f'<nodes count="{len(nodes)}">\n{body}\n</nodes>'


def _format_links(
    links: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
) -> str:
    if not links:
        return '<links count="0"/>'
    name_by_id = {n.get("id"): _node_display_name(n) for n in nodes}
    visible = links[:_MAX_LINKS]
    lines = []
    for link in visible:
        src_id = link.get("source_id") or ""
        dst_id = link.get("sink_id") or ""
        src_name = name_by_id.get(src_id, src_id)
        dst_name = name_by_id.get(dst_id, dst_id)
        src_out = link.get("source_name") or ""
        dst_in = link.get("sink_name") or ""
        lines.append(
            f"- {_sanitize_for_xml(src_name)}.{_sanitize_for_xml(src_out)} "
            f"-> {_sanitize_for_xml(dst_name)}.{_sanitize_for_xml(dst_in)}"
        )
    extra = len(links) - len(visible)
    if extra > 0:
        lines.append(f"({extra} more not shown)")
    body = "\n".join(lines)
    return f'<links count="{len(links)}">\n{body}\n</links>'


def _format_graph_block(agent_json: dict[str, Any], guide: str) -> str:
    graph_id = _sanitize_for_xml(agent_json.get("id") or "")
    version = _sanitize_for_xml(agent_json.get("version") or "")
    name = _sanitize_for_xml(agent_json.get("name") or "")
    raw_description = agent_json.get("description") or ""
    description = ""
    if isinstance(raw_description, str) and raw_description.strip():
        trimmed = raw_description.strip()[:_MAX_DESCRIPTION_CHARS]
        description = f"<description>{_sanitize_for_xml(trimmed)}</description>\n"

    nodes = agent_json.get("nodes") or []
    links = agent_json.get("links") or []
    nodes_block = _format_nodes(nodes)
    links_block = _format_links(links, nodes)

    # The guide is trusted server-side content (read from disk). We do NOT
    # escape it — the LLM needs the raw markdown to make sense of block ids,
    # code fences, and example JSON.
    guide_block = f"<building_guide>\n{guide}\n</building_guide>"

    inner = (
        f'<graph id="{graph_id}" version="{version}" name="{name}">\n'
        f"{description}"
        f"{nodes_block}\n"
        f"{links_block}\n"
        f"</graph>\n"
        f"{guide_block}"
    )
    return f"<{BUILDER_CONTEXT_TAG}>\n{inner}\n</{BUILDER_CONTEXT_TAG}>\n\n"


def _fetch_failed_block() -> str:
    return (
        f"<{BUILDER_CONTEXT_TAG}>\n"
        f"<status>fetch_failed</status>\n"
        f"</{BUILDER_CONTEXT_TAG}>\n\n"
    )


async def build_builder_context_block(
    session: ChatSession,
    user_id: str | None,
) -> str:
    """Return the per-turn ``<builder_context>`` block for *session*.

    Returns an empty string when the session is not builder-bound.  When the
    graph fetch fails, returns a minimal ``<builder_context><status>fetch_failed
    </status></builder_context>`` marker so the LLM can still reason about its
    binding instead of silently seeing no context.
    """
    metadata = getattr(session, "metadata", None)
    graph_id = getattr(metadata, "builder_graph_id", None) if metadata else None
    if not graph_id:
        return ""

    try:
        agent_json = await get_agent_as_json(graph_id, user_id)
    except Exception:
        logger.exception(
            "[builder_context] Failed to fetch graph %s for session %s",
            graph_id,
            getattr(session, "session_id", "?"),
        )
        return _fetch_failed_block()

    if not agent_json:
        logger.warning(
            "[builder_context] Graph %s not found for session %s",
            graph_id,
            getattr(session, "session_id", "?"),
        )
        return _fetch_failed_block()

    try:
        guide = _load_guide()
    except Exception:
        logger.exception("[builder_context] Failed to load agent-building guide")
        return _fetch_failed_block()

    return _format_graph_block(agent_json, guide)
