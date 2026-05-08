"""Builder-session context helpers — split cacheable system prompt from
the volatile per-turn snapshot so Claude's prompt cache stays warm."""

from __future__ import annotations

import logging
from typing import Any

from backend.copilot.model import ChatSession, ChatSessionInfo
from backend.copilot.permissions import CopilotPermissions
from backend.copilot.tools.agent_generator import get_agent_as_json
from backend.copilot.tools.get_agent_building_guide import _load_guide

logger = logging.getLogger(__name__)


BUILDER_CONTEXT_TAG = "builder_context"
BUILDER_SESSION_TAG = "builder_session"


# Tools hidden from builder-bound sessions: ``create_agent`` /
# ``customize_agent`` would mint a new graph (panel is bound to one),
# and ``get_agent_building_guide`` duplicates bytes already in the
# system-prompt suffix. Everything else (find_block, find_agent, …)
# stays available so the LLM can look up ids instead of hallucinating.
BUILDER_BLOCKED_TOOLS: tuple[str, ...] = (
    "create_agent",
    "customize_agent",
    "get_agent_building_guide",
)


def resolve_session_permissions(
    session: ChatSessionInfo | None,
) -> CopilotPermissions | None:
    """Blacklist :data:`BUILDER_BLOCKED_TOOLS` for builder-bound sessions,
    return ``None`` (unrestricted) otherwise.

    Reads ``metadata.builder_graph_id`` only — works on either the bare
    ``ChatSessionInfo`` (no messages) or the full ``ChatSession``.
    """
    if session is None or not session.metadata.builder_graph_id:
        return None
    return CopilotPermissions(
        tools=list(BUILDER_BLOCKED_TOOLS),
        tools_exclude=True,
    )


# Caps — mirror the frontend ``serializeGraphForChat`` defaults so the
# server-side block stays within a practical token budget for large graphs.
_MAX_NODES = 100
_MAX_LINKS = 200

_FETCH_FAILED_PREFIX = (
    f"<{BUILDER_CONTEXT_TAG}>\n"
    f"<status>fetch_failed</status>\n"
    f"</{BUILDER_CONTEXT_TAG}>\n\n"
)

# Embedded in the cacheable suffix so the LLM picks the right run_agent
# dispatch mode without forcing the user to watch a long-blocking call.
_BUILDER_RUN_AGENT_GUIDANCE = (
    "You are operating inside the builder panel, not the standalone "
    "copilot page. The builder page already subscribes to agent "
    "executions the moment you return an execution_id, so for REAL "
    "(non-dry) runs prefer `run_agent(dry_run=False, wait_for_result=0)` "
    "— the user will see the run stream in the builder's execution panel "
    "in-place and your turn ends immediately with the id. For DRY-RUNS "
    "keep `dry_run=True, wait_for_result=120`: blocking is required so "
    "you can inspect `execution.node_executions` and report the verdict "
    "in the same turn."
)


def _sanitize_for_xml(value: Any) -> str:
    """Escape XML special chars — mirrors ``sanitizeForXml`` in
    ``BuilderChatPanel/helpers.ts``."""
    s = "" if value is None else str(value)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _node_display_name(node: dict[str, Any]) -> str:
    """Prefer the user-set label (``input_default.name`` / ``metadata.title``);
    fall back to the block id."""
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
        return "<nodes>\n</nodes>"
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
    return f"<nodes>\n{body}\n</nodes>"


def _format_links(
    links: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
) -> str:
    if not links:
        return "<links>\n</links>"
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
    return f"<links>\n{body}\n</links>"


async def build_builder_system_prompt_suffix(session: ChatSession) -> str:
    """Return the cacheable system-prompt suffix for a builder session.

    Holds only static content (dispatch guidance + building guide) so the
    bytes are identical across turns AND across sessions for different
    graphs — the live id/name/version ride on the per-turn prefix.
    """
    if not session.metadata.builder_graph_id:
        return ""

    try:
        guide = _load_guide()
    except Exception:
        logger.exception("[builder_context] Failed to load agent-building guide")
        return ""

    # The guide is trusted server-side content (read from disk). We do NOT
    # escape it — the LLM needs the raw markdown to make sense of block ids,
    # code fences, and example JSON.
    return (
        f"\n\n<{BUILDER_SESSION_TAG}>\n"
        f"<run_agent_dispatch_mode>\n"
        f"{_BUILDER_RUN_AGENT_GUIDANCE}\n"
        f"</run_agent_dispatch_mode>\n"
        f"<building_guide>\n{guide}\n</building_guide>\n"
        f"</{BUILDER_SESSION_TAG}>"
    )


async def build_builder_context_turn_prefix(
    session: ChatSession,
    user_id: str | None,
) -> str:
    """Return the per-turn ``<builder_context>`` prefix with the live
    graph snapshot (id/name/version/nodes/links). ``""`` for non-builder
    sessions; fetch-failure marker if the graph cannot be read."""
    graph_id = session.metadata.builder_graph_id
    if not graph_id:
        return ""

    try:
        agent_json = await get_agent_as_json(graph_id, user_id)
    except Exception:
        logger.exception(
            "[builder_context] Failed to fetch graph %s for session %s",
            graph_id,
            session.session_id,
        )
        return _FETCH_FAILED_PREFIX

    if not agent_json:
        logger.warning(
            "[builder_context] Graph %s not found for session %s",
            graph_id,
            session.session_id,
        )
        return _FETCH_FAILED_PREFIX

    version = _sanitize_for_xml(agent_json.get("version") or "")
    raw_name = agent_json.get("name")
    graph_name = (
        raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else None
    )
    nodes = agent_json.get("nodes") or []
    links = agent_json.get("links") or []
    name_attr = f' name="{_sanitize_for_xml(graph_name)}"' if graph_name else ""
    graph_tag = (
        f'<graph id="{_sanitize_for_xml(graph_id)}"'
        f"{name_attr} "
        f'version="{version}" '
        f'node_count="{len(nodes)}" '
        f'edge_count="{len(links)}"/>'
    )

    inner = f"{graph_tag}\n{_format_nodes(nodes)}\n{_format_links(links, nodes)}"
    return f"<{BUILDER_CONTEXT_TAG}>\n{inner}\n</{BUILDER_CONTEXT_TAG}>\n\n"
