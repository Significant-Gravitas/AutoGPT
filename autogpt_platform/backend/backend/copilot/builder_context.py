"""Builder-session context helpers.

When a copilot session is bound to a builder graph (via
``session.metadata.builder_graph_id``), the assistant needs two distinct
pieces of context:

- A **static** session-long block — the bound graph's id/name and the
  full agent-building guide. This never changes turn-to-turn, so it
  belongs in the *system prompt* where Claude's cross-turn prompt cache
  keeps it warm across the whole session.
- A **dynamic** per-turn snapshot — the current graph version and a
  compact summary of its live nodes and links. The user may edit the
  graph between turns, so this is injected as a ``<builder_context>``
  prefix on every user message.

Splitting the two lets the ~20KB guide live in the cacheable system
prompt while only the small, volatile snapshot rides along with each
user turn — a large prompt-cache win for builder sessions.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.copilot.model import ChatSession
from backend.copilot.tools.agent_generator import get_agent_as_json
from backend.copilot.tools.get_agent_building_guide import _load_guide

logger = logging.getLogger(__name__)


# Tag that wraps the per-turn graph snapshot prepended to user messages.
BUILDER_CONTEXT_TAG = "builder_context"

# Tag that wraps the session-long static block appended to the system prompt.
BUILDER_SESSION_TAG = "builder_session"


# Caps — mirror the frontend ``serializeGraphForChat`` defaults so the
# server-side block stays within a practical token budget even for large
# graphs.
_MAX_NODES = 100
_MAX_LINKS = 200


def _sanitize_for_xml(value: Any) -> str:
    """Escape XML special characters so user-controlled strings cannot break
    out of the context wrappers.

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


def _fetch_failed_turn_prefix() -> str:
    return (
        f"<{BUILDER_CONTEXT_TAG}>\n"
        f"<status>fetch_failed</status>\n"
        f"</{BUILDER_CONTEXT_TAG}>\n\n"
    )


# Guidance embedded in the builder-session block so the LLM picks the
# right run_agent dispatch mode without forcing the user to watch a
# long-blocking call when the builder UI is already live-streaming the
# execution.
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


def _format_session_block(guide: str) -> str:
    """Render the session-long ``<builder_session>`` block.

    Holds only static content (dispatch guidance + building guide). The
    live graph id/name/version move to the per-turn ``<builder_context>``
    prefix — putting anything mutable here would invalidate Claude's
    prompt cache on every rename / version bump / cross-graph session.
    """
    # The guide is trusted server-side content (read from disk). We do NOT
    # escape it — the LLM needs the raw markdown to make sense of block ids,
    # code fences, and example JSON.
    return (
        f"<{BUILDER_SESSION_TAG}>\n"
        f"<run_agent_dispatch_mode>\n"
        f"{_BUILDER_RUN_AGENT_GUIDANCE}\n"
        f"</run_agent_dispatch_mode>\n"
        f"<building_guide>\n{guide}\n</building_guide>\n"
        f"</{BUILDER_SESSION_TAG}>"
    )


async def build_builder_system_prompt_suffix(session: ChatSession) -> str:
    """Return the system-prompt suffix for a builder-bound *session*.

    Returns ``"\\n\\n<builder_session>…</builder_session>"`` when the session
    is bound to a graph and the building guide can be loaded; otherwise
    ``""``. The suffix is byte-identical for every builder session (only
    depends on the on-disk guide), so Claude's prompt cache stays warm
    both across turns and across sessions for different graphs.
    """
    metadata = getattr(session, "metadata", None)
    graph_id = getattr(metadata, "builder_graph_id", None) if metadata else None
    if not graph_id:
        return ""

    try:
        guide = _load_guide()
    except Exception:
        logger.exception("[builder_context] Failed to load agent-building guide")
        return ""

    return "\n\n" + _format_session_block(guide)


async def build_builder_context_turn_prefix(
    session: ChatSession,
    user_id: str | None,
) -> str:
    """Return the per-turn ``<builder_context>`` prefix for *session*.

    Carries all the live, mutable bits: graph id+name, current version,
    node/edge counts, and the capped node + link lists. The static
    building guide lives in :func:`build_builder_system_prompt_suffix`
    so only this (small) block rides along with each user turn — and
    the cacheable system prefix stays identical across turns/sessions.

    Returns ``""`` for non-builder sessions. Returns a
    ``<builder_context><status>fetch_failed</status></builder_context>``
    marker when the graph cannot be fetched.
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
        return _fetch_failed_turn_prefix()

    if not agent_json:
        logger.warning(
            "[builder_context] Graph %s not found for session %s",
            graph_id,
            getattr(session, "session_id", "?"),
        )
        return _fetch_failed_turn_prefix()

    version = _sanitize_for_xml(agent_json.get("version") or "")
    raw_name = agent_json.get("name")
    graph_name = (
        raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else None
    )
    nodes = agent_json.get("nodes") or []
    links = agent_json.get("links") or []
    nodes_block = _format_nodes(nodes)
    links_block = _format_links(links, nodes)
    name_attr = f' name="{_sanitize_for_xml(graph_name)}"' if graph_name else ""
    graph_tag = (
        f'<graph id="{_sanitize_for_xml(graph_id)}"'
        f"{name_attr} "
        f'version="{version}" '
        f'node_count="{len(nodes)}" '
        f'edge_count="{len(links)}"/>'
    )

    inner = f"{graph_tag}\n{nodes_block}\n{links_block}"
    return f"<{BUILDER_CONTEXT_TAG}>\n{inner}\n</{BUILDER_CONTEXT_TAG}>\n\n"
