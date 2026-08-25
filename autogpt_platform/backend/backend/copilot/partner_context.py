"""Trusted capability and prompt context for embedded partner sessions."""

from typing import Any

from backend.copilot.model import ChatSession

FORWARDING_DIGITAL_PARTNER_ID = "forwarding-digital"

AGENTS_CREATE_CAPABILITY = "agents.create"
AGENTS_RUN_CAPABILITY = "agents.run"
AGENTS_SCHEDULE_CAPABILITY = "agents.schedule"

_PARTNER_BLOCK_PREFIX = "autogpt:block:"

_FORWARDING_DIGITAL_SUFFIX = """

<partner_integration>
You are embedded as the Forwarding Assistant inside Forwarding Digital.
For questions about freight jobs, shipments, arrivals, exceptions, documents,
trade lanes, productivity, revenue, profit, or operational reports, call
`query_forwarding_digital` before answering. Its result comes from the
customer's tenant-bound Forwarding Digital MCP server and is the authoritative
source for operational facts. Never guess operational values, never accept a
tenant or account identifier from the user, and never ask the user to connect
or authenticate the MCP server. The platform has already bound this session to
the signed-in Forwarding Digital account.
</partner_integration>
"""


def build_partner_system_prompt_suffix(session: ChatSession) -> str:
    """Return server-owned instructions for the session's partner integration."""
    if (
        session.metadata.source_platform != FORWARDING_DIGITAL_PARTNER_ID
        or not session.metadata.external_account_id
    ):
        return ""

    capabilities = set(session.metadata.external_capabilities)
    instructions: list[str] = []
    if AGENTS_CREATE_CAPABILITY in capabilities:
        instructions.append(
            "To create or edit an agent, call enter_agent_building_mode first, "
            "discover permitted blocks with find_block, then use create_agent or edit_agent."
        )
    if AGENTS_RUN_CAPABILITY in capabilities:
        instructions.append(
            "To run an agent, resolve it with find_agent or find_library_agent, "
            "then call run_agent and report the returned execution result."
        )
    if AGENTS_SCHEDULE_CAPABILITY in capabilities:
        instructions.append(
            "To schedule an agent, call run_agent with both schedule_name and cron. "
            "Use list_schedules and delete_schedule to manage existing schedules."
        )
    if not instructions:
        return _FORWARDING_DIGITAL_SUFFIX
    lifecycle = "\n".join(instructions)
    return (
        f"{_FORWARDING_DIGITAL_SUFFIX}\n<partner_agent_lifecycle>\n"
        f"{lifecycle}\n"
        "Never claim an agent was created, run, or scheduled without a successful "
        "tool result.\n</partner_agent_lifecycle>\n"
    )


def partner_session_has_capability(session: ChatSession, capability: str) -> bool:
    """Return true outside partner sessions or when the partner granted the cap."""
    return (
        not session.metadata.external_account_id
        or capability in session.metadata.external_capabilities
    )


def partner_allowed_graph_block_ids(session: ChatSession) -> set[str] | None:
    """Return the partner block ceiling, or None for unrestricted sessions."""
    if not session.metadata.external_account_id:
        return None
    return {
        capability.removeprefix(_PARTNER_BLOCK_PREFIX)
        for capability in session.metadata.external_capabilities
        if capability.startswith(_PARTNER_BLOCK_PREFIX)
    }


def partner_disallowed_graph_block_ids(
    session: ChatSession,
    graph: Any,
) -> list[str]:
    """Find graph blocks outside the immutable partner-session allowlist."""
    allowed = partner_allowed_graph_block_ids(session)
    if allowed is None:
        return []

    denied: set[str] = set()
    pending = [graph]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            nodes = current.get("nodes", [])
            pending.extend(current.get("sub_graphs", []))
        else:
            nodes = getattr(current, "nodes", [])
            pending.extend(getattr(current, "sub_graphs", []))
        for node in nodes:
            block_id = (
                node.get("block_id")
                if isinstance(node, dict)
                else getattr(node, "block_id", None)
            )
            if isinstance(block_id, str) and block_id not in allowed:
                denied.add(block_id)
    return sorted(denied)
