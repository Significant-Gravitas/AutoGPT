"""Trusted system-prompt context for embedded partner sessions."""

from backend.copilot.model import ChatSession

FORWARDING_DIGITAL_PARTNER_ID = "forwarding-digital"

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
        session.metadata.source_platform == FORWARDING_DIGITAL_PARTNER_ID
        and session.metadata.external_account_id
    ):
        return _FORWARDING_DIGITAL_SUFFIX
    return ""
