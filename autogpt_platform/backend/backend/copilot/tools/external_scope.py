"""Keep an External API call's tools inside the organization it acts in.

The tools look rows up by user, which is what a chat in the web app wants. An
External API credential is bound to one organization, and the v2 REST API
answers 404 for a row tagged with another one; the same API's MCP tools hide
such rows too. The MCP server checks the ids a call names before the tool
runs (`api/external/v2/mcp_tenancy.py`); these helpers filter what a tool
lists or finds on its own, such as a search by name.
"""

from backend.copilot.model import ChatSession


def external_tenant(session: ChatSession) -> str | None:
    """The organization an External API call is confined to; None for a chat."""
    return session.organization_id if session.external_caller else None


def external_tenancy(session: ChatSession) -> tuple[str | None, str | None]:
    """The organization and team an External API call writes new rows to.

    (None, None) for a chat, whose writes keep their existing defaults.
    """
    if not session.external_caller:
        return None, None
    return session.organization_id, session.team_id


def error_detail(error: BaseException, session: ChatSession) -> str:
    """An unexpected exception's text, for a tool's error response.

    AutoPilot's model reads it to recover; an External API client gets none,
    as the v2 REST API's 500 bodies carry none: it can name internal services,
    queries or a provider's response.
    """
    return "an internal error" if session.external_caller else str(error)


def in_tenant(organization_id: str | None, tenant: str | None) -> bool:
    """Whether a row tagged with ``organization_id`` is visible to ``tenant``.

    Untagged rows predate organization tagging and stay visible to their
    owner, as they do in the REST API.
    """
    return tenant is None or organization_id is None or organization_id == tenant
