"""The permissions the MCP server's tools can require.

Kept out of the MCP server so OAuth client registration can read them: the
server imports the OAuth metadata, which imports client registration.
"""

from backend.copilot.tools import TOOL_REGISTRY


def mcp_scopes() -> list[str]:
    """Every permission an exposed tool can require."""
    uses = [tool.allow_external_use for tool in TOOL_REGISTRY.values()]
    return sorted({p.value for allowed, perms in uses if allowed for p in perms or []})
