"""Provider registration for MCP — metadata only."""

from backend.sdk import ProviderBuilder

mcp = ProviderBuilder("mcp").with_description("Model Context Protocol servers").build()
