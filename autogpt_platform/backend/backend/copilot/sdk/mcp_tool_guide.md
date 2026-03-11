## MCP Tool Guide

### Workflow

`run_mcp_tool` follows a two-step pattern:

1. **Discover** — call with only `server_url` to list available tools on the server.
2. **Execute** — call again with `server_url`, `tool_name`, and `tool_arguments` to run a tool.

### Known hosted MCP servers

Use these URLs directly without asking the user:

| Service | URL |
|---|---|
| Notion | `https://mcp.notion.com/mcp` |
| Linear | `https://mcp.linear.app/mcp` |
| Stripe | `https://mcp.stripe.com` |
| Intercom | `https://mcp.intercom.com/mcp` |
| Cloudflare | `https://mcp.cloudflare.com/mcp` |
| Atlassian / Jira | `https://mcp.atlassian.com/mcp` |

For other services, search the MCP registry at https://registry.modelcontextprotocol.io/.

### Authentication

If the server requires credentials, a `SetupRequirementsResponse` is returned with an OAuth
login prompt. Once the user completes the flow and confirms, retry the same call immediately.
