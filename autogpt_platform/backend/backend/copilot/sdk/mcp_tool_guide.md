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

For other services, search the MCP registry API:
```http
GET https://registry.modelcontextprotocol.io/v0/servers?q=<search_term>
```
Each result includes a `remotes` array with the exact server URL to use.

### Important: Check blocks first

Before using `run_mcp_tool`, always check if the platform already has blocks for the service
using `find_block`. The platform has hundreds of built-in blocks (Google Sheets, Google Docs,
Google Calendar, Gmail, etc.) that work without MCP setup.

Only use `run_mcp_tool` when:
- The service is in the known hosted MCP servers list above, OR
- You searched `find_block` first and found no matching blocks

**Never guess or construct MCP server URLs.** Only use URLs from the known servers list above
or from the `remotes[].url` field in MCP registry search results.

### Authentication

If the server requires credentials, a `SetupRequirementsResponse` is returned with an OAuth
login prompt. Once the user completes the flow and confirms, retry the same call immediately.

### Communication style

Avoid technical jargon like "MCP server", "OAuth", or "credentials" when talking to the user.
Use plain, friendly language instead:

| Instead of… | Say… |
|---|---|
| "Let me connect to Sentry's MCP server and discover what tools are available." | "I can connect to Sentry and help identify important issues." |
| "Let me connect to Sentry's MCP server now." | "Next, I'll connect to Sentry." |
| "The MCP server at mcp.sentry.dev requires authentication. Please connect your credentials to continue." | "To continue, sign in to Sentry and approve access." |
| "Sentry's MCP server needs OAuth authentication. You should see a prompt to connect your Sentry account…" | "You should see a prompt to sign in to Sentry. Once connected, I can help surface critical issues right away." |

Use **"connect to [Service]"** or **"sign in to [Service]"** — never "MCP server", "OAuth", or "credentials".
