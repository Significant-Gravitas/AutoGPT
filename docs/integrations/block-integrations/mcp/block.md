# Mcp Block
<!-- MANUAL: file_description -->
Blocks for connecting to and executing tools on MCP (Model Context Protocol) servers.
<!-- END MANUAL -->

## MCP Tool

### What it is
Connect to any MCP server and execute its tools. Provide a server URL, select a tool, and pass arguments dynamically.

### How it works
<!-- MANUAL: how_it_works -->
The block uses JSON-RPC 2.0 over HTTP to communicate with MCP servers. When configuring, it sends an `initialize` request followed by `tools/list` to discover available tools and their input schemas. On execution, it calls `tools/call` with the selected tool name and arguments, then extracts text, image, or resource content from the response.

Authentication is handled via OAuth 2.0 when the server requires it. The block supports optional credentials — public servers work without authentication, while protected servers trigger a standard OAuth flow with PKCE. Tokens are automatically refreshed when they expire.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| server_url | URL of the MCP server (Streamable HTTP endpoint) | str | Yes |
| selected_tool | The MCP tool to execute | str | No |
| tool_arguments | Arguments to pass to the selected MCP tool. The fields here are defined by the tool's input schema. | Dict[str, Any] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the tool call failed | str |
| result | The result returned by the MCP tool | Result |

### Possible use case
<!-- MANUAL: use_case -->
- **Connecting to third-party APIs**: Use an MCP server like Sentry or Linear to query issues, create tickets, or manage projects without building custom integrations.
- **AI-powered tool execution**: Chain MCP tool calls with AI blocks to let agents dynamically discover and use external tools based on task requirements.
- **Data retrieval from knowledge bases**: Connect to MCP servers like DeepWiki to search documentation, retrieve code context, or query structured knowledge bases.
<!-- END MANUAL -->

---
