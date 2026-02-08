# MCP Block Implementation Plan

## Overview

Create a single **MCPBlock** that dynamically integrates with any MCP (Model Context Protocol)
server. Users provide a server URL, the block discovers available tools, presents them as a
dropdown, and dynamically adjusts input/output schema based on the selected tool — exactly like
`AgentExecutorBlock` handles dynamic schemas.

## Architecture

```
User provides MCP server URL + credentials
         ↓
MCPBlock fetches tools via MCP protocol (tools/list)
         ↓
User selects tool from dropdown (stored in constantInput)
         ↓
Input schema dynamically updates based on selected tool's inputSchema
         ↓
On execution: MCPBlock calls the tool via MCP protocol (tools/call)
         ↓
Result yielded as block output
```

## Design Decisions

1. **Single block, not many blocks** — One `MCPBlock` handles all MCP servers/tools
2. **Dynamic schema via AgentExecutorBlock pattern** — Override `get_input_schema()`,
   `get_input_defaults()`, `get_missing_input()` on the Input class
3. **Auth via API key or OAuth2 credentials** — Use existing `APIKeyCredentials` or
   `OAuth2Credentials` with `ProviderName.MCP` provider. API keys are sent as Bearer tokens;
   OAuth2 uses the access token.
4. **HTTP-based MCP client** — Use `aiohttp` (already a dependency) to implement MCP Streamable
   HTTP transport directly. No need for the `mcp` Python SDK — the protocol is simple JSON-RPC
   over HTTP. Handles both JSON and SSE response formats.
5. **No new DB tables** — Everything fits in existing `AgentBlock` + `AgentNode` tables

## Implementation Files

### New Files
- `backend/blocks/mcp/` — MCP block package
  - `__init__.py`
  - `block.py` — MCPToolBlock implementation
  - `client.py` — MCP HTTP client (list_tools, call_tool)
  - `oauth.py` — MCP OAuth handler for dynamic endpoint discovery
  - `test_mcp.py` — Unit tests
  - `test_oauth.py` — OAuth handler tests
  - `test_integration.py` — Integration tests with local test server
  - `test_e2e.py` — E2E tests against real MCP servers

### Modified Files
- `backend/integrations/providers.py` — Add `MCP = "mcp"` to ProviderName

## Dev Loop

```bash
cd autogpt_platform/backend
poetry run pytest backend/blocks/mcp/test_mcp.py -xvs        # Unit tests
poetry run pytest backend/blocks/mcp/test_oauth.py -xvs       # OAuth tests
poetry run pytest backend/blocks/mcp/test_integration.py -xvs  # Integration tests
poetry run pytest backend/blocks/mcp/ -xvs                     # All MCP tests
```

## Status

- [x] Research & Design
- [x] Add ProviderName.MCP
- [x] Implement MCP client (client.py)
- [x] Implement MCPToolBlock (block.py)
- [x] Add OAuth2 support (oauth.py)
- [x] Write unit tests
- [x] Write integration tests
- [x] Write E2E tests
- [x] Run tests & fix issues
- [x] Create PR
