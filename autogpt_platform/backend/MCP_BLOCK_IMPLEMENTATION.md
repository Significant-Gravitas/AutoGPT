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
3. **Auth via API key credentials** — Use existing `APIKeyCredentials` with `ProviderName.MCP`
   provider. The API key is sent as Bearer token in the HTTP Authorization header to the MCP
   server. This keeps it simple and uses existing infrastructure.
4. **HTTP-based MCP client** — Use `aiohttp` (already a dependency) to implement MCP Streamable
   HTTP transport directly. No need for the `mcp` Python SDK — the protocol is simple JSON-RPC
   over HTTP.
5. **No new DB tables** — Everything fits in existing `AgentBlock` + `AgentNode` tables

## Implementation Files

### New Files
- `backend/blocks/mcp/` — MCP block package
  - `__init__.py`
  - `block.py` — MCPToolBlock implementation
  - `client.py` — MCP HTTP client (list_tools, call_tool)
  - `test_mcp.py` — Tests (34 tests)

### Modified Files
- `backend/integrations/providers.py` — Add `MCP = "mcp"` to ProviderName
- `pyproject.toml` — No changes needed (using aiohttp which is already a dep)

## Detailed Design

### MCP Client (`client.py`)

Simple async HTTP client for MCP Streamable HTTP protocol:

```python
class MCPClient:
    async def list_tools(server_url: str, headers: dict) -> list[MCPTool]
    async def call_tool(server_url: str, tool_name: str, arguments: dict, headers: dict) -> Any
```

Uses JSON-RPC 2.0 over HTTP POST:
- `tools/list` → `{"jsonrpc": "2.0", "method": "tools/list", "id": 1}`
- `tools/call` → `{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "...", "arguments": {...}}, "id": 2}`

### MCPBlock (`block.py`)

Key fields:
- `server_url: str` — MCP server endpoint URL
- `credentials: MCPCredentialsInput` — API key for auth (optional)
- `available_tools: dict` — Cached tools list from server (populated by frontend API call)
- `selected_tool: str` — Which tool the user selected
- `tool_input_schema: dict` — JSON schema of the selected tool's inputs
- `tool_arguments: dict` — The actual tool arguments (dynamic, validated against tool_input_schema)

Dynamic schema pattern (like AgentExecutorBlock):
```python
@classmethod
def get_input_schema(cls, data: BlockInput) -> dict[str, Any]:
    return data.get("tool_input_schema", {})

@classmethod
def get_input_defaults(cls, data: BlockInput) -> BlockInput:
    return data.get("tool_arguments", {})

@classmethod
def get_missing_input(cls, data: BlockInput) -> set[str]:
    required = cls.get_input_schema(data).get("required", [])
    return set(required) - set(data)
```

### Auth

Use existing `APIKeyCredentials` with provider `"mcp"`:
- User creates an API key credential for their MCP server
- Block sends it as `Authorization: Bearer <key>` header
- Credentials are optional (some MCP servers don't need auth)

## Dev Loop

```bash
cd /Users/majdyz/Code/AutoGPT2/autogpt_platform/backend
poetry run pytest backend/blocks/test/test_mcp_block.py -xvs  # Run MCP-specific tests
poetry run pytest backend/blocks/test/test_block.py -xvs -k "MCP"  # Run block test suite for MCP
```

## Dev Loop

```bash
cd /Users/majdyz/Code/AutoGPT2/autogpt_platform/backend
poetry run pytest backend/blocks/mcp/test_mcp.py -xvs        # Run MCP-specific tests (34 tests)
poetry run pytest backend/blocks/test/test_block.py -xvs -k "MCP"  # Run block test suite for MCP
```

## Status

- [x] Research & Design
- [x] Add ProviderName.MCP
- [x] Implement MCP client (client.py)
- [x] Implement MCPToolBlock (block.py)
- [x] Write unit tests (34 tests — all passing)
- [x] Run tests & fix issues
- [ ] Create PR
