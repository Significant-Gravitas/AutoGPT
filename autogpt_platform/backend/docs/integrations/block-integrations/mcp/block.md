# Mcp Block
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## MCP Tool

### What it is
Connect to any MCP server and execute its tools. Provide a server URL, select a tool, and pass arguments dynamically.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
