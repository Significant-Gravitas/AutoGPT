"""Prompt constants for CoPilot - workflow guidance and supplementary documentation.

This module contains workflow patterns and guidance that supplement the main system prompt.
These are appended dynamically to the prompt along with auto-generated tool documentation.
"""

# Workflow guidance for key tool patterns
# This is appended after the auto-generated tool list to provide usage patterns
KEY_WORKFLOWS = """

## KEY WORKFLOWS

### MCP Integration Workflow
When using `run_mcp_tool`:
1. **Known servers** (use directly): Notion (https://mcp.notion.com/mcp), Linear (https://mcp.linear.app/mcp), Stripe (https://mcp.stripe.com), Intercom (https://mcp.intercom.com/mcp), Cloudflare (https://mcp.cloudflare.com/mcp), Atlassian (https://mcp.atlassian.com/mcp)
2. **Unknown servers**: Use `web_search("{{service}} MCP server URL")` to find the endpoint
3. **Discovery**: Call `run_mcp_tool(server_url)` to see available tools
4. **Execution**: Call `run_mcp_tool(server_url, tool_name, tool_arguments)`
5. **Authentication**: If credentials needed, user will be prompted. When they confirm, retry immediately with same arguments.

### Agent Creation Workflow
When using `create_agent`:
1. Always check `find_library_agent` first for existing solutions
2. Call `create_agent` with description
3. **If `suggested_goal` returned**: Present to user, ask for confirmation, call again with suggested goal if accepted
4. **If `clarifying_questions` returned**: After user answers, call again with original description AND answers in `context` parameter

### Folder Management
Use folder tools (`create_folder`, `list_folders`, `move_agents_to_folder`) to organize agents in the user's library for better discoverability."""
