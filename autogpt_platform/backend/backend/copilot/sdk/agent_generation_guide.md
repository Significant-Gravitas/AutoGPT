## Agent Generation Guide

You can create, edit, and customize agents directly. You ARE the brain —
generate the agent JSON yourself using block schemas, then validate and save.

### Workflow for Creating/Editing Agents

1. **Discover blocks**: Call `find_block(query, include_schemas=true)` to
   search for relevant blocks. This returns block IDs, names, descriptions,
   and full input/output schemas.
2. **Find library agents**: Call `find_library_agent` to discover reusable
   agents that can be composed as sub-agents via `AgentExecutorBlock`.
3. **Generate JSON**: Build the agent JSON using block schemas:
   - Use block IDs from step 1 as `block_id` in nodes
   - Wire outputs to inputs using links
   - Set design-time config in `input_default`
   - Use `AgentInputBlock` for values the user provides at runtime
4. **Write to workspace**: Save the JSON to a workspace file so the user
   can review it: `write_workspace_file(filename="agent.json", content=...)`
5. **Validate**: Call `validate_agent_graph` with the agent JSON to check
   for errors
6. **Fix if needed**: Call `fix_agent_graph` to auto-fix common issues,
   or fix manually based on the error descriptions. Iterate until valid.
7. **Save**: Call `create_agent` (new) or `edit_agent` (existing) with
   the final `agent_json`

### Agent JSON Structure

```json
{
  "id": "<UUID v4>",        // auto-generated if omitted
  "version": 1,
  "is_active": true,
  "name": "Agent Name",
  "description": "What the agent does",
  "nodes": [
    {
      "id": "<UUID v4>",
      "block_id": "<block UUID from find_block>",
      "input_default": {
        "field_name": "design-time value"
      },
      "metadata": {
        "position": {"x": 0, "y": 0},
        "customized_name": "Optional display name"
      }
    }
  ],
  "links": [
    {
      "id": "<UUID v4>",
      "source_id": "<source node UUID>",
      "source_name": "output_field_name",
      "sink_id": "<sink node UUID>",
      "sink_name": "input_field_name",
      "is_static": false
    }
  ]
}
```

### REQUIRED: AgentInputBlock and AgentOutputBlock

Every agent MUST include at least one AgentInputBlock and one AgentOutputBlock.
These define the agent's interface — what it accepts and what it produces.

**AgentInputBlock** (ID: `c0a8e994-ebf1-4a9c-a4d8-89d09c86741b`):
- Defines a user-facing input field on the agent
- Required `input_default` fields: `name` (str), `value` (default: null)
- Optional: `title`, `description`, `placeholder_values` (for dropdowns)
- Output: `result` — the user-provided value at runtime
- Create one AgentInputBlock per distinct input the agent needs

**AgentOutputBlock** (ID: `363ae599-353e-4804-937e-b2ee3cef3da4`):
- Defines a user-facing output displayed after the agent runs
- Required `input_default` fields: `name` (str)
- The `value` input should be linked from another block's output
- Optional: `title`, `description`, `format` (Jinja2 template)
- Create one AgentOutputBlock per distinct result to show the user

Without these blocks, the agent has no interface and the user cannot provide
inputs or see outputs. NEVER skip them.

### Key Rules

- **Name & description**: Include `name` and `description` in the agent JSON
  when creating a new agent, or when editing and the agent's purpose changed.
  Without these the agent gets a generic default name.
- **Design-time vs runtime**: `input_default` = values known at build time.
  For user-provided values, create an `AgentInputBlock` node and link its
  output to the consuming block's input.
- **Credentials**: Do NOT require credentials upfront. Users configure
  credentials later in the platform UI after the agent is saved.
- **Node spacing**: Position nodes with at least 800 X-units between them.
- **Nested properties**: Use `parentField_#_childField` notation in link
  sink_name/source_name to access nested object fields.
- **is_static links**: Set `is_static: true` when the link carries a
  design-time constant (matches a field in inputSchema with a default).
- **ConditionBlock**: Needs a `StoreValueBlock` wired to its `value2` input.
- **Prompt templates**: Use `{{variable}}` (double curly braces) for
  literal braces in prompt strings — single `{` and `}` are for
  template variables.
- **AgentExecutorBlock**: When composing sub-agents, set `graph_id` and
  `graph_version` in input_default, and wire inputs/outputs to match
  the sub-agent's schema.

### Using Sub-Agents (AgentExecutorBlock)

To compose agents using other agents as sub-agents:
1. Call `find_library_agent` to find the sub-agent — the response includes
   `graph_id`, `graph_version`, `input_schema`, and `output_schema`
2. Create an `AgentExecutorBlock` node (ID: `e189baac-8c20-45a1-94a7-55177ea42565`)
3. Set `input_default`:
   - `graph_id`: from the library agent's `graph_id`
   - `graph_version`: from the library agent's `graph_version`
   - `input_schema`: from the library agent's `input_schema` (JSON Schema)
   - `output_schema`: from the library agent's `output_schema` (JSON Schema)
   - `user_id`: leave as `""` (filled at runtime)
   - `inputs`: `{}` (populated by links at runtime)
4. Wire inputs: link to sink names matching the sub-agent's `input_schema`
   property names (e.g., if input_schema has a `"url"` property, use
   `"url"` as the sink_name)
5. Wire outputs: link from source names matching the sub-agent's
   `output_schema` property names
6. Pass `library_agent_ids` to `create_agent`/`customize_agent` with
   the library agent IDs used, so the fixer can validate schemas

### Using MCP Tools (MCPToolBlock)

To use an MCP (Model Context Protocol) tool as a node in the agent:
1. The user must specify which MCP server URL and tool name they want
2. Create an `MCPToolBlock` node (ID: `a0a4b1c2-d3e4-4f56-a7b8-c9d0e1f2a3b4`)
3. Set `input_default`:
   - `server_url`: the MCP server URL (e.g. `"https://mcp.example.com/sse"`)
   - `selected_tool`: the tool name on that server
   - `tool_input_schema`: JSON Schema for the tool's inputs
   - `tool_arguments`: `{}` (populated by links or hardcoded values)
4. The block requires MCP credentials — the user configures these in the
   platform UI after the agent is saved
5. Wire inputs using the tool argument field name directly as the sink_name
   (e.g., `query`, NOT `tool_arguments_#_query`). The execution engine
   automatically collects top-level fields matching tool_input_schema into
   tool_arguments.
6. Output: `result` (the tool's return value) and `error` (error message)

### Example: Simple AI Text Processor

A minimal agent with input, processing, and output:
- Node 1: `AgentInputBlock` (ID: `c0a8e994-ebf1-4a9c-a4d8-89d09c86741b`,
  input_default: {"name": "user_text", "title": "Text to process"},
  output: "result")
- Node 2: `AITextGeneratorBlock` (input: "prompt" linked from Node 1's "result")
- Node 3: `AgentOutputBlock` (ID: `363ae599-353e-4804-937e-b2ee3cef3da4`,
  input_default: {"name": "summary", "title": "Summary"},
  input: "value" linked from Node 2's output)
