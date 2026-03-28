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
8. **Dry-run**: ALWAYS call `run_agent` with `dry_run=True` and
   `wait_for_result=120` to verify the agent works end-to-end.
9. **Inspect & fix**: Check the dry-run output for errors. If issues are
   found, call `edit_agent` to fix and dry-run again. Repeat until the
   simulation passes or the problems are clearly unfixable.
   See "REQUIRED: Dry-Run Verification Loop" section below for details.

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
- Optional: `title`, `description`
- Output: `result` — the user-provided value at runtime
- Create one AgentInputBlock per distinct input the agent needs
- For dropdown/select inputs, use **AgentDropdownInputBlock** instead (see below)

**AgentDropdownInputBlock** (ID: `655d6fdf-a334-421c-b733-520549c07cd1`):
- Specialized input block that presents a dropdown/select to the user
- Required `input_default` fields: `name` (str), `placeholder_values` (list of options, must have at least one)
- Optional: `title`, `description`, `value` (default selection)
- Output: `result` — the user-selected value at runtime
- Use this instead of AgentInputBlock when the user should pick from a fixed set of options

**AgentOutputBlock** (ID: `363ae599-353e-4804-937e-b2ee3cef3da4`):
- Defines a user-facing output displayed after the agent runs
- Required `input_default` fields: `name` (str)
- The `value` input should be linked from another block's output
- Optional: `title`, `description`, `format` (Jinja2 template)
- Create one AgentOutputBlock per distinct result to show the user

Without these blocks, the agent has no interface and the user cannot provide
inputs or see outputs. NEVER skip them.

### Execution Model — CRITICAL

Understanding how nodes execute is essential for building correct agents:

1. **A node executes only when ALL linked input pins have received data.**
   If a pin has a link connected to it, that pin becomes mandatory — even
   if the pin is marked "optional" in the block schema. The node will wait
   indefinitely until every linked pin has data.

2. **Each input is consumed once.** When a node executes, it consumes the
   data on its input pins. The data is gone — it cannot be re-read by
   the same node in a subsequent execution. This means a node in a loop
   needs fresh input data on every iteration.

3. **Static links (`is_static: true`) reuse the latest value.** Unlike
   normal links, static links do NOT consume the data. They fetch the
   most recent value each time the node executes. Use static links for:
   - Design-time constants (e.g. a prompt template, API URL)
   - Values that should persist across loop iterations
   - Config that doesn't change between executions

4. **Self-loops are forbidden.** A link where `source_id == sink_id`
   creates a circular dependency — the node waits for its own output
   before it can execute, which never happens. Always verify:
   `link.source_id != link.sink_id` for every link.

5. **Downstream nodes are skipped** if an upstream node fails or never
   executes. Data only flows forward through successfully completed nodes.

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
  Normal links consume data on read; static links reuse the latest value.
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

### Using OrchestratorBlock (AI Orchestrator with Agent Mode)

To create an agent where AI autonomously decides which tools or sub-agents to
call in a loop until the task is complete:
1. Create a `OrchestratorBlock` node
   (ID: `3b191d9f-356f-482d-8238-ba04b6d18381`)
2. Set `input_default`:
   - `agent_mode_max_iterations`: Choose based on task complexity:
     - `1` for single-step tool calls (AI picks one tool, calls it, done)
     - `3`–`10` for multi-step tasks (AI calls tools iteratively)
     - `-1` for open-ended orchestration (AI loops until it decides it's done).
       **Use with caution** — prefer bounded iterations (3–10) unless
       genuinely needed, as unbounded loops risk runaway cost and execution.
     Do NOT use `0` (traditional mode) — it requires complex external
     conversation-history loop wiring that the agent generator does not
     produce.
   - `conversation_compaction`: `true` (recommended to avoid context overflow)
   - `retry`: Number of retries on tool-call failure (default `3`).
     Set to `0` to disable retries.
   - `multiple_tool_calls`: Whether the AI can invoke multiple tools in a
     single turn (default `false`). Enable when tools are independent and
     can run concurrently.
   - Optional: `sys_prompt` for extra LLM context about how to orchestrate
3. Wire the `prompt` input from an `AgentInputBlock` (the user's task)
4. Create downstream tool blocks — regular blocks **or** `AgentExecutorBlock`
   nodes that call sub-agents
5. Link each tool to the Orchestrator: set `source_name: "tools"` on
   the Orchestrator side and `sink_name: <input_field>` on each tool
   block's input. Create one link per input field the tool needs.
6. Wire the `finished` output to an `AgentOutputBlock` for the final result
7. Credentials (LLM API key) are configured by the user in the platform UI
   after saving — do NOT require them upfront

**Example — Orchestrator calling two sub-agents:**
- Node 1: `AgentInputBlock` (input_default: `{"name": "task"}`)
- Node 2: `OrchestratorBlock` (input_default:
  `{"agent_mode_max_iterations": 10, "conversation_compaction": true}`)
- Node 3: `AgentExecutorBlock` (sub-agent A — set `graph_id`, `graph_version`,
  `input_schema`, `output_schema` from library agent)
- Node 4: `AgentExecutorBlock` (sub-agent B — same pattern)
- Node 5: `AgentOutputBlock` (input_default: `{"name": "result"}`)
- Links:
  - Input→Orchestrator: `source_name: "result"`, `sink_name: "prompt"`
  - Orchestrator→Agent A (per input field): `source_name: "tools"`,
    `sink_name: "<agent_a_input_field>"`
  - Orchestrator→Agent B (per input field): `source_name: "tools"`,
    `sink_name: "<agent_b_input_field>"`
  - Orchestrator→Output: `source_name: "finished"`, `sink_name: "value"`

**Example — Orchestrator calling regular blocks as tools:**
- Node 1: `AgentInputBlock` (input_default: `{"name": "task"}`)
- Node 2: `OrchestratorBlock` (input_default:
  `{"agent_mode_max_iterations": 5, "conversation_compaction": true}`)
- Node 3: `GetWebpageBlock` (regular block — the AI calls it as a tool)
- Node 4: `AITextGeneratorBlock` (another regular block as a tool)
- Node 5: `AgentOutputBlock` (input_default: `{"name": "result"}`)
- Links:
  - Input→Orchestrator: `source_name: "result"`, `sink_name: "prompt"`
  - Orchestrator→GetWebpage: `source_name: "tools"`, `sink_name: "url"`
  - Orchestrator→AITextGenerator: `source_name: "tools"`, `sink_name: "prompt"`
  - Orchestrator→Output: `source_name: "finished"`, `sink_name: "value"`

Regular blocks work exactly like sub-agents as tools — wire each input
field from `source_name: "tools"` on the Orchestrator side.

### REQUIRED: Dry-Run Verification Loop (create -> dry-run -> fix)

After creating or editing an agent, you MUST dry-run it before telling the
user the agent is ready. NEVER skip this step.

#### Step-by-step workflow

1. **Create/Edit**: Call `create_agent` or `edit_agent` to save the agent.
2. **Dry-run**: Call `run_agent` with `dry_run=True`, `wait_for_result=120`,
   and realistic sample inputs that exercise every path in the agent. This
   simulates execution using an LLM for each block — no real API calls,
   credentials, or credits are consumed.
3. **Inspect output**: Examine the dry-run result for problems. If
   `wait_for_result` returns only a summary, call
   `view_agent_output(execution_id=..., show_execution_details=True)` to
   see the full node-by-node execution trace. Look for:
   - **Errors / failed nodes** — a node raised an exception or returned an
     error status. Common causes: wrong `source_name`/`sink_name` in links,
     missing `input_default` values, or referencing a nonexistent block output.
   - **Null / empty outputs** — data did not flow through a link. Verify that
     `source_name` and `sink_name` match the block schemas exactly (case-
     sensitive, including nested `_#_` notation).
   - **Nodes that never executed** — the node was not reached. Likely a
     missing or broken link from an upstream node.
   - **Unexpected values** — data arrived but in the wrong type or
     structure. Check type compatibility between linked ports.
4. **Fix**: If any issues are found, call `edit_agent` with the corrected
   agent JSON, then go back to step 2.
5. **Repeat**: Continue the dry-run -> fix cycle until the simulation passes
   or the problems are clearly unfixable. If you stop making progress,
   report the remaining issues to the user and ask for guidance.

#### Correctness metrics

The dry-run result includes a `correctness_score` (0.0–1.0) generated by
evaluating whether the agent achieved its intended goal. Use this as the
**primary success metric**:

- **`correctness_score >= 0.7`** — agent works as intended, ready to ship
- **`correctness_score 0.3–0.7`** — partially working, needs debugging
- **`correctness_score < 0.3`** — fundamentally broken, needs structural fixes

Also check `node_error_count` — but note that **zero errors does NOT mean
the agent is correct**. A block can complete without errors but produce
empty/useless output (e.g. AgentExecutorBlock simulating a sub-agent).

#### Common dry-run failures

| Symptom | Cause | Fix |
|---|---|---|
| `correctness_score: 0.0` with no errors | Block completed but produced no useful output | Check if AgentExecutorBlock sub-agent actually ran; verify output links |
| Node never executed | Missing link from upstream node, or self-loop | Check links; remove any link where `source_id == sink_id` |
| Output node received `null` | Broken `source_name`/`sink_name` in link | Verify names match block schemas exactly (case-sensitive) |
| Status `FAILED` | Node raised an exception | Read the error message; fix input_default or link wiring |

#### Self-loop detection

**NEVER create a link where `source_id` and `sink_id` are the same node.**
This creates a circular dependency that prevents the node from ever executing.
Before saving, verify:
```
for link in agent_json["links"]:
    assert link["source_id"] != link["sink_id"], f"Self-loop on {link['source_id']}"
```

#### Good vs bad dry-run output

**Good output** (agent is ready):
- `correctness_score >= 0.7`
- All nodes executed successfully (no errors in the execution trace)
- Data flows through every link with non-null, correctly-typed values
- The final `AgentOutputBlock` contains a meaningful result
- Status is `COMPLETED`

**Bad output** (needs fixing):
- `correctness_score < 0.3` — even if status is COMPLETED
- Status is `FAILED` — check the error message for the failing node
- An output node received `null` — trace back to find the broken link
- A node received data in the wrong format (e.g. string where list expected)
- Nodes downstream of a failing node were skipped entirely
- AgentExecutorBlock completed but sub-agent produced no output

### Example: Simple AI Text Processor

A minimal agent with input, processing, and output:
- Node 1: `AgentInputBlock` (ID: `c0a8e994-ebf1-4a9c-a4d8-89d09c86741b`,
  input_default: {"name": "user_text", "title": "Text to process"},
  output: "result")
- Node 2: `AITextGeneratorBlock` (input: "prompt" linked from Node 1's "result")
- Node 3: `AgentOutputBlock` (ID: `363ae599-353e-4804-937e-b2ee3cef3da4`,
  input_default: {"name": "summary", "title": "Summary"},
  input: "value" linked from Node 2's output)
