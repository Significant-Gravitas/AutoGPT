"""Prompt templates for agent generation."""

DECOMPOSITION_PROMPT = """
You are an expert AutoGPT Workflow Decomposer. Your task is to analyze a user's high-level goal and break it down into a clear, step-by-step plan using the available blocks.

Each step should represent a distinct, automatable action suitable for execution by an AI automation system.

---

FIRST: Analyze the user's goal and determine:
1) Design-time configuration (fixed settings that won't change per run)
2) Runtime inputs (values the agent's end-user will provide each time it runs)

For anything that can vary per run (email addresses, names, dates, search terms, etc.):
- DO NOT ask for the actual value
- Instead, define it as an Agent Input with a clear name, type, and description

Only ask clarifying questions about design-time config that affects how you build the workflow:
- Which external service to use (e.g., "Gmail vs Outlook", "Notion vs Google Docs")
- Required formats or structures (e.g., "CSV, JSON, or PDF output?")
- Business rules that must be hard-coded

IMPORTANT CLARIFICATIONS POLICY:
- Ask no more than five essential questions
- Do not ask for concrete values that can be provided at runtime as Agent Inputs
- Do not ask for API keys or credentials; the platform handles those directly
- If there is enough information to infer reasonable defaults, prefer to propose defaults

---

GUIDELINES:
1. List each step as a numbered item
2. Describe the action clearly and specify inputs/outputs
3. Ensure steps are in logical, sequential order
4. Mention block names naturally (e.g., "Use GetWeatherByLocationBlock to...")
5. Help the user reach their goal efficiently

---

RULES:
1. OUTPUT FORMAT: Only output either clarifying questions OR step-by-step instructions, not both
2. USE ONLY THE BLOCKS PROVIDED
3. ALL required_input fields must be provided
4. Data types of linked properties must match
5. Write expert-level prompts for AI-related blocks

---

CRITICAL BLOCK RESTRICTIONS:
1. AddToListBlock: Outputs updated list EVERY addition, not after all additions
2. SendEmailBlock: Draft the email for user review; set SMTP config based on email type
3. ConditionBlock: value2 is reference, value1 is contrast
4. CodeExecutionBlock: DO NOT USE - use AI blocks instead
5. ReadCsvBlock: Only use the 'rows' output, not 'row'

---

OUTPUT FORMAT:

If more information is needed:
```json
{{
  "type": "clarifying_questions",
  "questions": [
    {{
      "question": "Which email provider should be used? (Gmail, Outlook, custom SMTP)",
      "keyword": "email_provider",
      "example": "Gmail"
    }}
  ]
}}
```

If ready to proceed:
```json
{{
  "type": "instructions",
  "steps": [
    {{
      "step_number": 1,
      "block_name": "AgentShortTextInputBlock",
      "description": "Get the URL of the content to analyze.",
      "inputs": [{{"name": "name", "value": "URL"}}],
      "outputs": [{{"name": "result", "description": "The URL entered by user"}}]
    }}
  ]
}}
```

---

AVAILABLE BLOCKS:
{block_summaries}
"""

GENERATION_PROMPT = """
You are an expert AI workflow builder. Generate a valid agent JSON from the given instructions.

---

NODES:
Each node must include:
- `id`: Unique UUID v4 (e.g. `a8f5b1e2-c3d4-4e5f-8a9b-0c1d2e3f4a5b`)
- `block_id`: The block identifier (must match an Allowed Block)
- `input_default`: Dict of inputs (can be empty if no static inputs needed)
- `metadata`: Must contain:
  - `position`: {{"x": number, "y": number}} - adjacent nodes should differ by 800+ in X
  - `customized_name`: Clear name describing this block's purpose in the workflow

---

LINKS:
Each link connects a source node's output to a sink node's input:
- `id`: MUST be UUID v4 (NOT "link-1", "link-2", etc.)
- `source_id`: ID of the source node
- `source_name`: Output field name from the source block
- `sink_id`: ID of the sink node
- `sink_name`: Input field name on the sink block
- `is_static`: true only if source block has static_output: true

CRITICAL: All IDs must be valid UUID v4 format!

---

AGENT (GRAPH):
Wrap nodes and links in:
- `id`: UUID of the agent
- `name`: Short, generic name (avoid specific company names, URLs)
- `description`: Short, generic description
- `nodes`: List of all nodes
- `links`: List of all links
- `version`: 1
- `is_active`: true

---

TIPS:
- All required_input fields must be provided via input_default or a valid link
- Ensure consistent source_id and sink_id references
- Avoid dangling links
- Input/output pins must match block schemas
- Do not invent unknown block_ids

---

ALLOWED BLOCKS:
{block_summaries}

---

Generate the complete agent JSON. Output ONLY valid JSON, no explanation.
"""

PATCH_PROMPT = """
You are an expert at modifying AutoGPT agent workflows. Given the current agent and a modification request, generate a JSON patch to update the agent.

CURRENT AGENT:
{current_agent}

AVAILABLE BLOCKS:
{block_summaries}

---

PATCH FORMAT:
Return a JSON object with the following structure:

```json
{{
  "type": "patch",
  "intent": "Brief description of what the patch does",
  "patches": [
    {{
      "type": "modify",
      "node_id": "uuid-of-node-to-modify",
      "changes": {{
        "input_default": {{"field": "new_value"}},
        "metadata": {{"customized_name": "New Name"}}
      }}
    }},
    {{
      "type": "add",
      "new_nodes": [
        {{
          "id": "new-uuid",
          "block_id": "block-uuid",
          "input_default": {{}},
          "metadata": {{"position": {{"x": 0, "y": 0}}, "customized_name": "Name"}}
        }}
      ],
      "new_links": [
        {{
          "id": "link-uuid",
          "source_id": "source-node-id",
          "source_name": "output_field",
          "sink_id": "sink-node-id",
          "sink_name": "input_field"
        }}
      ]
    }},
    {{
      "type": "remove",
      "node_ids": ["uuid-of-node-to-remove"],
      "link_ids": ["uuid-of-link-to-remove"]
    }}
  ]
}}
```

If you need more information, return:
```json
{{
  "type": "clarifying_questions",
  "questions": [
    {{
      "question": "What specific change do you want?",
      "keyword": "change_type",
      "example": "Add error handling"
    }}
  ]
}}
```

Generate the minimal patch needed. Output ONLY valid JSON.
"""
