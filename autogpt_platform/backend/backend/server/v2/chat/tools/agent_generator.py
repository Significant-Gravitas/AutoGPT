"""Agent Generator - Core logic for creating agents from natural language."""

import json
import logging
import os
import re
import uuid
from typing import Any

from openai import AsyncOpenAI

from backend.data.block import get_blocks
from backend.data.graph import Graph, Link, Node, create_graph
from backend.server.v2.library import db as library_db

logger = logging.getLogger(__name__)

# Configuration - use OPEN_ROUTER_API_KEY for consistency with chat/config.py
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
AGENT_GENERATOR_MODEL = os.getenv("AGENT_GENERATOR_MODEL", "anthropic/claude-opus-4.5")

# OpenRouter client (OpenAI-compatible API)
_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    """Get or create the OpenRouter client."""
    global _client
    if _client is None:
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        _client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
    return _client


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

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


# =============================================================================
# UTILITIES
# =============================================================================


def get_block_summaries() -> str:
    """Generate block summaries for prompts."""
    blocks = get_blocks()
    summaries = []
    for block_id, block_cls in blocks.items():
        block = block_cls()
        name = block.name
        desc = getattr(block, "description", "") or ""
        # Truncate long descriptions
        if len(desc) > 200:
            desc = desc[:197] + "..."
        summaries.append(f"- {name} (id: {block_id}): {desc}")
    return "\n".join(summaries)


def parse_json_from_llm(text: str) -> dict[str, Any] | None:
    """Extract JSON from LLM response (handles markdown code blocks)."""
    if not text:
        return None

    # Try fenced code block
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try raw text
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try finding {...} span
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Try finding [...] span
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


# =============================================================================
# CORE FUNCTIONS
# =============================================================================


async def decompose_goal(description: str, context: str = "") -> dict[str, Any] | None:
    """Break down a goal into steps or return clarifying questions.

    Args:
        description: Natural language goal description
        context: Additional context (e.g., answers to previous questions)

    Returns:
        Dict with either:
        - {"type": "clarifying_questions", "questions": [...]}
        - {"type": "instructions", "steps": [...]}
        Or None on error
    """
    client = get_client()
    prompt = DECOMPOSITION_PROMPT.format(block_summaries=get_block_summaries())

    full_description = description
    if context:
        full_description = f"{description}\n\nAdditional context:\n{context}"

    try:
        response = await client.chat.completions.create(
            model=AGENT_GENERATOR_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": full_description},
            ],
            temperature=0,
        )

        content = response.choices[0].message.content
        if content is None:
            logger.error("LLM returned empty content for decomposition")
            return None

        result = parse_json_from_llm(content)

        if result is None:
            logger.error(f"Failed to parse decomposition response: {content[:200]}")
            return None

        return result

    except Exception as e:
        logger.error(f"Error decomposing goal: {e}")
        return None


async def generate_agent(instructions: dict[str, Any]) -> dict[str, Any] | None:
    """Generate agent JSON from instructions.

    Args:
        instructions: Structured instructions from decompose_goal

    Returns:
        Agent JSON dict or None on error
    """
    client = get_client()
    prompt = GENERATION_PROMPT.format(block_summaries=get_block_summaries())

    try:
        response = await client.chat.completions.create(
            model=AGENT_GENERATOR_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(instructions, indent=2)},
            ],
            temperature=0,
        )

        content = response.choices[0].message.content
        if content is None:
            logger.error("LLM returned empty content for agent generation")
            return None

        result = parse_json_from_llm(content)

        if result is None:
            logger.error(f"Failed to parse agent JSON: {content[:200]}")
            return None

        # Ensure required fields
        if "id" not in result:
            result["id"] = str(uuid.uuid4())
        if "version" not in result:
            result["version"] = 1
        if "is_active" not in result:
            result["is_active"] = True

        return result

    except Exception as e:
        logger.error(f"Error generating agent: {e}")
        return None


def json_to_graph(agent_json: dict[str, Any]) -> Graph:
    """Convert agent JSON dict to Graph model.

    Args:
        agent_json: Agent JSON with nodes and links

    Returns:
        Graph ready for saving
    """
    nodes = []
    for n in agent_json.get("nodes", []):
        node = Node(
            id=n.get("id", str(uuid.uuid4())),
            block_id=n["block_id"],
            input_default=n.get("input_default", {}),
            metadata=n.get("metadata", {}),
        )
        nodes.append(node)

    links = []
    for link_data in agent_json.get("links", []):
        link = Link(
            id=link_data.get("id", str(uuid.uuid4())),
            source_id=link_data["source_id"],
            sink_id=link_data["sink_id"],
            source_name=link_data["source_name"],
            sink_name=link_data["sink_name"],
            is_static=link_data.get("is_static", False),
        )
        links.append(link)

    return Graph(
        id=agent_json.get("id", str(uuid.uuid4())),
        version=agent_json.get("version", 1),
        is_active=agent_json.get("is_active", True),
        name=agent_json.get("name", "Generated Agent"),
        description=agent_json.get("description", ""),
        nodes=nodes,
        links=links,
    )


async def save_agent_to_library(
    agent_json: dict[str, Any], user_id: str
) -> tuple[Graph, Any]:
    """Save agent to database and user's library.

    Args:
        agent_json: Agent JSON dict
        user_id: User ID

    Returns:
        Tuple of (created Graph, LibraryAgent)
    """
    graph = json_to_graph(agent_json)

    # Ensure graph has a unique ID
    if not graph.id or graph.id == "":
        graph.id = str(uuid.uuid4())

    # Save to database
    created_graph = await create_graph(graph, user_id)

    # Add to user's library
    library_agents = await library_db.create_library_agent(
        graph=created_graph,
        user_id=user_id,
        create_library_agents_for_sub_graphs=False,
    )

    return created_graph, library_agents[0]


async def get_agent_as_json(
    graph_id: str, user_id: str | None
) -> dict[str, Any] | None:
    """Fetch an agent and convert to JSON format for editing.

    Args:
        graph_id: Graph ID or library agent ID
        user_id: User ID

    Returns:
        Agent as JSON dict or None if not found
    """
    from backend.data.graph import get_graph

    # Try to get the graph (version=None gets the active version)
    graph = await get_graph(graph_id, version=None, user_id=user_id)
    if not graph:
        return None

    # Convert to JSON format
    nodes = []
    for node in graph.nodes:
        nodes.append(
            {
                "id": node.id,
                "block_id": node.block_id,
                "input_default": node.input_default,
                "metadata": node.metadata,
            }
        )

    links = []
    for node in graph.nodes:
        for link in node.output_links:
            links.append(
                {
                    "id": link.id,
                    "source_id": link.source_id,
                    "sink_id": link.sink_id,
                    "source_name": link.source_name,
                    "sink_name": link.sink_name,
                    "is_static": link.is_static,
                }
            )

    return {
        "id": graph.id,
        "name": graph.name,
        "description": graph.description,
        "version": graph.version,
        "is_active": graph.is_active,
        "nodes": nodes,
        "links": links,
    }


# =============================================================================
# AGENT FIXER - Fixes common LLM generation errors
# =============================================================================

# Block IDs that need special handling
UUID_REGEX = re.compile(
    r"^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$"
)
STORE_VALUE_BLOCK_ID = "1ff065e9-88e8-4358-9d82-8dc91f622ba9"
CONDITION_BLOCK_ID = "715696a0-e1da-45c8-b209-c2fa9c3b0be6"
DOUBLE_CURLY_BRACES_BLOCK_IDS = [
    "44f6c8ad-d75c-4ae1-8209-aad1c0326928",  # FillTextTemplateBlock
    "6ab085e2-20b3-4055-bc3e-08036e01eca6",
    "90f8c45e-e983-4644-aa0b-b4ebe2f531bc",
    "363ae599-353e-4804-937e-b2ee3cef3da4",
    "3b191d9f-356f-482d-8238-ba04b6d18381",
    "db7d8f02-2f44-4c55-ab7a-eae0941f0c30",
    "1f292d4a-41a4-4977-9684-7c8d560b9f91",  # LLM blocks
]


def is_valid_uuid(value: str) -> bool:
    """Check if a string is a valid UUID v4."""
    return isinstance(value, str) and UUID_REGEX.match(value) is not None


def fix_agent_ids(agent: dict[str, Any]) -> dict[str, Any]:
    """Fix invalid UUIDs in agent and link IDs."""
    # Fix agent ID
    if not is_valid_uuid(agent.get("id", "")):
        agent["id"] = str(uuid.uuid4())
        logger.debug(f"Fixed agent ID: {agent['id']}")

    # Fix node IDs
    id_mapping = {}  # Old ID -> New ID
    for node in agent.get("nodes", []):
        if not is_valid_uuid(node.get("id", "")):
            old_id = node.get("id", "")
            new_id = str(uuid.uuid4())
            id_mapping[old_id] = new_id
            node["id"] = new_id
            logger.debug(f"Fixed node ID: {old_id} -> {new_id}")

    # Fix link IDs and update references
    for link in agent.get("links", []):
        if not is_valid_uuid(link.get("id", "")):
            link["id"] = str(uuid.uuid4())
            logger.debug(f"Fixed link ID: {link['id']}")

        # Update source/sink IDs if they were remapped
        if link.get("source_id") in id_mapping:
            link["source_id"] = id_mapping[link["source_id"]]
        if link.get("sink_id") in id_mapping:
            link["sink_id"] = id_mapping[link["sink_id"]]

    return agent


def fix_double_curly_braces(agent: dict[str, Any]) -> dict[str, Any]:
    """Fix single curly braces to double in template blocks."""
    for node in agent.get("nodes", []):
        if node.get("block_id") not in DOUBLE_CURLY_BRACES_BLOCK_IDS:
            continue

        input_data = node.get("input_default", {})
        for key in ("prompt", "format"):
            if key in input_data and isinstance(input_data[key], str):
                original = input_data[key]
                # Fix simple variable references: {var} -> {{var}}
                fixed = re.sub(
                    r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})",
                    r"{{\1}}",
                    original,
                )
                if fixed != original:
                    input_data[key] = fixed
                    logger.debug(f"Fixed curly braces in {key}")

    return agent


def fix_storevalue_before_condition(agent: dict[str, Any]) -> dict[str, Any]:
    """Add StoreValueBlock before ConditionBlock if needed for value2."""
    nodes = agent.get("nodes", [])
    links = agent.get("links", [])

    # Find all ConditionBlock nodes
    condition_node_ids = {
        node["id"] for node in nodes if node.get("block_id") == CONDITION_BLOCK_ID
    }

    if not condition_node_ids:
        return agent

    new_nodes = []
    new_links = []
    processed_conditions = set()

    for link in links:
        sink_id = link.get("sink_id")
        sink_name = link.get("sink_name")

        # Check if this link goes to a ConditionBlock's value2
        if sink_id in condition_node_ids and sink_name == "value2":
            source_node = next(
                (n for n in nodes if n["id"] == link.get("source_id")), None
            )

            # Skip if source is already a StoreValueBlock
            if source_node and source_node.get("block_id") == STORE_VALUE_BLOCK_ID:
                continue

            # Skip if we already processed this condition
            if sink_id in processed_conditions:
                continue

            processed_conditions.add(sink_id)

            # Create StoreValueBlock
            store_node_id = str(uuid.uuid4())
            store_node = {
                "id": store_node_id,
                "block_id": STORE_VALUE_BLOCK_ID,
                "input_default": {"data": None},
                "metadata": {"position": {"x": 0, "y": -100}},
            }
            new_nodes.append(store_node)

            # Create link: original source -> StoreValueBlock
            new_links.append(
                {
                    "id": str(uuid.uuid4()),
                    "source_id": link["source_id"],
                    "source_name": link["source_name"],
                    "sink_id": store_node_id,
                    "sink_name": "input",
                    "is_static": False,
                }
            )

            # Update original link: StoreValueBlock -> ConditionBlock
            link["source_id"] = store_node_id
            link["source_name"] = "output"

            logger.debug(f"Added StoreValueBlock before ConditionBlock {sink_id}")

    if new_nodes:
        agent["nodes"] = nodes + new_nodes

    return agent


def apply_all_fixes(agent: dict[str, Any]) -> dict[str, Any]:
    """Apply all fixes to an agent JSON."""
    agent = fix_agent_ids(agent)
    agent = fix_double_curly_braces(agent)
    agent = fix_storevalue_before_condition(agent)
    return agent


# =============================================================================
# PATCH GENERATION - For editing existing agents
# =============================================================================

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


async def generate_agent_patch(
    update_request: str, current_agent: dict[str, Any]
) -> dict[str, Any] | None:
    """Generate a patch to update an existing agent.

    Args:
        update_request: Natural language description of changes
        current_agent: Current agent JSON

    Returns:
        Patch dict or clarifying questions, or None on error
    """
    client = get_client()
    prompt = PATCH_PROMPT.format(
        current_agent=json.dumps(current_agent, indent=2),
        block_summaries=get_block_summaries(),
    )

    try:
        response = await client.chat.completions.create(
            model=AGENT_GENERATOR_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": update_request},
            ],
            temperature=0,
        )

        content = response.choices[0].message.content
        if content is None:
            logger.error("LLM returned empty content for patch generation")
            return None

        return parse_json_from_llm(content)

    except Exception as e:
        logger.error(f"Error generating patch: {e}")
        return None


def apply_agent_patch(
    current_agent: dict[str, Any], patch: dict[str, Any]
) -> dict[str, Any]:
    """Apply a patch to an existing agent.

    Args:
        current_agent: Current agent JSON
        patch: Patch dict with operations

    Returns:
        Updated agent JSON
    """
    import copy

    agent = copy.deepcopy(current_agent)
    patches = patch.get("patches", [])

    for p in patches:
        patch_type = p.get("type")

        if patch_type == "modify":
            node_id = p.get("node_id")
            changes = p.get("changes", {})

            for node in agent.get("nodes", []):
                if node["id"] == node_id:
                    _deep_update(node, changes)
                    logger.debug(f"Modified node {node_id}")
                    break

        elif patch_type == "add":
            new_nodes = p.get("new_nodes", [])
            new_links = p.get("new_links", [])

            agent["nodes"] = agent.get("nodes", []) + new_nodes
            agent["links"] = agent.get("links", []) + new_links
            logger.debug(f"Added {len(new_nodes)} nodes, {len(new_links)} links")

        elif patch_type == "remove":
            node_ids_to_remove = set(p.get("node_ids", []))
            link_ids_to_remove = set(p.get("link_ids", []))

            # Remove nodes
            agent["nodes"] = [
                n for n in agent.get("nodes", []) if n["id"] not in node_ids_to_remove
            ]

            # Remove links (both explicit and those referencing removed nodes)
            agent["links"] = [
                link
                for link in agent.get("links", [])
                if link["id"] not in link_ids_to_remove
                and link["source_id"] not in node_ids_to_remove
                and link["sink_id"] not in node_ids_to_remove
            ]

            logger.debug(
                f"Removed {len(node_ids_to_remove)} nodes, {len(link_ids_to_remove)} links"
            )

    return agent


def _deep_update(target: dict, source: dict) -> None:
    """Recursively update a dict with another dict."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
