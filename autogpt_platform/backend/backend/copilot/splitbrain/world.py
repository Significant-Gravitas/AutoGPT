"""The agent-building surface both arms act on.

Real AutoGPT data, no stack: the block catalogue comes from
``get_blocks_as_dicts()`` (573 blocks, real input/output schemas) and
``validate_graph`` runs the production :class:`AgentValidator`. Graph writes go
to memory instead of Postgres, which is the only thing simulated here — the
schemas the model has to read and the errors it has to fix are the real ones,
and those are what drive context growth.
"""

import json
import logging
from typing import Any

from backend.copilot.tools.agent_generator.blocks import get_blocks_as_dicts
from backend.copilot.tools.agent_generator.validator import AgentValidator

logger = logging.getLogger(__name__)

# A find_block page. Ten is what the production tool returns.
_SEARCH_RESULTS = 10
# Descriptions in the search page are trimmed; the full text is in the schema.
_SEARCH_DESC_CHARS = 180


class World:
    """One task's mutable state, shared by every transcript in a run."""

    def __init__(self) -> None:
        self.blocks: list[dict[str, Any]] = get_blocks_as_dicts()
        self._by_id = {b["id"]: b for b in self.blocks}
        self.graph: dict[str, Any] = {"nodes": [], "links": []}
        self.tool_calls: list[str] = []

    def call(self, name: str, args: dict[str, Any]) -> str:
        """Run a work tool and return the JSON string the model sees."""
        self.tool_calls.append(name)
        handler = {
            "find_block": self._find_block,
            "get_block_schema": self._get_block_schema,
            "write_graph": self._write_graph,
            "read_graph": self._read_graph,
            "validate_graph": self._validate_graph,
        }.get(name)
        if handler is None:
            return json.dumps({"error": f"unknown tool {name!r}"})
        try:
            return json.dumps(handler(args), default=str)
        except Exception as exc:  # a tool crash is a tool result, not a run failure
            logger.warning("splitbrain tool %s raised: %s", name, exc)
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    def validate(self) -> tuple[bool, str | None]:
        """The oracle, also exposed as a tool. Real production validator."""
        return AgentValidator().validate(self.graph, self.blocks)

    def _find_block(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).lower().strip()
        terms = [t for t in query.replace("_", " ").split() if t]
        scored: list[tuple[int, dict[str, Any]]] = []
        for block in self.blocks:
            haystack = f"{block['name']} {block.get('description', '')}".lower()
            score = sum(3 if t in block["name"].lower() else 0 for t in terms)
            score += sum(1 for t in terms if t in haystack)
            if score:
                scored.append((score, block))
        scored.sort(key=lambda pair: (-pair[0], pair[1]["name"]))
        hits = [
            {
                "id": b["id"],
                "name": b["name"],
                "description": (b.get("description") or "")[:_SEARCH_DESC_CHARS],
                "ui_type": b.get("uiType"),
            }
            for _, b in scored[:_SEARCH_RESULTS]
        ]
        return {"query": query, "count": len(hits), "blocks": hits}

    def _get_block_schema(self, args: dict[str, Any]) -> dict[str, Any]:
        block = self._by_id.get(str(args.get("block_id", "")))
        if block is None:
            return {"error": "no such block_id; call find_block first"}
        return {
            "id": block["id"],
            "name": block["name"],
            "description": block.get("description"),
            "ui_type": block.get("uiType"),
            "inputSchema": block.get("inputSchema"),
            "outputSchema": block.get("outputSchema"),
        }

    def _write_graph(self, args: dict[str, Any]) -> dict[str, Any]:
        nodes = args.get("nodes")
        links = args.get("links")
        if not isinstance(nodes, list) or not isinstance(links, list):
            return {"error": "nodes and links must both be arrays"}
        self.graph = {"nodes": nodes, "links": links}
        return {
            "written": True,
            "node_count": len(nodes),
            "link_count": len(links),
            "next": "call validate_graph to check it",
        }

    def _read_graph(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.graph

    def _validate_graph(self, args: dict[str, Any]) -> dict[str, Any]:
        valid, errors = self.validate()
        return {"valid": valid, "errors": errors}


WORK_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "find_block",
        "description": (
            "Search the block catalogue by keyword. Returns ids and short "
            "descriptions. Call this before get_block_schema."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_block_schema",
        "description": (
            "Full input and output schema for one block. You need the exact "
            "field names and types from here to wire links correctly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"block_id": {"type": "string"}},
            "required": ["block_id"],
        },
    },
    {
        "name": "write_graph",
        "description": (
            "Replace the working graph. A node is "
            '{"id","block_id","input_default":{}}; a link is '
            '{"id","source_id","sink_id","source_name","sink_name"} where the '
            "names are output/input field names from the block schemas."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "nodes": {"type": "array", "items": {"type": "object"}},
                "links": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["nodes", "links"],
        },
    },
    {
        "name": "read_graph",
        "description": "The working graph as currently stored.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "validate_graph",
        "description": (
            "Validate the working graph. Returns valid=true, or the list of "
            "errors to fix. This is the authority on whether the agent is "
            "correct."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]
