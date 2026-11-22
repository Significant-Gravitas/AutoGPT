"""The store agents seeded from ``autogpt_platform/backend/agents`` name only
models the catalog serves.

``load-store-agents`` copies each node's ``input_default`` into
``AgentNode.constantInput`` as it is, so a seeded node on a model its provider
has shut down (or the catalog has switched off) fails in every install that
loads it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from backend.data.graph import find_llm_model_fields
from backend.data.llm_registry.catalog import get_catalog

AGENTS_DIR = Path(__file__).parents[3] / "agents"


def _llm_node_models() -> Iterator[tuple[str, str, str]]:
    """(file, node id, model) for every LLM-block node of every seeded agent
    and its sub-graphs."""
    model_fields = find_llm_model_fields()
    for path in sorted(AGENTS_DIR.glob("*.json")):
        agent = json.loads(path.read_text(encoding="utf-8"))
        for graph in [agent, *agent.get("sub_graphs", [])]:
            for node in graph.get("nodes", []):
                field = model_fields.get(node["block_id"])
                inputs = node.get("input_default") or {}
                if field and field in inputs:
                    yield path.name, node["id"], inputs[field]


def test_seeded_agents_use_only_models_the_catalog_serves():
    enabled = {m.slug for m in get_catalog().models if m.is_enabled}
    nodes = list(_llm_node_models())

    assert len(nodes) > 30, "the seeded agents lost most of their LLM nodes"
    stale = [node for node in nodes if node[2] not in enabled]
    assert not stale, f"Seeded LLM nodes on a model the catalog has off: {stale}"
