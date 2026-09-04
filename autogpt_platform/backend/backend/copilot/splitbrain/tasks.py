"""The tasks both arms are measured on, and the oracle that scores them.

Success is never the model's own claim. A run passes only if the production
``AgentValidator`` accepts the graph AND the graph structurally does what was
asked: the required kinds of block are present and there is a link path from an
input to an output. Both halves are needed — a validator-clean graph of two
disconnected nodes is not the agent anyone asked for.
"""

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from .world import World

BlockPredicate = Callable[[dict[str, Any]], bool]


@dataclass
class TaskSpec:
    key: str
    brief: str
    # (label, predicate) — at least one node's block must satisfy each.
    requires: list[tuple[str, BlockPredicate]] = field(default_factory=list)
    min_nodes: int = 0
    # How many distinct nodes must satisfy a predicate, when one is not enough.
    min_matching: dict[str, int] = field(default_factory=dict)

    def score(self, world: World) -> dict[str, Any]:
        valid, errors = world.validate()
        node_blocks = [
            world._by_id[n["block_id"]]
            for n in world.graph.get("nodes", [])
            if isinstance(n, dict) and n.get("block_id") in world._by_id
        ]
        missing = [
            label
            for label, predicate in self.requires
            if sum(1 for b in node_blocks if predicate(b))
            < self.min_matching.get(label, 1)
        ]
        if len(node_blocks) < self.min_nodes:
            missing.append(f"at least {self.min_nodes} nodes")
        connected = has_input_to_output_path(world)
        return {
            "validator_passed": valid,
            "validator_errors": errors,
            "missing_requirements": missing,
            "input_to_output_path": connected,
            "node_count": len(node_blocks),
            "success": valid and not missing and connected,
        }


def has_input_to_output_path(world: World) -> bool:
    """Is some Input-uiType node linked, forwards, to some Output-uiType node?"""
    nodes = {
        n["id"]: n
        for n in world.graph.get("nodes", [])
        if isinstance(n, dict) and n.get("id")
    }
    ui_type_of = {
        node_id: (world._by_id.get(n.get("block_id", ""), {}) or {}).get("uiType")
        for node_id, n in nodes.items()
    }
    edges: dict[str, list[str]] = {}
    for link in world.graph.get("links", []):
        if isinstance(link, dict) and link.get("source_id") and link.get("sink_id"):
            edges.setdefault(link["source_id"], []).append(link["sink_id"])

    queue = deque(nid for nid, ui in ui_type_of.items() if ui == "Input")
    seen = set(queue)
    while queue:
        for nxt in edges.get(queue.popleft(), []):
            if ui_type_of.get(nxt) == "Output":
                return True
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def named(*patterns: str) -> BlockPredicate:
    compiled = [re.compile(p, re.I) for p in patterns]
    return lambda b: any(c.search(b.get("name", "")) for c in compiled)


def ui_type(value: str) -> BlockPredicate:
    return lambda b: b.get("uiType") == value


TASKS: dict[str, TaskSpec] = {
    "research": TaskSpec(
        key="research",
        brief=(
            "Build an AutoGPT agent that takes a research topic as its input, "
            "searches the web for that topic, has an LLM write a short summary "
            "of what it found, and returns that summary as the agent's output. "
            "The graph must pass validate_graph."
        ),
        requires=[
            ("an input block", ui_type("Input")),
            ("an output block", ui_type("Output")),
            ("a web search block", named("SearchTheWeb", "TavilySearch", "ExaSearch")),
            ("an LLM block", named("^AI.*Block$", "LLM")),
        ],
    ),
    "issues": TaskSpec(
        key="issues",
        brief=(
            "Build an AutoGPT agent that takes a GitHub repository URL as its "
            "input, lists the repository's open issues, has an LLM turn that "
            "list into a short markdown digest, and returns the digest as the "
            "agent's output. The graph must pass validate_graph."
        ),
        requires=[
            ("an input block", ui_type("Input")),
            ("an output block", ui_type("Output")),
            ("a GitHub issue-listing block", named("GithubListIssues")),
            ("an LLM block", named("^AI.*Block$", "LLM")),
        ],
    ),
    # The long-horizon case: enough distinct blocks that a single transcript
    # has to carry several full schemas at once to wire the last link.
    "pipeline": TaskSpec(
        key="pipeline",
        brief=(
            "Build an AutoGPT agent that takes a research topic as its input "
            "and produces a briefing. It must: search the web for the topic; "
            "have one LLM write a prose summary of the results; have a SECOND, "
            "separate LLM node extract the key points from those same results; "
            "combine the summary and the key points into a single piece of "
            "text; and return that combined text as the agent's output. Use at "
            "least six nodes. The graph must pass validate_graph."
        ),
        requires=[
            ("an input block", ui_type("Input")),
            ("an output block", ui_type("Output")),
            ("a web search block", named("SearchTheWeb", "TavilySearch", "ExaSearch")),
            ("two LLM blocks", named("^AI.*Block$", "LLM")),
        ],
        min_nodes=6,
        min_matching={"two LLM blocks": 2},
    ),
}
