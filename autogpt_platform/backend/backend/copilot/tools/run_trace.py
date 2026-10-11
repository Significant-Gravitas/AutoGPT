"""A graph run's per-node trace, and what of it came from outside AutoGPT.

A trace repeats each node's static inputs. An LLM block's system prompt and
prompt template are the graph's own (for an agent AutoPilot just built,
AutoPilot's), the same values ``find_library_agent(include_graph=true)``
returns unjudged. The content judge read them as a quoted prompt telling the
reader to act and held the user's first run.

Provenance decides, never text: a node input is left out of the judge's view
only when the graph itself sets that pin, no link feeds it, and the value is
the graph's own. Every block output, every link-fed input and every run input
stays declared whole. Anything this cannot account for (no graph, another
graph version, an unknown node, a malformed entry) declares nothing, so the
whole result is judged.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from backend.blocks import get_block
from backend.blocks.llm import AIBlockBase
from backend.data.dynamic_fields import extract_base_field_name
from backend.data.execution import NodeExecutionResult
from backend.data.graph import GraphModel

logger = logging.getLogger(__name__)

# The LLM blocks' ``prompt`` pin repeats what was sent to the model: the node's
# own inputs, which the trace already shows, and on a structured-output retry
# the model's earlier answers. A dry run simulates it with another LLM. It is
# left out of the trace rather than exempted from the judge.
_ECHO_PIN = "prompt"


def node_trace(node_executions: Sequence[NodeExecutionResult]) -> list[dict[str, Any]]:
    """The per-node trace a model reads: what each node received and produced."""
    return [
        {
            "node_id": ne.node_id,
            "block_id": ne.block_id,
            "status": ne.status.value,
            "input_data": ne.input_data,
            "output_data": _without_echo(ne.block_id, dict(ne.output_data)),
            "start_time": ne.start_time.isoformat() if ne.start_time else None,
            "end_time": ne.end_time.isoformat() if ne.end_time else None,
        }
        for ne in node_executions
    ]


def run_outside(
    graph: GraphModel | None,
    graph_version: int | None,
    trace: list[dict[str, Any]] | None,
    *parts: Any,
) -> tuple[Any, ...] | None:
    """What to declare with ``from_outside``: ``parts`` whole, and the trace
    less the inputs the graph set itself. None declares nothing, so the whole
    result is judged; it is the answer whenever the trace cannot be accounted
    for against ``graph`` at ``graph_version``."""
    if not trace:
        return parts
    try:
        declared = _declared_trace(graph, graph_version, trace)
    except Exception:
        logger.warning("Run trace not accounted for; judged whole", exc_info=True)
        return None
    if declared is None:
        logger.warning("Run trace not accounted for; judged whole")
        return None
    return (*parts, declared)


def _declared_trace(
    graph: GraphModel | None, graph_version: int | None, trace: list[dict[str, Any]]
) -> list[Any] | None:
    if graph is None or graph_version is None or graph.version != graph_version:
        return None
    nodes = {node.id: node for node in graph.nodes}
    linked = {(link.sink_id, _pin(link.sink_name)) for link in graph.links}
    declared: list[Any] = []
    for entry in trace:
        if not isinstance(entry, Mapping):
            return None
        node = nodes.get(str(entry.get("node_id")))
        inputs, outputs = (entry.get("input_data"), entry.get("output_data"))
        if (
            node is None
            or entry.get("block_id") != node.block_id
            or not isinstance(inputs, Mapping)
            or not isinstance(outputs, Mapping)
        ):
            return None
        fed = {
            key: value
            for key, value in inputs.items()
            if (node.id, _pin(key)) in linked
            or key not in node.input_default
            or node.input_default[key] != value
        }
        declared.append([fed, outputs])
    return declared


def _pin(name: str) -> str:
    """The pin a link or input key belongs to: ``values_#_name`` is ``values``."""
    return extract_base_field_name(name)


def _without_echo(block_id: str, outputs: dict[str, Any]) -> dict[str, Any]:
    if isinstance(get_block(block_id), AIBlockBase):
        outputs.pop(_ECHO_PIN, None)
    return outputs
