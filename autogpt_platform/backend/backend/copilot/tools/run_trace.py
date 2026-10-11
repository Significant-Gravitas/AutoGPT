"""A graph run's per-node trace, and what of it came from outside AutoGPT.

A trace repeats each node's static inputs. An LLM block's system prompt and
prompt template are the graph's own (for an agent AutoPilot just built,
AutoPilot's), the same values ``find_library_agent(include_graph=true)``
returns unjudged. The content judge read them as a quoted prompt telling the
reader to act and held the user's first run.

Provenance decides, never text: a node input is left out of the judge's view
only when the graph itself sets that pin, no link feeds it, the run did not
override it (``nodes_input_masks``), the node is not an input or trigger block
(the run's own inputs land there), and the value is the graph's own. Every
retained block output, every link-fed or overridden input and every run input
stays declared whole. Anything this cannot account for (no graph, another graph
version, an unknown node or block, a malformed entry or override, an error)
declares nothing, so the whole result is judged.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from backend.blocks import get_block
from backend.blocks._base import BlockType
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
# Blocks whose inputs the run supplies: graph inputs, trigger payloads.
_RUN_FED = frozenset({BlockType.INPUT, BlockType.WEBHOOK, BlockType.WEBHOOK_MANUAL})


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
    overrides: Any,
    trace: list[dict[str, Any]] | None,
    *parts: Any,
) -> tuple[Any, ...] | None:
    """What to declare with ``from_outside``: ``parts`` whole, and the trace
    less the inputs the graph set itself. ``overrides`` is the run's
    ``nodes_input_masks``. None declares nothing, so the whole result is
    judged; it is the answer whenever the trace cannot be accounted for against
    ``graph`` at ``graph_version``."""
    if not trace:
        return parts
    try:
        declared = _declared_trace(graph, graph_version, overrides, trace)
    except Exception:
        logger.warning("Run trace not accounted for; judged whole", exc_info=True)
        return None
    if declared is None:
        logger.warning("Run trace not accounted for; judged whole")
        return None
    return (*parts, declared)


def _declared_trace(
    graph: GraphModel | None,
    graph_version: int | None,
    overrides: Any,
    trace: list[dict[str, Any]],
) -> list[Any] | None:
    if graph is None or graph_version is None or graph.version != graph_version:
        return None
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping) or not all(
        isinstance(mask, Mapping) for mask in overrides.values()
    ):
        return None
    nodes = {node.id: node for node in graph.nodes}
    # Pins something other than the graph's own default sets: a link, or the run.
    fed_pins = {(link.sink_id, _pin(link.sink_name)) for link in graph.links}
    fed_pins |= {
        (node_id, _pin(key)) for node_id, mask in overrides.items() for key in mask
    }
    declared: list[Any] = []
    for entry in trace:
        if not isinstance(entry, Mapping):
            return None
        node = nodes.get(str(entry.get("node_id")))
        inputs, outputs = (entry.get("input_data"), entry.get("output_data"))
        block = get_block(node.block_id) if node is not None else None
        if (
            node is None
            or block is None
            or entry.get("block_id") != node.block_id
            or not isinstance(inputs, Mapping)
            or not isinstance(outputs, Mapping)
        ):
            return None
        if block.block_type in _RUN_FED:
            declared.append([inputs, outputs])
            continue
        fed = {
            key: value
            for key, value in inputs.items()
            if (node.id, _pin(key)) in fed_pins
            or key not in node.input_default
            or node.input_default[key] != value
        }
        declared.append([fed, outputs])
    return declared


def _pin(name: str) -> str:
    """The pin a link, override or input key belongs to: ``values_#_name`` is
    ``values``."""
    return extract_base_field_name(name)


def _without_echo(block_id: str, outputs: dict[str, Any]) -> dict[str, Any]:
    """``outputs`` less the LLM echo pin; whole if the block cannot be told."""
    try:
        is_llm = isinstance(get_block(block_id), AIBlockBase)
    except Exception:
        logger.warning(f"Block {block_id} not resolved; its outputs kept whole")
        return outputs
    if is_llm:
        outputs.pop(_ECHO_PIN, None)
    return outputs
