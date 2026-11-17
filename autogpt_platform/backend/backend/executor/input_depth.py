from collections.abc import Iterable, Mapping

from backend.data.block import BlockInput
from backend.data.execution import NodesInputMasks
from backend.data.model import GraphInput
from backend.util.exceptions import GraphValidationError
from backend.util.json import nesting_depth

# Prisma's JSON parser accepts 127 containers, including each column's wrappers.
MAX_STORED_JSON_DEPTH = 127


def validate_graph_input_depth(
    graph_inputs: GraphInput, nodes_input_masks: NodesInputMasks
) -> None:
    for name, value in graph_inputs.items():
        if nesting_depth(value) + 1 > MAX_STORED_JSON_DEPTH:
            raise GraphValidationError(
                f"Workflow input '{name}' is nested too deeply to be saved "
                f"(maximum {MAX_STORED_JSON_DEPTH - 1} levels)"
            )
    _validate_node_input_depth(nodes_input_masks.items(), MAX_STORED_JSON_DEPTH - 2)


def validate_starting_node_input_depth(
    starting_nodes_input: list[tuple[str, BlockInput]],
) -> None:
    _validate_node_input_depth(starting_nodes_input, MAX_STORED_JSON_DEPTH)


def _validate_node_input_depth(
    node_inputs: Iterable[tuple[str, Mapping[str, object]]], max_depth: int
) -> None:
    node_errors = {
        node_id: {
            name: f"Input is nested too deeply to be saved (maximum {max_depth} levels)"
            for name, value in values.items()
            if nesting_depth(value) > max_depth
        }
        for node_id, values in node_inputs
    }
    if node_errors := {
        node_id: errors for node_id, errors in node_errors.items() if errors
    }:
        n_errors = sum(len(errors) for errors in node_errors.values())
        raise GraphValidationError(
            f"Workflow validation failed: {n_errors} node inputs are nested too deeply to be saved",
            node_errors=node_errors,
        )
