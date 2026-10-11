"""What of a graph run's result came from outside AutoGPT.

A run's per-node trace repeats what the graph itself says: each node's static
inputs, such as an LLM block's system prompt and prompt template, which an LLM
block also echoes back in its ``prompt`` output. That text is the graph's, the
same text ``find_library_agent(include_graph=true)`` returns unjudged, and for
an agent AutoPilot just built it is AutoPilot's own. The content judge read it
as a quoted prompt telling the reader to act and held the user's first run.

So a trace declares every block's output and every input a link or the run fed
a node, each with the graph's own text cut out of it; what is left is whatever
a block fetched, received or generated. The run's inputs are not the graph's:
the model may have copied them from something it read, so they stay judged.
"""

import re
from collections.abc import Iterable, Mapping
from typing import Any

from backend.data.graph import Graph

# Shorter static strings (a model name, a pin's title) stay in: cut out of a
# block's output, they would shred the page around them for the judge.
_MIN_AUTHORED = 16
# ``{{name}}`` (the platform's own form) or ``{name}`` (a format string).
_PLACEHOLDER = re.compile(r"\{\{[^{}]*\}\}|\{[^{}]*\}")


def run_outside(
    graph: Graph | None,
    node_executions: list[dict[str, Any]] | None,
    *parts: Any,
) -> tuple[Any, ...]:
    """``parts`` and the trace, to declare with ``from_outside``, less the text
    the graph wrote. Without the graph the trace is declared whole; a node it
    does not have keeps every input."""
    if graph is None:
        return (*parts, node_executions)
    nodes = [node for g in (graph, *graph.sub_graphs) for node in g.nodes]
    defaults = {node.id: node.input_default for node in nodes}
    authored = _authored([node.input_default for node in nodes])
    trace = [
        [
            _unset_inputs(ne.get("input_data"), defaults.get(ne.get("node_id", ""))),
            ne.get("output_data"),
        ]
        for ne in node_executions or ()
    ]
    return tuple(_cut(part, authored) for part in (*parts, trace))


def _unset_inputs(inputs: Any, default: Mapping[str, Any] | None) -> Any:
    """The inputs the graph does not set itself: a link or the run fed them."""
    if default is None or not isinstance(inputs, Mapping):
        return inputs
    return {
        key: value
        for key, value in inputs.items()
        if key not in default or default[key] != value
    }


def _authored(values: Iterable[Any]) -> re.Pattern[str] | None:
    """Every long static string, and each long literal run of a template, as
    one pattern, longest first so a whole prompt goes before a run of it."""
    texts = {
        piece
        for text in _strings(values)
        for piece in (text, *_PLACEHOLDER.split(text))
        if len(piece.strip()) >= _MIN_AUTHORED
    }
    if not texts:
        return None
    ordered = sorted(texts, key=len, reverse=True)
    return re.compile("|".join(re.escape(text) for text in ordered))


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _strings(child)


def _cut(value: Any, authored: re.Pattern[str] | None) -> Any:
    """``value`` with every authored string cut out: a string it touches
    becomes the pieces around it, each still a substring the judge can find."""
    if authored is None:
        return value
    if isinstance(value, str):
        pieces = authored.split(value)
        if len(pieces) == 1:
            return value
        return [piece for piece in pieces if piece.strip()]
    if isinstance(value, Mapping):
        return {key: _cut(child, authored) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cut(child, authored) for child in value]
    return value
