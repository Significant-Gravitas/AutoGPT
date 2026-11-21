"""Which model an AI node the agent generator writes gets in place of one the
catalog has switched off, and when an edit keeps the model a node was saved
on. The block's enum still lists a switched-off model, so stored workflows
keep validating, but no new work may start on one."""

import json
from collections import deque
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from backend.data.llm_registry.llm_models import (
    LLMModel,
    is_switched_off,
    llm_model_named,
    scheduled_model_fallback,
)

from .helpers import AgentDict

# A node's block and the model it names, the bucket pass 2 pairs within.
_Key = tuple[str | None, LLMModel | None]


class SwitchedOffMove(BaseModel):
    """A node the fixer moved off a switched-off model, as Otto's note names
    it: by its block and the models as the catalog spells them, never by the
    raw text of the node it came from."""

    model_config = ConfigDict(frozen=True)

    block: str
    old: str
    new: str


def canonical(model: Any) -> str:
    """*model* as the enum spells it, or as given when it names no model."""
    named = llm_model_named(model)
    return str(model) if named is None else named.value


def models_by_node(agent: AgentDict, saved_agent: AgentDict | None) -> dict[str, Any]:
    """The model each node of *agent* was saved on, by its id in *agent*.

    The library gives every node a new id each time it stores a version, so
    an edit made from the stored version's ids carries some of them, and the
    file Otto keeps editing carries none after its first save. In the first
    case a node is its saved self when the stored version has a node of that
    id on the same block; any other node is new or changed work. In the
    second, nothing names the saved selves, so each node is matched to a
    stored node of the same block and model, each stored node at most once
    (``_paired`` says which).
    """
    nodes = _nodes(agent)
    saved_nodes = _nodes(saved_agent)
    if not saved_nodes:
        return {}
    saved_by_id = {node["id"]: node for node in saved_nodes}
    if any(node["id"] in saved_by_id for node in nodes):
        return {
            node["id"]: _model(saved_by_id[node["id"]])
            for node in nodes
            if node["id"] in saved_by_id
            and saved_by_id[node["id"]].get("block_id") == node.get("block_id")
        }
    matched: dict[str, Any] = {}
    stored = _by_key(saved_nodes)
    for key, edited in _by_key(nodes).items():
        # A node that names no model has none to keep.
        if key[1] is not None:
            matched.update(_paired(edited, stored.get(key, [])))
    return matched


def _paired(
    edited: list[dict[str, Any]], stored: list[dict[str, Any]]
) -> dict[str, Any]:
    """Each *stored* node's model given to one node of *edited*, all of them on
    one block and model. A stored node goes first to an edited node whose
    other inputs equal its own, so a copy with new inputs doesn't take the
    model from the node it copies, wherever it sits in the edit. Nodes the
    inputs can't tell apart (several equal, or none) are interchangeable: the
    first in the edit gets the model, and exactly one keeps it per stored
    node."""
    by_inputs: dict[str, deque[dict[str, Any]]] = {}
    for node in edited:
        if (inputs := _inputs(node)) is not None:
            by_inputs.setdefault(inputs, deque()).append(node)
    taken: set[int] = set()
    paired: dict[str, Any] = {}
    unmatched: list[dict[str, Any]] = []
    for saved in stored:
        inputs = _inputs(saved)
        same = by_inputs.get(inputs) if inputs is not None else None
        if same:
            node = same.popleft()
            taken.add(id(node))
            paired[node["id"]] = _model(saved)
        else:
            unmatched.append(saved)
    left = [node for node in edited if id(node) not in taken]
    paired.update((node["id"], _model(saved)) for node, saved in zip(left, unmatched))
    return paired


def _by_key(nodes: Iterable[dict[str, Any]]) -> dict[_Key, list[dict[str, Any]]]:
    grouped: dict[_Key, list[dict[str, Any]]] = {}
    for node in nodes:
        grouped.setdefault(_key(node), []).append(node)
    return grouped


def _nodes(agent: AgentDict | None) -> list[dict[str, Any]]:
    """*agent*'s nodes that are objects with a string id; a malformed node
    is the validator's to report."""
    nodes = (agent or {}).get("nodes")
    if not isinstance(nodes, list):
        return []
    return [
        node
        for node in nodes
        if isinstance(node, dict) and isinstance(node.get("id"), str) and node["id"]
    ]


def _model(node: dict[str, Any]) -> Any:
    defaults = node.get("input_default")
    return defaults.get("model") if isinstance(defaults, dict) else None


def _key(node: dict[str, Any]) -> _Key:
    block_id = node.get("block_id")
    return (block_id if isinstance(block_id, str) else None), llm_model_named(
        _model(node)
    )


def _inputs(node: dict[str, Any]) -> str | None:
    """*node*'s inputs other than its model, as text that is equal for equal
    inputs, so a pool pairs in one pass however large the edit. None when
    they don't serialize, which no node parsed from JSON does."""
    defaults = node.get("input_default")
    if not isinstance(defaults, dict):
        defaults = {}
    others = {key: value for key, value in defaults.items() if key != "model"}
    try:
        return json.dumps(others, sort_keys=True, default=str)
    except (TypeError, ValueError, RecursionError):
        return None


def kept_model(model: Any, saved_model: Any, offered: set[str]) -> str | None:
    """The model an edit keeps, spelled as the block's schema lists it: the
    one its node was saved on, when *model* names it in any spelling the enum
    takes and the block offers it, switched off or not. None otherwise."""
    named = llm_model_named(model)
    if named is None or named != llm_model_named(saved_model):
        return None
    return named.value if named.value in offered else None


def offered_fallback(model: Any, offered: set[str]) -> str | None:
    """A switched-off model's catalog fallback, the model blocks run in its
    place, when the block offers it and the catalog hasn't switched it off
    too. The enum lists switched-off models, so being offered isn't enough."""
    scheduled = scheduled_model_fallback(model)
    if (
        scheduled is None
        or is_switched_off(scheduled[0])
        or scheduled[0].value not in offered
    ):
        return None
    return scheduled[0].value


def live_default(default: str, choices: Sequence[str]) -> str:
    """The block's default model, or, when the catalog has switched it off,
    its offered fallback, else the first of *choices* still switched on."""
    if not is_switched_off(default):
        return default
    return offered_fallback(default, set(choices)) or next(
        (choice for choice in choices if not is_switched_off(choice)), default
    )
