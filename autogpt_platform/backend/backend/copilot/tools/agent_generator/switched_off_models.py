"""Which model an AI node the agent generator writes gets in place of one the
catalog has switched off, and when an edit keeps the model a node was saved
on. The block's enum still lists a switched-off model, so stored workflows
keep validating, but no new work may start on one."""

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from backend.data.llm_registry.llm_models import (
    LLMModel,
    is_switched_off,
    llm_model_named,
    scheduled_model_fallback,
)

from .helpers import AgentDict


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

    A node is its saved self when the stored version has a node of that id on
    the same block. The library gives every node a new id each time it stores
    a version, so the ids in a file Otto keeps editing stop matching after its
    first edit. A node whose id the stored version lacks is therefore matched
    to a stored node of the same block and model that the edit no longer
    carries, each stored node at most once.
    """
    nodes = _nodes(agent)
    saved_nodes = _nodes(saved_agent)
    saved_by_id = {node["id"]: node for node in saved_nodes}
    carried = {node["id"] for node in nodes}
    unclaimed: dict[tuple[str | None, LLMModel | None], list[Any]] = {}
    for saved in saved_nodes:
        if saved["id"] not in carried:
            unclaimed.setdefault(_key(saved), []).append(_model(saved))

    matched: dict[str, Any] = {}
    for node in nodes:
        saved = saved_by_id.get(node["id"])
        if saved is not None and saved.get("block_id") == node.get("block_id"):
            matched[node["id"]] = _model(saved)
    for node in nodes:
        if node["id"] not in matched and (same := unclaimed.get(_key(node))):
            matched[node["id"]] = same.pop()
    return matched


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


def _key(node: dict[str, Any]) -> tuple[str | None, LLMModel | None]:
    block_id = node.get("block_id")
    return (block_id if isinstance(block_id, str) else None), llm_model_named(
        _model(node)
    )


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
