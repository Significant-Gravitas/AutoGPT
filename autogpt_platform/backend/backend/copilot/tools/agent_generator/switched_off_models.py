"""Which model an AI node the agent generator writes gets in place of one the
catalog has switched off, and when an edit keeps the model a node was saved
on. The block's enum still lists a switched-off model, so stored workflows
keep validating, but no new work may start on one."""

from collections.abc import Sequence
from typing import Any

from backend.data.llm_registry.llm_models import (
    is_switched_off,
    llm_model_named,
    scheduled_model_fallback,
)

from .helpers import AgentDict


def models_by_node(agent: AgentDict | None) -> dict[str, Any]:
    """Each node's ``model`` input, by node id."""
    return {
        node["id"]: (node.get("input_default") or {}).get("model")
        for node in (agent or {}).get("nodes", [])
        if node.get("id")
    }


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
