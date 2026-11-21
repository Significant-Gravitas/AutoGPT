"""Which model an AI node the agent generator writes gets in place of one the
catalog has switched off. The block's enum still lists such a model, so
stored workflows keep validating, but no new work may start on one."""

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


def is_new_switched_off_model(model: Any, saved_model: Any) -> bool:
    """Whether *model* is switched off and not the model its node was saved
    on, in whichever spelling the enum takes."""
    return is_switched_off(model) and (
        llm_model_named(model) != llm_model_named(saved_model)
    )


def offered_fallback(model: Any, offered: set[str]) -> str | None:
    """A switched-off model's catalog fallback, the model blocks run in its
    place, when the block offers it."""
    scheduled = scheduled_model_fallback(model)
    if scheduled is None or scheduled[0].value not in offered:
        return None
    return scheduled[0].value
