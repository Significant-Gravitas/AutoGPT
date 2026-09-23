"""What a call acts on, when the tool's name alone does not say.

``run_capability`` and ``run_agent`` run a block or a workflow; the gate
decides on that subject's effect, and the card's headline names it.
"""

import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from backend.blocks._base import Block, BlockEffect

from .effects import block_effect, graph_effect
from .policy import Effect

if TYPE_CHECKING:
    from backend.data.graph import GraphModel


class Subject(BaseModel):
    model_config = ConfigDict(frozen=True)

    # What a chat rule names: ``block:<id>`` or ``workflow:<graph id>``.
    key: str
    name: str
    effect: Effect
    # The line the card shows under the name; empty where nothing asks.
    reason: str = ""


# A schema lookup, ``validate_only``, a dry run, or a trigger workflow's details:
# nothing runs, so nothing asks, whatever the mode.
NO_OP = Subject(key="", name="", effect=Effect.UNGATED)
# An MCP call keeps its own review until MCP servers carry effect maps.
OWN_REVIEW = Subject(key="", name="", effect=Effect.UNGATED)


def block_subject(block: Block, inputs: dict[str, Any]) -> Subject:
    effect = block_effect(block, inputs)
    return Subject(
        key=f"block:{block.id}",
        name=display_name(block),
        effect=_gate_effect(effect),
        reason=_reason(effect, block, culprit=None),
    )


def workflow_subject(
    graph: "GraphModel", *, schedules: bool = False, saves_preset: bool = False
) -> Subject:
    """``schedules`` and ``saves_preset``: the call also creates a platform
    object, so a read workflow is at least a platform edit."""
    effect, decided_by = graph_effect(graph)
    subject = Subject(
        key=f"workflow:{graph.id}",
        name=graph.name or "Untitled workflow",
        effect=_gate_effect(effect),
        reason=_reason(effect, decided_by, culprit=decided_by),
    )
    creates = "creates a schedule" if schedules else "saves a preset"
    if (schedules or saves_preset) and subject.effect in (
        Effect.READ,
        Effect.WORKSPACE,
    ):
        return subject.model_copy(update={"effect": Effect.PLATFORM, "reason": creates})
    return subject


def display_name(block: Block) -> str:
    """``GmailSendBlock`` -> ``Gmail Send``; the card drops any headline
    containing "Block", so the suffix must go."""
    bare = block.name.removesuffix("Block")
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", bare)


_EFFECTS = {
    BlockEffect.NONE: Effect.READ,
    BlockEffect.READ: Effect.READ,
    BlockEffect.WORKSPACE: Effect.WORKSPACE,
    BlockEffect.PLATFORM: Effect.PLATFORM,
    BlockEffect.EXTERNAL: Effect.EXTERNAL,
}


def _gate_effect(effect: BlockEffect | None) -> Effect:
    # Unreadable is treated as external: it asks wherever an outward act asks.
    return Effect.EXTERNAL if effect is None else _EFFECTS[effect]


def _reason(
    effect: BlockEffect | None, block: Block | None, culprit: Block | None
) -> str:
    step = f": {display_name(culprit)}" if culprit is not None else ""
    if effect is None:
        return f"effect unknown{step}"
    if effect is BlockEffect.EXTERNAL:
        if block is not None and block.is_irreversible_action:
            return f"cannot be taken back{step}"
        return f"reaches outside the platform{step}"
    if effect is BlockEffect.PLATFORM:
        return f"changes your platform objects{step}"
    return ""
