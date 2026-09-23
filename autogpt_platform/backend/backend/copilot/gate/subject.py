"""What a call acts on, when the tool's name alone does not say.

``run_capability`` and ``run_agent`` run a block, a workflow or an MCP tool; the gate
decides on that subject's effect, and the card's headline names it.
"""

import re
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict

from backend.blocks._base import Block, BlockEffect
from backend.copilot.tree import MICRODOLLARS_PER_CREDIT
from backend.executor.utils import block_usage_cost
from backend.integrations.mcp_catalog import mcp_tool_effect

from .effects import block_effect, graph_effect
from .policy import Effect

if TYPE_CHECKING:
    from backend.data.graph import GraphModel


class Subject(BaseModel):
    model_config = ConfigDict(frozen=True)

    # What a chat rule names: ``block:<id>``, ``workflow:<graph id>`` or
    # ``mcp:<host>/<tool>``.
    key: str
    name: str
    effect: Effect
    # The line the card shows under the name; empty where nothing asks.
    reason: str = ""
    # What one call is expected to cost, in microdollars; a paid read asks
    # once the turn's tree is over its ceiling.
    estimate: int = 0


# A schema lookup, ``validate_only``, a dry run, or a trigger workflow's details:
# nothing runs, so nothing asks, whatever the mode.
NO_OP = Subject(key="", name="", effect=Effect.UNGATED)


def block_subject(block: Block, inputs: dict[str, Any]) -> Subject:
    effect = block_effect(block, inputs)
    return Subject(
        key=f"block:{block.id}",
        name=display_name(block),
        effect=_gate_effect(effect),
        reason=_reason(effect, block, culprit=None),
        estimate=_priced(effect, lambda: block_usage_cost(block, inputs)[0]),
    )


def workflow_subject(graph: "GraphModel", *, schedules: bool = False) -> Subject:
    """``schedules``: the call creates a schedule, itself a platform edit."""
    effect, decided_by = graph_effect(graph)
    subject = Subject(
        key=f"workflow:{graph.id}",
        name=graph.name or "Untitled workflow",
        effect=_gate_effect(effect),
        reason=_reason(effect, decided_by, culprit=decided_by),
        estimate=0 if schedules else _priced(effect, lambda: graph_cost_credits(graph)),
    )
    if schedules and subject.effect in (Effect.READ, Effect.WORKSPACE):
        return subject.model_copy(
            update={"effect": Effect.PLATFORM, "reason": "creates a schedule"}
        )
    return subject


def mcp_subject(server_url: str, tool: str) -> Subject:
    """Keyed on the host and the tool, so a rule on one tool leaves the host's
    others asking. Only the server's effect map decides; a tool's name is no
    evidence of what it does."""
    host = urlsplit(server_url).hostname or server_url
    effect = mcp_tool_effect(server_url, tool)
    subject = Subject(
        key=f"mcp:{host}/{tool}",
        name=f"{tool} on {host}",
        effect=Effect.EXTERNAL,
        reason="first use of this tool: its effect is unknown",
    )
    if effect == "read":
        return subject.model_copy(update={"effect": Effect.READ, "reason": ""})
    if effect == "external":
        return subject.model_copy(update={"reason": "reaches outside the platform"})
    if effect == "irreversible":
        return subject.model_copy(update={"reason": "cannot be taken back"})
    return subject


def graph_cost_credits(graph: "GraphModel") -> int:
    """A workflow run's pre-flight estimate: every node once, sub-graphs included."""
    return sum(
        block_usage_cost(node.block, node.input_default)[0]
        for each in (graph, *graph.sub_graphs)
        for node in each.nodes
    )


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


def _priced(effect: BlockEffect | None, credits: Callable[[], int]) -> int:
    # Pure computation never consults money.
    if effect is BlockEffect.NONE:
        return 0
    return credits() * MICRODOLLARS_PER_CREDIT


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
