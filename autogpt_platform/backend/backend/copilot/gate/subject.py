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
    # ``mcp:<host><path>::<tool>``.
    key: str
    name: str
    effect: Effect
    # The line the card shows under the name; empty where nothing asks.
    reason: str = ""
    # A judge rule still asks: the supervisor may not wave through what
    # cannot be taken back.
    irreversible: bool = False
    # What one call is expected to cost, in microdollars; a paid read asks
    # once the turn's tree is over its ceiling.
    estimate: int = 0


# A schema lookup, ``validate_only``, a dry run, or a trigger workflow's details:
# nothing runs, so nothing asks, whatever the mode.
NO_OP = Subject(key="", name="", effect=Effect.UNGATED)


def block_subject(block: Block, inputs: dict[str, Any]) -> Subject:
    effect = block_effect(block, inputs)
    name = display_name(block)
    return Subject(
        key=f"block:{block.id}",
        name=name,
        effect=_gate_effect(effect),
        reason=_reason(effect, name, culprit=None),
        irreversible=_irreversible(effect, block),
        estimate=_priced(effect, lambda: block_usage_cost(block, inputs)[0]),
    )


def workflow_subject(
    graph: "GraphModel", *, schedules: bool = False, saves_preset: bool = False
) -> Subject:
    """``schedules`` and ``saves_preset``: the call also creates a platform
    object, so a read workflow is at least a platform edit."""
    effect, decided_by = graph_effect(graph)
    name = graph.name or "Untitled workflow"
    subject = Subject(
        key=f"workflow:{graph.id}",
        name=name,
        effect=_gate_effect(effect),
        reason=_reason(effect, name, culprit=decided_by),
        irreversible=_irreversible(effect, decided_by),
        estimate=0 if schedules else _priced(effect, lambda: graph_cost_credits(graph)),
    )
    creates = (
        f"Runs {name} and creates a schedule."
        if schedules
        else f"Runs {name} and saves a preset."
    )
    if (schedules or saves_preset) and subject.effect in (
        Effect.READ,
        Effect.WORKSPACE,
    ):
        return subject.model_copy(update={"effect": Effect.PLATFORM, "reason": creates})
    return subject


def mcp_subject(server_url: str, tool: str) -> Subject:
    """Keyed on the server (host and path, since one host can serve many) and
    the tool, so a rule on one tool leaves every other asking. Only the
    server's effect map decides; a tool's name is no evidence of what it does."""
    url = urlsplit(server_url)
    host = url.hostname or server_url
    port = f":{url.port}" if url.port not in (None, 443) else ""
    mapped = mcp_tool_effect(server_url, tool)
    name = f"{tool} on {host}"
    if mapped is None:
        effect, reason = Effect.EXTERNAL, f"Its effect is unknown: {name}."
    elif mapped == "read":
        effect, reason = Effect.READ, ""
    else:
        effect = Effect.EXTERNAL
        reason = f"Runs {name}, which reaches outside the platform."
    return Subject(
        key=f"mcp:{host}{port}{url.path.rstrip('/')}::{tool}",
        name=name,
        effect=effect,
        reason=reason,
        irreversible=mapped == "irreversible",
    )


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


def _reason(effect: BlockEffect | None, name: str, culprit: Block | None) -> str:
    """The card's reason line: ``culprit`` is the workflow step that decided."""
    if effect is None:
        step = display_name(culprit) if culprit is not None else name
        return f"Its effect is unknown: {step}."
    does = {
        BlockEffect.EXTERNAL: "reaches outside the platform",
        BlockEffect.PLATFORM: "changes your platform objects",
    }.get(effect)
    if does is None:
        return ""
    if culprit is not None:
        return f"Runs {name}; its step {display_name(culprit)} {does}."
    return f"Runs {name}, which {does}."


def _irreversible(effect: BlockEffect | None, block: Block | None) -> bool:
    return (
        effect is BlockEffect.EXTERNAL
        and block is not None
        and block.is_irreversible_action
    )
