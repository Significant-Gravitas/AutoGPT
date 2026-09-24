"""What a block or a workflow run would do, for the gate to decide on.

``None`` is the answer whenever the effect cannot be read — an undeclared
block, an input another node supplies, a sub-graph that is not there — and
every caller treats it as EXTERNAL.
"""

from typing import TYPE_CHECKING, Any, Callable, Iterable, NamedTuple

from backend.blocks._base import Block, BlockEffect, BlockType

if TYPE_CHECKING:
    from backend.data.graph import BaseGraph, GraphModel

BlockInputs = dict[str, Any]


class Resolved(NamedTuple):
    effect: BlockEffect | None
    # The block that decided it: the worst node, or the one nobody can read.
    decided_by: Block | None = None


def block_effect(
    block: Block, inputs: BlockInputs, linked_inputs: Iterable[str] = ()
) -> BlockEffect | None:
    """``linked_inputs`` are the input names another node supplies at run time."""
    if block.block_type in _STRUCTURAL:
        return _STRUCTURAL[block.block_type]
    resolve = _INPUT_DECIDED.get(type(block).__name__)
    if resolve is not None:
        return resolve(inputs, frozenset(linked_inputs))
    return block.effect


def graph_effect(graph: "GraphModel") -> Resolved:
    """The worst effect of any node across a graph and its sub-graphs.

    A write iterated N times is still a write, so the loop structure is
    irrelevant; one node nobody can read makes the whole run unreadable. The
    walk never stops early: an irreversible node anywhere names the run, since
    approving the card lifts the pause that would otherwise have caught it.
    """
    graphs: list["BaseGraph"] = [graph, *graph.sub_graphs]
    present = {g.id for g in graphs}
    worst = Resolved(BlockEffect.NONE)
    unreadable: Block | None = None
    for each in graphs:
        linked = _linked_input_names(each)
        for node in each.nodes:
            block = node.block
            if block.block_type is BlockType.AGENT:
                # Its nodes are counted where the sub-graph itself is walked.
                if node.input_default.get("graph_id") not in present:
                    unreadable = unreadable or block
                continue
            effect = block_effect(
                block, node.input_default, linked.get(node.id, frozenset())
            )
            if effect is None:
                unreadable = unreadable or block
            elif _worse(effect, block, worst):
                worst = Resolved(effect, block)
    if unreadable is None or _irreversible(worst):
        return worst
    return Resolved(None, unreadable)


RANK = {effect: rank for rank, effect in enumerate(BlockEffect)}


def _worse(effect: BlockEffect, block: Block, than: Resolved) -> bool:
    """Among equals an irreversible node names the run: it is what the card warns of."""
    current = than.effect or BlockEffect.NONE
    if RANK[effect] != RANK[current]:
        return RANK[effect] > RANK[current]
    return block.is_irreversible_action and not _irreversible(than)


def _irreversible(resolved: Resolved) -> bool:
    return (
        resolved.decided_by is not None and resolved.decided_by.is_irreversible_action
    )


# Input, output, note and human-in-the-loop nodes do nothing; a trigger receives.
# A nested workflow (AGENT) is resolved through its sub-graph instead.
_STRUCTURAL: dict[BlockType, BlockEffect] = {
    BlockType.INPUT: BlockEffect.NONE,
    BlockType.OUTPUT: BlockEffect.NONE,
    BlockType.NOTE: BlockEffect.NONE,
    BlockType.HUMAN_IN_THE_LOOP: BlockEffect.NONE,
    BlockType.WEBHOOK: BlockEffect.READ,
    BlockType.WEBHOOK_MANUAL: BlockEffect.READ,
}
STRUCTURAL_TYPES = frozenset(_STRUCTURAL) | {BlockType.AGENT}

# Exfiltration by GET stays possible, the same accepted limit ``web_fetch``
# carries; the fix is egress control, not a narrower method list.
_SAFE_HTTP_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _web_request_effect(
    inputs: BlockInputs, linked: frozenset[str]
) -> BlockEffect | None:
    if "method" in linked:
        return None
    method = inputs.get("method", "POST")  # SendWebRequestBlock.Input default
    name = getattr(method, "value", method)
    if not isinstance(name, str):
        return None
    if name.upper() in _SAFE_HTTP_METHODS:
        return BlockEffect.READ
    return BlockEffect.EXTERNAL


def _sql_query_effect(
    inputs: BlockInputs, linked: frozenset[str]
) -> BlockEffect | None:
    if "read_only" in linked:
        return None
    read_only = inputs.get("read_only", True)  # SQLQueryBlock.Input default
    if not isinstance(read_only, bool):
        return None
    return BlockEffect.READ if read_only else BlockEffect.EXTERNAL


_INPUT_DECIDED: dict[
    str, Callable[[BlockInputs, frozenset[str]], BlockEffect | None]
] = {
    "SendWebRequestBlock": _web_request_effect,
    "SendAuthenticatedWebRequestBlock": _web_request_effect,
    "SQLQueryBlock": _sql_query_effect,
}

# Unreadable by construction: what they do lives in code the caller supplies,
# on a remote server, or in another model's choices. The input-decided three
# are here too because a declaration on them would bypass their input.
UNREADABLE_BLOCKS = frozenset(
    {
        "AutoPilotBlock",
        "ClaudeCodeBlock",
        "CodeGenerationBlock",
        "ExecuteCodeBlock",
        "ExecuteCodeStepBlock",
        "InstantiateCodeSandboxBlock",
        "MCPToolBlock",
        "OrchestratorBlock",
        *_INPUT_DECIDED,
    }
)


# Unreadable blocks whose effect is the code the call itself carries: the
# supervisor reads that code, as it reads a bash_exec command. Add a block here
# only when everything it does is in its own input.
JUDGED_BLOCKS = frozenset(
    {
        "ClaudeCodeBlock",
        "ExecuteCodeBlock",
        "ExecuteCodeStepBlock",
        "InstantiateCodeSandboxBlock",
    }
)


def _linked_input_names(graph: "BaseGraph") -> dict[str, frozenset[str]]:
    by_node: dict[str, set[str]] = {}
    for link in graph.links:
        by_node.setdefault(link.sink_id, set()).add(link.sink_name)
    return {node_id: frozenset(names) for node_id, names in by_node.items()}
