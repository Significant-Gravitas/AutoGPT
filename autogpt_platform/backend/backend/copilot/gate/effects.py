"""What a block or a workflow run would do, for the auto-mode gate to decide on.

``None`` is the answer whenever the effect cannot be read off the arguments —
an undeclared block, a linked input that decides it, a sub-graph that cannot be
fetched — and every caller must treat it as at least as dangerous as WRITE.

Three blocks take their effect from an input rather than a declaration, so they
carry no ``effect=`` of their own; a fourth class, the sandbox and MCP blocks,
is unresolvable by construction and stays undeclared.
"""

from typing import Any, Callable, Iterable

from backend.blocks._base import Block, BlockEffect

BlockInputs = dict[str, Any]


def block_effect(
    block: Block,
    inputs: BlockInputs,
    linked_inputs: Iterable[str] = (),
) -> BlockEffect | None:
    """``linked_inputs`` are the input names another node supplies at run time."""
    resolve = _INPUT_AWARE.get(type(block).__name__)
    if resolve:
        return resolve(inputs, frozenset(linked_inputs))
    return block.effect


def graph_effect(graphs: Iterable[Any]) -> BlockEffect | None:
    """The worst effect of any node across a graph and its flattened sub-graphs.

    A write iterated N times is still a write, so the loop structure is
    irrelevant; one unresolvable node makes the whole run unresolvable.
    """
    worst = BlockEffect.NONE
    for graph in graphs:
        linked = _linked_input_names(graph)
        for node in graph.nodes:
            effect = block_effect(
                node.block, node.input_default, linked.get(node.id, frozenset())
            )
            if effect is None:
                return None
            if _RANK[effect] > _RANK[worst]:
                worst = effect
    return worst


_RANK = {BlockEffect.NONE: 0, BlockEffect.READ: 1, BlockEffect.WRITE: 2}

# Exfiltration by GET stays possible and is the same accepted limit ``web_fetch``
# already carries; the fix is egress control, not a narrower method list.
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
    return BlockEffect.READ if name.upper() in _SAFE_HTTP_METHODS else BlockEffect.WRITE


def _sql_query_effect(
    inputs: BlockInputs, linked: frozenset[str]
) -> BlockEffect | None:
    if "read_only" in linked:
        return None
    read_only = inputs.get("read_only", True)  # SQLQueryBlock.Input default
    if not isinstance(read_only, bool):
        return None
    return BlockEffect.READ if read_only else BlockEffect.WRITE


_INPUT_AWARE: dict[str, Callable[[BlockInputs, frozenset[str]], BlockEffect | None]] = {
    "SendWebRequestBlock": _web_request_effect,
    "SendAuthenticatedWebRequestBlock": _web_request_effect,
    "SQLQueryBlock": _sql_query_effect,
}

# Unresolvable by construction: the semantics live on a remote server or in
# code the caller supplies. Pinned by a test so a stray declaration cannot
# quietly make one of them run without asking.
UNRESOLVABLE_BLOCKS = frozenset(
    {
        "AgentExecutorBlock",  # a graph run; the caller resolves it with graph_effect
        "ClaudeCodeBlock",
        "CodeGenerationBlock",
        "ExecuteCodeBlock",
        "ExecuteCodeStepBlock",
        "InstantiateCodeSandboxBlock",
        "MCPToolBlock",
    }
)


def _linked_input_names(graph: Any) -> dict[str, frozenset[str]]:
    by_node: dict[str, set[str]] = {}
    for link in graph.links:
        by_node.setdefault(link.sink_id, set()).add(link.sink_name)
    return {node_id: frozenset(names) for node_id, names in by_node.items()}
