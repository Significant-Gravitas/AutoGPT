from typing import Sequence, cast

import prisma.enums
import prisma.types

# No "AgentBlock": nothing reads it. `AgentNode.agentBlockId` has its id, and
# the block itself comes from the in-memory registry.
AGENT_NODE_INCLUDE: prisma.types.AgentNodeInclude = {
    "Input": True,
    "Output": True,
    "Webhook": True,
}

AGENT_GRAPH_INCLUDE: prisma.types.AgentGraphInclude = {
    "Nodes": {"include": AGENT_NODE_INCLUDE}
}


EXECUTION_RESULT_ORDER: list[prisma.types.AgentNodeExecutionOrderByInput] = [
    {"queuedTime": "desc"},
    # Fallback: Incomplete execs has no queuedTime.
    {"addedTime": "desc"},
]

# Defensive cap on per-execution Input/Output fanout. Set well above the
# realistic p99 (~tens of rows per node execution) so the cap only kicks in
# on pathological cases — `NodeExecutionResult.from_db` rebuilds input/output
# dicts from these rows, so silently truncating real data would corrupt the
# result. Matches `MAX_NODE_EXECUTIONS_FETCH` for symmetry.
# Writes are held to it too, so a stored value always comes back whole.
MAX_NODE_INPUT_OUTPUT_FETCH = 1000

# For node executions read under their graph execution: the parent is already
# at hand, so it isn't attached again to each of them.
NESTED_EXECUTION_RESULT_INCLUDE: prisma.types.AgentNodeExecutionInclude = {
    "Input": {"order_by": {"time": "asc"}, "take": MAX_NODE_INPUT_OUTPUT_FETCH},
    "Output": {"order_by": {"time": "asc"}, "take": MAX_NODE_INPUT_OUTPUT_FETCH},
    "Node": True,
}

EXECUTION_RESULT_INCLUDE: prisma.types.AgentNodeExecutionInclude = {
    **NESTED_EXECUTION_RESULT_INCLUDE,
    "GraphExecution": True,
}

# For a caller that reads each node once for the run and attaches it itself.
# No "GraphExecution" either: these reads are nested under their parent, which
# `GraphExecution.from_db` hands to each node execution.
EXECUTION_RESULT_INCLUDE_WITHOUT_NODE: prisma.types.AgentNodeExecutionInclude = {
    "Input": {"order_by": {"time": "asc"}, "take": MAX_NODE_INPUT_OUTPUT_FETCH},
    "Output": {"order_by": {"time": "asc"}, "take": MAX_NODE_INPUT_OUTPUT_FETCH},
}

MAX_NODE_EXECUTIONS_FETCH = 1000
MAX_LIBRARY_AGENT_EXECUTIONS_FETCH = 10

# Default limits for potentially large result sets
MAX_CREDIT_REFUND_REQUESTS_FETCH = 100
MAX_INTEGRATION_WEBHOOKS_FETCH = 100
MAX_USER_API_KEYS_FETCH = 500
MAX_GRAPH_VERSIONS_FETCH = 50

GRAPH_EXECUTION_INCLUDE_WITH_NODES: prisma.types.AgentGraphExecutionInclude = {
    "NodeExecutions": {
        "include": NESTED_EXECUTION_RESULT_INCLUDE,
        "order_by": EXECUTION_RESULT_ORDER,
        "take": MAX_NODE_EXECUTIONS_FETCH,  # Avoid loading excessive node executions.
    }
}


def graph_execution_include(
    include_block_ids: Sequence[str],
    node_execution_include: prisma.types.AgentNodeExecutionInclude | None = None,
) -> prisma.types.AgentGraphExecutionInclude:
    return {
        "NodeExecutions": {
            **cast(
                prisma.types.FindManyAgentNodeExecutionArgsFromAgentGraphExecution,
                GRAPH_EXECUTION_INCLUDE_WITH_NODES["NodeExecutions"],  # type: ignore
            ),
            **(
                {"include": node_execution_include}
                if node_execution_include is not None
                else {}
            ),
            "where": {
                "Node": {
                    "is": {"AgentBlock": {"is": {"id": {"in": include_block_ids}}}}
                },
                "NOT": [
                    {"executionStatus": prisma.enums.AgentExecutionStatus.INCOMPLETE}
                ],
            },
        }
    }


AGENT_PRESET_INCLUDE: prisma.types.AgentPresetInclude = {
    "InputPresets": {"order_by": {"time": "asc"}, "take": MAX_NODE_INPUT_OUTPUT_FETCH},
    "Webhook": True,
}


INTEGRATION_WEBHOOK_INCLUDE: prisma.types.IntegrationWebhookInclude = {
    "AgentNodes": {"include": AGENT_NODE_INCLUDE},
    "AgentPresets": {"include": AGENT_PRESET_INCLUDE},
}


LIBRARY_FOLDER_INCLUDE: prisma.types.LibraryFolderInclude = {
    "LibraryAgents": {"where": {"isDeleted": False}},
    "Children": {"where": {"isDeleted": False}},
}


def library_agent_include(
    include_nodes: bool = True,
) -> prisma.types.LibraryAgentInclude:
    """The relations a library agent read loads: its creator, its folder and its
    graph, with the graph's nodes when ``include_nodes`` (``get_sub_graphs``
    needs them).

    The graph's runs are left out, since an include cannot leave out a run's
    input columns; ``attach_latest_runs`` in ``api/features/library/db.py``
    loads the latest few for an agent's status.
    """
    return {
        "Creator": True,
        "Folder": True,
        "AgentGraph": (
            {"include": dict(AGENT_GRAPH_INCLUDE)} if include_nodes else True
        ),
    }
