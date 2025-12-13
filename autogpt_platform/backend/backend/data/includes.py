from typing import Sequence, cast

import prisma.enums
import prisma.types

AGENT_NODE_INCLUDE: prisma.types.AgentNodeInclude = {
    "Input": True,
    "Output": True,
    "Webhook": True,
    "AgentBlock": True,
}

AGENT_GRAPH_INCLUDE: prisma.types.AgentGraphInclude = {
    "Nodes": {"include": AGENT_NODE_INCLUDE}
}


EXECUTION_RESULT_ORDER: list[prisma.types.AgentNodeExecutionOrderByInput] = [
    {"queuedTime": "desc"},
    # Fallback: Incomplete execs has no queuedTime.
    {"addedTime": "desc"},
]

EXECUTION_RESULT_INCLUDE: prisma.types.AgentNodeExecutionInclude = {
    "Input": {"order_by": {"time": "asc"}},
    "Output": {"order_by": {"time": "asc"}},
    "Node": True,
    "GraphExecution": True,
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
        "include": EXECUTION_RESULT_INCLUDE,
        "order_by": EXECUTION_RESULT_ORDER,
        "take": MAX_NODE_EXECUTIONS_FETCH,  # Avoid loading excessive node executions.
    }
}


def graph_execution_include(
    include_block_ids: Sequence[str],
) -> prisma.types.AgentGraphExecutionInclude:
    return {
        "NodeExecutions": {
            **cast(
                prisma.types.FindManyAgentNodeExecutionArgsFromAgentGraphExecution,
                GRAPH_EXECUTION_INCLUDE_WITH_NODES["NodeExecutions"],  # type: ignore
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
    "InputPresets": True,
    "Webhook": True,
}


INTEGRATION_WEBHOOK_INCLUDE: prisma.types.IntegrationWebhookInclude = {
    "AgentNodes": {"include": AGENT_NODE_INCLUDE},
    "AgentPresets": {"include": AGENT_PRESET_INCLUDE},
}


def library_agent_include(
    user_id: str,
    include_nodes: bool = True,
    include_executions: bool = True,
    execution_limit: int = MAX_LIBRARY_AGENT_EXECUTIONS_FETCH,
) -> prisma.types.LibraryAgentInclude:
    """
    Fully configurable includes for library agent queries with performance optimization.

    Args:
        user_id: User ID for filtering user-specific data
        include_nodes: Whether to include graph nodes (default: True, needed for get_sub_graphs)
        include_executions: Whether to include executions (default: True, safe with execution_limit)
        execution_limit: Limit on executions to fetch (default: MAX_LIBRARY_AGENT_EXECUTIONS_FETCH)

    Defaults maintain backward compatibility and safety - includes everything needed for all functionality.
    For performance optimization, explicitly set include_nodes=False and include_executions=False
    for listing views where frontend fetches data separately.

    Performance impact:
    - Default (full nodes + limited executions): Original performance, works everywhere
    - Listing optimization (no nodes/executions): ~2s for 15 agents vs potential timeouts
    - Unlimited executions: varies by user (thousands of executions = timeouts)
    """
    result: prisma.types.LibraryAgentInclude = {
        "Creator": True,  # Always needed for creator info
    }

    # Build AgentGraph include based on requested options
    if include_nodes or include_executions:
        agent_graph_include = {}

        # Add nodes if requested (always full nodes)
        if include_nodes:
            agent_graph_include.update(AGENT_GRAPH_INCLUDE)  # Full nodes

        # Add executions if requested
        if include_executions:
            agent_graph_include["Executions"] = {
                "where": {"userId": user_id},
                "order_by": {"createdAt": "desc"},
                "take": execution_limit,
            }

        result["AgentGraph"] = cast(
            prisma.types.AgentGraphArgsFromLibraryAgent,
            {"include": agent_graph_include},
        )
    else:
        # Default: Basic metadata only (fast - recommended for most use cases)
        result["AgentGraph"] = True  # Basic graph metadata (name, description, id)

    return result
