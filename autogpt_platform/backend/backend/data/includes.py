from typing import cast

import prisma.enums
import prisma.types

from backend.blocks.io import IO_BLOCK_IDs
from backend.util.type import typed_cast

AGENT_NODE_INCLUDE: prisma.types.AgentNodeInclude = {
    "Input": True,
    "Output": True,
    "Webhook": True,
    "AgentBlock": True,
}

AGENT_GRAPH_INCLUDE: prisma.types.AgentGraphInclude = {
    "Nodes": {
        "include": typed_cast(
            prisma.types.AgentNodeIncludeFromAgentNodeRecursive1,
            prisma.types.AgentNodeIncludeFromAgentNode,
            AGENT_NODE_INCLUDE,
        )
    }
}

EXECUTION_RESULT_INCLUDE: prisma.types.AgentNodeExecutionInclude = {
    "Input": True,
    "Output": True,
    "Node": True,
    "GraphExecution": True,
}

MAX_NODE_EXECUTIONS_FETCH = 1000

GRAPH_EXECUTION_INCLUDE_WITH_NODES: prisma.types.AgentGraphExecutionInclude = {
    "NodeExecutions": {
        "include": {
            "Input": True,
            "Output": True,
            "Node": True,
            "GraphExecution": True,
        },
        "order_by": [
            {"queuedTime": "desc"},
            # Fallback: Incomplete execs has no queuedTime.
            {"addedTime": "desc"},
        ],
        "take": MAX_NODE_EXECUTIONS_FETCH,  # Avoid loading excessive node executions.
    }
}

GRAPH_EXECUTION_INCLUDE: prisma.types.AgentGraphExecutionInclude = {
    "NodeExecutions": {
        **cast(
            prisma.types.FindManyAgentNodeExecutionArgsFromAgentGraphExecution,
            GRAPH_EXECUTION_INCLUDE_WITH_NODES["NodeExecutions"],
        ),
        "where": {
            "Node": typed_cast(
                prisma.types.AgentNodeRelationFilter,
                prisma.types.AgentNodeWhereInput,
                {
                    "AgentBlock": {"id": {"in": IO_BLOCK_IDs}},
                },
            ),
            "NOT": [{"executionStatus": prisma.enums.AgentExecutionStatus.INCOMPLETE}],
        },
    }
}


INTEGRATION_WEBHOOK_INCLUDE: prisma.types.IntegrationWebhookInclude = {
    "AgentNodes": {
        "include": typed_cast(
            prisma.types.AgentNodeIncludeFromAgentNodeRecursive1,
            prisma.types.AgentNodeInclude,
            AGENT_NODE_INCLUDE,
        )
    }
}


def library_agent_include(user_id: str) -> prisma.types.LibraryAgentInclude:
    return {
        "AgentGraph": {
            "include": {
                **AGENT_GRAPH_INCLUDE,
                "Executions": {"where": {"userId": user_id}},
            }
        },
        "Creator": True,
    }
