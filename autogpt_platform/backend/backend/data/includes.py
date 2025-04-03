import prisma.enums
import prisma.types

from backend.blocks.io import IO_BLOCK_IDs

AGENT_NODE_INCLUDE: prisma.types.AgentNodeInclude = {
    "Input": True,
    "Output": True,
    "Webhook": True,
    "AgentBlock": True,
}

AGENT_GRAPH_INCLUDE: prisma.types.AgentGraphInclude = {
    "AgentNodes": {"include": AGENT_NODE_INCLUDE}  # type: ignore
}

EXECUTION_RESULT_INCLUDE: prisma.types.AgentNodeExecutionInclude = {
    "Input": True,
    "Output": True,
    "AgentNode": True,
    "AgentGraphExecution": True,
}

MAX_NODE_EXECUTIONS_FETCH = 1000

GRAPH_EXECUTION_INCLUDE_WITH_NODES: prisma.types.AgentGraphExecutionInclude = {
    "AgentNodeExecutions": {
        "include": {
            "Input": True,
            "Output": True,
            "AgentNode": True,
            "AgentGraphExecution": True,
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
    "AgentNodeExecutions": {
        **GRAPH_EXECUTION_INCLUDE_WITH_NODES["AgentNodeExecutions"],  # type: ignore
        "where": {
            "AgentNode": {
                "AgentBlock": {"id": {"in": IO_BLOCK_IDs}},  # type: ignore
            },
            "NOT": {
                "executionStatus": prisma.enums.AgentExecutionStatus.INCOMPLETE,
            },
        },
    }
}


INTEGRATION_WEBHOOK_INCLUDE: prisma.types.IntegrationWebhookInclude = {
    "AgentNodes": {"include": AGENT_NODE_INCLUDE}  # type: ignore
}


def library_agent_include(user_id: str) -> prisma.types.LibraryAgentInclude:
    return {
        "Agent": {
            "include": {
                **AGENT_GRAPH_INCLUDE,
                "AgentGraphExecution": {"where": {"userId": user_id}},
            }
        },
        "Creator": True,
    }
