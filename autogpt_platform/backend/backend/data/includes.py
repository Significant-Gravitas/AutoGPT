import prisma

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

GRAPH_EXECUTION_INCLUDE: prisma.types.AgentGraphExecutionInclude = {
    "AgentNodeExecutions": {
        "include": {
            "Input": True,
            "Output": True,
            "AgentNode": True,
            "AgentGraphExecution": True,
        }
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
