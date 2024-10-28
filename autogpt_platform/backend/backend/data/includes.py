import prisma

AGENT_NODE_INCLUDE: prisma.types.AgentNodeInclude = {
    "Input": True,
    "Output": True,
    "Webhook": True,
    "AgentBlock": True,
}

__SUBGRAPH_INCLUDE = {"AgentNodes": {"include": AGENT_NODE_INCLUDE}}

AGENT_GRAPH_INCLUDE: prisma.types.AgentGraphInclude = {
    **__SUBGRAPH_INCLUDE,
    "AgentSubGraphs": {"include": __SUBGRAPH_INCLUDE},  # type: ignore
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
