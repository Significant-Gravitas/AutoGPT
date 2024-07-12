import time

import pytest

from autogpt_server.blocks.sample import PrintingBlock
from autogpt_server.data import block, db, execution, graph
from autogpt_server.data.agent_block import AutoGPTAgentBlock
from autogpt_server.executor import ExecutionManager
from autogpt_server.server import AgentServer
from autogpt_server.util.service import PyroNameServer


async def create_test_graph() -> graph.Graph:
    """
    AgentBlock ---- PrintingBlock
    """
    nodes = [
        graph.Node(block_id=AutoGPTAgentBlock().id),
        graph.Node(block_id=PrintingBlock().id),
    ]
    links = [
        graph.Link(nodes[0].id, nodes[1].id, "result", "text"),
    ]

    test_graph = graph.Graph(
        name="TestGraph",
        description="Test graph",
        nodes=nodes,
        links=links,
    )
    result = await graph.create_graph(test_graph)

    # Assertions
    assert result.name == test_graph.name
    assert result.description == test_graph.description
    assert len(result.nodes) == len(test_graph.nodes)

    return test_graph


async def execute_agent(test_manager: ExecutionManager, test_graph: graph.Graph):
    input_data = {
        "task": "Make calculations and use output command to output the result.",
        "input": "5 + 3",
        "disabled_components": [
            "ActionHistoryComponent",
            "UserInteractionComponent",
            "FileManagerComponent",
            "CodeExecutorComponent",
            "GitOperationsComponent",
            "ImageGeneratorComponent",
            "ContextComponent",
        ],
        "disabled_commands": ["finish"],
        "fast_mode": True,
    }
    agent_server = AgentServer()
    response = await agent_server.execute_graph(test_graph.id, input_data)
    executions = response["executions"]
    graph_exec_id = response["id"]
    assert len(executions) == 1

    async def is_execution_completed():
        execs = await agent_server.get_run_execution_results(
            test_graph.id, graph_exec_id
        )
        return test_manager.queue.empty() and len(execs) == 2

    # Wait for the executions to complete
    for _ in range(20):
        if await is_execution_completed():
            break
        time.sleep(1)

    # Execution queue should be empty
    assert await is_execution_completed()

    executions = await agent_server.get_run_execution_results(
        test_graph.id, graph_exec_id
    )

    # Executing AgentBlock
    exec = executions[0]
    assert exec.status == execution.ExecutionStatus.COMPLETED
    assert exec.graph_exec_id == graph_exec_id
    assert "8" in exec.output_data["result"][0]
    assert exec.input_data == input_data
    assert exec.node_id == test_graph.nodes[0].id

    # Executing PrintingBlock
    exec = executions[1]
    assert exec.status == execution.ExecutionStatus.COMPLETED
    assert exec.graph_exec_id == graph_exec_id
    assert exec.output_data["status"] == ["printed"]
    assert "8" in exec.input_data["text"]
    assert exec.node_id == test_graph.nodes[1].id


@pytest.mark.asyncio(scope="session")
async def test_autogpt_agent():
    with PyroNameServer():
        with AgentServer():
            with ExecutionManager(1) as test_manager:
                await db.connect()
                await block.initialize_blocks()
                test_graph = await create_test_graph()
                await execute_agent(test_manager, test_graph)
