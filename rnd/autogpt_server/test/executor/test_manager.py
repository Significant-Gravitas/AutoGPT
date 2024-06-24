import time

import pytest

from autogpt_server.data import block, db, execution, graph
from autogpt_server.executor import ExecutionManager
from autogpt_server.server import AgentServer
from autogpt_server.util.service import PyroNameServer


async def create_test_graph() -> graph.Graph:
    """
    ParrotBlock
                \
                 ---- TextCombinerBlock ---- PrintingBlock
                /
    ParrotBlock
    """
    nodes = [
        graph.Node(block_id=block.ParrotBlock.id),
        graph.Node(block_id=block.ParrotBlock.id),
        graph.Node(
            block_id=block.TextCombinerBlock.id,
            input_default={"format": "{text1},{text2}"},
        ),
        graph.Node(block_id=block.PrintingBlock.id),
    ]
    nodes[0].connect(nodes[2], "output", "text1")
    nodes[1].connect(nodes[2], "output", "text2")
    nodes[2].connect(nodes[3], "combined_text", "text")

    test_graph = graph.Graph(
        name="TestGraph",
        description="Test graph",
        nodes=nodes,
    )
    await block.initialize_blocks()
    result = await graph.create_graph(test_graph)

    # Assertions
    assert result.name == test_graph.name
    assert result.description == test_graph.description
    assert len(result.nodes) == len(test_graph.nodes)

    return test_graph


async def execute_graph(test_manager: ExecutionManager, test_graph: graph.Graph):
    # --- Test adding new executions --- #
    text = "Hello, World!"
    input_data = {"input": text}
    agent_server = AgentServer()
    response = await agent_server.execute_graph(test_graph.id, input_data)
    executions = response["executions"]
    graph_exec_id = response["id"]
    assert len(executions) == 2

    async def is_execution_completed():
        execs = await agent_server.get_executions(test_graph.id, graph_exec_id)
        return test_manager.queue.empty() and len(execs) == 4

    # Wait for the executions to complete
    for i in range(10):
        if await is_execution_completed():
            break
        time.sleep(1)

    # Execution queue should be empty
    assert await is_execution_completed()
    executions = await agent_server.get_executions(test_graph.id, graph_exec_id)

    # Executing ParrotBlock1
    exec = executions[0]
    assert exec.status == execution.ExecutionStatus.COMPLETED
    assert exec.graph_exec_id == graph_exec_id
    assert exec.output_data == {"output": ["Hello, World!"]}
    assert exec.input_data == {"input": text}
    assert exec.node_id == test_graph.nodes[0].id

    # Executing ParrotBlock2
    exec = executions[1]
    assert exec.status == execution.ExecutionStatus.COMPLETED
    assert exec.graph_exec_id == graph_exec_id
    assert exec.output_data == {"output": ["Hello, World!"]}
    assert exec.input_data == {"input": text}
    assert exec.node_id == test_graph.nodes[1].id

    # Executing TextCombinerBlock
    exec = executions[2]
    assert exec.status == execution.ExecutionStatus.COMPLETED
    assert exec.graph_exec_id == graph_exec_id
    assert exec.output_data == {"combined_text": ["Hello, World!,Hello, World!"]}
    assert exec.input_data == {
        "text1": "Hello, World!",
        "text2": "Hello, World!",
    }
    assert exec.node_id == test_graph.nodes[2].id

    # Executing PrintingBlock
    exec = executions[3]
    assert exec.status == execution.ExecutionStatus.COMPLETED
    assert exec.graph_exec_id == graph_exec_id
    assert exec.output_data == {"status": ["printed"]}
    assert exec.input_data == {"text": "Hello, World!,Hello, World!"}
    assert exec.node_id == test_graph.nodes[3].id


@pytest.mark.asyncio(scope="session")
async def test_agent_execution():
    with PyroNameServer():
        with ExecutionManager(1) as test_manager:
            await db.connect()
            test_graph = await create_test_graph()
            await execute_graph(test_manager, test_graph)
