import time

import pytest

from autogpt_server.blocks.sample import ParrotBlock, PrintingBlock
from autogpt_server.blocks.text import TextFormatterBlock
from autogpt_server.data import block, db, execution, graph
from autogpt_server.executor import ExecutionManager
from autogpt_server.server import AgentServer
from autogpt_server.util.service import PyroNameServer


async def create_test_graph() -> graph.Graph:
    """
    ParrotBlock
                \
                 ---- TextFormatterBlock ---- PrintingBlock
                /
    ParrotBlock
    """
    nodes = [
        graph.Node(block_id=ParrotBlock().id),
        graph.Node(block_id=ParrotBlock().id),
        graph.Node(
            block_id=TextFormatterBlock().id,
            input_default={
                "format": "{texts[0]},{texts[1]},{texts[2]}",
                "texts_$_3": "!!!",
            },
        ),
        graph.Node(block_id=PrintingBlock().id),
    ]
    links = [
        graph.Link(nodes[0].id, nodes[2].id, "output", "texts_$_1"),
        graph.Link(nodes[1].id, nodes[2].id, "output", "texts_$_2"),
        graph.Link(nodes[2].id, nodes[3].id, "output", "text"),
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


async def execute_graph(test_manager: ExecutionManager, test_graph: graph.Graph) -> str:
    # --- Test adding new executions --- #
    text = "Hello, World!"
    input_data = {"input": text}
    agent_server = AgentServer()
    response = await agent_server.execute_graph(test_graph.id, input_data)
    executions = response["executions"]
    graph_exec_id = response["id"]
    assert len(executions) == 2

    async def is_execution_completed():
        execs = await agent_server.get_run_execution_results(
            test_graph.id, graph_exec_id
        )
        return (
            test_manager.queue.empty()
            and len(execs) == 4
            and all(
                exec.status == execution.ExecutionStatus.COMPLETED for exec in execs
            )
        )

    # Wait for the executions to complete
    for i in range(10):
        if await is_execution_completed():
            break
        time.sleep(1)

    # Execution queue should be empty
    assert await is_execution_completed()
    return graph_exec_id


async def assert_executions(test_graph: graph.Graph, graph_exec_id: str):
    text = "Hello, World!"
    agent_server = AgentServer()
    executions = await agent_server.get_run_execution_results(
        test_graph.id, graph_exec_id
    )

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

    # Executing TextFormatterBlock
    exec = executions[2]
    assert exec.status == execution.ExecutionStatus.COMPLETED
    assert exec.graph_exec_id == graph_exec_id
    assert exec.output_data == {"output": ["Hello, World!,Hello, World!,!!!"]}
    assert exec.input_data == {
        "texts_$_1": "Hello, World!",
        "texts_$_2": "Hello, World!",
    }
    assert exec.node_id == test_graph.nodes[2].id

    # Executing PrintingBlock
    exec = executions[3]
    assert exec.status == execution.ExecutionStatus.COMPLETED
    assert exec.graph_exec_id == graph_exec_id
    assert exec.output_data == {"status": ["printed"]}
    assert exec.input_data == {"text": "Hello, World!,Hello, World!,!!!"}
    assert exec.node_id == test_graph.nodes[3].id


@pytest.mark.asyncio(scope="session")
async def test_agent_execution():
    with PyroNameServer():
        with AgentServer():
            with ExecutionManager(1) as test_manager:
                await db.connect()
                await block.initialize_blocks()
                test_graph = await create_test_graph()
                graph_exec_id = await execute_graph(test_manager, test_graph)
                await assert_executions(test_graph, graph_exec_id)
