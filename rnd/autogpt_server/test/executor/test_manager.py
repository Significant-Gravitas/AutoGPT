import pytest

from autogpt_server.data import execution, graph
from autogpt_server.executor import ExecutionManager
from autogpt_server.server import AgentServer
from autogpt_server.util.test import SpinTestServer, wait_execution
from autogpt_server.usecases.sample import create_test_graph


async def execute_graph(test_manager: ExecutionManager, test_graph: graph.Graph) -> str:
    # --- Test adding new executions --- #
    text = "Hello, World!"
    input_data = {"input": text}
    agent_server = AgentServer()
    response = await agent_server.execute_graph(test_graph.id, input_data)
    executions = response["executions"]
    graph_exec_id = response["id"]
    assert len(executions) == 2

    # Execution queue should be empty
    assert await wait_execution(test_manager, test_graph.id, graph_exec_id, 4)
    return graph_exec_id


async def assert_executions(test_graph: graph.Graph, graph_exec_id: str):
    text = "Hello, World!"
    agent_server = AgentServer()
    executions = await agent_server.get_run_execution_results(
        test_graph.id, graph_exec_id
    )

    # Executing ConstantBlock1
    exec = executions[0]
    assert exec.status == execution.ExecutionStatus.COMPLETED
    assert exec.graph_exec_id == graph_exec_id
    assert exec.output_data == {"output": ["Hello, World!"]}
    assert exec.input_data == {"input": text}
    assert exec.node_id == test_graph.nodes[0].id

    # Executing ConstantBlock2
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
    async with SpinTestServer() as server:
        test_graph = create_test_graph()
        await graph.create_graph(test_graph)
        graph_exec_id = await execute_graph(server.exec_manager, test_graph)
        await assert_executions(test_graph, graph_exec_id)
