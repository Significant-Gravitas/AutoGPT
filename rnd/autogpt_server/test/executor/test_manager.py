import pytest

from autogpt_server.blocks.basic import ObjectLookupBlock, ValueBlock
from autogpt_server.data import execution, graph
from autogpt_server.executor import ExecutionManager
from autogpt_server.server import AgentServer
from autogpt_server.usecases.sample import create_test_graph
from autogpt_server.util.test import wait_execution


async def execute_graph(
    test_manager: ExecutionManager,
    test_graph: graph.Graph,
    input_data: dict[str, str],
    num_execs: int = 4,
) -> str:
    # --- Test adding new executions --- #
    agent_server = AgentServer()
    response = await agent_server.execute_graph(test_graph.id, input_data)
    graph_exec_id = response["id"]

    # Execution queue should be empty
    assert await wait_execution(test_manager, test_graph.id, graph_exec_id, num_execs)
    return graph_exec_id


async def assert_sample_graph_executions(test_graph: graph.Graph, graph_exec_id: str):
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
async def test_agent_execution(server):
    test_graph = create_test_graph()
    await graph.create_graph(test_graph)
    data = {"input": "Hello, World!"}
    graph_exec_id = await execute_graph(server.exec_manager, test_graph, data, 4)
    await assert_sample_graph_executions(test_graph, graph_exec_id)


@pytest.mark.asyncio(scope="session")
async def test_input_pin_always_waited(server):
    """
    This test is asserting that the input pin should always be waited for the execution,
    even when default value on that pin is defined, the value has to be ignored.

    Test scenario:
    ValueBlock1
                \\ input
                     >------- ObjectLookupBlock | input_default: key: "", input: {}
                // key
    ValueBlock2
    """
    nodes = [
        graph.Node(
            block_id=ValueBlock().id,
            input_default={"input": {"key1": "value1", "key2": "value2"}},
        ),
        graph.Node(
            block_id=ValueBlock().id,
            input_default={"input": "key2"},
        ),
        graph.Node(
            block_id=ObjectLookupBlock().id,
            input_default={"key": "", "input": {}},
        ),
    ]
    links = [
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[2].id,
            source_name="output",
            sink_name="input",
        ),
        graph.Link(
            source_id=nodes[1].id,
            sink_id=nodes[2].id,
            source_name="output",
            sink_name="key",
        ),
    ]
    test_graph = graph.Graph(
        name="TestGraph",
        description="Test graph",
        nodes=nodes,
        links=links,
    )

    test_graph = await graph.create_graph(test_graph)
    graph_exec_id = await execute_graph(server.exec_manager, test_graph, {}, 3)

    agent_server = AgentServer()
    executions = await agent_server.get_run_execution_results(
        test_graph.id, graph_exec_id
    )
    assert len(executions) == 3
    # ObjectLookupBlock should wait for the input pin to be provided,
    # Hence executing extraction of "key" from {"key1": "value1", "key2": "value2"}
    assert executions[2].status == execution.ExecutionStatus.COMPLETED
    assert executions[2].output_data == {"output": ["value2"]}
