import pytest

from autogpt_server.blocks.basic import ObjectLookupBlock, ValueBlock
from autogpt_server.blocks.if_block import ComparisonOperator, ConditionBlock
from autogpt_server.blocks.maths import MathsBlock, Operation
from autogpt_server.data import execution, graph
from autogpt_server.executor import ExecutionManager
from autogpt_server.server import AgentServer
from autogpt_server.usecases.sample import create_test_graph
from autogpt_server.util.test import wait_execution


async def execute_graph(
    agent_server: AgentServer,
    test_manager: ExecutionManager,
    test_graph: graph.Graph,
    input_data: dict,
    num_execs: int = 4,
) -> str:
    # --- Test adding new executions --- #
    response = await agent_server.execute_graph(test_graph.id, input_data)
    graph_exec_id = response["id"]

    # Execution queue should be empty
    assert await wait_execution(test_manager, test_graph.id, graph_exec_id, num_execs)
    return graph_exec_id


async def assert_sample_graph_executions(
    agent_server: AgentServer, test_graph: graph.Graph, graph_exec_id: str
):
    text = "Hello, World!"
    executions = await agent_server.get_run_execution_results(
        test_graph.id, graph_exec_id
    )

    # Executing ConstantBlock1
    exec = executions[0]
    assert exec.status == execution.ExecutionStatus.COMPLETED
    assert exec.graph_exec_id == graph_exec_id
    assert exec.output_data == {"output": ["Hello, World!"]}
    assert exec.input_data == {"input": text}
    assert exec.node_id in [test_graph.nodes[0].id, test_graph.nodes[1].id]

    # Executing ConstantBlock2
    exec = executions[1]
    assert exec.status == execution.ExecutionStatus.COMPLETED
    assert exec.graph_exec_id == graph_exec_id
    assert exec.output_data == {"output": ["Hello, World!"]}
    assert exec.input_data == {"input": text}
    assert exec.node_id in [test_graph.nodes[0].id, test_graph.nodes[1].id]

    # Executing TextFormatterBlock
    exec = executions[2]
    assert exec.status == execution.ExecutionStatus.COMPLETED
    assert exec.graph_exec_id == graph_exec_id
    assert exec.output_data == {"output": ["Hello, World!,Hello, World!,!!!"]}
    assert exec.input_data == {
        "format": "{texts[0]},{texts[1]},{texts[2]}",
        "texts": ["Hello, World!", "Hello, World!", "!!!"],
        "texts_$_1": "Hello, World!",
        "texts_$_2": "Hello, World!",
        "texts_$_3": "!!!",
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
    graph_exec_id = await execute_graph(
        server.agent_server, server.exec_manager, test_graph, data, 4
    )
    await assert_sample_graph_executions(server.agent_server, test_graph, graph_exec_id)


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
    graph_exec_id = await execute_graph(
        server.agent_server, server.exec_manager, test_graph, {}, 3
    )

    executions = await server.agent_server.get_run_execution_results(
        test_graph.id, graph_exec_id
    )
    assert len(executions) == 3
    # ObjectLookupBlock should wait for the input pin to be provided,
    # Hence executing extraction of "key" from {"key1": "value1", "key2": "value2"}
    assert executions[2].status == execution.ExecutionStatus.COMPLETED
    assert executions[2].output_data == {"output": ["value2"]}


@pytest.mark.asyncio(scope="session")
async def test_static_input_link_on_loop(server):
    """
    This test is asserting the behaviour of static input link, e.g: reusable input link.
    In this test, the input link is reused on the loop, and the input value is retained.

    Test scenario:
    ValueBlock --C MathBlock -- IfBlock --
                      |_________________|

    ValueBlock will be connected to both the input of MathBlock (`a` and `b`).
    Link to `a` will be static, IfBlock `no_output` will be also connected to `b`.
    This will create a loop to add keep adding `ValueBlock` until IF condition is met.
    """
    nodes = [
        graph.Node(block_id=ValueBlock().id),
        graph.Node(
            block_id=MathsBlock().id,
            input_default={"operation": Operation.ADD.value},
        ),
        graph.Node(
            block_id=ConditionBlock().id,
            input_default={
                "operator": ComparisonOperator.GREATER_THAN.value,
                "value2": 10,
            },
        ),
    ]
    links = [
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[1].id,
            source_name="output",
            sink_name="a",
            is_static=True,  # This is the static link to test.
        ),
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[1].id,
            source_name="output",
            sink_name="b",
        ),
        graph.Link(
            source_id=nodes[1].id,
            sink_id=nodes[2].id,
            source_name="result",
            sink_name="value1",
        ),
        graph.Link(
            source_id=nodes[2].id,
            sink_id=nodes[1].id,
            source_name="no_output",
            sink_name="b",
        ),
    ]
    test_graph = graph.Graph(
        name="TestGraph",
        description="Test graph",
        nodes=nodes,
        links=links,
    )

    test_graph = await graph.create_graph(test_graph)
    input_data = {"input": 3}
    graph_exec_id = await execute_graph(
        server.agent_server, server.exec_manager, test_graph, input_data, 7
    )
    executions = await server.agent_server.get_run_execution_results(
        test_graph.id, graph_exec_id
    )
    # Loop: 3+3=6, 6+3=9, 9+3=12
    assert len(executions) == 7
    assert executions[6].status == execution.ExecutionStatus.COMPLETED
    assert executions[6].output_data == {"result": [True], "yes_output": [12]}
