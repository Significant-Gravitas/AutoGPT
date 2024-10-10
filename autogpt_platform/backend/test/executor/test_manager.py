import pytest
from prisma.models import User

from backend.blocks.basic import FindInDictionaryBlock, StoreValueBlock
from backend.blocks.maths import CalculatorBlock, Operation
from backend.data import execution, graph
from backend.server.model import CreateGraph
from backend.server.rest_api import AgentServer
from backend.usecases.sample import create_test_graph, create_test_user
from backend.util.test import SpinTestServer, wait_execution


async def create_graph(s: SpinTestServer, g: graph.Graph, u: User) -> graph.Graph:
    return await s.agent_server.create_graph(CreateGraph(graph=g), False, u.id)


async def execute_graph(
    agent_server: AgentServer,
    test_graph: graph.Graph,
    test_user: User,
    input_data: dict,
    num_execs: int = 4,
) -> str:
    # --- Test adding new executions --- #
    response = await agent_server.execute_graph(test_graph.id, input_data, test_user.id)
    graph_exec_id = response["id"]

    # Execution queue should be empty
    result = await wait_execution(test_user.id, test_graph.id, graph_exec_id)
    assert result and len(result) == num_execs
    return graph_exec_id


async def assert_sample_graph_executions(
    agent_server: AgentServer,
    test_graph: graph.Graph,
    test_user: User,
    graph_exec_id: str,
):
    executions = await agent_server.get_graph_run_node_execution_results(
        test_graph.id,
        graph_exec_id,
        test_user.id,
    )

    output_list = [{"result": ["Hello"]}, {"result": ["World"]}]
    input_list = [
        {
            "name": "input_1",
            "value": "Hello",
        },
        {
            "name": "input_2",
            "value": "World",
        },
    ]

    # Executing StoreValueBlock
    exec = executions[0]
    assert exec.status == execution.ExecutionStatus.COMPLETED
    assert exec.graph_exec_id == graph_exec_id
    assert (
        exec.output_data in output_list
    ), f"Output data: {exec.output_data} and {output_list}"
    assert (
        exec.input_data in input_list
    ), f"Input data: {exec.input_data} and {input_list}"
    assert exec.node_id in [test_graph.nodes[0].id, test_graph.nodes[1].id]

    # Executing StoreValueBlock
    exec = executions[1]
    assert exec.status == execution.ExecutionStatus.COMPLETED
    assert exec.graph_exec_id == graph_exec_id
    assert (
        exec.output_data in output_list
    ), f"Output data: {exec.output_data} and {output_list}"
    assert (
        exec.input_data in input_list
    ), f"Input data: {exec.input_data} and {input_list}"
    assert exec.node_id in [test_graph.nodes[0].id, test_graph.nodes[1].id]

    # Executing FillTextTemplateBlock
    exec = executions[2]
    assert exec.status == execution.ExecutionStatus.COMPLETED
    assert exec.graph_exec_id == graph_exec_id
    assert exec.output_data == {"output": ["Hello, World!!!"]}
    assert exec.input_data == {
        "format": "{a}, {b}{c}",
        "values": {"a": "Hello", "b": "World", "c": "!!!"},
        "values_#_a": "Hello",
        "values_#_b": "World",
        "values_#_c": "!!!",
    }
    assert exec.node_id == test_graph.nodes[2].id

    # Executing PrintToConsoleBlock
    exec = executions[3]
    assert exec.status == execution.ExecutionStatus.COMPLETED
    assert exec.graph_exec_id == graph_exec_id
    assert exec.output_data == {"status": ["printed"]}
    assert exec.input_data == {"text": "Hello, World!!!"}
    assert exec.node_id == test_graph.nodes[3].id


@pytest.mark.asyncio(scope="session")
async def test_agent_execution(server: SpinTestServer):
    test_user = await create_test_user()
    test_graph = await create_graph(server, create_test_graph(), test_user)
    data = {"input_1": "Hello", "input_2": "World"}
    graph_exec_id = await execute_graph(
        server.agent_server,
        test_graph,
        test_user,
        data,
        4,
    )
    await assert_sample_graph_executions(
        server.agent_server, test_graph, test_user, graph_exec_id
    )


@pytest.mark.asyncio(scope="session")
async def test_input_pin_always_waited(server: SpinTestServer):
    """
    This test is asserting that the input pin should always be waited for the execution,
    even when default value on that pin is defined, the value has to be ignored.

    Test scenario:
    StoreValueBlock1
                \\ input
                     >------- FindInDictionaryBlock | input_default: key: "", input: {}
                // key
    StoreValueBlock2
    """
    nodes = [
        graph.Node(
            block_id=StoreValueBlock().id,
            input_default={"input": {"key1": "value1", "key2": "value2"}},
        ),
        graph.Node(
            block_id=StoreValueBlock().id,
            input_default={"input": "key2"},
        ),
        graph.Node(
            block_id=FindInDictionaryBlock().id,
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
    test_user = await create_test_user()
    test_graph = await create_graph(server, test_graph, test_user)
    graph_exec_id = await execute_graph(
        server.agent_server, test_graph, test_user, {}, 3
    )

    executions = await server.agent_server.get_graph_run_node_execution_results(
        test_graph.id, graph_exec_id, test_user.id
    )
    assert len(executions) == 3
    # FindInDictionaryBlock should wait for the input pin to be provided,
    # Hence executing extraction of "key" from {"key1": "value1", "key2": "value2"}
    assert executions[2].status == execution.ExecutionStatus.COMPLETED
    assert executions[2].output_data == {"output": ["value2"]}


@pytest.mark.asyncio(scope="session")
async def test_static_input_link_on_graph(server: SpinTestServer):
    """
    This test is asserting the behaviour of static input link, e.g: reusable input link.

    Test scenario:
    *StoreValueBlock1*===a=========\\
    *StoreValueBlock2*===a=====\\  ||
    *StoreValueBlock3*===a===*MathBlock*====b / static====*StoreValueBlock5*
    *StoreValueBlock4*=========================================//

    In this test, there will be three input waiting in the MathBlock input pin `a`.
    And later, another output is produced on input pin `b`, which is a static link,
    this input will complete the input of those three incomplete executions.
    """
    nodes = [
        graph.Node(block_id=StoreValueBlock().id, input_default={"input": 4}),  # a
        graph.Node(block_id=StoreValueBlock().id, input_default={"input": 4}),  # a
        graph.Node(block_id=StoreValueBlock().id, input_default={"input": 4}),  # a
        graph.Node(block_id=StoreValueBlock().id, input_default={"input": 5}),  # b
        graph.Node(block_id=StoreValueBlock().id),
        graph.Node(
            block_id=CalculatorBlock().id,
            input_default={"operation": Operation.ADD.value},
        ),
    ]
    links = [
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[5].id,
            source_name="output",
            sink_name="a",
        ),
        graph.Link(
            source_id=nodes[1].id,
            sink_id=nodes[5].id,
            source_name="output",
            sink_name="a",
        ),
        graph.Link(
            source_id=nodes[2].id,
            sink_id=nodes[5].id,
            source_name="output",
            sink_name="a",
        ),
        graph.Link(
            source_id=nodes[3].id,
            sink_id=nodes[4].id,
            source_name="output",
            sink_name="input",
        ),
        graph.Link(
            source_id=nodes[4].id,
            sink_id=nodes[5].id,
            source_name="output",
            sink_name="b",
            is_static=True,  # This is the static link to test.
        ),
    ]
    test_graph = graph.Graph(
        name="TestGraph",
        description="Test graph",
        nodes=nodes,
        links=links,
    )
    test_user = await create_test_user()
    test_graph = await create_graph(server, test_graph, test_user)
    graph_exec_id = await execute_graph(
        server.agent_server, test_graph, test_user, {}, 8
    )
    executions = await server.agent_server.get_graph_run_node_execution_results(
        test_graph.id, graph_exec_id, test_user.id
    )
    assert len(executions) == 8
    # The last 3 executions will be a+b=4+5=9
    for exec_data in executions[-3:]:
        assert exec_data.status == execution.ExecutionStatus.COMPLETED
        assert exec_data.output_data == {"result": [9]}
