import pytest
from prisma.models import User

from autogpt_server.blocks.basic import ObjectLookupBlock, ValueBlock
from autogpt_server.blocks.maths import MathsBlock, Operation
from autogpt_server.data import execution, graph
from autogpt_server.executor import ExecutionManager
from autogpt_server.server import AgentServer
from autogpt_server.usecases.sample import create_test_graph, create_test_user
from autogpt_server.util.test import wait_execution


async def execute_graph(
    agent_server: AgentServer,
    test_manager: ExecutionManager,
    test_graph: graph.Graph,
    test_user: User,
    input_data: dict,
    num_execs: int = 4,
) -> str:
    # --- Test adding new executions --- #
    response = await agent_server.execute_graph(test_graph.id, input_data, test_user.id)
    graph_exec_id = response["id"]

    # Execution queue should be empty
    assert await wait_execution(
        test_manager, test_user.id, test_graph.id, graph_exec_id, num_execs
    )
    return graph_exec_id


async def assert_sample_graph_executions(
    agent_server: AgentServer,
    test_graph: graph.Graph,
    test_user: User,
    graph_exec_id: str,
):
    input = {"input_1": "Hello", "input_2": "World"}
    executions = await agent_server.get_run_execution_results(
        test_graph.id, graph_exec_id, test_user.id
    )

    # Executing ValueBlock
    exec = executions[0]
    assert exec.status == execution.ExecutionStatus.COMPLETED
    assert exec.graph_exec_id == graph_exec_id
    assert exec.output_data == {"output": ["Hello"]}
    assert exec.input_data == {"input": input, "key": "input_1"}
    assert exec.node_id in [test_graph.nodes[0].id, test_graph.nodes[1].id]

    # Executing ValueBlock
    exec = executions[1]
    assert exec.status == execution.ExecutionStatus.COMPLETED
    assert exec.graph_exec_id == graph_exec_id
    assert exec.output_data == {"output": ["World"]}
    assert exec.input_data == {"input": input, "key": "input_2"}
    assert exec.node_id in [test_graph.nodes[0].id, test_graph.nodes[1].id]

    # Executing TextFormatterBlock
    exec = executions[2]
    assert exec.status == execution.ExecutionStatus.COMPLETED
    assert exec.graph_exec_id == graph_exec_id
    assert exec.output_data == {"output": ["Hello, World!!!"]}
    assert exec.input_data == {
        "format": "{texts[0]}, {texts[1]}{texts[2]}",
        "texts": ["Hello", "World", "!!!"],
        "texts_$_1": "Hello",
        "texts_$_2": "World",
        "texts_$_3": "!!!",
    }
    assert exec.node_id == test_graph.nodes[2].id

    # Executing PrintingBlock
    exec = executions[3]
    assert exec.status == execution.ExecutionStatus.COMPLETED
    assert exec.graph_exec_id == graph_exec_id
    assert exec.output_data == {"status": ["printed"]}
    assert exec.input_data == {"text": "Hello, World!!!"}
    assert exec.node_id == test_graph.nodes[3].id


@pytest.mark.asyncio(scope="session")
async def test_agent_execution(server):
    test_graph = create_test_graph()
    test_user = await create_test_user()
    await graph.create_graph(test_graph, user_id=test_user.id)
    data = {"input_1": "Hello", "input_2": "World"}
    graph_exec_id = await execute_graph(
        server.agent_server,
        server.exec_manager,
        test_graph,
        test_user,
        data,
        4,
    )
    await assert_sample_graph_executions(
        server.agent_server, test_graph, test_user, graph_exec_id
    )


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
    test_user = await create_test_user()
    test_graph = await graph.create_graph(test_graph, user_id=test_user.id)
    graph_exec_id = await execute_graph(
        server.agent_server, server.exec_manager, test_graph, test_user, {}, 3
    )

    executions = await server.agent_server.get_run_execution_results(
        test_graph.id, graph_exec_id, test_user.id
    )
    assert len(executions) == 3
    # ObjectLookupBlock should wait for the input pin to be provided,
    # Hence executing extraction of "key" from {"key1": "value1", "key2": "value2"}
    assert executions[2].status == execution.ExecutionStatus.COMPLETED
    assert executions[2].output_data == {"output": ["value2"]}


@pytest.mark.asyncio(scope="session")
async def test_static_input_link_on_graph(server):
    """
    This test is asserting the behaviour of static input link, e.g: reusable input link.

    Test scenario:
    *ValueBlock1*===a=========\\
    *ValueBlock2*===a=====\\  ||
    *ValueBlock3*===a===*MathBlock*====b / static====*ValueBlock5*
    *ValueBlock4*=========================================//

    In this test, there will be three input waiting in the MathBlock input pin `a`.
    And later, another output is produced on input pin `b`, which is a static link,
    this input will complete the input of those three incomplete executions.
    """
    nodes = [
        graph.Node(block_id=ValueBlock().id, input_default={"input": 4}),  # a
        graph.Node(block_id=ValueBlock().id, input_default={"input": 4}),  # a
        graph.Node(block_id=ValueBlock().id, input_default={"input": 4}),  # a
        graph.Node(block_id=ValueBlock().id, input_default={"input": 5}),  # b
        graph.Node(block_id=ValueBlock().id),
        graph.Node(
            block_id=MathsBlock().id,
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
    test_graph = await graph.create_graph(test_graph, user_id=test_user.id)
    graph_exec_id = await execute_graph(
        server.agent_server, server.exec_manager, test_graph, test_user, {}, 8
    )
    executions = await server.agent_server.get_run_execution_results(
        test_graph.id, graph_exec_id, test_user.id
    )
    assert len(executions) == 8
    # The last 3 executions will be a+b=4+5=9
    for exec_data in executions[-3:]:
        assert exec_data.status == execution.ExecutionStatus.COMPLETED
        assert exec_data.output_data == {"result": [9]}
