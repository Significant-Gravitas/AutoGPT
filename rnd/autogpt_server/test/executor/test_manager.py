import pytest
from prisma.models import User

from autogpt_server.blocks.basic import (
    ObjectLookupBlock,
    ValueBlock,
    CreateListBlock,
    ObjectLookupBase,
    DictionaryAddEntryBlock,
    PrintingBlock,
)
from autogpt_server.blocks.iteration import ForEachBlock
from autogpt_server.blocks.text import TextFormatterBlock
from autogpt_server.blocks.time_blocks import WaitBlock
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
    timeout: int = 20,
) -> str:
    # --- Test adding new executions --- #
    response = await agent_server.execute_graph(test_graph.id, input_data, test_user.id)
    graph_exec_id = response["id"]

    # Execution queue should be empty
    assert await wait_execution(
        test_manager, test_user.id, test_graph.id, graph_exec_id, num_execs, timeout=timeout
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


@pytest.mark.asyncio(scope="session")
async def test_async_bug_graph_behavior(server):
    """
    This test is asserting the behaviour of the Async Bug Graph.

    Test scenario:
    The graph has multiple nodes performing object lookups, formatting texts, and processing lists.
    The graph links them in a specific sequence to test asynchronous operations and dependencies.
    """
    nodes = [
        graph.Node( # Node 0 - executed once
            block_id=CreateListBlock().id,
            input_default={
                "items": [
                    '{"item": "one"}',
                    '{"item": "two"}',
                    '{"item": "three"}',
                    '{"item": "four"}',
                ]
            },
        ),
        graph.Node( # Node 1 - executed once
            block_id=ForEachBlock().id,
            input_default={"return_index": False},
        ),
        graph.Node( # Node 2 - executed once per loop
            block_id=ObjectLookupBlock().id,
            input_default={"key": "item", "input": {}},
        ),
        graph.Node( # Node 3 (TOP) - executed once per loop
            block_id=TextFormatterBlock().id,
            input_default={"format": "Fast Path Item is {item}"},
        ),
        graph.Node( # Node 4 (BOTTOM) - executed once per loop
            block_id=TextFormatterBlock().id,
            input_default={"format": "Slow Path Item is {item}"},
        ),
        graph.Node( # Node 5 (TOP) - executed once per loop
            block_id=DictionaryAddEntryBlock().id,
            input_default={"key": "path_string"},
        ),
        graph.Node( # Node 6 (BOTTOM) - executed once per loop
            block_id=WaitBlock().id,
            input_default={"seconds": 1},
        ),
        graph.Node( # Node 7 (TOP) - executed once per loop
            block_id=DictionaryAddEntryBlock().id,
            input_default={"key": "should_be_same"},
        ),
        graph.Node( # Node 8 (BOTTOM) - executed once per loop
            block_id=DictionaryAddEntryBlock().id,
            input_default={"key": "path_string"},
        ),
        graph.Node( # Node 9 (BOTTOM) - executed once per loop
            block_id=DictionaryAddEntryBlock().id,
            input_default={"key": "should_be_same"},
        ), 
        graph.Node( # Node 10 (TOP) - executed once per loop
            block_id=TextFormatterBlock().id,
            input_default={"format": "{path_string} same as {should_be_same}"},
        ),
        graph.Node( # Node 11 (BOTTOM) - executed once per loop
            block_id=TextFormatterBlock().id,
            input_default={"format": "{path_string} same as {should_be_same}"},
        ),
        graph.Node( # Node 12 (TOP) - executed once per loop
            block_id=PrintingBlock().id,
        ),
        graph.Node( # Node 13 (BOTTOM) - executed once per loop
            block_id=PrintingBlock().id,
        ),     
    ]
    # num execs  = 2 initial + 9 per loop = 2 + 9*3 = 29

    links = [
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[1].id,
            source_name="list",
            sink_name="items",
            is_static=False,
        ),
        # ForEachBlock needs to be connected to 2x text formmater blocks and object lookup block
        graph.Link(
            source_id=nodes[1].id, # ForEachBlock
            sink_id=nodes[2].id, # ObjectLookupBlock
            source_name="item",
            sink_name="input",
            is_static=False,
        ),
        graph.Link(
            source_id=nodes[1].id, # ForEachBlock
            sink_id=nodes[3].id, # TextFormatterBlock
            source_name="item",
            sink_name="named_texts",
            is_static=False,
        ),
        graph.Link(
            source_id=nodes[1].id, # ForEachBlock
            sink_id=nodes[4].id, # TextFormatterBlock
            source_name="item",
            sink_name="named_texts",
            is_static=False,
        ),
        # Top Execution Path
        graph.Link(
            source_id=nodes[3].id, # TextFormatterBlock
            sink_id=nodes[5].id, # DictionaryAddEntryBlock
            source_name="output",
            sink_name="value",
            is_static=False,
        ),
        graph.Link(
            source_id=nodes[5].id, # DictionaryAddEntryBlock
            sink_id=nodes[7].id, # DictionaryAddEntryBlock
            source_name="updated_dictionary",
            sink_name="dictionary",
            is_static=False,
        ),
        graph.Link(
            source_id=nodes[7].id, # DictionaryAddEntryBlock
            sink_id=nodes[10].id, # TextFormatterBlock
            source_name="updated_dictionary",
            sink_name="named_texts",
            is_static=False,
        ),
        graph.Link(
            source_id=nodes[10].id, # TextFormatterBlock
            sink_id=nodes[12].id, # PrintingBlock
            source_name="output",
            sink_name="text",
            is_static=False,
        ),
        # Object Lookup Block needs to be connected to the DictionaryAddEntryBlock
        graph.Link(
            source_id=nodes[2].id, # ObjectLookupBlock
            sink_id=nodes[7].id, # DictionaryAddEntryBlock
            source_name="output",
            sink_name="value",
        ),
        graph.Link(
            source_id=nodes[10].id, # TextFormatterBlock
            sink_id=nodes[12].id, # PrintingBlock
            source_name="output",
            sink_name="text",
            is_static=False,
        ),
        # Bottom Execution Path
        graph.Link(
            source_id=nodes[4].id, # TextFormatterBlock
            sink_id=nodes[6].id, # WaitBlock
            source_name="output",
            sink_name="data",
            is_static=False,
        ),
        graph.Link(
            source_id=nodes[6].id, # WaitBlock
            sink_id=nodes[8].id, # DictionaryAddEntryBlock
            source_name="data",
            sink_name="value",
            is_static=False,
        ),
        graph.Link(
            source_id=nodes[8].id, # DictionaryAddEntryBlock
            sink_id=nodes[9].id, # DictionaryAddEntryBlock
            source_name="updated_dictionary",
            sink_name="dictionary",
            is_static=False,
        ),
        # Object Lookup Block needs to be connected to the DictionaryAddEntryBlock
        graph.Link(
            source_id=nodes[2].id, # ObjectLookupBlock
            sink_id=nodes[9].id, # DictionaryAddEntryBlock
            source_name="output",
            sink_name="value",
        ),
        graph.Link(
            source_id=nodes[9].id, # DictionaryAddEntryBlock
            sink_id=nodes[11].id, # TextFormatterBlock
            source_name="updated_dictionary",
            sink_name="named_texts",
            is_static=False,
        ),
        graph.Link(
            source_id=nodes[11].id, # TextFormatterBlock
            sink_id=nodes[13].id, # PrintingBlock
            source_name="output",
            sink_name="text",
            is_static=False,
        ),
        
    ]

    test_graph = graph.Graph(
        name="Async Bug Graph",
        description="Agent Description",
        nodes=nodes,
        links=links,
    )

    test_user = await create_test_user()
    test_graph = await graph.create_graph(test_graph, user_id=test_user.id)
    graph_exec_id = await execute_graph(
        server.agent_server, server.exec_manager, test_graph, test_user, {}, 54
    )
    executions = await server.agent_server.get_run_execution_results(
        test_graph.id, graph_exec_id, test_user.id
    )
    assert len(executions) == 54

    expected_ouputs = set(
        [
            "Fast Path Item is one should be the same as one",
            "Fast Path Item is two should be the same as two",
            "Fast Path Item is three should be the same as three",
            "Fast Path Item is four should be the same as four",
            "Slow Path Item is one should be the same as one",
            "Slow Path Item is two should be the same as two",
            "Slow Path Item is three should be the same as three",
            "Slow Path Item is four should be the same as four",
        ]
    )
    
    actaul_outputs = set()
    
    for exec_data in executions:
        if "text" in exec_data.input_data:
            output = exec_data.input_data["text"]
            actaul_outputs.add(output)
    
    assert expected_ouputs.isdisjoint(actaul_outputs), f"Actual: {actaul_outputs}"
    
