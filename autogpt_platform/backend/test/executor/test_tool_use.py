import logging

import pytest
from prisma.models import User

import backend.blocks.llm as llm
from backend.blocks.agent import AgentExecutorBlock
from backend.blocks.basic import StoreValueBlock
from backend.blocks.smart_decision_maker import SmartDecisionMakerBlock
from backend.data import graph
from backend.data.model import ProviderName
from backend.server.model import CreateGraph
from backend.server.rest_api import AgentServer
from backend.usecases.sample import create_test_graph, create_test_user
from backend.util.test import SpinTestServer, wait_execution

logger = logging.getLogger(__name__)


async def create_graph(s: SpinTestServer, g: graph.Graph, u: User) -> graph.Graph:
    logger.info(f"Creating graph for user {u.id}")
    return await s.agent_server.test_create_graph(CreateGraph(graph=g), u.id)


def create_credentials(s: SpinTestServer, u: User):
    provider = ProviderName.OPENAI
    credentials = llm.TEST_CREDENTIALS
    try:
        s.agent_server.test_create_credentials(u.id, provider, credentials)
    except Exception:
        # Errors when trying to recreate the same credentials
        # so hidding the error
        pass


async def execute_graph(
    agent_server: AgentServer,
    test_graph: graph.Graph,
    test_user: User,
    input_data: dict,
    num_execs: int = 4,
) -> str:
    logger.info(f"Executing graph {test_graph.id} for user {test_user.id}")
    logger.info(f"Input data: {input_data}")

    # --- Test adding new executions --- #
    response = await agent_server.test_execute_graph(
        user_id=test_user.id,
        graph_id=test_graph.id,
        graph_version=test_graph.version,
        node_input=input_data,
    )
    graph_exec_id = response.graph_exec_id
    logger.info(f"Created execution with ID: {graph_exec_id}")

    # Execution queue should be empty
    logger.info("Waiting for execution to complete...")
    result = await wait_execution(test_user.id, test_graph.id, graph_exec_id, 30)
    logger.info(f"Execution completed with {len(result)} results")
    return graph_exec_id


@pytest.mark.asyncio(scope="session")
async def test_graph_validation_with_tool_nodes_correct(server: SpinTestServer):
    test_user = await create_test_user()
    test_tool_graph = await create_graph(server, create_test_graph(), test_user)
    create_credentials(server, test_user)

    nodes = [
        graph.Node(
            block_id=SmartDecisionMakerBlock().id,
            input_default={
                "prompt": "Hello, World!",
                "credentials": llm.TEST_CREDENTIALS_INPUT,
            },
        ),
        graph.Node(
            block_id=AgentExecutorBlock().id,
            input_default={
                "graph_id": test_tool_graph.id,
                "graph_version": test_tool_graph.version,
                "input_schema": test_tool_graph.input_schema,
                "output_schema": test_tool_graph.output_schema,
            },
        ),
    ]

    links = [
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[1].id,
            source_name="tools_sample_tool_#_input_1",
            sink_name="input_1",
        ),
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[1].id,
            source_name="tools_sample_tool_#_input_2",
            sink_name="input_2",
        ),
    ]

    test_graph = graph.Graph(
        name="TestGraph",
        description="Test graph",
        nodes=nodes,
        links=links,
    )
    test_graph = await create_graph(server, test_graph, test_user)


@pytest.mark.asyncio(scope="session")
async def test_graph_validation_with_tool_nodes_raises_error(server: SpinTestServer):

    test_user = await create_test_user()
    test_tool_graph = await create_graph(server, create_test_graph(), test_user)
    create_credentials(server, test_user)

    nodes = [
        graph.Node(
            block_id=SmartDecisionMakerBlock().id,
            input_default={
                "prompt": "Hello, World!",
                "credentials": llm.TEST_CREDENTIALS_INPUT,
            },
        ),
        graph.Node(
            block_id=AgentExecutorBlock().id,
            input_default={
                "graph_id": test_tool_graph.id,
                "graph_version": test_tool_graph.version,
                "input_schema": test_tool_graph.input_schema,
                "output_schema": test_tool_graph.output_schema,
            },
        ),
        graph.Node(
            block_id=StoreValueBlock().id,
        ),
    ]

    links = [
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[1].id,
            source_name="tools_sample_tool_#_input_1",
            sink_name="input_1",
        ),
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[1].id,
            source_name="tools_sample_tool_#_input_2",
            sink_name="input_2",
        ),
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[2].id,
            source_name="tools_store_value_#_input",
            sink_name="input",
        ),
    ]

    test_graph = graph.Graph(
        name="TestGraph",
        description="Test graph",
        nodes=nodes,
        links=links,
    )
    with pytest.raises(Exception):
        test_graph = await create_graph(server, test_graph, test_user)


@pytest.mark.asyncio(scope="session")
async def test_smart_decision_maker_function_signature(server: SpinTestServer):
    test_user = await create_test_user()
    test_tool_graph = await create_graph(server, create_test_graph(), test_user)
    create_credentials(server, test_user)

    nodes = [
        graph.Node(
            block_id=SmartDecisionMakerBlock().id,
            input_default={
                "prompt": "Hello, World!",
                "credentials": llm.TEST_CREDENTIALS_INPUT,
            },
        ),
        graph.Node(
            block_id=AgentExecutorBlock().id,
            input_default={
                "graph_id": test_tool_graph.id,
                "graph_version": test_tool_graph.version,
                "input_schema": test_tool_graph.input_schema,
                "output_schema": test_tool_graph.output_schema,
            },
        ),
    ]

    links = [
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[1].id,
            source_name="tools_sample_tool_#_input_1",
            sink_name="input_1",
        ),
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[1].id,
            source_name="tools_sample_tool_#_input_2",
            sink_name="input_2",
        ),
    ]

    test_graph = graph.Graph(
        name="TestGraph",
        description="Test graph",
        nodes=nodes,
        links=links,
    )
    test_graph = await create_graph(server, test_graph, test_user)

    tool_functions = SmartDecisionMakerBlock._create_function_signature(
        test_graph.nodes[0].id, test_graph, [test_tool_graph]
    )
    assert tool_functions is not None, "Tool functions should not be None"
    assert (
        len(tool_functions) == 1
    ), f"Expected 1 tool function, got {len(tool_functions)}"

    tool_function = next(
        filter(lambda x: x["function"]["name"] == "TestGraph", tool_functions),
        None,
    )
    assert tool_function is not None, "TestGraph function not found"
    assert (
        tool_function["function"]["name"] == "TestGraph"
    ), "Incorrect function name for TestGraph"
    assert (
        tool_function["function"]["parameters"]["properties"]["input_1"]["type"]
        == "string"
    ), "Input type for input_1 should be 'string'"
    assert (
        tool_function["function"]["parameters"]["properties"]["input_2"]["type"]
        == "string"
    ), "Input type for input_2 should be 'string'"
    assert (
        tool_function["function"]["parameters"]["required"] == []
    ), "Required parameters should be an empty list"
