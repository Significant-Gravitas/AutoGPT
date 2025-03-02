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
    logger.info("Creating graph for user %s", u.id)
    return await s.agent_server.test_create_graph(CreateGraph(graph=g), u.id)


def create_credentials(s: SpinTestServer, u: User):
    provider = ProviderName.OPENAI
    credentials = llm.TEST_CREDENTIALS
    try:
        s.agent_server.test_create_credentials(u.id, provider, credentials)
    except Exception:
        # ValueErrors is raised trying to recreate the same credentials
        # so hidding the error
        pass


async def execute_graph(
    agent_server: AgentServer,
    test_graph: graph.Graph,
    test_user: User,
    input_data: dict,
    num_execs: int = 4,
) -> str:
    logger.info("Executing graph %s for user %s", test_graph.id, test_user.id)
    logger.info("Input data: %s", input_data)

    # --- Test adding new executions --- #
    response = await agent_server.test_execute_graph(
        user_id=test_user.id,
        graph_id=test_graph.id,
        graph_version=test_graph.version,
        node_input=input_data,
    )
    graph_exec_id = response.graph_exec_id
    logger.info("Created execution with ID: %s", graph_exec_id)

    # Execution queue should be empty
    logger.info("Waiting for execution to complete...")
    result = await wait_execution(test_user.id, test_graph.id, graph_exec_id, 30)
    logger.info("Execution completed with %d results", len(result))
    return graph_exec_id


@pytest.mark.skip()
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
            source_name="tools_^_sample_tool_input_1",
            sink_name="input_1",
        ),
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[1].id,
            source_name="tools_^_sample_tool_input_2",
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


@pytest.mark.skip()
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
            source_name="tools_^_sample_tool_input_1",
            sink_name="input_1",
        ),
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[1].id,
            source_name="tools_^_sample_tool_input_2",
            sink_name="input_2",
        ),
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[2].id,
            source_name="tools_^_store_value_input",
            sink_name="input",
        ),
    ]

    test_graph = graph.Graph(
        name="TestGraph",
        description="Test graph",
        nodes=nodes,
        links=links,
    )
    with pytest.raises(ValueError):
        test_graph = await create_graph(server, test_graph, test_user)


@pytest.mark.skip()
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
        graph.Node(
            block_id=StoreValueBlock().id,
        ),
    ]

    links = [
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[1].id,
            source_name="tools_^_sample_tool_input_1",
            sink_name="input_1",
        ),
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[1].id,
            source_name="tools_^_sample_tool_input_2",
            sink_name="input_2",
        ),
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[2].id,
            source_name="tools_^_store_value_input",
            sink_name="input",
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
        test_graph.nodes[0].id
    )
    assert tool_functions is not None, "Tool functions should not be None"

    assert (
        len(tool_functions) == 2
    ), f"Expected 2 tool functions, got {len(tool_functions)}"

    # Check the first tool function (testgraph)
    assert tool_functions[0]["type"] == "function"
    assert tool_functions[0]["function"]["name"] == "testgraph"
    assert tool_functions[0]["function"]["description"] == "Test graph description"
    assert "input_1" in tool_functions[0]["function"]["parameters"]["properties"]
    assert "input_2" in tool_functions[0]["function"]["parameters"]["properties"]

    # Check the second tool function (storevalueblock)
    assert tool_functions[1]["type"] == "function"
    assert tool_functions[1]["function"]["name"] == "storevalueblock"
    assert "input" in tool_functions[1]["function"]["parameters"]["properties"]
    assert (
        tool_functions[1]["function"]["parameters"]["properties"]["input"][
            "description"
        ]
        == "Trigger the block to produce the output. The value is only used when `data` is None."
    )
