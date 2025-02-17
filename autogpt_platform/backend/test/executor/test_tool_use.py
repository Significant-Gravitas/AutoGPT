import logging

import pytest
from prisma.models import User

from backend.blocks.basic import PrintToConsoleBlock, StoreValueBlock
from backend.blocks.smart_decision_maker import SmartDecisionMakerBlock
from backend.data import graph
from backend.server.model import CreateGraph
from backend.usecases.sample import create_test_user
from backend.util.test import SpinTestServer

logger = logging.getLogger(__name__)


async def create_graph(s: SpinTestServer, g: graph.Graph, u: User) -> graph.Graph:
    logger.info(f"Creating graph for user {u.id}")
    return await s.agent_server.test_create_graph(CreateGraph(graph=g), u.id)


@pytest.mark.asyncio(scope="session")
async def test_smart_decision_maker_function_signature(server: SpinTestServer):

    nodes = [
        graph.Node(
            block_id=SmartDecisionMakerBlock().id,
            input_default={"text": "Hello, World!"},
        ),
        graph.Node(
            block_id=StoreValueBlock().id,
        ),
        graph.Node(
            block_id=PrintToConsoleBlock().id,
        ),
    ]

    smd_id = nodes[0].id

    links = [
        graph.Link(
            source_id=smd_id,
            sink_id=nodes[1].id,
            source_name="tool",
            sink_name="input",
        ),
        graph.Link(
            source_id=smd_id,
            sink_id=nodes[2].id,
            source_name="tool",
            sink_name="text",
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

    tool_functions = SmartDecisionMakerBlock._create_function_signature(
        test_graph.nodes[0].id, test_graph
    )
    assert tool_functions is not None
    assert (
        len(tool_functions) == 2
    ), f"Expected 2 tool functions, got {len(tool_functions)}"

    store_value_function = next(
        filter(lambda x: x["function"]["name"] == "StoreValueBlock", tool_functions)
    )
    print_to_console_function = next(
        filter(lambda x: x["function"]["name"] == "PrintToConsoleBlock", tool_functions)
    )

    assert store_value_function["function"]["name"] == "StoreValueBlock"
    assert (
        store_value_function["function"]["parameters"]["properties"]["input"]["type"]
        == "string"
    )
    assert store_value_function["function"]["parameters"]["required"] == ["input"]

    assert print_to_console_function["function"]["name"] == "PrintToConsoleBlock"
    assert (
        print_to_console_function["function"]["parameters"]["properties"]["text"][
            "type"
        ]
        == "string"
    )
    assert print_to_console_function["function"]["parameters"]["required"] == ["text"]
