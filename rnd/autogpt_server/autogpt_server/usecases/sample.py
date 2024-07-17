from autogpt_server.blocks.basic import PrintingBlock, ValueBlock
from autogpt_server.blocks.text import TextFormatterBlock
from autogpt_server.data import graph
from autogpt_server.data.graph import create_graph
from autogpt_server.util.test import SpinTestServer, wait_execution


def create_test_graph() -> graph.Graph:
    """
    ValueBlock
               \
                 ---- TextFormatterBlock ---- PrintingBlock
               /
    ValueBlock
    """
    nodes = [
        graph.Node(block_id=ValueBlock().id),
        graph.Node(block_id=ValueBlock().id),
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
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[2].id,
            source_name="output",
            sink_name="texts_$_1",
        ),
        graph.Link(
            source_id=nodes[1].id,
            sink_id=nodes[2].id,
            source_name="output",
            sink_name="texts_$_2",
        ),
        graph.Link(
            source_id=nodes[2].id,
            sink_id=nodes[3].id,
            source_name="output",
            sink_name="text",
        ),
    ]

    return graph.Graph(
        name="TestGraph",
        description="Test graph",
        nodes=nodes,
        links=links,
    )


async def sample_agent():
    async with SpinTestServer() as server:
        exec_man = server.exec_manager
        test_graph = await create_graph(create_test_graph())
        input_data = {"input": "test!!"}
        response = await server.agent_server.execute_graph(test_graph.id, input_data)
        print(response)
        result = await wait_execution(exec_man, test_graph.id, response["id"], 4, 10)
        print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(sample_agent())
