import pytest

from autogpt_server.data import block, db, graph
from autogpt_server.data.execution import Execution, ExecutionQueue, add_execution
from autogpt_server.executor import executor
from autogpt_server.server import server


async def create_test_graph() -> graph.Graph:
    """
    ParrotBlock
                \
                 ---- TextCombinerBlock ---- PrintingBlock
                /
    ParrotBlock
    """
    nodes = [
        graph.Node(block_name="ParrotBlock"),
        graph.Node(block_name="ParrotBlock"),
        graph.Node(
            block_name="TextCombinerBlock", input_default={"format": "{text1},{text2}"}
        ),
        graph.Node(block_name="PrintingBlock"),
    ]
    edges = [
        graph.Edge(
            input_node=nodes[0].id,
            input_name="output",
            output_node=nodes[2].id,
            output_name="text1",
        ),
        graph.Edge(
            input_node=nodes[1].id,
            input_name="output",
            output_node=nodes[2].id,
            output_name="text2",
        ),
        graph.Edge(
            input_node=nodes[2].id,
            input_name="combined_text",
            output_node=nodes[3].id,
            output_name="text",
        ),
    ]
    test_graph = graph.Graph(
        name="TestGraph", description="Test graph", nodes=nodes, edges=edges
    )
    await block.initialize_blocks()
    result = await graph.create_graph(test_graph)

    # Assertions
    assert result.name == test_graph.name
    assert result.description == test_graph.description
    assert len(result.nodes) == len(test_graph.nodes)
    assert len(result.edges) == len(test_graph.edges)

    return result


async def execute_node(queue: ExecutionQueue) -> Execution | None:
    next_exec = await executor.execute_node(queue.get())
    if next_exec:
        return await add_execution(next_exec, queue)
    return None


@pytest.mark.asyncio
async def test_agent_execution():
    await db.connect()
    test_graph = await create_test_graph()
    test_queue = ExecutionQueue()
    test_server = server.AgentServer(test_queue)

    # --- Test adding new executions --- #
    text = "Hello, World!"
    input_data = {"input": text}
    executions = await test_server.execute_agent(test_graph.id, input_data)

    # 2 executions should be created, one for each ParrotBlock, with same run_id.
    assert len(executions) == 2
    assert executions[0].run_id == executions[1].run_id
    assert executions[0].node_id != executions[1].node_id
    assert executions[0].data == executions[1].data == input_data

    # --- Test Executing added tasks --- #

    # Executing ParrotBlock1, TextCombinerBlock won't be enqueued yet.
    assert not test_queue.empty()
    next_execution = await execute_node(test_queue)
    assert next_execution is None

    # Executing ParrotBlock2, TextCombinerBlock will be enqueued.
    assert not test_queue.empty()
    next_execution = await execute_node(test_queue)
    assert test_queue.empty()
    assert next_execution.data.keys() == {"text1", "text2", "format"}
    assert next_execution.data["text1"] == text
    assert next_execution.data["text2"] == text
    assert next_execution.data["format"] == "{text1},{text2}"

    # Executing TextCombinerBlock, PrintingBlock will be enqueued.
    next_execution = await execute_node(test_queue)
    assert next_execution.data.keys() == {"text"}
    assert next_execution.data["text"] == f"{text},{text}"

    # Executing PrintingBlock, no more tasks will be enqueued.
    next_execution = await execute_node(test_queue)
    assert next_execution is None
