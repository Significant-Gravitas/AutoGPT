import pytest

from autogpt_server.data import block, db, graph
from autogpt_server.data.execution import ExecutionQueue, add_execution
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
        graph.Node(block_id=block.ParrotBlock.id),
        graph.Node(block_id=block.ParrotBlock.id),
        graph.Node(
            block_id=block.TextCombinerBlock.id,
            input_default={"format": "{text1},{text2}"}
        ),
        graph.Node(block_id=block.PrintingBlock.id),
    ]
    nodes[0].connect(nodes[2], "output", "text1")
    nodes[1].connect(nodes[2], "output", "text2")
    nodes[2].connect(nodes[3], "combined_text", "text")

    test_graph = graph.Graph(
        name="TestGraph",
        description="Test graph",
        nodes=nodes,
    )
    await block.initialize_blocks()
    result = await graph.create_graph(test_graph)

    # Assertions
    assert result.name == test_graph.name
    assert result.description == test_graph.description
    assert len(result.nodes) == len(test_graph.nodes)

    return result


async def execute_node(queue: ExecutionQueue) -> dict | None:
    next_exec = await executor.execute_node(queue.get())
    if not next_exec:
        return None
    await add_execution(next_exec, queue)
    return next_exec.data


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
    assert next_execution
    assert next_execution.keys() == {"text1", "text2", "format"}
    assert next_execution["text1"] == text
    assert next_execution["text2"] == text
    assert next_execution["format"] == "{text1},{text2}"

    # Executing TextCombinerBlock, PrintingBlock will be enqueued.
    next_execution = await execute_node(test_queue)
    assert next_execution
    assert next_execution.keys() == {"text"}
    assert next_execution["text"] == f"{text},{text}"

    # Executing PrintingBlock, no more tasks will be enqueued.
    next_execution = await execute_node(test_queue)
    assert next_execution is None
