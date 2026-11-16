"""End to end through the executor service: when one block's output feeds
many blocks, each of them runs once with that value. That includes a block fed
twice by the same output, and more blocks than one output registers at once."""

import pytest

from backend.api.model import CreateGraph
from backend.blocks.basic import StoreValueBlock
from backend.blocks.maths import CalculatorBlock, Operation
from backend.data import execution, graph
from backend.executor.manager import MAX_SINKS_REGISTERED_AT_ONCE
from backend.usecases.sample import create_test_user
from backend.util.test import SpinTestServer, wait_execution


@pytest.fixture(autouse=True)
def _runners_belong_to_their_org():
    """Check memberships for real, as in ``manager_test.py``: the executor
    service checks a run's org in its own process, out of a patch's reach."""


@pytest.fixture(autouse=True)
def _default_team_untenanted():
    """Look up the user's own org and team for real, as in ``manager_test.py``."""


@pytest.mark.asyncio(loop_scope="session")
async def test_an_output_that_feeds_many_blocks_runs_each_of_them_once(
    server: SpinTestServer,
) -> None:
    source = graph.Node(block_id=StoreValueBlock().id, input_default={"input": 3})
    doubled = graph.Node(
        block_id=CalculatorBlock().id,
        input_default={"operation": Operation.ADD.value},
    )
    tenfold = graph.Node(
        block_id=CalculatorBlock().id,
        input_default={"operation": Operation.MULTIPLY.value, "b": 10},
    )
    copies = [
        graph.Node(block_id=StoreValueBlock().id)
        for _ in range(MAX_SINKS_REGISTERED_AT_ONCE + 1)
    ]

    def from_source(sink: graph.Node, pin: str) -> graph.Link:
        return graph.Link(
            source_id=source.id, sink_id=sink.id, source_name="output", sink_name=pin
        )

    links = [
        from_source(doubled, "a"),
        from_source(doubled, "b"),
        from_source(tenfold, "a"),
        *(from_source(copy, "input") for copy in copies),
    ]
    user = await create_test_user()
    created = await server.agent_server.test_create_graph(
        CreateGraph(
            graph=graph.Graph(
                name="Fan-out",
                description="One output feeds many blocks",
                nodes=[source, doubled, tenfold, *copies],
                links=links,
            )
        ),
        user.id,
    )
    run = await server.agent_server.test_execute_graph(
        user_id=user.id,
        graph_id=created.id,
        graph_version=created.version,
        node_input={},
    )

    node_runs = await wait_execution(user.id, run.id, 60)

    source_id, doubled_id, tenfold_id, *copy_ids = [node.id for node in created.nodes]
    runs_by_node: dict[str, list[execution.NodeExecutionResult]] = {}
    for node_run in node_runs:
        runs_by_node.setdefault(node_run.node_id, []).append(node_run)
    assert sorted(runs_by_node) == sorted(
        [source_id, doubled_id, tenfold_id, *copy_ids]
    )
    assert all(len(runs) == 1 for runs in runs_by_node.values())
    assert all(
        runs[0].status == execution.ExecutionStatus.COMPLETED
        for runs in runs_by_node.values()
    )
    assert runs_by_node[doubled_id][0].output_data == {"result": [6]}
    assert runs_by_node[tenfold_id][0].output_data == {"result": [30]}
    assert all(
        runs_by_node[copy_id][0].output_data == {"output": [3]} for copy_id in copy_ids
    )
