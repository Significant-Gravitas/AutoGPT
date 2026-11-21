"""End to end through the executor service: a run keeps every key of the data
it is given, including keys starting with "_sa", such as ``_sales_total``.

FastAPI's ``jsonable_encoder`` drops such keys by default, at any depth, and a
run's data went through it on every hop: the database write, the executor's own
encoding of each block output, and the database manager's RPC both ways."""

import pytest

from backend.api.model import CreateGraph
from backend.blocks.io import AgentInputBlock, AgentOutputBlock
from backend.data import execution, graph
from backend.usecases.sample import create_test_user
from backend.util.test import SpinTestServer, wait_execution

REPORT = {"_sales_total": 42, "rows": [{"_sa_region": "EMEA"}]}


@pytest.fixture(autouse=True)
def _runners_belong_to_their_org():
    """Check memberships for real, as in ``manager_test.py``: the executor
    service checks a run's org in its own process, out of a patch's reach."""


@pytest.fixture(autouse=True)
def _default_team_untenanted():
    """Look up the user's own org and team for real, as in ``manager_test.py``."""


@pytest.mark.asyncio(loop_scope="session")
async def test_a_run_keeps_keys_starting_with_sa_from_its_input_to_its_output(
    server: SpinTestServer,
) -> None:
    given = graph.Node(block_id=AgentInputBlock().id, input_default={"name": "report"})
    shown = graph.Node(block_id=AgentOutputBlock().id, input_default={"name": "report"})
    user = await create_test_user()
    created = await server.agent_server.test_create_graph(
        CreateGraph(
            graph=graph.Graph(
                name="Report",
                description="Passes a report through",
                nodes=[given, shown],
                links=[
                    graph.Link(
                        source_id=given.id,
                        sink_id=shown.id,
                        source_name="result",
                        sink_name="value",
                    )
                ],
            )
        ),
        user.id,
    )
    run = await server.agent_server.test_execute_graph(
        user_id=user.id,
        graph_id=created.id,
        graph_version=created.version,
        node_input={"report": REPORT},
    )

    await wait_execution(user.id, run.id, 60)

    stored = await execution.get_graph_execution(user.id, run.id)
    assert stored is not None
    assert stored.inputs == {"report": REPORT}
    assert stored.outputs == {"report": [REPORT]}
