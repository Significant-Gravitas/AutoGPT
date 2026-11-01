"""A run's starting inputs are bounded before any row is written.

The names reaching a starting node come from the request, the node's own
defaults and the input masks of whatever started the run, and each becomes a
row of its own.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from backend.blocks.io import AgentInputBlock
from backend.data.graph import GraphModel, NodeModel
from backend.data.includes import MAX_NAMED_INPUTS
from backend.executor.utils import validate_and_construct_node_execution_input
from backend.util.exceptions import InvalidInputError

GRAPH_ID = "graph-1"
NODE_ID = "input-node"
USER_ID = "user-1"


@pytest.fixture
def graph_source(mocker):
    """Every caller reaches the graph through one of these two names."""
    gdb = AsyncMock()
    gdb.validate_graph_execution_permissions = AsyncMock(return_value=None)
    mocker.patch("backend.executor.utils.graph_db", gdb)
    mocker.patch(
        "backend.executor.utils.get_database_manager_async_client", return_value=gdb
    )
    return gdb


async def test_an_oversized_input_mask_is_refused(graph_source):
    """A mask's names land in the node's input, so they count towards it."""
    graph_source.get_graph = AsyncMock(return_value=_graph())
    mask = {f"extra_{i}": i for i in range(MAX_NAMED_INPUTS + 1)}

    with pytest.raises(InvalidInputError, match="at most"):
        await validate_and_construct_node_execution_input(
            graph_id=GRAPH_ID,
            user_id=USER_ID,
            graph_inputs={},
            nodes_input_masks={NODE_ID: mask},
        )


async def test_an_oversized_set_of_node_defaults_is_refused(graph_source):
    """A node's own defaults land there too, with no mask involved."""
    defaults = {f"extra_{i}": i for i in range(MAX_NAMED_INPUTS + 1)}
    graph_source.get_graph = AsyncMock(return_value=_graph(extra_defaults=defaults))

    with pytest.raises(InvalidInputError, match="at most"):
        await validate_and_construct_node_execution_input(
            graph_id=GRAPH_ID,
            user_id=USER_ID,
            graph_inputs={},
        )


async def test_a_mask_just_under_the_bound_still_runs(graph_source):
    graph_source.get_graph = AsyncMock(return_value=_graph())
    mask = {f"extra_{i}": i for i in range(MAX_NAMED_INPUTS - 10)}

    _, starting_nodes_input, _, _ = await validate_and_construct_node_execution_input(
        graph_id=GRAPH_ID,
        user_id=USER_ID,
        graph_inputs={},
        nodes_input_masks={NODE_ID: mask},
    )

    assert len(dict(starting_nodes_input)[NODE_ID]) >= MAX_NAMED_INPUTS - 10


async def test_an_ordinary_run_is_untouched(graph_source):
    graph_source.get_graph = AsyncMock(return_value=_graph())

    _, starting_nodes_input, _, _ = await validate_and_construct_node_execution_input(
        graph_id=GRAPH_ID,
        user_id=USER_ID,
        graph_inputs={"topic": "weather"},
    )

    assert dict(starting_nodes_input)[NODE_ID]["value"] == "weather"


def _graph(extra_defaults: dict | None = None) -> GraphModel:
    return GraphModel(
        id=GRAPH_ID,
        version=1,
        is_active=True,
        name="One input",
        description="",
        user_id=USER_ID,
        created_at=datetime(2026, 11, 1, tzinfo=UTC),
        nodes=[
            NodeModel(
                id=NODE_ID,
                block_id=AgentInputBlock().id,
                graph_id=GRAPH_ID,
                graph_version=1,
                input_default={"name": "topic", **(extra_defaults or {})},
            )
        ],
    )
