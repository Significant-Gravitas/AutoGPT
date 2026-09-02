"""A graph started by an agent turn stays inside that turn's tree.

`run_agent` hands work to the graph executor, which is a different process, so
the spawning turn's envelope contextvar cannot reach the `AutoPilotBlock`
inside the graph. Without the execution context carrying the tree, that block's
copilot turn looks parentless and roots a fresh tree — escaping the depth bound
and the per-tree spend ceiling of the tree that started it.
"""

from __future__ import annotations

import pytest

from backend.blocks.autopilot import _spawner_envelope_from
from backend.copilot.tree import (
    MAX_DEPTH,
    SpawnRequest,
    TreeRefusal,
    derive_child_envelope,
    root_envelope,
)
from backend.data.execution import ExecutionContext


def test_a_user_started_graph_roots_its_own_tree() -> None:
    """No turn started it, so there is nothing to inherit."""
    assert _spawner_envelope_from(ExecutionContext()) is None


def test_the_spawning_turns_tree_is_rebuilt_across_the_process_boundary() -> None:
    context = ExecutionContext(
        copilot_tree_id="tree-1", copilot_tree_depth=2, copilot_tree_tainted=True
    )
    rebuilt = _spawner_envelope_from(context)
    assert rebuilt is not None
    assert (rebuilt.tree_id, rebuilt.depth, rebuilt.tainted) == ("tree-1", 2, True)


def test_a_nested_graph_turn_stays_in_the_tree_and_keeps_counting_depth() -> None:
    """The whole point: the block's turn is a child of the spawning turn, not
    a new root, so the tree id carries and the depth keeps climbing."""
    spawner = derive_child_envelope(
        root_envelope("tree-1"), SpawnRequest(may_spawn=True)
    )
    context = ExecutionContext(
        copilot_tree_id=spawner.tree_id,
        copilot_tree_depth=spawner.depth,
        copilot_tree_tainted=spawner.tainted,
    )
    rebuilt = _spawner_envelope_from(context)
    assert rebuilt is not None
    nested = derive_child_envelope(rebuilt, SpawnRequest(may_spawn=True))
    assert nested.tree_id == "tree-1"
    assert nested.depth == spawner.depth + 1


def test_the_depth_bound_cannot_be_reset_by_going_through_a_graph() -> None:
    """Before this, each hop through run_agent restarted depth at 0."""
    node = root_envelope("tree-1")
    for _ in range(MAX_DEPTH):
        node = derive_child_envelope(node, SpawnRequest(may_spawn=True))
    # Round-trip the exhausted envelope through a graph execution…
    context = ExecutionContext(
        copilot_tree_id=node.tree_id,
        copilot_tree_depth=node.depth,
        copilot_tree_tainted=node.tainted,
    )
    rebuilt = _spawner_envelope_from(context)
    assert rebuilt is not None
    # …and it is still exhausted on the other side.
    with pytest.raises(TreeRefusal):
        derive_child_envelope(rebuilt, SpawnRequest(may_spawn=True))


def test_taint_survives_the_graph_boundary() -> None:
    context = ExecutionContext(copilot_tree_id="t", copilot_tree_tainted=True)
    rebuilt = _spawner_envelope_from(context)
    assert rebuilt is not None
    assert derive_child_envelope(rebuilt, SpawnRequest()).tainted
