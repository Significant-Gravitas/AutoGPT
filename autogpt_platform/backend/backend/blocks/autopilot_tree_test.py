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


def _cross_the_boundary(envelope):
    """Round-trip an envelope through the execution payload, as run_agent →
    graph executor → AutoPilotBlock does."""
    context = ExecutionContext(
        copilot_tree_id=envelope.tree_id,
        copilot_tree_depth=envelope.depth,
        copilot_tree_tainted=envelope.tainted,
        copilot_tree_tools=(
            sorted(envelope.tools) if envelope.tools is not None else None
        ),
    )
    return _spawner_envelope_from(context)


@pytest.mark.parametrize(
    "spawn",
    [
        SpawnRequest(may_spawn=True),
        SpawnRequest(shares_memory=True, may_spawn=True),
        # Pinned sets must keep a spawn tool, or the rebuilt turn cannot spawn
        # at all and there is nothing to compare on the far side.
        SpawnRequest(tools=["read_workspace_file", "run_sub_session"], may_spawn=True),
    ],
    ids=["delegate", "isolate", "pinned"],
)
def test_the_tool_ceiling_survives_the_graph_boundary(spawn: SpawnRequest) -> None:
    """The rebuilt turn may never hold more than it did before crossing.

    ``TurnEnvelope.tools=None`` is the ROOT sentinel meaning unrestricted, so a
    rebuild that leaves it defaulted silently reopens the whole registry — the
    exact amplification the envelope exists to prevent, arriving through the
    one door that has no contextvar.
    """
    before = derive_child_envelope(root_envelope("t"), spawn)
    assert before.tools is not None
    rebuilt = _cross_the_boundary(before)
    assert rebuilt is not None
    assert rebuilt.tools == before.tools

    after = derive_child_envelope(rebuilt, SpawnRequest(may_spawn=True))
    assert after.tools is not None
    assert after.tools <= before.tools, sorted(after.tools - before.tools)


@pytest.mark.parametrize("denied", ["DESCENT", "ISOLATE"])
def test_denied_tools_are_not_regained_by_crossing(denied: str) -> None:
    """Neither denial list is undone by a trip through the graph executor."""
    from backend.copilot.tree import DESCENT_DENIED_TOOLS, ISOLATE_DENIED_TOOLS

    denied_set = DESCENT_DENIED_TOOLS if denied == "DESCENT" else ISOLATE_DENIED_TOOLS
    before = derive_child_envelope(
        root_envelope("t"), SpawnRequest(shares_memory=True, may_spawn=True)
    )
    assert before.tools is not None and before.tools.isdisjoint(denied_set)
    rebuilt = _cross_the_boundary(before)
    assert rebuilt is not None
    after = derive_child_envelope(rebuilt, SpawnRequest(may_spawn=True))
    assert after.tools is not None
    assert after.tools.isdisjoint(denied_set), sorted(after.tools & denied_set)


def test_an_unrestricted_root_stays_unrestricted_across_the_boundary() -> None:
    """None must survive as None — a root genuinely holds everything, and
    turning it into an empty set would strip a legitimate run of every tool."""
    rebuilt = _cross_the_boundary(root_envelope("t"))
    assert rebuilt is not None
    assert rebuilt.tools is None


def test_recovery_re_derives_a_child_rather_than_an_unrestricted_root() -> None:
    """Retrying an orphaned sub-agent must not widen its authority.

    The orphaned turn's own envelope died with its worker, but the tree that
    spawned it is recoverable from the execution context. Re-rooting instead
    would hand the retry the full registry — a restricted child coming back
    unrestricted purely because its worker crashed.
    """
    spawner = derive_child_envelope(
        root_envelope("t"), SpawnRequest(shares_memory=True, may_spawn=True)
    )
    context = ExecutionContext(
        copilot_tree_id=spawner.tree_id,
        copilot_tree_depth=spawner.depth,
        copilot_tree_tainted=spawner.tainted,
        copilot_tree_tools=sorted(spawner.tools or ()),
    )
    rebuilt = _spawner_envelope_from(context)
    assert rebuilt is not None

    recovery = derive_child_envelope(rebuilt, SpawnRequest(may_spawn=True))
    assert recovery.tree_id == spawner.tree_id
    assert recovery.depth == spawner.depth + 1
    assert recovery.tools is not None
    assert recovery.tools <= (spawner.tools or frozenset())
    # It is emphatically not a root: None is the root sentinel.
    assert recovery.tools is not None
