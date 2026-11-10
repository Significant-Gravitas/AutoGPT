"""Where a run sits in a chain of sub-workflow runs, and how deep chains go.

A run waiting on a sub-workflow keeps its graph worker, so a chain holds one
worker per level at once. The executor lets each level below a top-level run
use one more of its owner's workers than the level above it (see
`ExecutionManager._has_worker_share_for`), so the deepest run of a chain can
always start its next level; MAX_SUB_WORKFLOW_DEPTH bounds how many more that
can be.

A chain names workflows, not versions: a run of a workflow already in its
chain is refused whatever version either names. Saving builds its chains from
the version each node names (see `GraphModel._sub_workflow_cycle_errors`), so
this check is the backstop for versions that aren't loaded together.
"""

from collections.abc import Sequence

from backend.data.execution import ExecutionContext, GraphExecutionEntry
from backend.data.graph import SUB_WORKFLOW_RUNS_ITSELF
from backend.util.exceptions import GraphValidationError

# How many levels of sub-workflows may run below a top-level run. With the
# default share of 5 of 10 workers, one account's runs then hold at most 9.
MAX_SUB_WORKFLOW_DEPTH = 4

# A level of a chain whose run no longer exists, so its workflow isn't known.
# No workflow has this id.
_GONE_RUN = ""


def nested_context(parent: ExecutionContext, graph_id: str) -> ExecutionContext:
    """The context of a run of `graph_id` that a node of `parent`'s run starts.

    Raises GraphValidationError for a workflow already running in the chain,
    which would start itself again without end, and for a run deeper than
    MAX_SUB_WORKFLOW_DEPTH.
    """
    lineage = [*parent.ancestor_graph_ids]
    if parent.graph_id:
        lineage.append(parent.graph_id)
    if _GONE_RUN in lineage:
        raise GraphValidationError(
            "This run can't start sub-workflows: a run it is nested in no "
            "longer exists"
        )
    if graph_id in lineage:
        raise GraphValidationError(SUB_WORKFLOW_RUNS_ITSELF)
    if len(lineage) > MAX_SUB_WORKFLOW_DEPTH:
        raise GraphValidationError(
            f"Sub-workflows can be nested at most {MAX_SUB_WORKFLOW_DEPTH} "
            "levels deep"
        )
    return parent.model_copy(update={"ancestor_graph_ids": lineage})


def resumed_chain(nesting: Sequence[str | None]) -> list[str]:
    """The chain of a resumed sub-workflow run, outermost first, from the
    workflows of the runs it is nested in, nearest first, ending in None
    where a parent run no longer exists.

    A run that's gone takes the levels above it with it, so how deep the run
    sits isn't known: it is placed as deep as chains go. It carries on,
    counted in its owner's share, but starts no sub-workflow of its own."""
    known = [graph_id for graph_id in reversed(nesting) if graph_id is not None]
    if nesting and nesting[-1] is None:
        missing = max(1, MAX_SUB_WORKFLOW_DEPTH - len(known))
        return [_GONE_RUN] * missing + known
    return known


def sub_workflow_depth(run: GraphExecutionEntry) -> int | None:
    """How many levels of sub-workflows a queued run sits below its top-level
    run, or None for a sub-workflow run queued before runs carried their
    chain, whose depth isn't known.

    Every sub-workflow run started since carries its chain, a resumed one
    included, so None is only ever an earlier run still in the queue."""
    context = run.execution_context
    if context.ancestor_graph_ids:
        return min(len(context.ancestor_graph_ids), MAX_SUB_WORKFLOW_DEPTH)
    if context.root_execution_id not in (None, run.graph_exec_id):
        return None
    return 0
