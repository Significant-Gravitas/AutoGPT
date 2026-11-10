"""Where a run sits in a chain of sub-workflow runs, and how deep chains go.

A run waiting on a sub-workflow keeps its graph worker, so a chain holds one
worker per level at once. The executor lets each level below a top-level run
use one more of its owner's workers than the level above it (see
`ExecutionManager._has_worker_share_for`), so the deepest run of a chain can
always start its next level; MAX_SUB_WORKFLOW_DEPTH bounds how many more that
can be.
"""

from backend.data.execution import ExecutionContext, GraphExecutionEntry
from backend.data.graph import SUB_WORKFLOW_RUNS_ITSELF
from backend.util.exceptions import GraphValidationError

# How many levels of sub-workflows may run below a top-level run. With the
# default share of 5 of 10 workers, one account's runs then hold at most 9.
MAX_SUB_WORKFLOW_DEPTH = 4


def nested_context(parent: ExecutionContext, graph_id: str) -> ExecutionContext:
    """The context of a run of `graph_id` that a node of `parent`'s run starts.

    Raises GraphValidationError for a workflow already running in the chain,
    which would start itself again without end, and for a run deeper than
    MAX_SUB_WORKFLOW_DEPTH.
    """
    lineage = [*parent.ancestor_graph_ids]
    if parent.graph_id:
        lineage.append(parent.graph_id)
    if graph_id in lineage:
        raise GraphValidationError(SUB_WORKFLOW_RUNS_ITSELF)
    if len(lineage) > MAX_SUB_WORKFLOW_DEPTH:
        raise GraphValidationError(
            f"Sub-workflows can be nested at most {MAX_SUB_WORKFLOW_DEPTH} "
            "levels deep"
        )
    return parent.model_copy(update={"ancestor_graph_ids": lineage})


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
