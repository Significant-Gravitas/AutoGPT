"""Reading, previewing and changing a workflow page, for the routes and Otto's tool.

Data is read through ``db_accessors``, so the same code runs in the API, where
the database is local, and in the copilot executor, where it is an RPC.
"""

from pydantic import ValidationError

from backend.api.features.library.model import LibraryAgent
from backend.data.db_accessors import execution_db, library_db, workflow_pages_db
from backend.data.execution import ExecutionStatus, GraphExecution
from backend.util.exceptions import NotFoundError

from . import db as pages_db
from .design import WorkflowPageDesign
from .page import ResolvedWorkflowPage, resolve_page

_RECENT_RUNS_FETCHED = 15


async def read_page(
    user_id: str, library_agent_id: str, *, run_id: str | None = None
) -> ResolvedWorkflowPage:
    """The stored page, resolved; *run_id* feeds the result from that run."""
    agent = await library_db().get_library_agent(library_agent_id, user_id)
    stored = await workflow_pages_db().get_workflow_page(user_id, library_agent_id)
    if stored is None:
        raise NotFoundError(f"No page for library agent #{library_agent_id}")
    try:
        design = WorkflowPageDesign.model_validate(stored.design)
    except ValidationError as e:
        raise NotFoundError("This page's design can no longer be read") from e
    return await build_page(
        user_id,
        agent,
        design,
        run_id=run_id,
        has_previous=stored.previous_design is not None,
    )


async def build_page(
    user_id: str,
    agent: LibraryAgent,
    design: WorkflowPageDesign,
    *,
    run_id: str | None = None,
    has_previous: bool = False,
) -> ResolvedWorkflowPage:
    result_run = await _result_run(user_id, agent, run_id)
    recent = await execution_db().get_graph_executions(
        graph_id=agent.graph_id, user_id=user_id, limit=_RECENT_RUNS_FETCHED
    )
    return resolve_page(
        design,
        agent,
        result_run,
        [run for run in recent if not run.is_dry_run],
        has_previous=has_previous,
    )


async def remove_section(
    user_id: str, library_agent_id: str, index: int
) -> ResolvedWorkflowPage:
    stored = await pages_db.get_workflow_page(user_id, library_agent_id)
    if stored is None:
        raise NotFoundError(f"No page for library agent #{library_agent_id}")
    design = WorkflowPageDesign.model_validate(stored.design)
    if not 0 <= index < len(design.sections):
        raise NotFoundError(f"This page has no section {index}")
    if len(design.sections) == 1:
        raise ValueError("A page needs one section; reset to the stock view instead")
    kept = design.model_copy(
        update={"sections": [s for i, s in enumerate(design.sections) if i != index]}
    )
    await pages_db.save_workflow_page(
        user_id, library_agent_id, kept.model_dump(exclude_none=True)
    )
    return await read_page(user_id, library_agent_id)


async def restore_previous(user_id: str, library_agent_id: str) -> ResolvedWorkflowPage:
    await pages_db.restore_workflow_page(user_id, library_agent_id)
    return await read_page(user_id, library_agent_id)


async def _result_run(
    user_id: str, agent: LibraryAgent, run_id: str | None
) -> GraphExecution | None:
    if run_id is None:
        finished = await execution_db().get_graph_executions(
            graph_id=agent.graph_id,
            user_id=user_id,
            statuses=[ExecutionStatus.COMPLETED],
            limit=5,
        )
        run_id = next((run.id for run in finished if not run.is_dry_run), None)
        if run_id is None:
            return None
    run = await execution_db().get_graph_execution(user_id=user_id, execution_id=run_id)
    if run is None or run.graph_id != agent.graph_id:
        raise NotFoundError(f"Run #{run_id} not found for this workflow")
    return run
