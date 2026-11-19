"""Admin-only, read-only routes for the Evals page: the saved runs of the task
eval and its feature eval, read from the files they write. Nothing here runs
an eval (see ``eval_commands`` for why)."""

import asyncio

from autogpt_libs.auth import requires_admin_user
from fastapi import APIRouter, HTTPException, Path, Security

from .eval_files import RUN_ID_PATTERN
from .eval_models import EvalRunDetail, EvalRunList
from .eval_results import get_run, list_runs

router = APIRouter(
    prefix="/admin/evals",
    tags=["admin", "evals"],
    dependencies=[Security(requires_admin_user)],
)


@router.get("/runs", response_model=EvalRunList, summary="List eval runs")
async def list_eval_runs() -> EvalRunList:
    return await asyncio.to_thread(list_runs)


@router.get("/runs/{run_id}", response_model=EvalRunDetail, summary="Get eval run")
async def get_eval_run(
    run_id: str = Path(pattern=RUN_ID_PATTERN, max_length=90),
) -> EvalRunDetail:
    try:
        detail = await asyncio.to_thread(get_run, run_id)
    except (OSError, ValueError) as e:
        raise HTTPException(
            status_code=500, detail="The saved results of this run could not be read."
        ) from e
    if detail is None:
        raise HTTPException(status_code=404, detail="No saved eval run has that id.")
    return detail
