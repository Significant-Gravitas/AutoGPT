"""Experiment assignment API.

The backend buckets the signed-in user by their id and stores the arm on first
read, so the frontend shows the same one on every page load, consent or not.
"""

import logging
from typing import Annotated

import fastapi
from autogpt_libs.auth import get_user_id
from autogpt_libs.auth.dependencies import requires_user

from backend.data import experiments as experiments_db
from backend.data.experiments import Experiment, ExperimentAssignment

router = fastapi.APIRouter(dependencies=[fastapi.Security(requires_user)])
logger = logging.getLogger(__name__)


@router.get(
    "/assignments",
    summary="List experiment assignments",
    response_model=list[ExperimentAssignment],
)
async def list_experiment_assignments(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
) -> list[ExperimentAssignment]:
    return await experiments_db.list_assignments(user_id)


@router.get(
    "/assignments/{experiment}",
    summary="Get experiment assignment",
    response_model=ExperimentAssignment | None,
)
async def get_experiment_assignment(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    experiment: Experiment,
) -> ExperimentAssignment | None:
    """The user's arm, bucketed on first read and fixed from then on; None if not enrolled."""
    return await experiments_db.assign(user_id, experiment)
