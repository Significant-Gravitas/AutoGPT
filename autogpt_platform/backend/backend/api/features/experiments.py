"""Experiment assignment API.

The frontend evaluates experiment flags through PostHog and reports the arm
it showed the user here, once per experiment, so the assignment is available
to server-side analytics and the SQL views.
"""

import logging
from typing import Annotated

import fastapi
import pydantic
from autogpt_libs.auth import get_user_id
from autogpt_libs.auth.dependencies import requires_user

from backend.data import experiments as experiments_db
from backend.data.experiments import AssignmentSource, ExperimentAssignment

router = fastapi.APIRouter(dependencies=[fastapi.Security(requires_user)])
logger = logging.getLogger(__name__)


class RecordAssignmentRequest(pydantic.BaseModel):
    experiment_key: str = pydantic.Field(..., min_length=1, max_length=128)
    variant: str = pydantic.Field(..., min_length=1, max_length=128)
    source: AssignmentSource = "posthog"


@router.get(
    "/assignments",
    summary="List experiment assignments",
    response_model=list[ExperimentAssignment],
)
async def list_experiment_assignments(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
) -> list[ExperimentAssignment]:
    return await experiments_db.list_assignments(user_id)


@router.post(
    "/assignments",
    summary="Record experiment assignment",
    response_model=ExperimentAssignment,
)
async def record_experiment_assignment(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    request: RecordAssignmentRequest,
) -> ExperimentAssignment:
    return await experiments_db.record_assignment(
        user_id=user_id,
        experiment_key=request.experiment_key,
        variant=request.variant,
        source=request.source,
    )
