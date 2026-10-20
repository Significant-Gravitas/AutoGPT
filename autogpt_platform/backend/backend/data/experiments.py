"""Experiment arm assignments: which variant a user saw, recorded once.

PostHog owns bucketing and significance testing. This table is the durable,
joinable copy of the arm so the ``analytics.*`` views can split activation,
retention and cost by variant without depending on the flag provider's
history. First write wins: a user's arm is fixed the moment it is observed.
"""

import logging
from datetime import datetime
from typing import Literal

import prisma.models
import prisma.types
from prisma.errors import UniqueViolationError
from pydantic import BaseModel

logger = logging.getLogger(__name__)

AssignmentSource = Literal["posthog", "launchdarkly", "backend"]


class ExperimentAssignment(BaseModel):
    experiment_key: str
    variant: str
    source: str
    assigned_at: datetime

    @classmethod
    def from_db(cls, row: prisma.models.ExperimentAssignment) -> "ExperimentAssignment":
        return cls(
            experiment_key=row.experimentKey,
            variant=row.variant,
            source=row.source,
            assigned_at=row.createdAt,
        )


async def record_assignment(
    user_id: str,
    experiment_key: str,
    variant: str,
    source: AssignmentSource = "posthog",
) -> ExperimentAssignment:
    where: prisma.types.ExperimentAssignmentWhereUniqueInput = {
        "userId_experimentKey": {"userId": user_id, "experimentKey": experiment_key}
    }
    existing = await prisma.models.ExperimentAssignment.prisma().find_unique(
        where=where
    )
    if existing is None:
        existing = await _create_first_assignment(
            where, user_id, experiment_key, variant, source
        )
    if existing.variant != variant:
        # The flag provider re-bucketed the user (rollout changed, flag
        # edited). Keep the arm they actually experienced first so the
        # analysis stays intention-to-treat.
        logger.info(
            f"Experiment {experiment_key}: keeping first-seen variant "
            f"{existing.variant} for user {user_id} (now {variant})"
        )
    return ExperimentAssignment.from_db(existing)


async def _create_first_assignment(
    where: prisma.types.ExperimentAssignmentWhereUniqueInput,
    user_id: str,
    experiment_key: str,
    variant: str,
    source: AssignmentSource,
) -> prisma.models.ExperimentAssignment:
    try:
        return await prisma.models.ExperimentAssignment.prisma().create(
            data=prisma.types.ExperimentAssignmentCreateInput(
                userId=user_id,
                experimentKey=experiment_key,
                variant=variant,
                source=source,
            )
        )
    except UniqueViolationError:
        # Two tabs reported the same experiment at once; the other write won.
        existing = await prisma.models.ExperimentAssignment.prisma().find_unique(
            where=where
        )
        if existing is None:
            raise
        return existing


async def list_assignments(user_id: str) -> list[ExperimentAssignment]:
    rows = await prisma.models.ExperimentAssignment.prisma().find_many(
        where={"userId": user_id}, order={"createdAt": "asc"}
    )
    return [ExperimentAssignment.from_db(row) for row in rows]
