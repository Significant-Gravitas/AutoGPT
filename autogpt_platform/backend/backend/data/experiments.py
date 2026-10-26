"""Experiment arm assignments: which variant a user saw, recorded once.

PostHog owns bucketing and significance testing. The backend reads the arm by
user id on first request and stores it here, so it stays the same whether or not
the browser may tell PostHog who the user is, and the ``analytics.*`` views can
split activation, retention and cost by variant. First write wins.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Literal

import prisma.models
import prisma.types
from prisma.errors import UniqueViolationError
from pydantic import BaseModel, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from backend.util.feature_flag import evaluate_experiment_variant

logger = logging.getLogger(__name__)

AssignmentSource = Literal["posthog", "launchdarkly", "backend"]


class Experiment(str, Enum):
    """Experiments, each a multivariate PostHog flag read by the frontend or backend."""

    SUBSCRIPTION_PRICING = "subscription-pricing-page-initial-state"
    # Backend-decided: the arm sets the trial's Stripe Checkout text.
    TRIAL_CHECKOUT_NO_CHARGE_COPY = "trial-checkout-no-charge-copy"

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # Orval names the generated const's members from this.
        schema = handler(core_schema)
        handler.resolve_ref_schema(schema)["x-enum-varnames"] = [m.name for m in cls]
        return schema


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


async def assign(user_id: str, experiment: Experiment) -> ExperimentAssignment | None:
    """The user's arm: the stored one, else PostHog's bucket for them, stored now.

    None while they are not enrolled, with nothing stored, so a later read can enrol them.
    """
    stored = await prisma.models.ExperimentAssignment.prisma().find_unique(
        where={
            "userId_experimentKey": {
                "userId": user_id,
                "experimentKey": experiment.value,
            }
        }
    )
    if stored is not None:
        return ExperimentAssignment.from_db(stored)
    variant = await evaluate_experiment_variant(experiment.value, user_id)
    if variant is None:
        return None
    return await record_assignment(user_id, experiment.value, variant, source="backend")


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
