"""Keep delayed foreground usage attached to the trial that started the turn.

This is attribution, not a spend reservation or a durable settlement queue.
Async child tasks inherit the snapshot even after their parent turn exits.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar

from pydantic import BaseModel, ConfigDict

from backend.data import db_accessors


class TrialCostContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str | None
    trial_id: str | None


_trial_cost_context: ContextVar[TrialCostContext | None] = ContextVar(
    "trial_cost_context", default=None
)


@asynccontextmanager
async def trial_cost_context(user_id: str | None) -> AsyncIterator[None]:
    trial = (
        await db_accessors.credit_db().get_subscription_trial(user_id)
        if user_id is not None
        else None
    )
    context = TrialCostContext(
        user_id=user_id,
        trial_id=(
            trial.id
            if trial is not None and trial.active and trial.consumed_at is not None
            else None
        ),
    )
    token = _trial_cost_context.set(context)
    try:
        yield
    finally:
        _trial_cost_context.reset(token)


def get_trial_cost_context(user_id: str | None) -> TrialCostContext | None:
    context = _trial_cost_context.get()
    if context is not None and context.user_id != user_id:
        raise ValueError("Trial cost attribution belongs to a different user")
    return context


async def record_attributed_trial_cost(user_id: str, cost_microdollars: int) -> bool:
    """Return whether the turn has a snapshot, including explicitly non-trial work."""
    context = get_trial_cost_context(user_id)
    if context is None:
        return False
    if context.trial_id is not None:
        await db_accessors.credit_db().record_subscription_trial_cost(
            user_id, cost_microdollars, trial_id=context.trial_id
        )
    return True
