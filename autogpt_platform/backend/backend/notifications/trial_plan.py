"""The plan a trial notice is about."""

from typing import Literal

from pydantic import BaseModel

from backend.data.subscription_trial import TrialState
from backend.data.subscription_trial_payment import SubscriptionPlanTerms


class NoticePlan(BaseModel):
    tier: str
    cycle: Literal["monthly", "yearly"]
    unit_amount: int | None
    currency: str


def notice_plan(trial: TrialState, kind: str, current: dict | None) -> NoticePlan:
    """The accepted plan, or for a conversion the plan the customer chose to
    end the trial on in its place."""
    offer = trial.offer
    chosen = (
        SubscriptionPlanTerms.model_validate(current).chosen_plan()
        if kind == "converted" and current
        else None
    )
    if chosen is None:
        return NoticePlan(
            tier=offer.tier,
            cycle=offer.billing_cycle,
            unit_amount=offer.unit_amount,
            currency=offer.currency,
        )
    recurring = chosen.price.recurring
    return NoticePlan(
        tier=chosen.tier.value,
        cycle="yearly" if recurring and recurring.interval == "year" else "monthly",
        unit_amount=chosen.price.unit_amount,
        currency=chosen.price.currency or offer.currency,
    )
