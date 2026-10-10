from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from backend.data.subscription_trial_config import AcceptedTrialOffer
from backend.data.subscription_trial_rejection import TrialRejectionReason


class TrialOfferResponse(BaseModel):
    token: str
    version: str
    duration_days: int
    tier: Literal["BASIC", "PRO", "MAX", "BUSINESS"]
    billing_cycle: Literal["monthly", "yearly"]
    unit_amount: int
    currency: str
    onboarding_credit_amount: int

    @classmethod
    def from_offer(cls, offer: AcceptedTrialOffer) -> "TrialOfferResponse":
        return cls(**offer.model_dump(), token=offer.token)


class TrialStatusResponse(BaseModel):
    eligible: bool = False
    offer: TrialOfferResponse | None = None
    status: str | None = None
    rejection_reason: TrialRejectionReason | None = None
    ends_at: datetime | None = None
    cancel_at_period_end: bool = False
    cancel_keeps_access: bool = Field(
        default=False,
        description=(
            "True when canceling schedules the trial's end: access runs to"
            " ends_at, the card is never charged, and the trial can be resumed"
            " until then. False when canceling ends trial access immediately."
        ),
    )
    allowance_used_percent: float | None = None
    active: bool = False
    converted: bool = False
    onboarding_credits_previously_received: bool = False


class TrialCheckoutRequest(BaseModel):
    offer_token: str = Field(pattern=r"^[a-f0-9]{64}$")
    return_to: Literal["onboarding", "billing"] = "billing"


class TrialCheckoutResponse(BaseModel):
    url: str
