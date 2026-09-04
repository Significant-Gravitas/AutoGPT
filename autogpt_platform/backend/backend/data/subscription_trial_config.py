"""Validated, versioned terms for a card-required subscription trial."""

import logging
from datetime import UTC, datetime
from hashlib import sha256
from typing import Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from backend.util.feature_flag import Flag, get_feature_flag_value

logger = logging.getLogger(__name__)


class TrialOffer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_.-]+$")
    new_users_from: AwareDatetime
    duration_days: int = Field(ge=1, le=730, strict=True)
    tier: Literal["BASIC", "PRO", "MAX", "BUSINESS"]
    billing_cycle: Literal["monthly", "yearly"]
    daily_cost_limit: int = Field(gt=0, strict=True)
    weekly_cost_limit: int = Field(gt=0, strict=True)
    total_cost_limit: int = Field(gt=0, strict=True)
    onboarding_credit_amount: int = Field(ge=0, le=2_147_483_647, strict=True)
    allow_existing_beta_users: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def ordered_limits(self) -> "TrialOffer":
        if not self.daily_cost_limit <= self.weekly_cost_limit <= self.total_cost_limit:
            raise ValueError("Trial limits must satisfy daily <= weekly <= total")
        return self

    def is_eligible(
        self,
        *,
        created_at: datetime,
        current_tier: str,
        has_subscription_history: bool,
    ) -> bool:
        if current_tier != "NO_TIER" or has_subscription_history:
            return False
        return created_at >= self.new_users_from or self.allow_existing_beta_users


class AcceptedTrialOffer(TrialOffer):
    price_id: str = Field(pattern=r"^price_", min_length=7)
    unit_amount: int = Field(gt=0, strict=True)
    currency: str = Field(pattern=r"^[a-z]{3}$")

    @property
    def token(self) -> str:
        return sha256(self.model_dump_json().encode()).hexdigest()


async def get_trial_offer(user_id: str) -> TrialOffer | None:
    try:
        raw = await get_feature_flag_value(
            Flag.CARD_REQUIRED_TRIAL_OFFER, user_id, None
        )
        if raw is None:
            return None
        return TrialOffer.model_validate(raw)
    except (ValidationError, ValueError, TypeError):
        logger.error("Invalid card-required-trial-offer; refusing trial enrollment")
        return None
    except Exception:
        logger.exception("Trial offer unavailable; refusing trial enrollment")
        return None


def trial_is_active(
    *,
    status: str,
    trial_end: datetime | None,
    card_verified: bool,
    now: datetime | None = None,
) -> bool:
    return bool(
        status == "trialing"
        and card_verified
        and trial_end is not None
        and trial_end > (now or datetime.now(UTC))
    )
