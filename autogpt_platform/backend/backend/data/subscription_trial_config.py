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

from backend.util.country import country_code
from backend.util.feature_flag import Flag, get_feature_flag_value, is_feature_enabled

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

    # Omit (or null) to run the trial uncapped. 0 closes enrolment without
    # ending the trials already running: the cap is only ever consulted when
    # a new seat is taken, never to revoke a seat already held.
    max_active_trials: int | None = Field(default=None, ge=0, strict=True)

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
        """A hash of the terms the user is shown and accepts at checkout.

        The cap is our capacity setting, not one of those terms, so it is left
        out: changing it must not bounce everyone mid-checkout with "the offer
        changed", and an offer hashes exactly as it did before the cap existed.
        """
        terms = self.model_dump_json(exclude={"max_active_trials"})
        return sha256(terms.encode()).hexdigest()


async def get_trial_offer(
    user_id: str, *, country: str | None = None
) -> TrialOffer | None:
    """The trial this user may see right now, or None.

    Who is offered a trial is the flag's targeting, not this code: the
    country is handed to the flag as the ``country`` attribute so that rules
    such as "country is not one of IN" decide it. A missing country never
    matches a country rule, negated or not, so the recommended shape -- a
    rule serving the offer when country is not one of the excluded list,
    falling through to off -- also withholds it when the country is unknown.

    A country that is present but not an assigned ISO 3166-1 alpha-2 code is
    refused outright rather than handed on: under a negated rule any value
    other than the excluded ones -- ``"IN, US"`` from a duplicated header, say
    -- would be served the offer, so it fails closed instead.
    """
    code = country_code(country)
    if code is None and (country or "").strip():
        logger.info("Unrecognised client country; withholding the trial offer")
        return None
    try:
        if not await is_feature_enabled(
            Flag.ENABLE_PLATFORM_PAYMENT, user_id, default=False
        ):
            return None
        raw = await get_feature_flag_value(
            Flag.CARD_REQUIRED_TRIAL_OFFER,
            user_id,
            None,
            attributes={"country": code} if code else None,
        )
        # The flag's disabled variation is an object ({"enabled": false}), not
        # null, so only a payload claiming to be an offer is worth an error.
        if not isinstance(raw, dict) or "version" not in raw:
            logger.debug("No card-required trial offer configured")
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
