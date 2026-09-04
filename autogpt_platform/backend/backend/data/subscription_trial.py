"""Durable enrollment and usage state, scoped to the authenticated user."""

from datetime import datetime

from prisma.models import CreditTransaction, SubscriptionTrial
from pydantic import BaseModel, TypeAdapter

from backend.data.subscription_trial_config import AcceptedTrialOffer, trial_is_active
from backend.util.json import SafeJson


class TrialState(BaseModel):
    id: str
    user_id: str
    offer: AcceptedTrialOffer
    customer_id: str
    checkout_session_id: str | None
    subscription_id: str | None
    checkout_attempt: int
    success_url: str
    cancel_url: str
    checkout_metadata: dict[str, str]
    status: str
    card_verified_at: datetime | None
    started_at: datetime | None
    ends_at: datetime | None
    consumed_at: datetime | None
    converted_at: datetime | None
    cancel_at_period_end: bool
    cost_microdollars: int

    @property
    def active(self) -> bool:
        return trial_is_active(
            status=self.status,
            trial_end=self.ends_at,
            card_verified=self.card_verified_at is not None,
        )

    @classmethod
    def from_db(cls, row: SubscriptionTrial) -> "TrialState":
        return cls(
            id=row.id,
            user_id=row.userId,
            offer=AcceptedTrialOffer.model_validate(row.offer),
            customer_id=row.stripeCustomerId,
            checkout_session_id=row.stripeCheckoutSessionId,
            subscription_id=row.stripeSubscriptionId,
            checkout_attempt=row.checkoutAttempt,
            success_url=row.checkoutSuccessUrl,
            cancel_url=row.checkoutCancelUrl,
            checkout_metadata=TypeAdapter(dict[str, str]).validate_python(
                row.checkoutMetadata
            ),
            status=row.status,
            card_verified_at=row.cardVerifiedAt,
            started_at=row.startedAt,
            ends_at=row.endsAt,
            consumed_at=row.consumedAt,
            converted_at=row.convertedAt,
            cancel_at_period_end=row.cancelAtPeriodEnd,
            cost_microdollars=row.costMicrodollars,
        )


async def get_subscription_trial(user_id: str) -> TrialState | None:
    row = await SubscriptionTrial.prisma().find_unique(where={"userId": user_id})
    return TrialState.from_db(row) if row else None


async def has_received_onboarding_credit(user_id: str) -> bool:
    return (
        await CreditTransaction.prisma().find_unique(
            where={
                "creditTransactionIdentifier": {
                    "userId": user_id,
                    "transactionKey": f"REWARD-{user_id}-ONBOARDING_COMPLETE",
                }
            }
        )
        is not None
    )


async def reserve_subscription_trial(
    user_id: str,
    offer: AcceptedTrialOffer,
    customer_id: str,
    success_url: str,
    cancel_url: str,
    metadata: dict[str, str],
) -> TrialState:
    row = await SubscriptionTrial.prisma().upsert(
        where={"userId": user_id},
        data={
            "create": {
                "userId": user_id,
                "offer": SafeJson(offer.model_dump(mode="json")),
                "stripeCustomerId": customer_id,
                "checkoutSuccessUrl": success_url,
                "checkoutCancelUrl": cancel_url,
                "checkoutMetadata": SafeJson(metadata),
            },
            "update": {},
        },
    )
    return TrialState.from_db(row)


async def record_subscription_trial_cost(user_id: str, cost_microdollars: int) -> None:
    if cost_microdollars <= 0:
        return
    await SubscriptionTrial.prisma().update_many(
        where={"userId": user_id, "status": "trialing"},
        data={"costMicrodollars": {"increment": cost_microdollars}},
    )
