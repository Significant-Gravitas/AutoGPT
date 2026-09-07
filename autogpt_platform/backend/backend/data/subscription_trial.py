"""Durable enrollment and usage state, scoped to the authenticated user."""

from datetime import datetime

from prisma.errors import UniqueViolationError
from prisma.models import CreditTransaction, SubscriptionTrial
from prisma.types import SubscriptionTrialWhereInput
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
    conversion_invoice_id: str | None = None
    notification_revision: int = 0
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
            conversion_invoice_id=row.stripeConversionInvoiceId,
            notification_revision=row.notificationRevision,
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
    try:
        row = await SubscriptionTrial.prisma().create(
            data={
                "userId": user_id,
                "offer": SafeJson(offer.model_dump(mode="json")),
                "stripeCustomerId": customer_id,
                "checkoutSuccessUrl": success_url,
                "checkoutCancelUrl": cancel_url,
                "checkoutMetadata": SafeJson(metadata),
            },
        )
    except UniqueViolationError:
        row = await SubscriptionTrial.prisma().find_unique_or_raise(
            where={"userId": user_id}
        )
    return TrialState.from_db(row)


async def record_subscription_trial_cost(
    user_id: str, cost_microdollars: int, trial_id: str | None = None
) -> None:
    if cost_microdollars <= 0:
        return
    where: SubscriptionTrialWhereInput = {"userId": user_id, "status": "trialing"}
    if trial_id is not None:
        where = {"userId": user_id, "id": trial_id, "consumedAt": {"not": None}}
    updated = await SubscriptionTrial.prisma().update_many(
        where=where,
        data={"costMicrodollars": {"increment": cost_microdollars}},
    )
    if trial_id is not None and updated != 1:
        raise ValueError("Trial cost attribution was not found for this user")
