"""Durable enrollment and usage state, scoped to the authenticated user."""

from datetime import datetime

from prisma.models import CreditTransaction, SubscriptionTrial
from prisma.types import SubscriptionTrialWhereInput
from pydantic import BaseModel, TypeAdapter

from backend.data.db import transaction
from backend.data.subscription_trial_capacity import (
    TRIAL_FULL,
    TrialCapacityReached,
    lock_trial_capacity,
    trial_seat_available,
)
from backend.data.subscription_trial_config import AcceptedTrialOffer, trial_is_active
from backend.data.subscription_trial_rejection import TrialRejectionReason
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
    rejection_reason: TrialRejectionReason | None = None
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
            rejection_reason=(
                TrialRejectionReason(row.rejectionReason)
                if row.rejectionReason
                else None
            ),
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
    """Take this user's seat under the offer's cap, or return the one they hold.

    Raises :class:`TrialCapacityReached` when the trial is full.

    The seat is read and taken inside its own short transaction rather than
    the caller's: the caller keeps a transaction open across its Stripe
    round-trips, and holding the global capacity lock for that long would
    queue every other enrolment behind one person's network latency.
    """
    async with transaction() as tx:
        await lock_trial_capacity(tx)
        existing = await tx.subscriptiontrial.find_unique(where={"userId": user_id})
        if existing:
            # An enrolment already on file keeps whatever seat it has; it is
            # resuming, not competing for a new one.
            return TrialState.from_db(existing)
        if not await trial_seat_available(offer, client=tx):
            raise TrialCapacityReached(TRIAL_FULL)
        row = await tx.subscriptiontrial.create(
            data={
                "userId": user_id,
                # The cap is live capacity config, read from the flag each
                # time, never from here. Leaving it out keeps these rows
                # readable by code that predates it, so a revert is safe.
                "offer": SafeJson(
                    offer.model_dump(mode="json", exclude={"max_active_trials"})
                ),
                "stripeCustomerId": customer_id,
                "checkoutSuccessUrl": success_url,
                "checkoutCancelUrl": cancel_url,
                "checkoutMetadata": SafeJson(metadata),
            },
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
