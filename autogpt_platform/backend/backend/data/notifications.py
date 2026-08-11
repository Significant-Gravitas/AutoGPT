"""Payloads for every email the platform sends.

Four product-notification families (Briefing, Alert, Verdict, Ops) plus the
account/billing service messages. Each model is exactly the render context its
Jinja template expects, minus the per-recipient values (`user_email`, `urls`)
which the renderer injects — so a payload can be queued, persisted and replayed
without carrying a copy of our own URLs around.

There is deliberately no per-run notification type. A finished run is evidence
for the Briefing's highlights, not a message of its own.
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Generic, Literal, Optional, TypeVar, Union

from prisma.enums import BriefingFrequency, NotificationType
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from backend.util.logging import TruncatedLogger

logger = TruncatedLogger(logging.getLogger(__name__), prefix="[NotificationService]")

NotificationDataType_co = TypeVar(
    "NotificationDataType_co", bound="BaseNotificationData", covariant=True
)


class BaseNotificationData(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ─────────────────────────── The Briefing ───────────────────────────


class BriefingPeriod(BaseModel):
    label: str = Field(description='Range shown in the eyebrow, "Mon 27 Jul – Sun 2 Aug"')
    noun: str = Field(description='"this week" / "yesterday" / "in July"')
    adjective: str = Field(description='"week" / "day" / "month"')
    frequency: Literal["daily", "weekly", "monthly"]


class BriefingTotals(BaseModel):
    """Server-side invariants: `runs` equals the sum of the ledger's runs, and
    `agents_active` equals `len(ledger)` — the lede counts the agents that ran
    and the ledger lists exactly those."""

    runs: int
    agents_active: int
    agents_idle: int = 0
    failed: int = 0
    credits_used: float
    credits_balance: float
    usd_estimate: float | None = None


class BriefingAttentionItem(BaseModel):
    agent: str
    title: str
    tag: str | None = None
    body: str
    cta_label: str
    cta_url: str


class BriefingHighlight(BaseModel):
    agent: str
    gist: str
    link_label: str
    url: str


class BriefingLedgerRow(BaseModel):
    agent: str
    runs: int
    credits: float
    issues_label: str | None = None
    issues_kind: Literal["fail", "warn"] | None = None


class BriefingData(BaseNotificationData):
    mode: Literal["standard", "quiet"] = "standard"
    period: BriefingPeriod
    totals: BriefingTotals
    standout: str | None = None
    subject_note: str | None = None
    # Order is load-bearing: sorted by severity, because the first card gets
    # the strong amber rule and the rest a lighter one.
    attention: list[BriefingAttentionItem] = Field(default_factory=list)
    highlights: list[BriefingHighlight] = Field(default_factory=list)
    # Arrives pre-sorted by interestingness.
    ledger: list[BriefingLedgerRow] = Field(default_factory=list)
    only_agent: str | None = None
    quiet_summary: str | None = None


# ───────────────────────────── The Alert ─────────────────────────────


class AlertFact(BaseModel):
    label: str
    value: str


class AlertPrimary(BaseModel):
    headline: str
    body: str
    cta_label: str
    cta_url: str
    # Written from the cause catalog, not left to the template to guess: the
    # subject states the blockage and its cause, and the preheader carries the
    # detail that decides whether this needs opening now.
    subject: str
    preheader: str
    microcopy: str | None = None
    facts: list[AlertFact] = Field(default_factory=list)


class AlertAlsoItem(BaseModel):
    agent: str
    text: str
    link_label: str
    url: str


class AlertData(BaseNotificationData):
    timestamp_label: str
    primary: AlertPrimary
    also: list[AlertAlsoItem] = Field(default_factory=list)
    also_label: str | None = None


# ──────────────────────────── The Verdict ────────────────────────────


class VerdictData(BaseNotificationData):
    """A store submission was reviewed. `changes` is the preferred shape — the
    review UI should collect discrete items — with free-text `comments` as the
    fallback. Empty and whitespace-only feedback are both handled by the
    template rather than papered over here."""

    outcome: Literal["approved", "changes"]
    agent_name: str
    version: int = 1
    reviewer_name: str
    reviewed_at_label: str
    comments: str = ""
    changes: list[str] = Field(default_factory=list)
    closing_note: str | None = None
    store_url: str | None = None
    share_url: str | None = None
    resubmit_url: str | None = None


# ────────────────────────── Ops (internal) ──────────────────────────


class OpsData(BaseNotificationData):
    """Internal refunds-team mail. Not customer-facing, and the only family
    that must NOT carry List-Unsubscribe headers."""

    kind: Literal["request", "processed"]
    user_name: str
    user_email: str
    user_id: str
    transaction_id: str
    refund_request_id: str
    amount_cents: int
    balance_cents: int
    reason: str = ""
    recipient: str
    stripe_url: str
    admin_url: str
    # Absolute, never a duration — the email is read hours later.
    age_label: str | None = None
    requested_at_label: str | None = None
    processed_at_label: str | None = None


# ───────────────────── Account & billing messages ─────────────────────


class SubscriptionPlan(BaseModel):
    name: str = Field(description='"Pro" / "Max"')
    cycle: Literal["monthly", "yearly"]
    cycle_noun: Literal["month", "year"]
    label: str = Field(description='"Pro — monthly"')
    price_display: str = Field(description='"$50.00 / month"')


class CardDetails(BaseModel):
    brand: str
    last4: str


class LifecycleData(BaseNotificationData):
    """Shared shape for the billing emails. `user_name` is the greeting name;
    every other sentence is assembled from the slots below by the template."""

    user_name: str
    plan: SubscriptionPlan


class SubscriptionWelcomeData(LifecycleData):
    renews_label: str


class PaymentFailedData(LifecycleData):
    amount_display: str
    card: CardDetails
    next_retry_label: str


class PaymentFinalNoticeData(LifecycleData):
    amount_display: str
    pauses_label: str


class SubscriptionCancelledData(LifecycleData):
    access_until_label: str


class SubscriptionResumedData(LifecycleData):
    renews_label: str


class SubscriptionEndedData(LifecycleData):
    ended_label: str
    due_to_payment: bool


NotificationData = Union[
    BriefingData,
    AlertData,
    VerdictData,
    OpsData,
    SubscriptionWelcomeData,
    PaymentFailedData,
    PaymentFinalNoticeData,
    SubscriptionCancelledData,
    SubscriptionResumedData,
    SubscriptionEndedData,
]

_DATA_TYPES: dict[NotificationType, type[BaseNotificationData]] = {
    NotificationType.BRIEFING: BriefingData,
    NotificationType.ALERT: AlertData,
    NotificationType.VERDICT: VerdictData,
    NotificationType.OPS: OpsData,
    NotificationType.SUBSCRIPTION_WELCOME: SubscriptionWelcomeData,
    NotificationType.PAYMENT_FAILED: PaymentFailedData,
    NotificationType.PAYMENT_FINAL_NOTICE: PaymentFinalNoticeData,
    NotificationType.SUBSCRIPTION_CANCELLED: SubscriptionCancelledData,
    NotificationType.SUBSCRIPTION_RESUMED: SubscriptionResumedData,
    NotificationType.SUBSCRIPTION_ENDED: SubscriptionEndedData,
}

# Which Jinja template family renders each type. The billing messages all share
# `lifecycle`, branching internally on `kind`.
_TEMPLATES: dict[NotificationType, str] = {
    NotificationType.BRIEFING: "briefing",
    NotificationType.ALERT: "alert",
    NotificationType.VERDICT: "verdict",
    NotificationType.OPS: "ops",
    NotificationType.SUBSCRIPTION_WELCOME: "lifecycle",
    NotificationType.PAYMENT_FAILED: "lifecycle",
    NotificationType.PAYMENT_FINAL_NOTICE: "lifecycle",
    NotificationType.SUBSCRIPTION_CANCELLED: "lifecycle",
    NotificationType.SUBSCRIPTION_RESUMED: "lifecycle",
    NotificationType.SUBSCRIPTION_ENDED: "lifecycle",
}

# The `kind` slot the lifecycle template branches on.
_LIFECYCLE_KINDS: dict[NotificationType, str] = {
    NotificationType.SUBSCRIPTION_WELCOME: "welcome",
    NotificationType.PAYMENT_FAILED: "payment_failed",
    NotificationType.PAYMENT_FINAL_NOTICE: "final_notice",
    NotificationType.SUBSCRIPTION_CANCELLED: "cancel_confirmed",
    NotificationType.SUBSCRIPTION_RESUMED: "cancel_reversed",
    NotificationType.SUBSCRIPTION_ENDED: "ended",
}


def get_notif_data_type(
    notification_type: NotificationType,
) -> type[BaseNotificationData]:
    return _DATA_TYPES[notification_type]


def get_template_family(notification_type: NotificationType) -> str:
    return _TEMPLATES[notification_type]


def get_lifecycle_kind(notification_type: NotificationType) -> str | None:
    """The `kind` the shared lifecycle template branches on, or None for the
    four product-notification families, which have a template each."""
    return _LIFECYCLE_KINDS.get(notification_type)


class DeliveryStream(Enum):
    """Which sender identity and reputation the message goes out on. Marketing
    mail (the onboarding tour, the monthly changelog) is not here: it is sent
    from MailerLite, and the backend only manages who is in its audience."""

    BILLING = "billing"
    PRODUCT = "product"
    OPS = "ops"


_STREAMS: dict[NotificationType, DeliveryStream] = {
    NotificationType.BRIEFING: DeliveryStream.PRODUCT,
    NotificationType.ALERT: DeliveryStream.PRODUCT,
    NotificationType.VERDICT: DeliveryStream.PRODUCT,
    NotificationType.OPS: DeliveryStream.OPS,
    NotificationType.SUBSCRIPTION_WELCOME: DeliveryStream.BILLING,
    NotificationType.PAYMENT_FAILED: DeliveryStream.BILLING,
    NotificationType.PAYMENT_FINAL_NOTICE: DeliveryStream.BILLING,
    NotificationType.SUBSCRIPTION_CANCELLED: DeliveryStream.BILLING,
    NotificationType.SUBSCRIPTION_RESUMED: DeliveryStream.BILLING,
    NotificationType.SUBSCRIPTION_ENDED: DeliveryStream.BILLING,
}


def get_delivery_stream(notification_type: NotificationType) -> DeliveryStream:
    return _STREAMS[notification_type]


# Ops is internal mail and is deliberately the one family without one-click
# unsubscribe headers; every other family gets them.
def supports_list_unsubscribe(notification_type: NotificationType) -> bool:
    return notification_type is not NotificationType.OPS


class BaseEventModel(BaseModel):
    type: NotificationType
    user_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class NotificationEventModel(BaseEventModel, Generic[NotificationDataType_co]):
    id: Optional[str] = None
    data: NotificationDataType_co

    @field_validator("type", mode="before")
    @classmethod
    def uppercase_type(cls, v: object) -> object:
        return v.upper() if isinstance(v, str) else v


class NotificationResult(BaseModel):
    success: bool
    message: Optional[str] = None


class AudienceAction(Enum):
    """Membership changes the backend owns. The tour → changelog handoff is
    deliberately absent: MailerLite's automation owns that edge."""

    ENROLL_TOUR = "enroll_tour"
    ADD_CHANGELOG = "add_changelog"
    REMOVE_CHANGELOG = "remove_changelog"


class AudienceEventModel(BaseModel):
    """A MailerLite audience change, queued rather than called inline so a
    MailerLite outage can never fail payment processing."""

    action: AudienceAction
    email: EmailStr
    user_id: str


class NotificationPreference(BaseModel):
    """The volume knob from the Briefing footer, not a checkbox list. Billing
    and account messages are service mail and are not represented here — they
    are sent regardless of these settings."""

    user_id: str
    email: EmailStr
    briefing_frequency: BriefingFrequency = BriefingFrequency.WEEKLY
    alerts_enabled: bool = True
    store_verdicts_enabled: bool = True
    daily_limit: int = 10

    @property
    def wants_briefing(self) -> bool:
        return self.briefing_frequency is not BriefingFrequency.OFF


class NotificationPreferenceDTO(BaseModel):
    email: EmailStr
    briefing_frequency: BriefingFrequency
    alerts_enabled: bool
    store_verdicts_enabled: bool
    daily_limit: int = Field(default=10, description="Max emails per day")
