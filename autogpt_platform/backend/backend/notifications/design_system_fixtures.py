"""Public-safe scenarios for the transactional email design checks and previews."""

from prisma.enums import NotificationType
from pydantic import BaseModel

from backend.data.notifications import (
    AlertAlsoItem,
    AlertData,
    AlertFact,
    AlertPrimary,
    BaseNotificationData,
    BriefingAttentionItem,
    BriefingData,
    BriefingHighlight,
    BriefingLedgerRow,
    BriefingPeriod,
    BriefingTotals,
    CardDetails,
    OpsData,
    PaymentFailedData,
    PaymentFinalNoticeData,
    SubscriptionCancelledData,
    SubscriptionEndedData,
    SubscriptionPlan,
    SubscriptionResumedData,
    SubscriptionWelcomeData,
    TrialUpdateData,
    VerdictData,
)
from backend.notifications.renderer import EmailUrls

URLS = EmailUrls(
    chat="https://platform.example/copilot",
    dashboard="https://platform.example/library",
    settings="https://platform.example/settings/account",
    unsubscribe="https://platform.example/unsubscribe?token=example-token",
    attention="https://platform.example/library?filter=needs-attention",
    billing="https://platform.example/settings/billing",
    prefs="https://platform.example/settings/account",
    marketplace="https://platform.example/marketplace",
    docs="https://docs.example",
    discord="https://discord.example/autogpt",
    volume={
        choice: f"https://platform.example/preferences?token=example-{choice}"
        for choice in ("daily", "weekly", "monthly", "alerts", "off")
    },
)
PLAN = SubscriptionPlan(
    name="Max",
    cycle="monthly",
    cycle_noun="month",
    label="Max · monthly",
    price_display="$50.00 / month",
)


class EmailScenario(BaseModel):
    name: str
    notification_type: NotificationType
    data: BaseNotificationData


def scenarios() -> list[EmailScenario]:
    return [
        *_lifecycle(),
        *_trials(),
        *_briefings(),
        *_alerts(),
        *_verdicts(),
        *_ops(),
        EmailScenario(
            name="welcome-experts-enabled",
            notification_type=NotificationType.SUBSCRIPTION_WELCOME,
            data=SubscriptionWelcomeData(
                user_name="Sam",
                plan=PLAN,
                renews_label="3 Nov 2026",
                experts_enabled=True,
            ),
        ),
    ]


def _lifecycle() -> list[EmailScenario]:
    common = {"user_name": "Sam", "plan": PLAN}
    rows = [
        (
            NotificationType.SUBSCRIPTION_WELCOME,
            SubscriptionWelcomeData(**common, renews_label="3 Nov 2026"),
        ),
        (
            NotificationType.PAYMENT_FAILED,
            PaymentFailedData(
                **common,
                amount_display="$50.00",
                next_retry_label="7 Oct 2026",
                card=CardDetails(brand="Visa", last4="4242"),
            ),
        ),
        (
            NotificationType.PAYMENT_FINAL_NOTICE,
            PaymentFinalNoticeData(
                **common, amount_display="$50.00", pauses_label="10 Oct 2026"
            ),
        ),
        (
            NotificationType.SUBSCRIPTION_CANCELLED,
            SubscriptionCancelledData(**common, access_until_label="3 Nov 2026"),
        ),
        (
            NotificationType.SUBSCRIPTION_RESUMED,
            SubscriptionResumedData(**common, renews_label="3 Nov 2026"),
        ),
    ]
    return [
        EmailScenario(name=str(kind).lower(), notification_type=kind, data=data)
        for kind, data in rows
    ] + [
        EmailScenario(
            name=f"ended-payment-{payment}",
            notification_type=NotificationType.SUBSCRIPTION_ENDED,
            data=SubscriptionEndedData(
                **common, ended_label="3 Oct 2026", due_to_payment=payment
            ),
        )
        for payment in (True, False)
    ]


def _trials() -> list[EmailScenario]:
    return [
        EmailScenario(
            name=f"trial-{kind}",
            notification_type=NotificationType.TRIAL_UPDATE,
            data=TrialUpdateData(
                user_name="Sam",
                plan=PLAN,
                kind=kind,
                ends_label="10 Oct 2026",
                onboarding_credit_amount=300,
                offer_version="example-offer",
            ),
        )
        for kind in (
            "started",
            "ending",
            "canceled",
            "resumed",
            "ended",
            "converted",
            "payment_failed",
        )
    ]


def _briefings() -> list[EmailScenario]:
    base = BriefingData(
        period=BriefingPeriod(
            label="26 Sep to 2 Oct 2026",
            noun="this week",
            adjective="week",
            frequency="weekly",
        ),
        totals=BriefingTotals(
            runs=42, agents_active=2, credits_used=4.2, credits_balance=295.8
        ),
        ledger=[
            BriefingLedgerRow(agent="Lead Scout", runs=21, credits=2.1),
            BriefingLedgerRow(agent="Invoice Follow-up", runs=21, credits=2.1),
        ],
        highlights=[
            BriefingHighlight(
                agent="Lead Scout",
                gist="found five prospects.",
                link_label="See the prospects",
                url="https://platform.example/library/run-1",
            )
        ],
    )
    attention = [
        BriefingAttentionItem(
            agent=f"Workflow {i}",
            title=f"Workflow {i} needs a connection",
            tag="CONNECTION",
            body="Reconnect the account to continue.",
            cta_label="Reconnect account",
            cta_url=f"https://platform.example/integrations/{i}",
        )
        for i in range(4)
    ]
    variants = {
        "weekly": base,
        "daily": base.model_copy(
            update={
                "period": BriefingPeriod(
                    label="2 Oct 2026",
                    noun="yesterday",
                    adjective="day",
                    frequency="daily",
                )
            }
        ),
        "monthly": base.model_copy(
            update={
                "period": BriefingPeriod(
                    label="September 2026",
                    noun="in September",
                    adjective="month",
                    frequency="monthly",
                )
            }
        ),
        "quiet": base.model_copy(
            update={
                "mode": "quiet",
                "only_agent": "Lead Scout",
                "quiet_summary": "The scheduled work is complete.",
            }
        ),
        "attention-overflow": base.model_copy(
            update={
                "attention": attention,
                "ledger_overflow": 8,
                "ledger_overflow_runs": 16,
                "ledger_overflow_issues": 2,
            }
        ),
        "rough": base.model_copy(
            update={
                "totals": base.totals.model_copy(update={"failed": 20}),
                "ledger": [
                    BriefingLedgerRow(
                        agent="Lead Scout",
                        runs=42,
                        credits=4.2,
                        issues_label="20 failed",
                        issues_kind="fail",
                    )
                ],
            }
        ),
    }
    return [
        EmailScenario(
            name=f"briefing-{name}",
            notification_type=NotificationType.BRIEFING,
            data=data,
        )
        for name, data in variants.items()
    ]


def _alerts() -> list[EmailScenario]:
    base = AlertData(
        timestamp_label="3 Oct 2026 at 09:26 UTC",
        primary=AlertPrimary(
            headline="Invoice Follow-up is paused",
            body="Gmail needs a new connection.",
            subject="Invoice Follow-up needs Gmail reconnected",
            preheader="Reconnect Gmail to continue your scheduled work.",
            cta_label="Reconnect Gmail",
            cta_url="https://platform.example/integrations/gmail",
            facts=[AlertFact(label="Skipped runs", value="2")],
            microcopy="Scheduled runs resume after you reconnect.",
        ),
    )
    return [
        EmailScenario(
            name=f"alert-{count}",
            notification_type=NotificationType.ALERT,
            data=base.model_copy(
                update={
                    "also": [
                        AlertAlsoItem(
                            agent=f"Workflow {i}",
                            text="has an output waiting",
                            link_label="Review output",
                            url=f"https://platform.example/library/review-{i}",
                        )
                        for i in range(count)
                    ]
                }
            ),
        )
        for count in (0, 3)
    ]


def _verdicts() -> list[EmailScenario]:
    base = VerdictData(
        outcome="approved",
        agent_name="Lead Scout",
        version=3,
        reviewer_name="Alex",
        reviewed_at_label="3 Oct 2026",
        comments="Ready to share.",
        store_url="https://platform.example/marketplace/lead-scout",
        share_url="https://platform.example/marketplace/lead-scout",
        resubmit_url="https://platform.example/library/lead-scout/edit",
    )
    variants = {
        "approved": base,
        "changes": base.model_copy(
            update={
                "outcome": "changes",
                "changes": ["Clarify the inputs.", "Add a sample output."],
            }
        ),
        "no-feedback": base.model_copy(update={"outcome": "changes", "comments": " "}),
    }
    return [
        EmailScenario(
            name=f"verdict-{name}",
            notification_type=NotificationType.VERDICT,
            data=data,
        )
        for name, data in variants.items()
    ]


def _ops() -> list[EmailScenario]:
    return [
        EmailScenario(
            name=f"ops-{kind}",
            notification_type=NotificationType.OPS,
            data=OpsData(
                kind=kind,
                user_name="Sam",
                user_email="sam@example.com",
                user_id="example-user",
                transaction_id="example-transaction",
                refund_request_id="example-request",
                amount_cents=1200,
                balance_cents=5000,
                reason="The workflow did not produce the expected output.",
                recipient="refunds@example.com",
                stripe_url="https://stripe.example/payment",
                admin_url="https://admin.example/refund",
                age_label="3 Oct 2026 at 09:26 UTC",
                requested_at_label="3 Oct 2026 at 09:26 UTC",
                processed_at_label="3 Oct 2026 at 10:26 UTC",
            ),
        )
        for kind in ("request", "processed")
    ]
