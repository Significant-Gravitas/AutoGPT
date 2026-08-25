"""Every family renders a complete document, a plain-text twin, and a subject
that states the news.

These are complete documents rather than content fragments, so the old
bleach-into-a-base-template path is gone; the guard that user-supplied values
stay inert is Jinja's autoescape, which is what the injection test below
checks.
"""

import pytest
from prisma.enums import BriefingFrequency, NotificationType

from backend.data.notifications import (
    AlertData,
    AlertPrimary,
    BriefingData,
    BriefingLedgerRow,
    BriefingPeriod,
    BriefingTotals,
    CardDetails,
    NotificationPreference,
    OpsData,
    PaymentFailedData,
    PaymentFinalNoticeData,
    SubscriptionCancelledData,
    SubscriptionEndedData,
    SubscriptionPlan,
    SubscriptionResumedData,
    SubscriptionWelcomeData,
    VerdictData,
    supports_list_unsubscribe,
)
from backend.notifications.preferences import SERVICE_MESSAGES, wants_notification
from backend.notifications.renderer import EmailUrls, render

URLS = EmailUrls(
    dashboard="https://p.example/library",
    settings="https://p.example/settings/account",
    unsubscribe="https://p.example/api/email/unsubscribe?token=x",
    attention="https://p.example/library?filter=needs-attention",
    billing="https://p.example/settings/billing",
    prefs="https://p.example/settings/account",
    marketplace="https://p.example/marketplace",
    docs="https://docs.example",
    discord="https://discord.gg/autogpt",
)
PLAN = SubscriptionPlan(
    name="Pro",
    cycle="monthly",
    cycle_noun="month",
    label="Pro — monthly",
    price_display="$50.00 / month",
)

BRIEFING = BriefingData(
    period=BriefingPeriod(
        label="Mon 27 Jul – Sun 2 Aug",
        noun="this week",
        adjective="week",
        frequency="weekly",
    ),
    totals=BriefingTotals(
        runs=136,
        agents_active=1,
        failed=0,
        credits_used=21.72,
        credits_balance=118.28,
    ),
    ledger=[BriefingLedgerRow(agent="Lead Scout", runs=136, credits=21.72)],
)
ALERT = AlertData(
    timestamp_label="Thu 6 Aug, 9:26",
    primary=AlertPrimary(
        headline="Invoice Chaser is stuck",
        body="Gmail's connection expired at 9:14.",
        cta_label="Reconnect Gmail",
        cta_url="https://p.example/integrations/gmail",
        subject="Invoice Chaser is stuck — Gmail needs a reconnect",
        preheader="2 runs skipped; the next try is at 16:00.",
    ),
)
VERDICT = VerdictData(
    outcome="approved",
    agent_name="Lead Scout",
    version=4,
    reviewer_name="Morgan Reyes",
    reviewed_at_label="5 August",
    comments="Clean submission.",
    store_url="https://p.example/store/lead-scout",
    share_url="https://p.example/store/lead-scout",
)
OPS = OpsData(
    kind="request",
    user_name="Priya Natarajan",
    user_email="priya@example.com",
    user_id="usr_1",
    transaction_id="txn_1",
    refund_request_id="rr_1",
    amount_cents=1200,
    balance_cents=340,
    reason="Charged but no output.",
    recipient="refunds@agpt.co",
    stripe_url="https://dashboard.stripe.com/payments/txn_1",
    admin_url="https://admin.example/refunds/rr_1",
    age_label="6 August at 09:14",
    requested_at_label="6 August at 09:14",
)
WELCOME = SubscriptionWelcomeData(user_name="Sam", plan=PLAN, renews_label="8 Sep 2026")
ENDED = SubscriptionEndedData(
    user_name="Sam", plan=PLAN, ended_label="8 Aug 2026", due_to_payment=True
)

ALL = [
    (NotificationType.BRIEFING, BRIEFING),
    (NotificationType.ALERT, ALERT),
    (NotificationType.VERDICT, VERDICT),
    (NotificationType.OPS, OPS),
    (NotificationType.SUBSCRIPTION_WELCOME, WELCOME),
    (NotificationType.SUBSCRIPTION_ENDED, ENDED),
]


@pytest.mark.parametrize("notification_type,data", ALL)
def test_every_family_renders_a_complete_document(notification_type, data):
    email = render(notification_type, data, "sam@example.com", URLS)
    assert email.html.startswith("<!doctype html>")
    assert "<table" in email.html
    assert email.html.rstrip().endswith("</html>")


@pytest.mark.parametrize("notification_type,data", ALL)
def test_every_family_has_a_plain_text_part_from_the_same_data(notification_type, data):
    email = render(notification_type, data, "sam@example.com", URLS)
    assert len(email.text) > 200
    assert "<table" not in email.text


@pytest.mark.parametrize("notification_type,data", ALL)
def test_subjects_state_the_news(notification_type, data):
    email = render(notification_type, data, "sam@example.com", URLS)
    assert email.subject
    for banned in ("Summary", "Report", "Update", "Newsletter"):
        assert banned not in email.subject
    assert "View this email" not in email.preheader


def test_the_briefing_subject_carries_the_actual_numbers():
    email = render(NotificationType.BRIEFING, BRIEFING, "sam@example.com", URLS)
    assert "136 runs" in email.subject


def test_the_footer_offers_a_volume_knob_not_just_a_trapdoor():
    email = render(NotificationType.BRIEFING, BRIEFING, "sam@example.com", URLS)
    for choice in ("f=daily", "f=monthly", "f=alerts", "f=off"):
        assert choice in email.html


def test_only_the_product_families_carry_one_click_unsubscribe():
    assert not supports_list_unsubscribe(NotificationType.OPS)
    for notification_type in (
        NotificationType.BRIEFING,
        NotificationType.ALERT,
        NotificationType.VERDICT,
    ):
        assert supports_list_unsubscribe(notification_type)


@pytest.mark.parametrize("notification_type", sorted(SERVICE_MESSAGES, key=str))
def test_we_never_advertise_an_unsubscribe_we_do_not_honour(notification_type):
    """A header we cannot act on is a broken unsubscribe.

    Service messages are sent whatever the preferences say, so if one carried
    List-Unsubscribe the subscriber could click it in Gmail, have every
    preference switched off, and still keep receiving the mail — on the
    billing sender's own reputation.
    """
    unsubscribed = NotificationPreference(
        user_id="u1",
        email="sam@example.com",
        briefing_frequency=BriefingFrequency.OFF,
        alerts_enabled=False,
        store_verdicts_enabled=False,
        daily_limit=0,
    )
    still_sends = wants_notification(unsubscribed, notification_type)
    assert still_sends, "fixture assumes service mail ignores preferences"
    assert not supports_list_unsubscribe(notification_type)


def test_ops_says_it_is_internal_and_offers_no_unsubscribe():
    email = render(NotificationType.OPS, OPS, "refunds@agpt.co", URLS)
    assert "Not customer-facing" in email.html
    assert URLS.unsubscribe not in email.html


def test_user_supplied_values_are_escaped_not_executed():
    hostile = VERDICT.model_copy(update={"agent_name": "<script>alert('x')</script>"})
    email = render(NotificationType.VERDICT, hostile, "sam@example.com", URLS)
    assert "<script>" not in email.html
    assert "&lt;script&gt;" in email.html


def test_hero_art_is_hosted_rather_than_inlined():
    email = render(NotificationType.BRIEFING, BRIEFING, "sam@example.com", URLS)
    assert "<img" in email.html
    assert "data:image" not in email.html


def test_a_newline_in_user_data_cannot_hijack_the_preheader():
    """The subject/preheader split is positional, so the two lines must come
    from the template. An agent name carrying a newline would otherwise cut the
    subject short and push its own remainder into the preheader slot."""
    hostile = VERDICT.model_copy(update={"agent_name": "Lead Scout\nBUY CHEAP PILLS"})
    email = render(NotificationType.VERDICT, hostile, "sam@example.com", URLS)

    assert "\n" not in email.subject
    assert "BUY CHEAP PILLS" not in email.preheader
    assert "Lead Scout BUY CHEAP PILLS" in email.subject


def test_the_preheader_still_comes_from_the_templates_second_line():
    """Flattening must not collapse the template's own two lines into one."""
    email = render(NotificationType.BRIEFING, BRIEFING, "sam@example.com", URLS)
    assert email.subject
    assert email.preheader
    assert email.preheader != email.subject


SERVICE_PAYLOADS = {
    NotificationType.SUBSCRIPTION_WELCOME: WELCOME,
    NotificationType.SUBSCRIPTION_ENDED: ENDED,
    NotificationType.PAYMENT_FAILED: PaymentFailedData(
        user_name="Sam",
        plan=PLAN,
        amount_display="$50.00",
        card=CardDetails(brand="Visa", last4="4242"),
        next_retry_label="11 Aug 2026",
    ),
    NotificationType.PAYMENT_FINAL_NOTICE: PaymentFinalNoticeData(
        user_name="Sam", plan=PLAN, amount_display="$50.00", pauses_label="22 Aug 2026"
    ),
    NotificationType.SUBSCRIPTION_CANCELLED: SubscriptionCancelledData(
        user_name="Sam", plan=PLAN, access_until_label="8 Sep 2026"
    ),
    NotificationType.SUBSCRIPTION_RESUMED: SubscriptionResumedData(
        user_name="Sam", plan=PLAN, renews_label="8 Sep 2026"
    ),
}


@pytest.mark.parametrize("notification_type", sorted(SERVICE_MESSAGES, key=str))
def test_service_mail_offers_preferences_not_an_unsubscribe(notification_type):
    """The plain-text part has to agree with the header and the HTML.

    Service mail sends whatever the preferences say, so an unsubscribe link in
    any part of it is a promise we do not keep — it just happened to be the
    text MIME part that still carried one.
    """
    email = render(
        notification_type, SERVICE_PAYLOADS[notification_type], "sam@example.com", URLS
    )

    assert URLS.unsubscribe not in email.text
    assert URLS.unsubscribe not in email.html
    assert URLS.prefs in email.text


def test_every_service_message_has_a_payload_here():
    """Keeps the check above honest if a seventh service type is added."""
    assert set(SERVICE_PAYLOADS) == set(SERVICE_MESSAGES)
