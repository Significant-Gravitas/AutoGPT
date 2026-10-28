"""Regression cases for the transactional-email approval review."""

import re

import pytest
from prisma.enums import NotificationType

from backend.data.notifications import PaymentFinalNoticeData
from backend.notifications.design_system_fixtures import PLAN, URLS, scenarios
from backend.notifications.renderer import render


def _email(name):
    scenario = next(s for s in scenarios() if s.name == name)
    return render(scenario.notification_type, scenario.data, "sam@example.com", URLS)


def test_failed_payment_explains_that_access_is_already_paused():
    email = _email("payment_failed")
    for body in (email.html, email.text):
        assert "Paid features and scheduled runs are paused" in body
        assert "keep running while we retry" not in body


def test_briefing_money_is_dollars_even_for_legacy_queued_estimates():
    scenario = next(s for s in scenarios() if s.name == "briefing-weekly")
    data = scenario.data.model_copy(deep=True)
    data.totals.usd_estimate = 0.04
    email = render(NotificationType.BRIEFING, data, "sam@example.com", URLS)
    for body in (email.html, email.text):
        assert "$4.20" in body
        assert "$295.80" in body
        assert "$2.10" in body
        assert "$0.04" not in body


def test_attention_preheader_omits_zero_failed_runs():
    email = _email("briefing-attention-overflow")
    assert "4 items need your attention" in email.preheader
    assert "0 runs failed" not in email.preheader


@pytest.mark.parametrize("kind", ["started", "ending", "resumed"])
def test_trial_instructions_name_the_cancel_action_and_retained_access(kind):
    email = _email(f"trial-{kind}")
    for body in (email.html, email.text):
        assert "Cancel trial" in body
        assert "keep access until" in body
        assert "trial allowance" not in body


def test_ended_trial_offers_a_plan_instead_of_a_past_trial_end():
    email = _email("trial-ended")
    for body in (email.html, email.text):
        assert "Trial ended" in body
        assert "Choose a plan" in body
        assert "Trial ends" not in body


def test_trial_conversion_welcomes_the_customer_in_both_formats():
    email = _email("trial-converted")
    for body in (email.html, email.text):
        assert "Open AutoGPT" in body
        assert "three good places to start" in body.lower()
        assert "Trial ends" not in body
        assert URLS.docs in body
        assert URLS.marketplace in body


def test_inline_content_column_survives_style_removal():
    html = _email("payment_failed").html
    stripped = re.sub(r"<style\b[^>]*>.*?</style>", "", html, flags=re.S)
    assert 'class="content"' in stripped
    assert "max-width:464px" in stripped
    assert "padding:0 22px" in stripped
    assert "@import" not in html
    assert '<link href="https://fonts.googleapis.com/' in html


def test_legacy_final_notice_never_relabels_the_renewal_date_as_an_attempt():
    email = render(
        NotificationType.PAYMENT_FINAL_NOTICE,
        PaymentFinalNoticeData(
            user_name="Sam",
            plan=PLAN,
            amount_display="$50.00",
            pauses_label="3 Oct 2026",
        ),
        "sam@example.com",
        URLS,
    )
    for body in (email.html, email.text):
        assert "3 Oct 2026" not in body
        assert "None" not in body
        assert "Last attempt" not in body
