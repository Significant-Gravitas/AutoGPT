import pytest

from backend.util.link_checkout.models import SpendRequest
from backend.util.link_checkout.status import (
    MAX_ACTION_MESSAGE_CHARS,
    TEST_PAYMENT_SUBMITTED,
    payment_status,
)


@pytest.mark.parametrize(
    "status",
    [
        "created",
        "pending_approval",
        "requires_action",
        "approved",
        "submitted",
        "denied",
        "expired",
        "canceled",
        "succeeded",
        "failed",
        "a_status_link_adds_later",
    ],
)
def test_only_link_success_confirms_payment(intent, status):
    spend = SpendRequest(id=intent.spend_request_id, status=status)
    result = payment_status(spend)
    assert result.paid == (status == "succeeded")
    assert result.message


def action_required(
    resolution: str, url: str, message: str, type: str = "three_d_secure"
) -> SpendRequest:
    return SpendRequest.model_validate(
        {
            "id": "lsrq_test",
            "status": "requires_action",
            "status_details": {
                "requires_action": {
                    "next_action": {
                        "type": type,
                        "resolution": resolution,
                        "action_url": url,
                        "display_message": message,
                    }
                }
            },
        }
    )


@pytest.mark.parametrize(
    "resolution",
    [
        "auto_resume",
        "create_new_spend_request",
        "create_new_spend_request_after_completion",
    ],
)
def test_required_action_keeps_its_resolution(resolution):
    result = payment_status(
        action_required(resolution, "https://app.link.com/verify", "Verify it's you")
    )
    assert result.resolution == resolution
    assert result.action_url == "https://app.link.com/verify"
    assert result.action_message == "Verify it's you"


def test_action_link_to_another_site_is_dropped():
    result = payment_status(
        action_required("auto_resume", "https://evil.example/collect", "Go here")
    )
    assert result.action_url == ""


def test_link_message_is_bounded_and_a_long_one_still_reads():
    result = payment_status(
        action_required("auto_resume", "https://app.link.com/verify", "x" * 10_000)
    )
    assert len(result.action_message) == MAX_ACTION_MESSAGE_CHARS


def test_three_d_secure_resumes_the_same_request():
    result = payment_status(
        action_required("auto_resume", "https://hooks.stripe.com/3ds", "Verify")
    )
    assert result.final is False
    assert "resumes" in result.message
    assert result.action_url == "https://hooks.stripe.com/3ds"


@pytest.mark.parametrize(
    "type,resolution,words",
    [
        ("select_payment_method", "create_new_spend_request", "another Link payment"),
        ("re_authorize", "create_new_spend_request", "correct total"),
        ("add_payment_method", "create_new_spend_request_after_completion", "add one"),
        ("contact_support", "create_new_spend_request_after_completion", "support"),
    ],
)
def test_an_action_that_ends_the_request_is_final_and_says_what_next(
    type, resolution, words
):
    result = payment_status(
        action_required(resolution, "https://app.link.com/x", "Fix it", type)
    )
    assert result.final is True
    assert words in result.message
    assert "new checkout" in result.message


def test_the_resolution_decides_even_when_the_type_says_otherwise():
    """Link: branch on resolution, not on type. A type paired with another
    resolution than usual is worded by the resolution."""
    resumes = payment_status(
        action_required("auto_resume", "", "Pick a card", "select_payment_method")
    )
    assert resumes.final is False
    assert "resumes" in resumes.message
    unknown = payment_status(
        action_required("a_resolution_link_adds_later", "", "Hmm", "three_d_secure")
    )
    assert unknown.final is True


def test_a_failed_payment_names_only_well_formed_codes(intent):
    spend = SpendRequest.model_validate(
        {
            "id": "lsrq_test",
            "status": "failed",
            "payment_status_details": {
                "outcome": "failure",
                "decline_code": "insufficient_funds",
                "code": "<script>alert(1)</script>",
            },
        }
    )
    result = payment_status(spend)
    assert result.final is True
    assert "insufficient_funds" in result.message
    assert "script" not in result.message


def test_an_attempted_test_payment_left_approved_is_final():
    """Link never charges test cards, so an attempted test payment stays
    approved; that must not keep the chat from starting another purchase."""
    approved = SpendRequest(id="lsrq_test", status="approved")
    result = payment_status(approved, attempted=True, test_mode=True)
    assert result.final is True
    assert result.paid is False
    assert result.message == TEST_PAYMENT_SUBMITTED
    assert payment_status(approved, attempted=True, test_mode=False).final is False
    assert payment_status(approved, attempted=False, test_mode=True).final is False
