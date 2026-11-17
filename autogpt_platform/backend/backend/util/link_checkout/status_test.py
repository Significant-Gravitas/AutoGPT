import pytest

from backend.util.link_checkout.models import SpendRequest
from backend.util.link_checkout.status import MAX_ACTION_MESSAGE_CHARS, payment_status


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


def action_required(resolution: str, url: str, message: str) -> SpendRequest:
    return SpendRequest.model_validate(
        {
            "id": "lsrq_test",
            "status": "requires_action",
            "status_details": {
                "requires_action": {
                    "next_action": {
                        "type": "three_d_secure",
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
