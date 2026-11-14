import pytest
from pydantic import ValidationError

from backend.util.link_checkout.checkout_record import (
    consume_intent,
    read_intent,
    save_intent,
)
from backend.util.link_checkout.models import CheckoutPlan, SpendRequest, WorkerReceipt


@pytest.mark.parametrize(
    "extra", [{"script": "return card.number"}, {"card_number": "4242424242424242"}]
)
def test_checkout_plan_rejects_script_and_card_arguments(plan, extra):
    with pytest.raises(ValidationError):
        CheckoutPlan.model_validate({**plan.model_dump(), **extra})


def test_plan_validation_errors_never_echo_the_input(plan):
    with pytest.raises(ValidationError) as failure:
        CheckoutPlan.model_validate(
            {**plan.model_dump(), "checkout_url": "http://4242424242424242.example"}
        )
    assert "4242424242424242" not in str(failure.value)


def test_link_is_shown_the_checkout_page_without_its_query(plan):
    plan.checkout_url = "https://shop.example/checkout?session=secret#pay"
    assert plan.merchant_url() == "https://shop.example/checkout"


def test_worker_receipt_cannot_return_card_or_merchant_output():
    with pytest.raises(ValidationError):
        WorkerReceipt.model_validate(
            {"status": "submitted", "card_number": "4242424242424242"}
        )


def test_an_unknown_link_status_still_reads():
    spend = SpendRequest.model_validate(
        {"id": "lsrq_new", "status": "a_future_status", "brand_new_field": 1}
    )
    assert spend.status == "a_future_status"


@pytest.mark.parametrize(
    "field,value",
    [
        ("user_id", "someone_else"),
        ("key", "other_chat"),
        ("checkout_id", "b" * 32),
    ],
)
def test_intent_ownership_is_checked(tmp_path, intent, field, value):
    save_intent(tmp_path, intent)
    args = {
        "checkout_id": intent.id,
        "user_id": intent.user_id,
        "key": intent.session_id,
        field: value,
    }
    with pytest.raises(RuntimeError):
        read_intent(tmp_path, **args)


def test_consumed_checkout_cannot_be_replayed_after_reload(tmp_path, intent):
    save_intent(tmp_path, intent)
    consume_intent(tmp_path, intent)
    with pytest.raises(RuntimeError):
        read_intent(tmp_path, intent.id, intent.user_id, intent.session_id)
    assert (tmp_path / "sensitive").exists()
