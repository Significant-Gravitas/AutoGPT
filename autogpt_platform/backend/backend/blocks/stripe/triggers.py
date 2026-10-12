import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    BlockWebhookConfig,
)
from backend.data.model import SchemaField
from backend.integrations.providers import ProviderName

from ._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    StripeCredentialsField,
    StripeCredentialsInput,
)

_PAYLOAD_DIR = Path(__file__).parent / "example_payloads"


def load_example_payload(event_type: str) -> dict[str, Any]:
    """Load a bundled example event payload, e.g. `customer.subscription.created`."""
    return json.loads((_PAYLOAD_DIR / f"{event_type}.json").read_text(encoding="utf-8"))


# Read once at import; `Block.__init__` runs on every block-registry cache miss.
EXAMPLE_SUBSCRIPTION_CREATED = load_example_payload("customer.subscription.created")


class StripeSubscriptionTriggerBlock(Block):
    """
    Triggers whenever a Stripe subscription is created, upgraded, or cancelled.
    Connects directly to Stripe — no internal DB polling, so internal/demo
    accounts are naturally excluded.
    """

    class Input(BlockSchemaInput):
        credentials: StripeCredentialsInput = StripeCredentialsField()
        payload: dict = SchemaField(hidden=True, default_factory=dict)

        class EventsFilter(BaseModel):
            """
            The `customer.subscription.*` lifecycle events:
            https://docs.stripe.com/api/events/types#event_types-customer.subscription.created
            """

            created: bool = True
            updated: bool = True
            deleted: bool = False

        events: EventsFilter = SchemaField(
            title="Events",
            description="Subscription lifecycle events to subscribe to. "
            "Cancellation and churn workflows need `deleted`, which is off by "
            "default. Note that `updated` is high-volume — Stripe sends it for "
            "any change to the subscription, including renewals, payment-method "
            "changes and metadata edits, not just upgrades. Use "
            "`previous_attributes` to tell an upgrade from routine churn.",
            default_factory=EventsFilter,
            advanced=False,
        )

    class Output(BlockSchemaOutput):
        payload: dict = SchemaField(
            description="Full Stripe event object as received from the webhook"
        )
        event_type: str = SchemaField(
            description="Stripe event type, e.g. customer.subscription.created"
        )
        subscription_id: str = SchemaField(
            description="Stripe subscription ID (sub_...)"
        )
        customer_id: str = SchemaField(description="Stripe customer ID (cus_...)")
        status: str = SchemaField(
            description="Subscription status: active, trialing, past_due, canceled, etc."
        )
        cancel_at_period_end: bool = SchemaField(
            description="True if the subscription is scheduled to end when the "
            "current billing period does, rather than having ended already"
        )
        canceled_at: int = SchemaField(
            description="Unix timestamp of when the subscription was canceled, "
            "or 0 if it has not been canceled"
        )
        previous_attributes: dict = SchemaField(
            description="On `updated` events, the changed fields' prior values, as "
            "sent by Stripe. Empty for other events. Compare against the "
            "subscription in `payload` to tell an upgrade from a renewal — e.g. a "
            "key of `items` or `plan` means the plan itself changed."
        )
        plan_name: str = SchemaField(
            description=(
                "Nickname of the subscription's first item price. Prices without "
                "a nickname fall back to the raw price ID (price_...). Only the "
                "first item is read; see `payload` for multi-item subscriptions."
            )
        )
        plan_interval: str = SchemaField(
            description="Billing interval of the first subscription item: "
            "day, week, month or year"
        )
        amount_cents: int = SchemaField(
            description="Unit amount of the first subscription item, in the smallest "
            "currency unit — cents for USD, but whole units for zero-decimal "
            "currencies like JPY and KRW. This is not the subscription total when "
            "there is more than one item."
        )
        currency: str = SchemaField(description="Three-letter ISO currency code")
        livemode: bool = SchemaField(
            description="True for live Stripe data, False for test mode"
        )
        error: str = SchemaField(
            description="Error message if the payload could not be parsed"
        )

    def __init__(self):
        # Imported here (as in the other trigger blocks) to avoid the import
        # cycle between the block package and the webhook managers.
        from backend.integrations.webhooks.stripe import StripeWebhookType

        super().__init__(
            id="bc05f7ef-ba6f-4cb7-a899-3913b745ed11",
            description=(
                "Triggers on Stripe subscription events (new, upgrade, cancel). "
                "Uses Stripe webhooks directly — real external customers only, "
                "no internal or demo account noise."
            ),
            categories={BlockCategory.INPUT, BlockCategory.DATA},
            input_schema=StripeSubscriptionTriggerBlock.Input,
            output_schema=StripeSubscriptionTriggerBlock.Output,
            webhook_config=BlockWebhookConfig(
                provider=ProviderName.STRIPE,
                webhook_type=StripeWebhookType.ACCOUNT,
                resource_format="",
                event_filter_input="events",
                event_format="customer.subscription.{event}",
            ),
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "events": {"created": True, "updated": True, "deleted": False},
                "payload": EXAMPLE_SUBSCRIPTION_CREATED,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("payload", EXAMPLE_SUBSCRIPTION_CREATED),
                ("event_type", "customer.subscription.created"),
                ("subscription_id", "sub_1OxK2fLkdIwHu7ixABCDEFGH"),
                ("customer_id", "cus_Pq1234ABCDEF"),
                ("status", "active"),
                ("cancel_at_period_end", False),
                ("canceled_at", 0),
                ("previous_attributes", {}),
                ("plan_name", "Pro Monthly"),
                ("plan_interval", "month"),
                ("amount_cents", 2000),
                ("currency", "usd"),
                ("livemode", False),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        payload = input_data.payload

        # Parse before yielding anything: a yield on "error" aborts the block
        # with a BlockExecutionError, so emitting outputs first would leave the
        # node both partially succeeded and failed.
        try:
            subscription = payload["data"]["object"]

            # Plan info lives on the first subscription item; older Stripe API
            # versions only expose it as a top-level `plan`. The two shapes
            # differ only in where the interval and amount live.
            if items := subscription.get("items", {}).get("data", []):
                price_source = items[0].get("price", {})
                plan_interval = price_source.get("recurring", {}).get("interval", "")
                amount_cents = price_source.get("unit_amount") or 0
            else:
                price_source = subscription.get("plan", {})
                plan_interval = price_source.get("interval", "")
                amount_cents = price_source.get("amount") or 0
            plan_name = price_source.get("nickname") or price_source.get("id", "")
        except (KeyError, TypeError) as e:
            yield "error", f"Failed to parse Stripe subscription payload: {e}"
            return

        yield "payload", payload
        yield "event_type", payload.get("type", "")
        yield "subscription_id", subscription.get("id", "")
        yield "customer_id", subscription.get("customer", "")
        yield "status", subscription.get("status", "")
        yield "cancel_at_period_end", bool(subscription.get("cancel_at_period_end"))
        yield "canceled_at", subscription.get("canceled_at") or 0
        yield "previous_attributes", payload.get("data", {}).get(
            "previous_attributes"
        ) or {}
        yield "plan_name", plan_name
        yield "plan_interval", plan_interval
        yield "amount_cents", amount_cents
        yield "currency", subscription.get("currency", "")
        yield "livemode", payload.get("livemode", False)
