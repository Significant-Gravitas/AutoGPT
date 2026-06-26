import json
from pathlib import Path

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
            https://docs.stripe.com/api/events/types#event_types-customer.subscription.created
            """

            created: bool = True
            updated: bool = True
            deleted: bool = False

        events: EventsFilter = SchemaField(
            title="Events",
            description="Subscription lifecycle events to subscribe to",
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
        customer_id: str = SchemaField(
            description="Stripe customer ID (cus_...)"
        )
        status: str = SchemaField(
            description="Subscription status: active, trialing, past_due, canceled, etc."
        )
        plan_name: str = SchemaField(
            description="Plan nickname from the subscription's first item price"
        )
        plan_interval: str = SchemaField(
            description="Billing interval: month or year"
        )
        amount_cents: int = SchemaField(
            description="Plan unit amount in the smallest currency unit (e.g. cents for USD)"
        )
        currency: str = SchemaField(description="Three-letter ISO currency code")
        livemode: bool = SchemaField(
            description="True for live Stripe data, False for test mode"
        )
        error: str = SchemaField(
            description="Error message if the payload could not be parsed"
        )

    def __init__(self):
        from backend.integrations.webhooks.stripe import StripeWebhookType

        example_payload = json.loads(
            (_PAYLOAD_DIR / "customer.subscription.created.json").read_text(
                encoding="utf-8"
            )
        )

        super().__init__(
            id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
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
                "payload": example_payload,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("payload", example_payload),
                ("event_type", "customer.subscription.created"),
                ("subscription_id", "sub_1OxK2fLkdIwHu7ixABCDEFGH"),
                ("customer_id", "cus_Pq1234ABCDEF"),
                ("status", "active"),
                ("plan_name", "Pro Monthly"),
                ("plan_interval", "month"),
                ("amount_cents", 2000),
                ("currency", "usd"),
                ("livemode", False),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        payload = input_data.payload
        yield "payload", payload

        try:
            subscription = payload["data"]["object"]
            event_type = payload.get("type", "")

            yield "event_type", event_type
            yield "subscription_id", subscription.get("id", "")
            yield "customer_id", subscription.get("customer", "")
            yield "status", subscription.get("status", "")
            yield "currency", subscription.get("currency", "")
            yield "livemode", payload.get("livemode", False)

            # Extract plan info from the first subscription item
            items = subscription.get("items", {}).get("data", [])
            if items:
                price = items[0].get("price", {})
                recurring = price.get("recurring", {})
                yield "plan_name", price.get("nickname") or price.get("id", "")
                yield "plan_interval", recurring.get("interval", "")
                yield "amount_cents", price.get("unit_amount", 0)
            else:
                # Fall back to top-level plan (older Stripe API versions)
                plan = subscription.get("plan", {})
                yield "plan_name", plan.get("nickname") or plan.get("id", "")
                yield "plan_interval", plan.get("interval", "")
                yield "amount_cents", plan.get("amount", 0)
        except (KeyError, TypeError) as e:
            yield "error", f"Failed to parse Stripe subscription payload: {e}"
