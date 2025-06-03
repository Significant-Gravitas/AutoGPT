"""
Example Webhook Block using the new SDK

This demonstrates webhook auto-registration without modifying
files outside the blocks folder.
"""

from backend.sdk import *  # noqa: F403, F405


# First, define a simple webhook manager for our example service
class ExampleWebhookManager(BaseWebhooksManager):
    """Example webhook manager for demonstration."""

    PROVIDER_NAME = ProviderName.GITHUB  # Reuse GitHub for example

    class WebhookType(str, Enum):
        EXAMPLE = "example"

    async def validate_payload(self, webhook, request) -> tuple[dict, str]:
        """Validate incoming webhook payload."""
        payload = await request.json()
        event_type = request.headers.get("X-Example-Event", "unknown")
        return payload, event_type

    async def _register_webhook(self, webhook, credentials) -> tuple[str, dict]:
        """Register webhook with external service."""
        # In real implementation, this would call the external API
        return "example-webhook-id", {"registered": True}

    async def _deregister_webhook(self, webhook, credentials) -> None:
        """Deregister webhook from external service."""
        # In real implementation, this would call the external API
        pass


# Now create the webhook block with auto-registration
@provider("examplewebhook")
@webhook_config("examplewebhook", ExampleWebhookManager)
@cost_config(
    BlockCost(
        cost_amount=0, cost_type=BlockCostType.RUN
    )  # Webhooks typically free to receive
)
class ExampleWebhookSDKBlock(Block):
    """
    Example webhook block demonstrating SDK webhook capabilities.

    With the new SDK:
    - Webhook manager registered via @webhook_config decorator
    - No need to modify webhooks/__init__.py
    - Fully self-contained webhook implementation
    """

    class Input(BlockSchema):
        webhook_url: String = SchemaField(
            description="URL to receive webhooks (auto-generated)",
            default="",
            hidden=True,
        )
        event_filter: Boolean = SchemaField(
            description="Filter for specific events", default=True
        )
        payload: Dict = SchemaField(
            description="Webhook payload data", default={}, hidden=True
        )

    class Output(BlockSchema):
        event_type: String = SchemaField(description="Type of webhook event")
        event_data: Dict = SchemaField(description="Event payload data")
        timestamp: String = SchemaField(description="Event timestamp")
        error: String = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="example-webhook-sdk-block-87654321-4321-4321-4321-210987654321",
            description="Example webhook block with auto-registration",
            categories={BlockCategory.INPUT},
            input_schema=ExampleWebhookSDKBlock.Input,
            output_schema=ExampleWebhookSDKBlock.Output,
            block_type=BlockType.WEBHOOK,
            webhook_config=BlockWebhookConfig(
                provider=ProviderName.GITHUB,  # Using GitHub for example
                webhook_type="example",
                event_filter_input="event_filter",
                resource_format="{event}",
            ),
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            # Extract webhook payload
            payload = input_data.payload

            # Get event type and timestamp
            event_type = payload.get("action", "unknown")
            timestamp = payload.get("timestamp", "")

            # Filter events if enabled
            if input_data.event_filter and event_type not in ["created", "updated"]:
                yield "event_type", "filtered"
                yield "event_data", {}
                yield "timestamp", timestamp
                return

            yield "event_type", event_type
            yield "event_data", payload
            yield "timestamp", timestamp

        except Exception as e:
            yield "error", str(e)
            yield "event_type", "error"
            yield "event_data", {}
            yield "timestamp", ""
