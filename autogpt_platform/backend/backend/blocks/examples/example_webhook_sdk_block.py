"""
Example Webhook Block using the new SDK

This demonstrates webhook auto-registration without modifying
files outside the blocks folder.
"""

from backend.sdk import *  # noqa: F403, F405


# Define event filter model
class ExampleEventFilter(BaseModel):
    created: bool = Field(default=True, description="Listen for created events")
    updated: bool = Field(default=True, description="Listen for updated events")
    deleted: bool = Field(default=False, description="Listen for deleted events")


# First, define a simple webhook manager for our example service
class ExampleWebhookManager(BaseWebhooksManager):
    """Example webhook manager for demonstration."""

    PROVIDER_NAME = ProviderName.GITHUB  # Reuse GitHub for example

    class WebhookType(str, Enum):
        EXAMPLE = "example"

    @classmethod
    async def validate_payload(cls, webhook, request) -> tuple[dict, str]:
        """Validate incoming webhook payload."""
        payload = await request.json()
        event_type = request.headers.get("X-Example-Event", "unknown")
        return payload, event_type

    async def _register_webhook(
        self,
        credentials,
        webhook_type: str,
        resource: str,
        events: list[str],
        ingress_url: str,
        secret: str,
    ) -> tuple[str, dict]:
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
        credentials: CredentialsMetaInput = CredentialsField(
            provider="examplewebhook",
            supported_credential_types={"api_key"},
            description="Credentials for webhook service",
        )
        webhook_url: String = SchemaField(
            description="URL to receive webhooks (auto-generated)",
            default="",
            hidden=True,
        )
        event_filter: ExampleEventFilter = SchemaField(
            description="Filter for specific events", default=ExampleEventFilter()
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
            id="7e50eb33-f854-4b73-99a1-50cad2819ae0",
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

            # Filter events based on event filter settings
            event_filter = input_data.event_filter
            should_process = False

            if event_type == "created" and event_filter.created:
                should_process = True
            elif event_type == "updated" and event_filter.updated:
                should_process = True
            elif event_type == "deleted" and event_filter.deleted:
                should_process = True

            if not should_process:
                yield "event_type", "filtered"
                yield "event_data", {}
                yield "timestamp", timestamp
                yield "error", ""
                return

            yield "event_type", event_type
            yield "event_data", payload
            yield "timestamp", timestamp

        except Exception as e:
            yield "error", str(e)
            yield "event_type", "error"
            yield "event_data", {}
            yield "timestamp", ""
