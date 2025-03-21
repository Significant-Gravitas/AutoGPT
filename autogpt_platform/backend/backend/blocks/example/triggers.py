import logging

from backend.data.block import (
    Block,
    BlockCategory,
    BlockManualWebhookConfig,
    BlockOutput,
    BlockSchema,
)
from backend.data.model import SchemaField
from backend.integrations.webhooks.example import ExampleWebhookEventType

logger = logging.getLogger(__name__)


class ExampleTriggerBlock(Block):
    """
    A trigger block that is activated by an external webhook event.

    Unlike standard blocks that are manually executed, trigger blocks are automatically
    activated when a webhook event is received from the specified provider.
    """

    class Input(BlockSchema):
        # The payload field is hidden because it's automatically populated by the webhook
        # system rather than being manually entered by the user
        payload: dict = SchemaField(hidden=True)

    class Output(BlockSchema):
        event_data: dict = SchemaField(
            description="The contents of the example webhook event."
        )

    def __init__(self):
        super().__init__(
            id="7c5933ce-d60c-42dd-9c4e-db82496474a3",
            description="This block will output the contents of an example webhook event.",
            categories={BlockCategory.BASIC},
            input_schema=ExampleTriggerBlock.Input,
            output_schema=ExampleTriggerBlock.Output,
            # The webhook_config is a key difference from standard blocks
            # It defines which external service can trigger this block and what type of events it responds to
            webhook_config=BlockManualWebhookConfig(
                provider="example_provider",  # The external service that will send webhook events
                webhook_type=ExampleWebhookEventType.EXAMPLE_EVENT,  # The specific event type this block responds to
            ),
            # Test input for trigger blocks should mimic the payload structure that would be received from the webhook
            test_input=[
                {
                    "payload": {
                        "event_type": "example",
                        "data": "Sample webhook data",
                    }
                }
            ],
            test_output=[
                ("event_data", {"event_type": "example", "data": "Sample webhook data"})
            ],
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        # For trigger blocks, the run method is called automatically when a webhook event is received
        # The payload from the webhook is passed in as input_data.payload
        logger.info("Example trigger block run with payload: %s", input_data.payload)
        yield "event_data", input_data.payload
