from backend.sdk import (
    Block,
    BlockCategory,
    BlockManualWebhookConfig,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    ProviderBuilder,
    ProviderName,
    SchemaField,
)

from ._webhook import GenericWebhooksManager, GenericWebhookType

generic_webhook = (
    ProviderBuilder("generic_webhook")
    .with_webhook_manager(GenericWebhooksManager)
    .build()
)


class GenericWebhookTriggerBlock(Block):
    class Input(BlockSchemaInput):
        payload: dict = SchemaField(hidden=True, default_factory=dict)
        constants: dict = SchemaField(
            description="The constants to be set when the block is put on the graph",
            default_factory=dict,
        )

    class Output(BlockSchemaOutput):
        payload: dict = SchemaField(
            description="The complete webhook payload that was received from the generic webhook."
        )
        constants: dict = SchemaField(
            description="The constants to be set when the block is put on the graph"
        )

    example_payload = {"message": "Hello, World!"}

    def __init__(self):
        super().__init__(
            id="8fa8c167-2002-47ce-aba8-97572fc5d387",
            description="This block will output the contents of the generic input for the webhook.",
            categories={BlockCategory.INPUT},
            input_schema=GenericWebhookTriggerBlock.Input,
            output_schema=GenericWebhookTriggerBlock.Output,
            webhook_config=BlockManualWebhookConfig(
                provider=ProviderName(generic_webhook.name),
                webhook_type=GenericWebhookType.PLAIN,
            ),
            test_input={"constants": {"key": "value"}, "payload": self.example_payload},
            test_output=[
                ("constants", {"key": "value"}),
                ("payload", self.example_payload),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        yield "constants", input_data.constants
        yield "payload", input_data.payload
