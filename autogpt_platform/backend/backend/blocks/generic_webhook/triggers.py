from backend.data.block import (
    Block,
    BlockCategory,
    BlockManualWebhookConfig,
    BlockOutput,
    BlockSchema,
)
from backend.data.model import SchemaField
from backend.integrations.providers import ProviderName
from backend.integrations.webhooks.generic import GenericWebhookType


class GenericWebhookTriggerBlock(Block):
    class Input(BlockSchema):
        payload: dict = SchemaField(hidden=True)
        constants: dict = SchemaField(
            description="The constants to be set when the block is put on the graph",
            default={},
        )

    class Output(BlockSchema):
        output: dict = SchemaField(
            description="The contents of the message AutoGPT received."
        )
        constants: dict = SchemaField(
            description="The constants to be set when the block is put on the graph"
        )

    example_payload = {"message": "Hello, World!"}

    def __init__(self):
        super().__init__(
            id="8fa8c167-2002-47ce-aba8-97572fc5d387",
            description="This block will output the contents of the compass transcription.",
            categories={BlockCategory.HARDWARE},
            input_schema=GenericWebhookTriggerBlock.Input,
            output_schema=GenericWebhookTriggerBlock.Output,
            webhook_config=BlockManualWebhookConfig(
                provider=ProviderName.GENERIC_WEBHOOK,
                webhook_type=GenericWebhookType.PLAIN,
            ),
            test_input=[
                {"constants": {"key": "value"}},
                {"payload": self.example_payload},
            ],
            test_output=[
                ("output", self.example_payload),
                ("constants", {"key": "value"}),
            ],
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        yield "output", input_data.payload
        yield "constants", input_data.constants
