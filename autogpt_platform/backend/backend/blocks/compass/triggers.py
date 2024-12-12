from pydantic import BaseModel

from backend.data.block import (
    Block,
    BlockCategory,
    BlockManualWebhookConfig,
    BlockOutput,
    BlockSchema,
)
from backend.data.model import SchemaField
from backend.integrations.webhooks.compass import CompassWebhookType


class Transcription(BaseModel):
    text: str
    speaker: str
    end: float
    start: float
    duration: float


class TranscriptionDataModel(BaseModel):
    date: str
    transcription: str
    transcriptions: list[Transcription]


class CompassAITriggerBlock(Block):
    class Input(BlockSchema):
        payload: TranscriptionDataModel = SchemaField(hidden=True)

    class Output(BlockSchema):
        transcription: str = SchemaField(
            description="The contents of the compass transcription."
        )

    def __init__(self):
        super().__init__(
            id="9464a020-ed1d-49e1-990f-7f2ac924a2b7",
            description="This block will output the contents of the compass transcription.",
            categories={BlockCategory.HARDWARE},
            input_schema=CompassAITriggerBlock.Input,
            output_schema=CompassAITriggerBlock.Output,
            webhook_config=BlockManualWebhookConfig(
                provider="compass",
                webhook_type=CompassWebhookType.TRANSCRIPTION,
            ),
            test_input=[
                {"input": "Hello, World!"},
                {"input": "Hello, World!", "data": "Existing Data"},
            ],
            # test_output=[
            #     ("output", "Hello, World!"),  # No data provided, so trigger is returned
            #     ("output", "Existing Data"),  # Data is provided, so data is returned.
            # ],
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        yield "transcription", input_data.payload.transcription
