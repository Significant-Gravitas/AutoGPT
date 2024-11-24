from typing import Literal
from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    BlockWebhookConfig,
)
from backend.data.model import CredentialsField, CredentialsMetaInput, SchemaField
from backend.integrations.webhooks.simple_webhook_manager import CompassWebhookType
from pydantic import BaseModel


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
        class EventsFilter(BaseModel):
            all: bool = True

        payload: TranscriptionDataModel = SchemaField(hidden=True)
        events: EventsFilter = SchemaField(
            description="Filter the events to be triggered on."
        )

        credentials: CredentialsMetaInput[Literal["compass"], Literal["api_key"]] = (
            CredentialsField(
                provider="compass",
                supported_credential_types={"api_key"},
            )
        )

    class Output(BlockSchema):
        transcription: str = SchemaField(
            description="The transcription of the compass transcription."
        )

    def __init__(self):
        super().__init__(
            id="9464a020-ed1d-49e1-990f-7f2ac924a2b7",
            description="This block forwards an input value as output, allowing reuse without change.",
            categories={BlockCategory.HARDWARE},
            input_schema=CompassAITriggerBlock.Input,
            output_schema=CompassAITriggerBlock.Output,
            webhook_config=BlockWebhookConfig(
                provider="simple_hook_manager",
                webhook_type=CompassWebhookType.TRANSCRIPTION,
                resource_format="{transcription_id}",
                event_filter_input="events",
                event_format="transcription.{event}",
            ),
            # test_input=[
            #     {"input": "Hello, World!"},
            #     {"input": "Hello, World!", "data": "Existing Data"},
            # ],
            # test_output=[
            #     ("output", "Hello, World!"),  # No data provided, so trigger is returned
            #     ("output", "Existing Data"),  # Data is provided, so data is returned.
            # ],
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        yield "transcription", input_data.payload.transcription
