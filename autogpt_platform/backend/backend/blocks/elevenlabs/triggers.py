"""
ElevenLabs webhook trigger blocks.
"""

from pydantic import BaseModel

from backend.sdk import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    BlockType,
    BlockWebhookConfig,
    CredentialsMetaInput,
    ProviderName,
    SchemaField,
)

from ._config import elevenlabs


class ElevenLabsWebhookTriggerBlock(Block):
    """
    Starts a flow when ElevenLabs POSTs an event (STT finished, voice removal, etc.).
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = elevenlabs.credentials_field(
            description="ElevenLabs API credentials"
        )
        webhook_url: str = SchemaField(
            description="URL to receive webhooks (auto-generated)",
            default="",
            hidden=True,
        )

        class EventsFilter(BaseModel):
            """ElevenLabs event types to subscribe to"""

            speech_to_text_completed: bool = SchemaField(
                description="Speech-to-text transcription completed", default=True
            )
            post_call_transcription: bool = SchemaField(
                description="Conversational AI call transcription completed",
                default=True,
            )
            voice_removal_notice: bool = SchemaField(
                description="Voice scheduled for removal", default=True
            )
            voice_removed: bool = SchemaField(
                description="Voice has been removed", default=True
            )
            voice_removal_notice_withdrawn: bool = SchemaField(
                description="Voice removal cancelled", default=True
            )

        events: EventsFilter = SchemaField(
            title="Events", description="The events to subscribe to"
        )

        # Webhook payload - populated by the system
        payload: dict = SchemaField(
            description="Webhook payload data",
            default={},
            hidden=True,
        )

    class Output(BlockSchema):
        type: str = SchemaField(description="Event type")
        event_timestamp: int = SchemaField(description="Unix timestamp of the event")
        data: dict = SchemaField(description="Event-specific data payload")

    def __init__(self):
        super().__init__(
            id="c1d2e3f4-a5b6-c7d8-e9f0-a1b2c3d4e5f6",
            description="Receive webhook events from ElevenLabs",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
            block_type=BlockType.WEBHOOK,
            webhook_config=BlockWebhookConfig(
                provider=ProviderName("elevenlabs"),
                webhook_type="notification",
                event_filter_input="events",
                resource_format="",
            ),
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        # Extract webhook data
        payload = input_data.payload

        # Extract event type
        event_type = payload.get("type", "unknown")

        # Map event types to filter fields
        event_filter_map = {
            "speech_to_text_completed": input_data.events.speech_to_text_completed,
            "post_call_transcription": input_data.events.post_call_transcription,
            "voice_removal_notice": input_data.events.voice_removal_notice,
            "voice_removed": input_data.events.voice_removed,
            "voice_removal_notice_withdrawn": input_data.events.voice_removal_notice_withdrawn,
        }

        # Check if this event type is enabled
        if not event_filter_map.get(event_type, False):
            # Skip this event
            return

        # Extract common fields
        yield "type", event_type
        yield "event_timestamp", payload.get("event_timestamp", 0)

        # Extract event-specific data
        data = payload.get("data", {})

        # Process based on event type
        if event_type == "speech_to_text_completed":
            # STT transcription completed
            processed_data = {
                "transcription_id": data.get("transcription_id"),
                "text": data.get("text"),
                "words": data.get("words", []),
                "language_code": data.get("language_code"),
                "language_probability": data.get("language_probability"),
            }
        elif event_type == "post_call_transcription":
            # Conversational AI call transcription
            processed_data = {
                "agent_id": data.get("agent_id"),
                "conversation_id": data.get("conversation_id"),
                "transcript": data.get("transcript"),
                "metadata": data.get("metadata", {}),
            }
        elif event_type == "voice_removal_notice":
            # Voice scheduled for removal
            processed_data = {
                "voice_id": data.get("voice_id"),
                "voice_name": data.get("voice_name"),
                "removal_date": data.get("removal_date"),
                "reason": data.get("reason"),
            }
        elif event_type == "voice_removal_notice_withdrawn":
            # Voice removal cancelled
            processed_data = {
                "voice_id": data.get("voice_id"),
                "voice_name": data.get("voice_name"),
            }
        elif event_type == "voice_removed":
            # Voice has been removed
            processed_data = {
                "voice_id": data.get("voice_id"),
                "voice_name": data.get("voice_name"),
                "removed_at": data.get("removed_at"),
            }
        else:
            # Unknown event type, pass through raw data
            processed_data = data

        yield "data", processed_data
