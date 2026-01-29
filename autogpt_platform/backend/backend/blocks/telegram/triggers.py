"""
Telegram trigger blocks for receiving messages via webhooks.
"""

import logging

from pydantic import BaseModel

from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    BlockWebhookConfig,
)
from backend.data.model import SchemaField
from backend.integrations.providers import ProviderName
from backend.integrations.webhooks.telegram import TelegramWebhookType

from ._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    TelegramCredentialsField,
    TelegramCredentialsInput,
)

logger = logging.getLogger(__name__)


# Example payload for testing
EXAMPLE_MESSAGE_PAYLOAD = {
    "update_id": 123456789,
    "message": {
        "message_id": 1,
        "from": {
            "id": 12345678,
            "is_bot": False,
            "first_name": "John",
            "last_name": "Doe",
            "username": "johndoe",
            "language_code": "en",
        },
        "chat": {
            "id": 12345678,
            "first_name": "John",
            "last_name": "Doe",
            "username": "johndoe",
            "type": "private",
        },
        "date": 1234567890,
        "text": "Hello, bot!",
    },
}


class TelegramTriggerBase:
    """Base class for Telegram trigger blocks."""

    class Input(BlockSchemaInput):
        credentials: TelegramCredentialsInput = TelegramCredentialsField()
        payload: dict = SchemaField(hidden=True, default_factory=dict)


class TelegramMessageTriggerBlock(TelegramTriggerBase, Block):
    """
    Triggers when a message is received by your Telegram bot.

    Supports text, photos, voice messages, and audio files.
    Connect the outputs to other blocks to process messages and send responses.
    """

    class Input(TelegramTriggerBase.Input):
        class EventsFilter(BaseModel):
            """Filter for message types to receive."""

            text: bool = True
            photo: bool = False
            voice: bool = False
            audio: bool = False
            document: bool = False
            video: bool = False

        events: EventsFilter = SchemaField(
            title="Message Types", description="Types of messages to receive"
        )

    class Output(BlockSchemaOutput):
        payload: dict = SchemaField(
            description="The complete webhook payload from Telegram"
        )
        chat_id: int = SchemaField(
            description="The chat ID where the message was received. "
            "Use this to send replies."
        )
        message_id: int = SchemaField(description="The unique message ID")
        user_id: int = SchemaField(description="The user ID who sent the message")
        username: str = SchemaField(
            description="Username of the sender (may be empty)"
        )
        first_name: str = SchemaField(description="First name of the sender")
        event: str = SchemaField(
            description="The message type (text, photo, voice, audio, etc.)"
        )
        text: str = SchemaField(
            description="Text content of the message (for text messages)"
        )
        photo_file_id: str = SchemaField(
            description="File ID of the photo (for photo messages). "
            "Use GetTelegramFileBlock to download."
        )
        voice_file_id: str = SchemaField(
            description="File ID of the voice message (for voice messages). "
            "Use GetTelegramFileBlock to download."
        )
        audio_file_id: str = SchemaField(
            description="File ID of the audio file (for audio messages). "
            "Use GetTelegramFileBlock to download."
        )
        caption: str = SchemaField(description="Caption for media messages")

    def __init__(self):
        super().__init__(
            id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            description="Triggers when a message is received by your Telegram bot. "
            "Supports text, photos, voice messages, and audio files.",
            categories={BlockCategory.SOCIAL},
            input_schema=TelegramMessageTriggerBlock.Input,
            output_schema=TelegramMessageTriggerBlock.Output,
            webhook_config=BlockWebhookConfig(
                provider=ProviderName.TELEGRAM,
                webhook_type=TelegramWebhookType.BOT,
                resource_format="bot",
                event_filter_input="events",
                event_format="message.{event}",
            ),
            test_input={
                "events": {"text": True, "photo": True},
                "credentials": TEST_CREDENTIALS_INPUT,
                "payload": EXAMPLE_MESSAGE_PAYLOAD,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("payload", EXAMPLE_MESSAGE_PAYLOAD),
                ("chat_id", 12345678),
                ("message_id", 1),
                ("user_id", 12345678),
                ("username", "johndoe"),
                ("first_name", "John"),
                ("event", "text"),
                ("text", "Hello, bot!"),
                ("photo_file_id", ""),
                ("voice_file_id", ""),
                ("audio_file_id", ""),
                ("caption", ""),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        payload = input_data.payload
        message = payload.get("message", {})

        # Extract common fields
        chat = message.get("chat", {})
        sender = message.get("from", {})

        yield "payload", payload
        yield "chat_id", chat.get("id", 0)
        yield "message_id", message.get("message_id", 0)
        yield "user_id", sender.get("id", 0)
        yield "username", sender.get("username", "")
        yield "first_name", sender.get("first_name", "")

        # Determine message type and extract content
        if "text" in message:
            yield "event", "text"
            yield "text", message.get("text", "")
            yield "photo_file_id", ""
            yield "voice_file_id", ""
            yield "audio_file_id", ""
            yield "caption", ""
        elif "photo" in message:
            # Get the largest photo (last in array)
            photos = message.get("photo", [])
            file_id = photos[-1]["file_id"] if photos else ""
            yield "event", "photo"
            yield "text", ""
            yield "photo_file_id", file_id
            yield "voice_file_id", ""
            yield "audio_file_id", ""
            yield "caption", message.get("caption", "")
        elif "voice" in message:
            voice = message.get("voice", {})
            yield "event", "voice"
            yield "text", ""
            yield "photo_file_id", ""
            yield "voice_file_id", voice.get("file_id", "")
            yield "audio_file_id", ""
            yield "caption", message.get("caption", "")
        elif "audio" in message:
            audio = message.get("audio", {})
            yield "event", "audio"
            yield "text", ""
            yield "photo_file_id", ""
            yield "voice_file_id", ""
            yield "audio_file_id", audio.get("file_id", "")
            yield "caption", message.get("caption", "")
        elif "document" in message:
            document = message.get("document", {})
            yield "event", "document"
            yield "text", ""
            yield "photo_file_id", ""
            yield "voice_file_id", ""
            yield "audio_file_id", document.get("file_id", "")
            yield "caption", message.get("caption", "")
        elif "video" in message:
            video = message.get("video", {})
            yield "event", "video"
            yield "text", ""
            yield "photo_file_id", ""
            yield "voice_file_id", ""
            yield "audio_file_id", video.get("file_id", "")
            yield "caption", message.get("caption", "")
        else:
            yield "event", "other"
            yield "text", ""
            yield "photo_file_id", ""
            yield "voice_file_id", ""
            yield "audio_file_id", ""
            yield "caption", ""
