"""
Telegram trigger blocks for receiving messages via webhooks.
"""

import logging

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
    Triggers when a message is received or edited in your Telegram bot.

    Supports text, photos, voice messages, audio files, documents, and videos.
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
            edited_message: bool = False

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
        username: str = SchemaField(description="Username of the sender (may be empty)")
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
        file_id: str = SchemaField(
            description="File ID for document/video messages. "
            "Use GetTelegramFileBlock to download."
        )
        file_name: str = SchemaField(
            description="Original filename (for document/audio messages)"
        )
        caption: str = SchemaField(description="Caption for media messages")
        is_edited: bool = SchemaField(
            description="Whether this is an edit of a previously sent message"
        )

    def __init__(self):
        super().__init__(
            id="4435e4e0-df6e-4301-8f35-ad70b12fc9ec",
            description="Triggers when a message is received or edited in your Telegram bot. "
            "Supports text, photos, voice messages, audio files, documents, and videos.",
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
                ("is_edited", False),
                ("event", "text"),
                ("text", "Hello, bot!"),
                ("photo_file_id", ""),
                ("voice_file_id", ""),
                ("audio_file_id", ""),
                ("file_id", ""),
                ("file_name", ""),
                ("caption", ""),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        payload = input_data.payload
        is_edited = "edited_message" in payload
        message = payload.get("message") or payload.get("edited_message", {})

        # Extract common fields
        chat = message.get("chat", {})
        sender = message.get("from", {})

        yield "payload", payload
        yield "chat_id", chat.get("id", 0)
        yield "message_id", message.get("message_id", 0)
        yield "user_id", sender.get("id", 0)
        yield "username", sender.get("username", "")
        yield "first_name", sender.get("first_name", "")
        yield "is_edited", is_edited

        # For edited messages, yield event as "edited_message" and extract
        # all content fields from the edited message body
        if is_edited:
            yield "event", "edited_message"
            yield "text", message.get("text", "")
            photos = message.get("photo", [])
            yield "photo_file_id", photos[-1].get("file_id", "") if photos else ""
            voice = message.get("voice", {})
            yield "voice_file_id", voice.get("file_id", "")
            audio = message.get("audio", {})
            yield "audio_file_id", audio.get("file_id", "")
            document = message.get("document", {})
            video = message.get("video", {})
            yield "file_id", (document.get("file_id", "") or video.get("file_id", ""))
            yield "file_name", (
                document.get("file_name", "") or audio.get("file_name", "")
            )
            yield "caption", message.get("caption", "")
        # Determine message type and extract content
        elif "text" in message:
            yield "event", "text"
            yield "text", message.get("text", "")
            yield "photo_file_id", ""
            yield "voice_file_id", ""
            yield "audio_file_id", ""
            yield "file_id", ""
            yield "file_name", ""
            yield "caption", ""
        elif "photo" in message:
            # Get the largest photo (last in array)
            photos = message.get("photo", [])
            photo_fid = photos[-1].get("file_id", "") if photos else ""
            yield "event", "photo"
            yield "text", ""
            yield "photo_file_id", photo_fid
            yield "voice_file_id", ""
            yield "audio_file_id", ""
            yield "file_id", ""
            yield "file_name", ""
            yield "caption", message.get("caption", "")
        elif "voice" in message:
            voice = message.get("voice", {})
            yield "event", "voice"
            yield "text", ""
            yield "photo_file_id", ""
            yield "voice_file_id", voice.get("file_id", "")
            yield "audio_file_id", ""
            yield "file_id", ""
            yield "file_name", ""
            yield "caption", message.get("caption", "")
        elif "audio" in message:
            audio = message.get("audio", {})
            yield "event", "audio"
            yield "text", ""
            yield "photo_file_id", ""
            yield "voice_file_id", ""
            yield "audio_file_id", audio.get("file_id", "")
            yield "file_id", ""
            yield "file_name", audio.get("file_name", "")
            yield "caption", message.get("caption", "")
        elif "document" in message:
            document = message.get("document", {})
            yield "event", "document"
            yield "text", ""
            yield "photo_file_id", ""
            yield "voice_file_id", ""
            yield "audio_file_id", ""
            yield "file_id", document.get("file_id", "")
            yield "file_name", document.get("file_name", "")
            yield "caption", message.get("caption", "")
        elif "video" in message:
            video = message.get("video", {})
            yield "event", "video"
            yield "text", ""
            yield "photo_file_id", ""
            yield "voice_file_id", ""
            yield "audio_file_id", ""
            yield "file_id", video.get("file_id", "")
            yield "file_name", video.get("file_name", "")
            yield "caption", message.get("caption", "")
        else:
            yield "event", "other"
            yield "text", ""
            yield "photo_file_id", ""
            yield "voice_file_id", ""
            yield "audio_file_id", ""
            yield "file_id", ""
            yield "file_name", ""
            yield "caption", ""


# Example payload for reaction trigger testing
EXAMPLE_REACTION_PAYLOAD = {
    "update_id": 123456790,
    "message_reaction": {
        "chat": {
            "id": 12345678,
            "first_name": "John",
            "last_name": "Doe",
            "username": "johndoe",
            "type": "private",
        },
        "message_id": 42,
        "user": {
            "id": 12345678,
            "is_bot": False,
            "first_name": "John",
            "username": "johndoe",
        },
        "date": 1234567890,
        "new_reaction": [{"type": "emoji", "emoji": "👍"}],
        "old_reaction": [],
    },
}


class TelegramMessageReactionTriggerBlock(TelegramTriggerBase, Block):
    """
    Triggers when a reaction to a message is changed.

    Works automatically in private chats. In group chats, the bot must be
    an administrator to receive reaction updates.
    """

    class Input(TelegramTriggerBase.Input):
        pass

    class Output(BlockSchemaOutput):
        payload: dict = SchemaField(
            description="The complete webhook payload from Telegram"
        )
        chat_id: int = SchemaField(
            description="The chat ID where the reaction occurred"
        )
        message_id: int = SchemaField(description="The message ID that was reacted to")
        user_id: int = SchemaField(description="The user ID who changed the reaction")
        username: str = SchemaField(description="Username of the user (may be empty)")
        new_reactions: list = SchemaField(
            description="List of new reactions on the message"
        )
        old_reactions: list = SchemaField(
            description="List of previous reactions on the message"
        )

    def __init__(self):
        super().__init__(
            id="82525328-9368-4966-8f0c-cd78e80181fd",
            description="Triggers when a reaction to a message is changed. "
            "Works in private chats automatically. "
            "In groups, the bot must be an administrator.",
            categories={BlockCategory.SOCIAL},
            input_schema=TelegramMessageReactionTriggerBlock.Input,
            output_schema=TelegramMessageReactionTriggerBlock.Output,
            webhook_config=BlockWebhookConfig(
                provider=ProviderName.TELEGRAM,
                webhook_type=TelegramWebhookType.BOT,
                resource_format="bot",
                event_filter_input="",
                event_format="message_reaction",
            ),
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "payload": EXAMPLE_REACTION_PAYLOAD,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("payload", EXAMPLE_REACTION_PAYLOAD),
                ("chat_id", 12345678),
                ("message_id", 42),
                ("user_id", 12345678),
                ("username", "johndoe"),
                ("new_reactions", [{"type": "emoji", "emoji": "👍"}]),
                ("old_reactions", []),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        payload = input_data.payload
        reaction = payload.get("message_reaction", {})

        chat = reaction.get("chat", {})
        user = reaction.get("user", {})

        yield "payload", payload
        yield "chat_id", chat.get("id", 0)
        yield "message_id", reaction.get("message_id", 0)
        yield "user_id", user.get("id", 0)
        yield "username", user.get("username", "")
        yield "new_reactions", reaction.get("new_reaction", [])
        yield "old_reactions", reaction.get("old_reaction", [])
