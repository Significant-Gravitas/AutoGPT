"""
Telegram action blocks for sending messages, photos, and voice messages.
"""

import base64
import logging
from enum import Enum
from typing import Optional

from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.execution import ExecutionContext
from backend.data.model import APIKeyCredentials, SchemaField
from backend.util.file import store_media_file
from backend.util.type import MediaFileType

from ._api import call_telegram_api, download_telegram_file
from ._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    TelegramCredentialsField,
    TelegramCredentialsInput,
)

logger = logging.getLogger(__name__)


class ParseMode(str, Enum):
    """Telegram message parse modes."""

    NONE = ""
    MARKDOWN = "Markdown"
    MARKDOWNV2 = "MarkdownV2"
    HTML = "HTML"


class SendTelegramMessageBlock(Block):
    """Send a text message to a Telegram chat."""

    class Input(BlockSchemaInput):
        credentials: TelegramCredentialsInput = TelegramCredentialsField()
        chat_id: int = SchemaField(
            description="The chat ID to send the message to. "
            "Get this from the trigger block's chat_id output."
        )
        text: str = SchemaField(
            description="The text message to send (max 4096 characters)"
        )
        parse_mode: ParseMode = SchemaField(
            description="Message formatting mode (Markdown, HTML, or none)",
            default=ParseMode.NONE,
            advanced=True,
        )
        reply_to_message_id: Optional[int] = SchemaField(
            description="Message ID to reply to",
            default=None,
            advanced=True,
        )
        disable_notification: bool = SchemaField(
            description="Send message silently (no notification sound)",
            default=False,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        message_id: int = SchemaField(description="The ID of the sent message")
        status: str = SchemaField(description="Status of the operation")

    def __init__(self):
        super().__init__(
            id="b2c3d4e5-f6a7-8901-bcde-f23456789012",
            description="Send a text message to a Telegram chat.",
            categories={BlockCategory.SOCIAL},
            input_schema=SendTelegramMessageBlock.Input,
            output_schema=SendTelegramMessageBlock.Output,
            test_input={
                "chat_id": 12345678,
                "text": "Hello from AutoGPT!",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("message_id", 123),
                ("status", "Message sent"),
            ],
            test_mock={
                "_send_message": lambda *args, **kwargs: {
                    "message_id": 123,
                }
            },
        )

    async def _send_message(
        self,
        credentials: APIKeyCredentials,
        chat_id: int,
        text: str,
        parse_mode: str,
        reply_to_message_id: Optional[int],
        disable_notification: bool,
    ) -> dict:
        data: dict = {
            "chat_id": chat_id,
            "text": text,
        }
        if parse_mode:
            data["parse_mode"] = parse_mode
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id
        if disable_notification:
            data["disable_notification"] = True

        return await call_telegram_api(credentials, "sendMessage", data)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            result = await self._send_message(
                credentials=credentials,
                chat_id=input_data.chat_id,
                text=input_data.text,
                parse_mode=input_data.parse_mode.value,
                reply_to_message_id=input_data.reply_to_message_id,
                disable_notification=input_data.disable_notification,
            )
            yield "message_id", result.get("message_id", 0)
            yield "status", "Message sent"
        except Exception as e:
            raise ValueError(f"Failed to send message: {e}")


class SendTelegramPhotoBlock(Block):
    """Send a photo to a Telegram chat."""

    class Input(BlockSchemaInput):
        credentials: TelegramCredentialsInput = TelegramCredentialsField()
        chat_id: int = SchemaField(
            description="The chat ID to send the photo to"
        )
        photo: MediaFileType = SchemaField(
            description="Photo to send (URL, data URI, or workspace:// reference). "
            "URLs are preferred as Telegram will fetch them directly."
        )
        caption: str = SchemaField(
            description="Caption for the photo (max 1024 characters)",
            default="",
            advanced=True,
        )
        parse_mode: ParseMode = SchemaField(
            description="Caption formatting mode",
            default=ParseMode.NONE,
            advanced=True,
        )
        reply_to_message_id: Optional[int] = SchemaField(
            description="Message ID to reply to",
            default=None,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        message_id: int = SchemaField(description="The ID of the sent message")
        status: str = SchemaField(description="Status of the operation")

    def __init__(self):
        super().__init__(
            id="c3d4e5f6-a7b8-9012-cdef-345678901234",
            description="Send a photo to a Telegram chat.",
            categories={BlockCategory.SOCIAL},
            input_schema=SendTelegramPhotoBlock.Input,
            output_schema=SendTelegramPhotoBlock.Output,
            test_input={
                "chat_id": 12345678,
                "photo": "https://example.com/image.jpg",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("message_id", 123),
                ("status", "Photo sent"),
            ],
            test_mock={
                "_send_photo": lambda *args, **kwargs: {"message_id": 123}
            },
        )

    async def _send_photo(
        self,
        credentials: APIKeyCredentials,
        chat_id: int,
        photo_data: str,
        caption: str,
        parse_mode: str,
        reply_to_message_id: Optional[int],
    ) -> dict:
        data: dict = {
            "chat_id": chat_id,
            "photo": photo_data,
        }
        if caption:
            data["caption"] = caption
        if parse_mode:
            data["parse_mode"] = parse_mode
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id

        return await call_telegram_api(credentials, "sendPhoto", data)

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        try:
            photo_input = input_data.photo

            # If it's a URL, pass it directly to Telegram (it will fetch it)
            if photo_input.startswith(("http://", "https://")):
                photo_data = photo_input
            else:
                # For data URIs or workspace:// references, use store_media_file
                photo_data = await store_media_file(
                    file=photo_input,
                    execution_context=execution_context,
                    return_format="for_external_api",
                )

            result = await self._send_photo(
                credentials=credentials,
                chat_id=input_data.chat_id,
                photo_data=photo_data,
                caption=input_data.caption,
                parse_mode=input_data.parse_mode.value,
                reply_to_message_id=input_data.reply_to_message_id,
            )
            yield "message_id", result.get("message_id", 0)
            yield "status", "Photo sent"
        except Exception as e:
            raise ValueError(f"Failed to send photo: {e}")


class SendTelegramVoiceBlock(Block):
    """Send a voice message to a Telegram chat."""

    class Input(BlockSchemaInput):
        credentials: TelegramCredentialsInput = TelegramCredentialsField()
        chat_id: int = SchemaField(
            description="The chat ID to send the voice message to"
        )
        voice: MediaFileType = SchemaField(
            description="Voice message to send (OGG format with OPUS codec). "
            "Can be URL, data URI, or workspace:// reference."
        )
        caption: str = SchemaField(
            description="Caption for the voice message",
            default="",
            advanced=True,
        )
        duration: Optional[int] = SchemaField(
            description="Duration in seconds",
            default=None,
            advanced=True,
        )
        reply_to_message_id: Optional[int] = SchemaField(
            description="Message ID to reply to",
            default=None,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        message_id: int = SchemaField(description="The ID of the sent message")
        status: str = SchemaField(description="Status of the operation")

    def __init__(self):
        super().__init__(
            id="d4e5f6a7-b8c9-0123-def0-456789012345",
            description="Send a voice message to a Telegram chat. "
            "Voice must be OGG format with OPUS codec.",
            categories={BlockCategory.SOCIAL},
            input_schema=SendTelegramVoiceBlock.Input,
            output_schema=SendTelegramVoiceBlock.Output,
            test_input={
                "chat_id": 12345678,
                "voice": "https://example.com/voice.ogg",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("message_id", 123),
                ("status", "Voice sent"),
            ],
            test_mock={
                "_send_voice": lambda *args, **kwargs: {"message_id": 123}
            },
        )

    async def _send_voice(
        self,
        credentials: APIKeyCredentials,
        chat_id: int,
        voice_data: str,
        caption: str,
        duration: Optional[int],
        reply_to_message_id: Optional[int],
    ) -> dict:
        data: dict = {
            "chat_id": chat_id,
            "voice": voice_data,
        }
        if caption:
            data["caption"] = caption
        if duration:
            data["duration"] = duration
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id

        return await call_telegram_api(credentials, "sendVoice", data)

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        try:
            voice_input = input_data.voice

            # If it's a URL, pass it directly to Telegram
            if voice_input.startswith(("http://", "https://")):
                voice_data = voice_input
            else:
                voice_data = await store_media_file(
                    file=voice_input,
                    execution_context=execution_context,
                    return_format="for_external_api",
                )

            result = await self._send_voice(
                credentials=credentials,
                chat_id=input_data.chat_id,
                voice_data=voice_data,
                caption=input_data.caption,
                duration=input_data.duration,
                reply_to_message_id=input_data.reply_to_message_id,
            )
            yield "message_id", result.get("message_id", 0)
            yield "status", "Voice sent"
        except Exception as e:
            raise ValueError(f"Failed to send voice: {e}")


class ReplyToTelegramMessageBlock(Block):
    """Reply to a specific Telegram message."""

    class Input(BlockSchemaInput):
        credentials: TelegramCredentialsInput = TelegramCredentialsField()
        chat_id: int = SchemaField(
            description="The chat ID where the message is"
        )
        reply_to_message_id: int = SchemaField(
            description="The message ID to reply to"
        )
        text: str = SchemaField(description="The reply text")
        parse_mode: ParseMode = SchemaField(
            description="Message formatting mode",
            default=ParseMode.NONE,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        message_id: int = SchemaField(description="The ID of the reply message")
        status: str = SchemaField(description="Status of the operation")

    def __init__(self):
        super().__init__(
            id="e5f6a7b8-c9d0-1234-ef01-567890123456",
            description="Reply to a specific message in a Telegram chat.",
            categories={BlockCategory.SOCIAL},
            input_schema=ReplyToTelegramMessageBlock.Input,
            output_schema=ReplyToTelegramMessageBlock.Output,
            test_input={
                "chat_id": 12345678,
                "reply_to_message_id": 42,
                "text": "This is a reply!",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("message_id", 123),
                ("status", "Reply sent"),
            ],
            test_mock={
                "_send_reply": lambda *args, **kwargs: {"message_id": 123}
            },
        )

    async def _send_reply(
        self,
        credentials: APIKeyCredentials,
        chat_id: int,
        reply_to_message_id: int,
        text: str,
        parse_mode: str,
    ) -> dict:
        data: dict = {
            "chat_id": chat_id,
            "text": text,
            "reply_to_message_id": reply_to_message_id,
        }
        if parse_mode:
            data["parse_mode"] = parse_mode

        return await call_telegram_api(credentials, "sendMessage", data)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            result = await self._send_reply(
                credentials=credentials,
                chat_id=input_data.chat_id,
                reply_to_message_id=input_data.reply_to_message_id,
                text=input_data.text,
                parse_mode=input_data.parse_mode.value,
            )
            yield "message_id", result.get("message_id", 0)
            yield "status", "Reply sent"
        except Exception as e:
            raise ValueError(f"Failed to send reply: {e}")


class GetTelegramFileBlock(Block):
    """Download a file from Telegram by file_id."""

    class Input(BlockSchemaInput):
        credentials: TelegramCredentialsInput = TelegramCredentialsField()
        file_id: str = SchemaField(
            description="The Telegram file_id to download. "
            "Get this from trigger outputs (photo_file_id, voice_file_id, etc.)"
        )

    class Output(BlockSchemaOutput):
        file: MediaFileType = SchemaField(
            description="The downloaded file (workspace:// reference or data URI)"
        )
        status: str = SchemaField(description="Status of the operation")

    def __init__(self):
        super().__init__(
            id="f6a7b8c9-d0e1-2345-f012-678901234567",
            description="Download a file from Telegram using its file_id. "
            "Use this to process photos, voice messages, or documents received.",
            categories={BlockCategory.SOCIAL},
            input_schema=GetTelegramFileBlock.Input,
            output_schema=GetTelegramFileBlock.Output,
            test_input={
                "file_id": "AgACAgIAAxkBAAI...",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("file", "data:application/octet-stream;base64,dGVzdA=="),
                ("status", "File downloaded"),
            ],
            test_mock={
                "_download_file": lambda *args, **kwargs: b"test"
            },
        )

    async def _download_file(
        self,
        credentials: APIKeyCredentials,
        file_id: str,
    ) -> bytes:
        return await download_telegram_file(credentials, file_id)

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        try:
            # Download the file from Telegram
            file_content = await self._download_file(
                credentials=credentials,
                file_id=input_data.file_id,
            )

            # Convert to data URI
            mime_type = "application/octet-stream"
            data_uri = (
                f"data:{mime_type};base64,"
                f"{base64.b64encode(file_content).decode()}"
            )

            # Store and get appropriate output format
            file_result = await store_media_file(
                file=data_uri,
                execution_context=execution_context,
                return_format="for_block_output",
            )

            yield "file", file_result
            yield "status", "File downloaded"
        except Exception as e:
            raise ValueError(f"Failed to download file: {e}")
