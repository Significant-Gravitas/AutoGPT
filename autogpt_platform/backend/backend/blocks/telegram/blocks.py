"""
Telegram action blocks for sending messages, media, and managing chat content.
"""

import base64
import logging
import mimetypes
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.execution import ExecutionContext
from backend.data.model import APIKeyCredentials, SchemaField
from backend.util.file import get_exec_file_path, store_media_file
from backend.util.type import MediaFileType

from ._api import (
    TelegramMessageResult,
    call_telegram_api,
    call_telegram_api_with_file,
    download_telegram_file,
)
from ._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    TelegramCredentialsField,
    TelegramCredentialsInput,
)

logger = logging.getLogger(__name__)


class ParseMode(str, Enum):
    """Telegram message parse modes."""

    NONE = "none"
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
            id="787678ad-1f47-4efc-89df-643f9908621a",
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
                "_send_message": lambda *args, **kwargs: TelegramMessageResult(
                    message_id=123,
                )
            },
        )

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
            yield "message_id", result.message_id
            yield "status", "Message sent"
        except Exception as e:
            raise ValueError(f"Failed to send message: {e}") from e

    async def _send_message(
        self,
        credentials: APIKeyCredentials,
        chat_id: int,
        text: str,
        parse_mode: str,
        reply_to_message_id: Optional[int],
        disable_notification: bool,
    ) -> TelegramMessageResult:
        data: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
        }
        if parse_mode and parse_mode != "none":
            data["parse_mode"] = parse_mode
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id
        if disable_notification:
            data["disable_notification"] = True

        return await call_telegram_api(credentials, "sendMessage", data)


class SendTelegramPhotoBlock(Block):
    """Send a photo to a Telegram chat."""

    class Input(BlockSchemaInput):
        credentials: TelegramCredentialsInput = TelegramCredentialsField()
        chat_id: int = SchemaField(description="The chat ID to send the photo to")
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
            id="ceb9fd95-fd95-49ff-b57b-278957255716",
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
                "_send_photo_url": lambda *args, **kwargs: TelegramMessageResult(
                    message_id=123
                )
            },
        )

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
                result = await self._send_photo_url(
                    credentials=credentials,
                    chat_id=input_data.chat_id,
                    photo_url=photo_input,
                    caption=input_data.caption,
                    parse_mode=input_data.parse_mode.value,
                    reply_to_message_id=input_data.reply_to_message_id,
                )
            else:
                # For data URIs or workspace:// references, resolve to local
                # file and upload via multipart/form-data
                relative_path = await store_media_file(
                    file=input_data.photo,
                    execution_context=execution_context,
                    return_format="for_local_processing",
                )
                if not execution_context.graph_exec_id:
                    raise ValueError("graph_exec_id is required")
                abs_path = get_exec_file_path(
                    execution_context.graph_exec_id, relative_path
                )
                result = await self._send_photo_file(
                    credentials=credentials,
                    chat_id=input_data.chat_id,
                    file_path=abs_path,
                    caption=input_data.caption,
                    parse_mode=input_data.parse_mode.value,
                    reply_to_message_id=input_data.reply_to_message_id,
                )

            yield "message_id", result.message_id
            yield "status", "Photo sent"
        except Exception as e:
            raise ValueError(f"Failed to send photo: {e}") from e

    async def _send_photo_url(
        self,
        credentials: APIKeyCredentials,
        chat_id: int,
        photo_url: str,
        caption: str,
        parse_mode: str,
        reply_to_message_id: Optional[int],
    ) -> TelegramMessageResult:
        data: dict[str, Any] = {
            "chat_id": chat_id,
            "photo": photo_url,
        }
        if caption:
            data["caption"] = caption
        if parse_mode and parse_mode != "none":
            data["parse_mode"] = parse_mode
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id

        return await call_telegram_api(credentials, "sendPhoto", data)

    async def _send_photo_file(
        self,
        credentials: APIKeyCredentials,
        chat_id: int,
        file_path: str,
        caption: str,
        parse_mode: str,
        reply_to_message_id: Optional[int],
    ) -> TelegramMessageResult:
        path = Path(file_path)
        file_bytes = path.read_bytes()
        data: dict[str, Any] = {"chat_id": str(chat_id)}
        if caption:
            data["caption"] = caption
        if parse_mode and parse_mode != "none":
            data["parse_mode"] = parse_mode
        if reply_to_message_id:
            data["reply_to_message_id"] = str(reply_to_message_id)

        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        return await call_telegram_api_with_file(
            credentials,
            "sendPhoto",
            file_field="photo",
            file_data=file_bytes,
            filename=path.name,
            content_type=mime_type,
            data=data,
        )


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
            id="7a0ef660-1a5b-4951-8642-c13a0c8d6d93",
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
                "_send_voice_url": lambda *args, **kwargs: TelegramMessageResult(
                    message_id=123
                )
            },
        )

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
                result = await self._send_voice_url(
                    credentials=credentials,
                    chat_id=input_data.chat_id,
                    voice_url=voice_input,
                    caption=input_data.caption,
                    duration=input_data.duration,
                    reply_to_message_id=input_data.reply_to_message_id,
                )
            else:
                # For data URIs or workspace:// references, resolve to local
                # file and upload via multipart/form-data
                relative_path = await store_media_file(
                    file=input_data.voice,
                    execution_context=execution_context,
                    return_format="for_local_processing",
                )
                if not execution_context.graph_exec_id:
                    raise ValueError("graph_exec_id is required")
                abs_path = get_exec_file_path(
                    execution_context.graph_exec_id, relative_path
                )
                result = await self._send_voice_file(
                    credentials=credentials,
                    chat_id=input_data.chat_id,
                    file_path=abs_path,
                    caption=input_data.caption,
                    duration=input_data.duration,
                    reply_to_message_id=input_data.reply_to_message_id,
                )

            yield "message_id", result.message_id
            yield "status", "Voice sent"
        except Exception as e:
            raise ValueError(f"Failed to send voice: {e}") from e

    async def _send_voice_url(
        self,
        credentials: APIKeyCredentials,
        chat_id: int,
        voice_url: str,
        caption: str,
        duration: Optional[int],
        reply_to_message_id: Optional[int],
    ) -> TelegramMessageResult:
        data: dict[str, Any] = {
            "chat_id": chat_id,
            "voice": voice_url,
        }
        if caption:
            data["caption"] = caption
        if duration:
            data["duration"] = duration
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id

        return await call_telegram_api(credentials, "sendVoice", data)

    async def _send_voice_file(
        self,
        credentials: APIKeyCredentials,
        chat_id: int,
        file_path: str,
        caption: str,
        duration: Optional[int],
        reply_to_message_id: Optional[int],
    ) -> TelegramMessageResult:
        path = Path(file_path)
        file_bytes = path.read_bytes()
        data: dict[str, Any] = {"chat_id": str(chat_id)}
        if caption:
            data["caption"] = caption
        if duration:
            data["duration"] = str(duration)
        if reply_to_message_id:
            data["reply_to_message_id"] = str(reply_to_message_id)

        mime_type = mimetypes.guess_type(path.name)[0] or "audio/ogg"
        return await call_telegram_api_with_file(
            credentials,
            "sendVoice",
            file_field="voice",
            file_data=file_bytes,
            filename=path.name,
            content_type=mime_type,
            data=data,
        )


class ReplyToTelegramMessageBlock(Block):
    """Reply to a specific Telegram message."""

    class Input(BlockSchemaInput):
        credentials: TelegramCredentialsInput = TelegramCredentialsField()
        chat_id: int = SchemaField(description="The chat ID where the message is")
        reply_to_message_id: int = SchemaField(description="The message ID to reply to")
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
            id="c2b1c976-844f-4b6c-ab21-4973d2ceab15",
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
                "_send_reply": lambda *args, **kwargs: TelegramMessageResult(
                    message_id=123
                )
            },
        )

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
            yield "message_id", result.message_id
            yield "status", "Reply sent"
        except Exception as e:
            raise ValueError(f"Failed to send reply: {e}") from e

    async def _send_reply(
        self,
        credentials: APIKeyCredentials,
        chat_id: int,
        reply_to_message_id: int,
        text: str,
        parse_mode: str,
    ) -> TelegramMessageResult:
        data: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "reply_to_message_id": reply_to_message_id,
        }
        if parse_mode and parse_mode != "none":
            data["parse_mode"] = parse_mode

        return await call_telegram_api(credentials, "sendMessage", data)


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
            id="b600aaf2-6272-40c6-b973-c3984747c5bd",
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
            test_mock={"_download_file": lambda *args, **kwargs: b"test"},
        )

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

            # Convert to data URI and wrap as MediaFileType
            mime_type = "application/octet-stream"
            data_uri = MediaFileType(
                f"data:{mime_type};base64," f"{base64.b64encode(file_content).decode()}"
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
            raise ValueError(f"Failed to download file: {e}") from e

    async def _download_file(
        self,
        credentials: APIKeyCredentials,
        file_id: str,
    ) -> bytes:
        return await download_telegram_file(credentials, file_id)


class DeleteTelegramMessageBlock(Block):
    """Delete a message from a Telegram chat."""

    class Input(BlockSchemaInput):
        credentials: TelegramCredentialsInput = TelegramCredentialsField()
        chat_id: int = SchemaField(description="The chat ID containing the message")
        message_id: int = SchemaField(description="The ID of the message to delete")

    class Output(BlockSchemaOutput):
        status: str = SchemaField(description="Status of the operation")

    def __init__(self):
        super().__init__(
            id="bb4fd91a-883e-4d29-9879-b06c4bb79d30",
            description="Delete a message from a Telegram chat. "
            "Bots can delete their own messages and incoming messages "
            "in private chats at any time.",
            categories={BlockCategory.SOCIAL},
            input_schema=DeleteTelegramMessageBlock.Input,
            output_schema=DeleteTelegramMessageBlock.Output,
            test_input={
                "chat_id": 12345678,
                "message_id": 42,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("status", "Message deleted"),
            ],
            test_mock={"_delete_message": lambda *args, **kwargs: True},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            await self._delete_message(
                credentials=credentials,
                chat_id=input_data.chat_id,
                message_id=input_data.message_id,
            )
            yield "status", "Message deleted"
        except Exception as e:
            raise ValueError(f"Failed to delete message: {e}") from e

    async def _delete_message(
        self,
        credentials: APIKeyCredentials,
        chat_id: int,
        message_id: int,
    ) -> bool:
        await call_telegram_api(
            credentials,
            "deleteMessage",
            {"chat_id": chat_id, "message_id": message_id},
        )
        return True


class EditTelegramMessageBlock(Block):
    """Edit the text of an existing message."""

    class Input(BlockSchemaInput):
        credentials: TelegramCredentialsInput = TelegramCredentialsField()
        chat_id: int = SchemaField(description="The chat ID containing the message")
        message_id: int = SchemaField(description="The ID of the message to edit")
        text: str = SchemaField(
            description="New text for the message (max 4096 characters)"
        )
        parse_mode: ParseMode = SchemaField(
            description="Message formatting mode",
            default=ParseMode.NONE,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        message_id: int = SchemaField(description="The ID of the edited message")
        status: str = SchemaField(description="Status of the operation")

    def __init__(self):
        super().__init__(
            id="c55816a2-37af-4901-ba19-36edcf2acfc1",
            description="Edit the text of an existing message sent by the bot.",
            categories={BlockCategory.SOCIAL},
            input_schema=EditTelegramMessageBlock.Input,
            output_schema=EditTelegramMessageBlock.Output,
            test_input={
                "chat_id": 12345678,
                "message_id": 42,
                "text": "Updated text!",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("message_id", 42),
                ("status", "Message edited"),
            ],
            test_mock={
                "_edit_message": lambda *args, **kwargs: TelegramMessageResult(
                    message_id=42
                )
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            result = await self._edit_message(
                credentials=credentials,
                chat_id=input_data.chat_id,
                message_id=input_data.message_id,
                text=input_data.text,
                parse_mode=input_data.parse_mode.value,
            )
            yield "message_id", result.message_id
            yield "status", "Message edited"
        except Exception as e:
            raise ValueError(f"Failed to edit message: {e}") from e

    async def _edit_message(
        self,
        credentials: APIKeyCredentials,
        chat_id: int,
        message_id: int,
        text: str,
        parse_mode: str,
    ) -> TelegramMessageResult:
        data: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
        }
        if parse_mode and parse_mode != "none":
            data["parse_mode"] = parse_mode

        return await call_telegram_api(credentials, "editMessageText", data)


class SendTelegramAudioBlock(Block):
    """Send an audio file to a Telegram chat, displayed in the music player."""

    class Input(BlockSchemaInput):
        credentials: TelegramCredentialsInput = TelegramCredentialsField()
        chat_id: int = SchemaField(description="The chat ID to send the audio to")
        audio: MediaFileType = SchemaField(
            description="Audio file to send (MP3 or M4A format). "
            "Can be URL, data URI, or workspace:// reference."
        )
        caption: str = SchemaField(
            description="Caption for the audio file",
            default="",
            advanced=True,
        )
        title: str = SchemaField(
            description="Track title",
            default="",
            advanced=True,
        )
        performer: str = SchemaField(
            description="Track performer/artist",
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
            id="9044829a-7915-4ab4-a50f-89922ba3679f",
            description="Send an audio file to a Telegram chat. "
            "The file is displayed in the music player. "
            "For voice messages, use the Send Voice block instead.",
            categories={BlockCategory.SOCIAL},
            input_schema=SendTelegramAudioBlock.Input,
            output_schema=SendTelegramAudioBlock.Output,
            test_input={
                "chat_id": 12345678,
                "audio": "https://example.com/track.mp3",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("message_id", 123),
                ("status", "Audio sent"),
            ],
            test_mock={
                "_send_audio_url": lambda *args, **kwargs: TelegramMessageResult(
                    message_id=123
                )
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        try:
            audio_input = input_data.audio

            if audio_input.startswith(("http://", "https://")):
                result = await self._send_audio_url(
                    credentials=credentials,
                    chat_id=input_data.chat_id,
                    audio_url=audio_input,
                    caption=input_data.caption,
                    title=input_data.title,
                    performer=input_data.performer,
                    duration=input_data.duration,
                    reply_to_message_id=input_data.reply_to_message_id,
                )
            else:
                relative_path = await store_media_file(
                    file=input_data.audio,
                    execution_context=execution_context,
                    return_format="for_local_processing",
                )
                if not execution_context.graph_exec_id:
                    raise ValueError("graph_exec_id is required")
                abs_path = get_exec_file_path(
                    execution_context.graph_exec_id, relative_path
                )
                result = await self._send_audio_file(
                    credentials=credentials,
                    chat_id=input_data.chat_id,
                    file_path=abs_path,
                    caption=input_data.caption,
                    title=input_data.title,
                    performer=input_data.performer,
                    duration=input_data.duration,
                    reply_to_message_id=input_data.reply_to_message_id,
                )

            yield "message_id", result.message_id
            yield "status", "Audio sent"
        except Exception as e:
            raise ValueError(f"Failed to send audio: {e}") from e

    async def _send_audio_url(
        self,
        credentials: APIKeyCredentials,
        chat_id: int,
        audio_url: str,
        caption: str,
        title: str,
        performer: str,
        duration: Optional[int],
        reply_to_message_id: Optional[int],
    ) -> TelegramMessageResult:
        data: dict[str, Any] = {
            "chat_id": chat_id,
            "audio": audio_url,
        }
        if caption:
            data["caption"] = caption
        if title:
            data["title"] = title
        if performer:
            data["performer"] = performer
        if duration:
            data["duration"] = duration
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id

        return await call_telegram_api(credentials, "sendAudio", data)

    async def _send_audio_file(
        self,
        credentials: APIKeyCredentials,
        chat_id: int,
        file_path: str,
        caption: str,
        title: str,
        performer: str,
        duration: Optional[int],
        reply_to_message_id: Optional[int],
    ) -> TelegramMessageResult:
        path = Path(file_path)
        file_bytes = path.read_bytes()
        data: dict[str, Any] = {"chat_id": str(chat_id)}
        if caption:
            data["caption"] = caption
        if title:
            data["title"] = title
        if performer:
            data["performer"] = performer
        if duration:
            data["duration"] = str(duration)
        if reply_to_message_id:
            data["reply_to_message_id"] = str(reply_to_message_id)

        mime_type = mimetypes.guess_type(path.name)[0] or "audio/mpeg"
        return await call_telegram_api_with_file(
            credentials,
            "sendAudio",
            file_field="audio",
            file_data=file_bytes,
            filename=path.name,
            content_type=mime_type,
            data=data,
        )


class SendTelegramDocumentBlock(Block):
    """Send a document to a Telegram chat."""

    class Input(BlockSchemaInput):
        credentials: TelegramCredentialsInput = TelegramCredentialsField()
        chat_id: int = SchemaField(description="The chat ID to send the document to")
        document: MediaFileType = SchemaField(
            description="Document to send (any file type). "
            "Can be URL, data URI, or workspace:// reference."
        )
        filename: str = SchemaField(
            description="Filename shown to the recipient. "
            "If empty, the original filename is used (may be a random ID "
            "for uploaded files).",
            default="",
        )
        caption: str = SchemaField(
            description="Caption for the document",
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
            id="554f4bad-4c99-48ba-9d1c-75840b71c5ad",
            description="Send a document (any file type) to a Telegram chat.",
            categories={BlockCategory.SOCIAL},
            input_schema=SendTelegramDocumentBlock.Input,
            output_schema=SendTelegramDocumentBlock.Output,
            test_input={
                "chat_id": 12345678,
                "document": "https://example.com/file.pdf",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("message_id", 123),
                ("status", "Document sent"),
            ],
            test_mock={
                "_send_document_url": lambda *args, **kwargs: TelegramMessageResult(
                    message_id=123
                )
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        try:
            doc_input = input_data.document

            if doc_input.startswith(("http://", "https://")):
                result = await self._send_document_url(
                    credentials=credentials,
                    chat_id=input_data.chat_id,
                    document_url=doc_input,
                    caption=input_data.caption,
                    parse_mode=input_data.parse_mode.value,
                    reply_to_message_id=input_data.reply_to_message_id,
                )
            else:
                relative_path = await store_media_file(
                    file=input_data.document,
                    execution_context=execution_context,
                    return_format="for_local_processing",
                )
                if not execution_context.graph_exec_id:
                    raise ValueError("graph_exec_id is required")
                abs_path = get_exec_file_path(
                    execution_context.graph_exec_id, relative_path
                )
                result = await self._send_document_file(
                    credentials=credentials,
                    chat_id=input_data.chat_id,
                    file_path=abs_path,
                    display_filename=input_data.filename,
                    caption=input_data.caption,
                    parse_mode=input_data.parse_mode.value,
                    reply_to_message_id=input_data.reply_to_message_id,
                )

            yield "message_id", result.message_id
            yield "status", "Document sent"
        except Exception as e:
            raise ValueError(f"Failed to send document: {e}") from e

    async def _send_document_url(
        self,
        credentials: APIKeyCredentials,
        chat_id: int,
        document_url: str,
        caption: str,
        parse_mode: str,
        reply_to_message_id: Optional[int],
    ) -> TelegramMessageResult:
        data: dict[str, Any] = {
            "chat_id": chat_id,
            "document": document_url,
        }
        if caption:
            data["caption"] = caption
        if parse_mode and parse_mode != "none":
            data["parse_mode"] = parse_mode
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id

        return await call_telegram_api(credentials, "sendDocument", data)

    async def _send_document_file(
        self,
        credentials: APIKeyCredentials,
        chat_id: int,
        file_path: str,
        display_filename: str,
        caption: str,
        parse_mode: str,
        reply_to_message_id: Optional[int],
    ) -> TelegramMessageResult:
        path = Path(file_path)
        file_bytes = path.read_bytes()
        data: dict[str, Any] = {"chat_id": str(chat_id)}
        if caption:
            data["caption"] = caption
        if parse_mode and parse_mode != "none":
            data["parse_mode"] = parse_mode
        if reply_to_message_id:
            data["reply_to_message_id"] = str(reply_to_message_id)

        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return await call_telegram_api_with_file(
            credentials,
            "sendDocument",
            file_field="document",
            file_data=file_bytes,
            filename=display_filename or path.name,
            content_type=mime_type,
            data=data,
        )


class SendTelegramVideoBlock(Block):
    """Send a video to a Telegram chat."""

    class Input(BlockSchemaInput):
        credentials: TelegramCredentialsInput = TelegramCredentialsField()
        chat_id: int = SchemaField(description="The chat ID to send the video to")
        video: MediaFileType = SchemaField(
            description="Video to send (MP4 format). "
            "Can be URL, data URI, or workspace:// reference."
        )
        caption: str = SchemaField(
            description="Caption for the video",
            default="",
            advanced=True,
        )
        parse_mode: ParseMode = SchemaField(
            description="Caption formatting mode",
            default=ParseMode.NONE,
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
            id="cda075af-3f31-47b0-baa9-0050b3bd78bd",
            description="Send a video to a Telegram chat.",
            categories={BlockCategory.SOCIAL},
            input_schema=SendTelegramVideoBlock.Input,
            output_schema=SendTelegramVideoBlock.Output,
            test_input={
                "chat_id": 12345678,
                "video": "https://example.com/video.mp4",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("message_id", 123),
                ("status", "Video sent"),
            ],
            test_mock={
                "_send_video_url": lambda *args, **kwargs: TelegramMessageResult(
                    message_id=123
                )
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        try:
            video_input = input_data.video

            if video_input.startswith(("http://", "https://")):
                result = await self._send_video_url(
                    credentials=credentials,
                    chat_id=input_data.chat_id,
                    video_url=video_input,
                    caption=input_data.caption,
                    parse_mode=input_data.parse_mode.value,
                    duration=input_data.duration,
                    reply_to_message_id=input_data.reply_to_message_id,
                )
            else:
                relative_path = await store_media_file(
                    file=input_data.video,
                    execution_context=execution_context,
                    return_format="for_local_processing",
                )
                if not execution_context.graph_exec_id:
                    raise ValueError("graph_exec_id is required")
                abs_path = get_exec_file_path(
                    execution_context.graph_exec_id, relative_path
                )
                result = await self._send_video_file(
                    credentials=credentials,
                    chat_id=input_data.chat_id,
                    file_path=abs_path,
                    caption=input_data.caption,
                    parse_mode=input_data.parse_mode.value,
                    duration=input_data.duration,
                    reply_to_message_id=input_data.reply_to_message_id,
                )

            yield "message_id", result.message_id
            yield "status", "Video sent"
        except Exception as e:
            raise ValueError(f"Failed to send video: {e}") from e

    async def _send_video_url(
        self,
        credentials: APIKeyCredentials,
        chat_id: int,
        video_url: str,
        caption: str,
        parse_mode: str,
        duration: Optional[int],
        reply_to_message_id: Optional[int],
    ) -> TelegramMessageResult:
        data: dict[str, Any] = {
            "chat_id": chat_id,
            "video": video_url,
        }
        if caption:
            data["caption"] = caption
        if parse_mode and parse_mode != "none":
            data["parse_mode"] = parse_mode
        if duration:
            data["duration"] = duration
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id

        return await call_telegram_api(credentials, "sendVideo", data)

    async def _send_video_file(
        self,
        credentials: APIKeyCredentials,
        chat_id: int,
        file_path: str,
        caption: str,
        parse_mode: str,
        duration: Optional[int],
        reply_to_message_id: Optional[int],
    ) -> TelegramMessageResult:
        path = Path(file_path)
        file_bytes = path.read_bytes()
        data: dict[str, Any] = {"chat_id": str(chat_id)}
        if caption:
            data["caption"] = caption
        if parse_mode and parse_mode != "none":
            data["parse_mode"] = parse_mode
        if duration:
            data["duration"] = str(duration)
        if reply_to_message_id:
            data["reply_to_message_id"] = str(reply_to_message_id)

        mime_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
        return await call_telegram_api_with_file(
            credentials,
            "sendVideo",
            file_field="video",
            file_data=file_bytes,
            filename=path.name,
            content_type=mime_type,
            data=data,
        )
