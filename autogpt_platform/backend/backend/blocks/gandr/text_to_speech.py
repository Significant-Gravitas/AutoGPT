"""Gandr text to speech block."""

import base64
from typing import Literal

from pydantic import SecretStr

from backend.data.execution import ExecutionContext
from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    MediaFileType,
    Requests,
    SchemaField,
)
from backend.util.exceptions import BlockExecutionError
from backend.util.file import store_media_file

from ._config import gandr

GANDR_TTS_URL = "https://tts.gandr.ai/v1/audio/speech"
MAX_INPUT_CHARS = 2000

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="gandr",
    api_key=SecretStr("mock-gandr-api-key"),
    title="Mock Gandr API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


class GandrTextToSpeechBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = gandr.credentials_field(
            description="Gandr API key. Get one at https://gandr.ai. "
            "The free tier is 50,000 tokens.",
        )
        text: str = SchemaField(
            description="The text to convert to speech. "
            "Up to 2000 characters per request.",
            placeholder="Enter the text you want to convert to speech",
        )
        voice: Literal[
            "gandr-mia",
            "gandr-ava",
            "gandr-jenny",
            "gandr-dane",
            "gandr-leo",
            "gandr-lewis",
        ] = SchemaField(
            description="The Gandr voice to use",
            default="gandr-mia",
        )

    class Output(BlockSchemaOutput):
        audio_file: MediaFileType = SchemaField(
            description="Generated MP3 audio (path or data URI)"
        )

    def __init__(self):
        super().__init__(
            id="80c6ebc9-302b-4614-ad2b-08813907af15",
            description="Converts text to speech using the Gandr API",
            categories={BlockCategory.AI, BlockCategory.TEXT, BlockCategory.MULTIMEDIA},
            input_schema=GandrTextToSpeechBlock.Input,
            output_schema=GandrTextToSpeechBlock.Output,
            test_input={
                "text": "This is a test of the text to speech API.",
                "voice": "gandr-mia",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("audio_file", str)],
            test_mock={
                "call_gandr_tts_api": lambda *args, **kwargs: b"mock mp3 bytes",
                "_store_output_audio": lambda *args, **kwargs: (
                    "data:audio/mpeg;base64,bW9jayBtcDMgYnl0ZXM="
                ),
            },
        )

    @staticmethod
    async def call_gandr_tts_api(api_key: SecretStr, text: str, voice: str) -> bytes:
        headers = {
            "Authorization": f"Bearer {api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "tts-1",
            "input": text,
            "voice": voice,
            "response_format": "mp3",
        }
        response = await Requests().post(GANDR_TTS_URL, headers=headers, json=data)
        return response.content

    @staticmethod
    async def _store_output_audio(
        execution_context: ExecutionContext, file: MediaFileType
    ) -> MediaFileType:
        return await store_media_file(
            file=file,
            execution_context=execution_context,
            return_format="for_block_output",
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        if len(input_data.text) > MAX_INPUT_CHARS:
            raise BlockExecutionError(
                message=(
                    f"Text is {len(input_data.text)} characters. The Gandr API "
                    f"accepts up to {MAX_INPUT_CHARS} characters per request. "
                    "Split the text and run the block once per chunk."
                ),
                block_name=self.name,
                block_id=str(self.id),
            )

        audio_bytes = await self.call_gandr_tts_api(
            credentials.api_key,
            input_data.text,
            input_data.voice,
        )
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        audio_file = await self._store_output_audio(
            execution_context,
            MediaFileType(f"data:audio/mpeg;base64,{audio_b64}"),
        )
        yield "audio_file", audio_file
