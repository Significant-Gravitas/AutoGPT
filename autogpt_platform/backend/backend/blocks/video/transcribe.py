"""TranscribeVideoBlock - Transcribe speech from a video file using Replicate."""

from __future__ import annotations

import logging

from replicate.client import Client as ReplicateClient
from replicate.helpers import FileOutput

from backend.blocks.replicate._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    ReplicateCredentials,
    ReplicateCredentialsInput,
)
from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.execution import ExecutionContext
from backend.data.model import CredentialsField, SchemaField
from backend.util.exceptions import BlockExecutionError
from backend.util.file import MediaFileType, store_media_file

logger = logging.getLogger(__name__)


class TranscribeVideoBlock(Block):
    """Transcribe speech from a video file to text via Replicate API."""

    class Input(BlockSchemaInput):
        credentials: ReplicateCredentialsInput = CredentialsField(
            description="Replicate API key for video transcription.",
        )
        video_in: MediaFileType = SchemaField(
            description="Input video file to transcribe (URL, data URI, or local path)",
        )

    class Output(BlockSchemaOutput):
        transcription: str = SchemaField(
            description="Text transcription extracted from the video",
        )

    def __init__(self):
        super().__init__(
            id="fa49dad0-a5fc-441c-ba04-2ac206e392d8",
            description="Transcribe speech from a video file to text",
            categories={BlockCategory.MULTIMEDIA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "video_in": "data:video/mp4;base64,AAAA",
            },
            test_output=[("transcription", "example transcript")],
            test_mock={
                "_transcribe": lambda *args: "example transcript",
                "_store_input_video": lambda *args, **kwargs: "test.mp4",
            },
            test_credentials=TEST_CREDENTIALS,
        )

    async def _store_input_video(
        self, execution_context: ExecutionContext, file: MediaFileType
    ) -> MediaFileType:
        """Store input video locally. Extracted for testability."""
        return await store_media_file(
            file=file,
            execution_context=execution_context,
            return_format="for_external_api",
        )

    async def _transcribe(self, data_uri: str, api_key: str) -> str:
        """Call Replicate API to transcribe the video."""
        client = ReplicateClient(api_token=api_key)

        output = await client.async_run(
            "jd7h/edit-video-by-editing-text:e010b880347314d07e3ce3b21cbd4c57add51fea3474677a6cb1316751c4cb90",
            input={
                "mode": "transcribe",
                "video_in": data_uri,
            },
        )

        # Handle dictionary response format
        if isinstance(output, dict):
            if "transcription" in output:
                return str(output["transcription"])
            if "error" in output:
                raise ValueError(f"API returned error: {output['error']}")

        # Handle list formats
        if isinstance(output, list) and len(output) > 0:
            if isinstance(output[0], FileOutput):
                return output[0].url
            if isinstance(output[0], dict) and "text" in output[0]:
                return " ".join(
                    segment.get("text", "") for segment in output  # type: ignore
                )
            return str(output[0])

        if isinstance(output, FileOutput):
            return output.url

        if isinstance(output, str):
            return output

        raise ValueError(f"Unexpected output format from Replicate API: {output}")

    async def run(
        self,
        input_data: Input,
        *,
        credentials: ReplicateCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        try:
            # Store video and get data URI for API submission
            data_uri = await self._store_input_video(
                execution_context, input_data.video_in
            )

            transcript = await self._transcribe(
                data_uri, credentials.api_key.get_secret_value()
            )
            yield "transcription", transcript

        except BlockExecutionError:
            raise
        except Exception as e:
            raise BlockExecutionError(
                message=f"Failed to transcribe video: {e}",
                block_name=self.name,
                block_id=str(self.id),
            ) from e
