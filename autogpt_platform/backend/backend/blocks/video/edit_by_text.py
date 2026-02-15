"""EditVideoByTextBlock - Edit a video by modifying its transcript via Replicate."""

from __future__ import annotations

import logging
from typing import Literal

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


class EditVideoByTextBlock(Block):
    """Edit a video by modifying its transcript, cutting segments via Replicate API."""

    class Input(BlockSchemaInput):
        credentials: ReplicateCredentialsInput = CredentialsField(
            description="Replicate API key for video editing.",
        )
        video_in: MediaFileType = SchemaField(
            description="Input video file to edit (URL, data URI, or local path)",
        )
        transcription: str = SchemaField(
            description="Desired transcript for the output video",
        )
        split_at: Literal["word", "character"] = SchemaField(
            description="Granularity for transcript matching",
            default="word",
        )

    class Output(BlockSchemaOutput):
        video_url: str = SchemaField(
            description="URL of the edited video",
        )
        transcription: str = SchemaField(
            description="Transcription used for editing",
        )

    def __init__(self):
        super().__init__(
            id="98d40049-a1de-465f-bba1-47411298ad1a",
            description="Edit a video by modifying its transcript",
            categories={BlockCategory.MULTIMEDIA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "video_in": "data:video/mp4;base64,AAAA",
                "transcription": "edited transcript",
            },
            test_output=[
                ("video_url", "https://replicate.com/output/video.mp4"),
                ("transcription", "edited transcript"),
            ],
            test_mock={
                "_edit_video": lambda *args: "https://replicate.com/output/video.mp4",
                "_store_input_video": lambda *args, **kwargs: "data:video/mp4;base64,AAAA",
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

    async def _edit_video(
        self, data_uri: str, transcription: str, split_at: str, api_key: str
    ) -> str:
        """Call Replicate API to edit the video based on the transcript."""
        client = ReplicateClient(api_token=api_key)

        output = await client.async_run(
            "jd7h/edit-video-by-editing-text:e010b880347314d07e3ce3b21cbd4c57add51fea3474677a6cb1316751c4cb90",
            input={
                "mode": "edit",
                "video_in": data_uri,
                "transcription": transcription,
                "split_at": split_at,
            },
        )

        # Get video URL from output
        if isinstance(output, dict) and "video" in output:
            video_output = output["video"]
            if isinstance(video_output, FileOutput):
                return video_output.url
            return str(video_output)

        if isinstance(output, list) and len(output) > 0:
            video_url = output[0]
            if isinstance(video_url, FileOutput):
                return video_url.url
            return str(video_url)

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

            video_url = await self._edit_video(
                data_uri,
                input_data.transcription,
                input_data.split_at,
                credentials.api_key.get_secret_value(),
            )

            yield "video_url", video_url
            yield "transcription", input_data.transcription

        except BlockExecutionError:
            raise
        except Exception as e:
            raise BlockExecutionError(
                message=f"Failed to edit video: {e}",
                block_name=self.name,
                block_id=str(self.id),
            ) from e
