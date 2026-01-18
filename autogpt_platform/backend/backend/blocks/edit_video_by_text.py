from __future__ import annotations

import base64
from typing import Literal

from pydantic import SecretStr
from replicate.client import Client as ReplicateClient
from replicate.helpers import FileOutput

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util.file import get_exec_file_path, store_media_file
from backend.util.type import MediaFileType

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="replicate",
    api_key=SecretStr("mock-replicate-api-key"),
    title="Mock Replicate API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


class EditVideoByTextBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.REPLICATE], Literal["api_key"]
        ] = CredentialsField(
            description="The Replicate integration can be used with "
            "any API key with sufficient permissions for the blocks it is used on.",
        )
        video_in: MediaFileType = SchemaField(
            description="Video file to edit",
        )
        transcription: str = SchemaField(
            description="Desired transcript for the output video",
        )
        split_at: str = SchemaField(
            description="Granularity for transcript matching",
            default="word",
        )

    class Output(BlockSchema):
        video_url: str = SchemaField(
            description="URL of the edited video",
        )
        transcription: str = SchemaField(
            description="Transcription used for editing",
        )
        error: str = SchemaField(
            description="Error message if something fails",
            default="",
        )

    def __init__(self) -> None:
        super().__init__(
            id="98d40049-a1de-465f-bba1-47411298ad1a",
            description="Edits a video by modifying its transcript.",
            categories={BlockCategory.MULTIMEDIA},
            input_schema=EditVideoByTextBlock.Input,
            output_schema=EditVideoByTextBlock.Output,
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
                "edit_video": lambda file_path, transcription, split_at, api_key: "https://replicate.com/output/video.mp4"
            },
            test_credentials=TEST_CREDENTIALS,
        )

    async def edit_video(
        self, file_path: str, transcription: str, split_at: str, api_key: SecretStr
    ) -> str:
        """Use Replicate's API to edit the video."""
        try:
            client = ReplicateClient(api_token=api_key.get_secret_value())

            # Convert file path to file URL
            with open(file_path, "rb") as f:
                file_data = f.read()
                file_b64 = base64.b64encode(file_data).decode()
                file_url = f"data:video/mp4;base64,{file_b64}"

            output = await client.async_run(
                "jd7h/edit-video-by-editing-text:e010b880347314d07e3ce3b21cbd4c57add51fea3474677a6cb1316751c4cb90",
                input={
                    "mode": "edit",
                    "video_in": file_url,
                    "transcription": transcription,
                    "split_at": split_at,
                },
                wait=False,
            )

            # Get video URL from output
            if isinstance(output, dict) and "video" in output:
                video_output = output["video"]
                if isinstance(video_output, FileOutput):
                    return video_output.url
                return str(video_output)
            elif isinstance(output, list) and len(output) > 0:
                video_url = output[0]
                if isinstance(video_url, FileOutput):
                    return video_url.url
                return str(video_url)
            elif isinstance(output, FileOutput):
                return output.url
            elif isinstance(output, str):
                return output

            raise ValueError(f"Unexpected output format from Replicate API: {output}")
        except Exception:
            raise

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        graph_exec_id: str,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        try:
            local_path = await store_media_file(
                graph_exec_id=graph_exec_id,
                file=input_data.video_in,
                user_id=user_id,
                return_content=False,
            )
            abs_path = get_exec_file_path(graph_exec_id, local_path)

            video_url = await self.edit_video(
                abs_path,
                input_data.transcription,
                input_data.split_at,
                credentials.api_key,
            )

            yield "video_url", video_url
            yield "transcription", input_data.transcription
        except Exception as e:
            error_msg = f"Failed to edit video: {str(e)}"
            yield "error", error_msg
