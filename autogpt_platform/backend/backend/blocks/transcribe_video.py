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


class TranscribeVideoBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.REPLICATE], Literal["api_key"]
        ] = CredentialsField(
            description="The Replicate integration can be used with "
            "any API key with sufficient permissions for the blocks it is used on.",
        )
        video_in: MediaFileType = SchemaField(
            description="Video file to transcribe",
        )

    class Output(BlockSchema):
        transcription: str = SchemaField(
            description="Text transcription of the video",
        )
        error: str = SchemaField(
            description="Error message if something fails",
            default="",
        )

    def __init__(self) -> None:
        super().__init__(
            id="fa49dad0-a5fc-441c-ba04-2ac206e392d8",
            description="Transcribes speech from a video file.",
            categories={BlockCategory.MULTIMEDIA},
            input_schema=TranscribeVideoBlock.Input,
            output_schema=TranscribeVideoBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "video_in": "data:video/mp4;base64,AAAA",
            },
            test_output=("transcription", "example transcript"),
            test_mock={"transcribe": lambda file_path, api_key: "example transcript"},
            test_credentials=TEST_CREDENTIALS,
        )

    async def transcribe(self, file_path: str, api_key: SecretStr) -> str:
        """Use Replicate's API to transcribe the video."""
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
                    "mode": "transcribe",
                    "video_in": file_url,
                },
                wait=False,
            )

            # Handle dictionary response format
            if isinstance(output, dict):
                if "transcription" in output:
                    return output["transcription"]
                elif "error" in output:
                    raise ValueError(f"API returned error: {output['error']}")
            # Handle list/string formats as before
            elif isinstance(output, list) and len(output) > 0:
                if isinstance(output[0], FileOutput):
                    return output[0].url
                return output[0]
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

            transcript = await self.transcribe(abs_path, credentials.api_key)
            yield "transcription", transcript
        except Exception as e:
            error_msg = f"Failed to transcribe video: {str(e)}"
            yield "error", error_msg
