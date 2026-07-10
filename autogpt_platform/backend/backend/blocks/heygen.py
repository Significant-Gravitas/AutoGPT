import asyncio
from typing import Literal, Optional

from pydantic import SecretStr

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.execution import ExecutionContext
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util.file import store_media_file
from backend.util.request import Requests
from backend.util.type import MediaFileType

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="heygen",
    api_key=SecretStr("mock-heygen-api-key"),
    title="Mock HeyGen API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.type,
}


class CreateHeyGenAvatarVideoBlock(Block):
    """
    Creates an avatar video via HeyGen's v3 API (POST /v3/videos, type=avatar)
    and polls until it's ready. The v2 endpoint this feature was originally
    requested against is on a deprecation path (support ends 2026-10-31), so
    this targets the current v3 structured-avatar equivalent instead.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.HEYGEN], Literal["api_key"]
        ] = CredentialsField(
            description="API key from https://app.heygen.com/home?from=&nav=API",
        )
        avatar_id: str = SchemaField(
            description="The HeyGen avatar ID to use",
            placeholder="Angela-inTshirt-20220820",
        )
        script: str = SchemaField(
            description="The script the avatar will speak",
            placeholder="Welcome to AutoGPT",
        )
        voice_id: Optional[str] = SchemaField(
            description="The HeyGen voice ID to use. If omitted, falls back "
            "to the avatar's default voice.",
            default=None,
        )
        title: Optional[str] = SchemaField(
            description="Optional title for the video", default=None
        )
        resolution: Literal["4k", "1080p", "720p"] = SchemaField(
            description="Output video resolution", default="1080p"
        )
        aspect_ratio: Literal["auto", "16:9", "9:16", "4:5", "5:4", "1:1"] = (
            SchemaField(description="Output video aspect ratio", default="16:9")
        )
        output_format: Literal["mp4", "webm"] = SchemaField(
            description="Output video file format", default="mp4"
        )
        max_polling_attempts: int = SchemaField(
            description="Maximum number of polling attempts", default=30, ge=5
        )
        polling_interval: int = SchemaField(
            description="Interval between polling attempts in seconds", default=10, ge=5
        )

    class Output(BlockSchemaOutput):
        video_url: str = SchemaField(description="The URL of the created video")

    def __init__(self):
        super().__init__(
            id="fa1f093e-e425-4a30-8bad-b00f758a33b8",
            description="This block integrates with HeyGen to create avatar videos and retrieve their URLs.",
            categories={BlockCategory.AI, BlockCategory.MULTIMEDIA},
            input_schema=CreateHeyGenAvatarVideoBlock.Input,
            output_schema=CreateHeyGenAvatarVideoBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "avatar_id": "Angela-inTshirt-20220820",
                "script": "Welcome to AutoGPT",
                "voice_id": "1bd001e7e50f421d891986aad5158bc8",
                "resolution": "1080p",
                "aspect_ratio": "16:9",
                "output_format": "mp4",
                "max_polling_attempts": 5,
                "polling_interval": 5,
            },
            test_output=[
                (
                    "video_url",
                    lambda x: x.startswith(("workspace://", "data:")),
                ),
            ],
            test_mock={
                "create_video": lambda *args, **kwargs: {
                    "data": {"video_id": "abcd1234-5678-efgh-ijkl-mnopqrstuvwx"}
                },
                # Use data URI to avoid HTTP requests during tests
                "get_video_status": lambda *args, **kwargs: {
                    "status": "completed",
                    "video_url": "data:video/mp4;base64,AAAA",
                },
            },
            test_credentials=TEST_CREDENTIALS,
        )

    async def create_video(self, api_key: SecretStr, payload: dict) -> dict:
        url = "https://api.heygen.com/v3/videos"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": api_key.get_secret_value(),
        }
        response = await Requests().post(url, json=payload, headers=headers)
        return response.json()

    async def get_video_status(self, api_key: SecretStr, video_id: str) -> dict:
        url = f"https://api.heygen.com/v3/videos/{video_id}"
        headers = {
            "accept": "application/json",
            "x-api-key": api_key.get_secret_value(),
        }
        response = await Requests().get(url, headers=headers)
        return response.json()

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        payload = {
            "type": "avatar",
            "avatar_id": input_data.avatar_id,
            "script": input_data.script,
            "resolution": input_data.resolution,
            "aspect_ratio": input_data.aspect_ratio,
            "output_format": input_data.output_format,
        }
        if input_data.voice_id:
            payload["voice_id"] = input_data.voice_id
        if input_data.title:
            payload["title"] = input_data.title

        response = await self.create_video(credentials.api_key, payload)
        video_id = response["data"]["video_id"]

        for _ in range(input_data.max_polling_attempts):
            status_response = await self.get_video_status(credentials.api_key, video_id)
            if status_response["status"] == "completed":
                stored_url = await store_media_file(
                    file=MediaFileType(status_response["video_url"]),
                    execution_context=execution_context,
                    return_format="for_block_output",
                )
                yield "video_url", stored_url
                return
            elif status_response["status"] == "failed":
                raise RuntimeError(
                    f"Video creation failed: "
                    f"{status_response.get('failure_message', 'Unknown error')}"
                )

            await asyncio.sleep(input_data.polling_interval)

        raise TimeoutError("Video creation timed out")
