import asyncio
import logging
from typing import Any, Literal

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

logger = logging.getLogger(__name__)

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
    "title": TEST_CREDENTIALS.title,
}


class HeyGenCreateVideoBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.HEYGEN], Literal["api_key"]
        ] = CredentialsField(
            description="The HeyGen integration can be used with "
            "any API key with sufficient permissions for the blocks it is used on.",
        )
        text: str = SchemaField(
            description="The text script for the avatar to speak",
            placeholder="e.g., 'Hello, welcome to our platform!'",
            title="Text Script",
        )
        avatar_id: str = SchemaField(
            description="The HeyGen avatar ID to use for the video",
            default="Daisy-inskirt-20220818",
            title="Avatar ID",
        )
        voice_id: str = SchemaField(
            description="The voice ID to use for the avatar",
            default="1bd001e7e50f421d891986aad5571571",
            title="Voice ID",
        )
        max_polling_attempts: int = SchemaField(
            description="Maximum number of polling attempts",
            default=30,
            ge=5,
            title="Max Polling Attempts",
            advanced=True,
        )
        polling_interval: int = SchemaField(
            description="Interval between polling attempts in seconds",
            default=10,
            ge=5,
            title="Polling Interval",
            advanced=True,
        )
        test: bool = SchemaField(
            description="Enable test mode to generate a free video with watermark",
            default=False,
            title="Test Mode",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        video_url: str = SchemaField(
            description="The URL of the generated avatar video"
        )
        error: str = SchemaField(description="Error message if video generation failed")

    def __init__(self):
        super().__init__(
            id="5b315ef2-f755-4637-a1ba-503924232737",
            description="This block uses HeyGen to create avatar videos "
            "from a text script.",
            categories={BlockCategory.AI, BlockCategory.MULTIMEDIA},
            input_schema=HeyGenCreateVideoBlock.Input,
            output_schema=HeyGenCreateVideoBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "text": "Hello, welcome to our platform!",
                "avatar_id": "Daisy-inskirt-20220818",
                "voice_id": "1bd001e7e50f421d891986aad5571571",
                "max_polling_attempts": 5,
                "polling_interval": 5,
                "test": False,
            },
            test_output=[
                (
                    "video_url",
                    "https://files.heygen.ai/video/test-video.mp4",
                ),
            ],
            test_mock={
                "create_video": lambda *args, **kwargs: {
                    "video_id": "test-video-id-1234",
                },
                "get_video_status": lambda *args, **kwargs: {
                    "status": "completed",
                    "video_url": "https://files.heygen.ai/video/test-video.mp4",
                },
            },
            test_credentials=TEST_CREDENTIALS,
        )

    async def create_video(self, api_key: SecretStr, payload: dict) -> dict:
        url = "https://api.heygen.com/v2/video/generate"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Api-Key": api_key.get_secret_value(),
        }
        response = await Requests().post(url, json=payload, headers=headers)
        result = response.json()
        if result.get("error"):
            raise RuntimeError(f"HeyGen API error: {result['error']}")
        return result.get("data", {})

    async def get_video_status(self, api_key: SecretStr, video_id: str) -> dict:
        url = f"https://api.heygen.com/v1/video_status.get?video_id={video_id}"
        headers = {
            "Accept": "application/json",
            "X-Api-Key": api_key.get_secret_value(),
        }
        response = await Requests().get(url, headers=headers)
        result = response.json()
        if result.get("error"):
            raise RuntimeError(f"HeyGen API error: {result['error']}")
        return result.get("data", {})

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        payload: dict[str, Any] = {
            "video_inputs": [
                {
                    "character": {
                        "type": "avatar",
                        "avatar_id": input_data.avatar_id,
                        "avatar_style": "normal",
                    },
                    "voice": {
                        "type": "text",
                        "input_text": input_data.text,
                        "voice_id": input_data.voice_id,
                    },
                }
            ],
        }
        if input_data.test:
            payload["test"] = True

        try:
            response = await self.create_video(credentials.api_key, payload)
            video_id = response.get("video_id")
            if not video_id:
                raise RuntimeError("HeyGen API did not return a video_id")

            for attempt in range(input_data.max_polling_attempts):
                status_response = await self.get_video_status(
                    credentials.api_key, video_id
                )
                status = status_response.get("status")
                logger.debug(
                    f"Polling HeyGen video {video_id}: status={status} "
                    f"(attempt {attempt + 1}/{input_data.max_polling_attempts})"
                )

                if status == "completed":
                    video_url = status_response.get("video_url")
                    if not video_url:
                        raise RuntimeError(
                            "HeyGen API returned completed status without video_url"
                        )
                    stored_url = await store_media_file(
                        file=MediaFileType(video_url),
                        execution_context=execution_context,
                        return_format="for_block_output",
                    )
                    yield "video_url", stored_url
                    return
                elif status == "failed":
                    error_msg = status_response.get("error", "Unknown error")
                    raise RuntimeError(f"Video generation failed: {error_msg}")

                await asyncio.sleep(input_data.polling_interval)

            raise TimeoutError("Video generation timed out")
        except Exception as e:
            yield "error", str(e)
