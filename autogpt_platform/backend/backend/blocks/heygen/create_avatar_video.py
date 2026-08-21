import asyncio
import logging
import uuid
from typing import Any

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.heygen._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    HeyGenCredentialsInput,
    HeyGenCredentialsField,
)
from backend.data.model import SchemaField
from backend.util.request import ClientResponseError, Requests

logger = logging.getLogger(__name__)

class CreateAvatarVideoBlock(Block):
    class Input(BlockSchemaInput):
        avatar_id: str = SchemaField(
            description="The ID of the avatar to use.",
            placeholder="Daisy-Endless-casual-20220816",
        )
        voice_id: str = SchemaField(
            description="The ID of the voice to use.",
            placeholder="1bd001e7e50f421d891986aad5158bc8",
        )
        input_text: str = SchemaField(
            description="The text for the avatar to speak.",
            placeholder="Welcome to the HeyGen API!",
        )
        credentials: HeyGenCredentialsInput = HeyGenCredentialsField()

    class Output(BlockSchemaOutput):
        video_id: str = SchemaField(description="The ID of the generated video.")
        error: str = SchemaField(description="Error message if video generation failed.")

    def __init__(self):
        super().__init__(
            id="8354c0cf-3f2d-42bc-9d32-eb52c676d5dc",
            description="Create an Avatar Video using HeyGen.",
            categories={BlockCategory.AI},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "avatar_id": "Daisy-Endless-casual-20220816",
                "voice_id": "1bd001e7e50f421d891986aad5158bc8",
                "input_text": "Welcome to the HeyGen API!",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("video_id", "mock-video-id"),
            ],
            test_mock={
                "generate_video": lambda *args, **kwargs: {"data": {"video_id": "mock-video-id"}}
            },
        )

    def _get_headers(self, api_key: str) -> dict[str, str]:
        return {
            "X-Api-Key": api_key,
            "Content-Type": "application/json",
        }

    async def generate_video(self, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        url = "https://api.heygen.com/v2/video/generate"
        response = await Requests().post(url, headers=headers, json=payload)
        return response.json()

    async def run(
        self, input_data: Input, **kwargs
    ) -> BlockOutput:
        headers = self._get_headers(input_data.credentials.api_key.get_secret_value())
        payload = {
            "video_inputs": [
                {
                    "character": {
                        "type": "avatar",
                        "avatar_id": input_data.avatar_id,
                        "avatar_style": "normal"
                    },
                    "voice": {
                        "type": "text",
                        "input_text": input_data.input_text,
                        "voice_id": input_data.voice_id
                    }
                }
            ],
            "dimension": {
                "width": 1280,
                "height": 720
            }
        }
        try:
            result = await self.generate_video(headers, payload)
            if "error" in result and result["error"]:
                return "error", str(result["error"])
            if "data" in result and "video_id" in result["data"]:
                return "video_id", result["data"]["video_id"]
            return "error", "Failed to get video_id from response."
        except Exception as e:
            logger.error(f"HeyGen video generation failed: {str(e)}")
            return "error", str(e)
