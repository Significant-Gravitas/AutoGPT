import asyncio
import logging
from enum import Enum
from typing import Any

from backend.blocks.fal._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    FalCredentials,
    FalCredentialsField,
    FalCredentialsInput,
)
from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField
from backend.util.request import ClientResponseError, Requests

logger = logging.getLogger(__name__)


class FalModel(str, Enum):
    MOCHI = "fal-ai/mochi-v1"
    LUMA = "fal-ai/luma-dream-machine"
    VEO3 = "fal-ai/veo3"


class AIVideoGeneratorBlock(Block):
    class Input(BlockSchemaInput):
        prompt: str = SchemaField(
            description="Description of the video to generate.",
            placeholder="A dog running in a field.",
        )
        model: FalModel = SchemaField(
            title="FAL Model",
            default=FalModel.MOCHI,
            description="The FAL model to use for video generation.",
        )
        credentials: FalCredentialsInput = FalCredentialsField()

    class Output(BlockSchemaOutput):
        video_url: str = SchemaField(description="The URL of the generated video.")
        error: str = SchemaField(
            description="Error message if video generation failed."
        )
        logs: list[str] = SchemaField(
            description="Generation progress logs.",
        )

    def __init__(self):
        super().__init__(
            id="530cf046-2ce0-4854-ae2c-659db17c7a46",
            description="Generate videos using FAL AI models.",
            categories={BlockCategory.AI},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "prompt": "A dog running in a field.",
                "model": FalModel.MOCHI,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("video_url", "https://fal.media/files/example/video.mp4")],
            test_mock={
                "generate_video": lambda *args, **kwargs: "https://fal.media/files/example/video.mp4"
            },
        )

    def _get_headers(self, api_key: str) -> dict[str, str]:
        """Get headers for FAL API Requests."""
        return {
            "Authorization": f"Key {api_key}",
            "Content-Type": "application/json",
        }

    async def _submit_request(
        self, url: str, headers: dict[str, str], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Submit a request to the FAL API."""
        try:
            response = await Requests().post(url, headers=headers, json=data)
            return response.json()
        except ClientResponseError as e:
            logger.error(f"FAL API request failed: {str(e)}")
            raise RuntimeError(f"Failed to submit request: {str(e)}")

    async def _poll_status(
        self, status_url: str, headers: dict[str, str]
    ) -> dict[str, Any]:
        """Poll the status endpoint until completion or failure."""
        try:
            response = await Requests().get(status_url, headers=headers)
            return response.json()
        except ClientResponseError as e:
            logger.error(f"Failed to get status: {str(e)}")
            raise RuntimeError(f"Failed to get status: {str(e)}")

    async def generate_video(
        self, input_data: Input, credentials: FalCredentials
    ) -> str:
        """Generate video using the specified FAL model."""
        base_url = "https://queue.fal.run"
        api_key = credentials.api_key.get_secret_value()
        headers = self._get_headers(api_key)

        # Submit generation request
        submit_url = f"{base_url}/{input_data.model.value}"
        submit_data = {"prompt": input_data.prompt}
        if input_data.model == FalModel.VEO3:
            submit_data["generate_audio"] = True  # type: ignore

        seen_logs = set()

        try:
            # Submit request to queue
            submit_response = await Requests().post(
                submit_url, headers=headers, json=submit_data
            )
            request_data = submit_response.json()

            # Get request_id and urls from initial response
            request_id = request_data.get("request_id")
            status_url = request_data.get("status_url")
            result_url = request_data.get("response_url")

            if not all([request_id, status_url, result_url]):
                raise ValueError("Missing required data in submission response")

            # Ensure status_url is a string
            if not isinstance(status_url, str):
                raise ValueError("Invalid status URL format")

            # Ensure result_url is a string
            if not isinstance(result_url, str):
                raise ValueError("Invalid result URL format")

            # Poll for status with exponential backoff
            max_attempts = 30
            attempt = 0
            base_wait_time = 5

            while attempt < max_attempts:
                status_response = await Requests().get(
                    f"{status_url}?logs=1", headers=headers
                )
                status_data = status_response.json()

                # Process new logs only
                logs = status_data.get("logs", [])
                if logs and isinstance(logs, list):
                    for log in logs:
                        if isinstance(log, dict):
                            # Create a unique key for this log entry
                            log_key = (
                                f"{log.get('timestamp', '')}-{log.get('message', '')}"
                            )
                            if log_key not in seen_logs:
                                seen_logs.add(log_key)
                                message = log.get("message", "")
                                if message:
                                    logger.debug(
                                        f"[FAL Generation] [{log.get('level', 'INFO')}] [{log.get('source', '')}] [{log.get('timestamp', '')}] {message}"
                                    )

                status = status_data.get("status")
                if status == "COMPLETED":
                    # Get the final result
                    result_response = await Requests().get(result_url, headers=headers)
                    result_data = result_response.json()

                    if "video" not in result_data or not isinstance(
                        result_data["video"], dict
                    ):
                        raise ValueError("Invalid response format - missing video data")

                    video_url = result_data["video"].get("url")
                    if not video_url or not isinstance(video_url, str):
                        raise ValueError("No valid video URL in response")

                    return video_url

                elif status == "FAILED":
                    error_msg = status_data.get("error", "No error details provided")
                    raise RuntimeError(f"Video generation failed: {error_msg}")
                elif status == "IN_QUEUE":
                    position = status_data.get("queue_position", "unknown")
                    logger.debug(
                        f"[FAL Generation] Status: In queue, position: {position}"
                    )
                elif status == "IN_PROGRESS":
                    logger.debug(
                        "[FAL Generation] Status: Request is being processed..."
                    )
                else:
                    logger.info(f"[FAL Generation] Status: Unknown status: {status}")

                wait_time = min(base_wait_time * (2**attempt), 60)  # Cap at 60 seconds
                await asyncio.sleep(wait_time)
                attempt += 1

            raise RuntimeError("Maximum polling attempts reached")

        except ClientResponseError as e:
            raise RuntimeError(f"API request failed: {str(e)}")

    async def run(
        self, input_data: Input, *, credentials: FalCredentials, **kwargs
    ) -> BlockOutput:
        try:
            video_url = await self.generate_video(input_data, credentials)
            yield "video_url", video_url
        except Exception as e:
            error_message = str(e)
            yield "error", error_message
