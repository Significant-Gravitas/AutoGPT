import asyncio
import logging
from typing import Any

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.wavespeed._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    WaveSpeedCredentials,
    WaveSpeedCredentialsField,
    WaveSpeedCredentialsInput,
)
from backend.data.execution import ExecutionContext
from backend.data.model import SchemaField
from backend.util.file import store_media_file
from backend.util.request import ClientResponseError, Requests
from backend.util.type import MediaFileType

logger = logging.getLogger(__name__)

WAVESPEED_API_BASE = "https://api.wavespeed.ai/api/v3"


class AIMediaGeneratorBlock(Block):
    """
    Block for running any image or video model on the WaveSpeed catalog.

    This block allows you to:
    - Run any model from the live wavespeed.ai catalog (Seedream, Seedance,
      FLUX, Wan, ...)
    - Pass a text prompt plus arbitrary extra model inputs
    - Get back the generated media URL(s)
    """

    class Input(BlockSchemaInput):
        credentials: WaveSpeedCredentialsInput = WaveSpeedCredentialsField()
        model: str = SchemaField(
            description="The WaveSpeed model ID (format: 'owner/model-name'). "
            "See https://wavespeed.ai for the full catalog.",
            default="bytedance/seedream-v5.0-pro",
            placeholder="bytedance/seedream-v5.0-pro",
            advanced=False,
        )
        prompt: str = SchemaField(
            description="Text prompt describing the image or video to generate.",
            placeholder="A serene mountain lake at sunrise, ultra detailed.",
            advanced=False,
        )
        extra_inputs: dict[str, Any] = SchemaField(
            default={},
            description=(
                "Additional model-specific inputs to include in the request "
                "body, e.g. size, seed, or an image URL for image-to-image / "
                "image-to-video models. Check the model's page on wavespeed.ai "
                "for its input schema."
            ),
            placeholder='{"size": "2048*2048", "seed": -1}',
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        media_url: str = SchemaField(
            description="The URL of the (first) generated image or video."
        )
        media_urls: list[str] = SchemaField(
            description="URLs of all generated outputs."
        )
        error: str = SchemaField(description="Error message if generation failed.")

    def __init__(self):
        super().__init__(
            id="b9030fff-6434-48b9-9618-9e65044185fd",
            description="Generate images and videos using WaveSpeed AI models.",
            categories={BlockCategory.AI},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "model": "bytedance/seedream-v5.0-pro",
                "prompt": "A serene mountain lake at sunrise.",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                # Output will be a workspace ref or data URI depending on context
                ("media_url", lambda x: x.startswith(("workspace://", "data:"))),
                (
                    "media_urls",
                    lambda x: all(
                        url.startswith(("workspace://", "data:")) for url in x
                    ),
                ),
            ],
            test_mock={
                # Use data URIs to avoid HTTP requests during tests
                "generate_media": lambda *args, **kwargs: [
                    "data:image/png;base64,iVBORw0KGgo="
                ]
            },
        )

    def _get_headers(self, api_key: str) -> dict[str, str]:
        """Get headers for WaveSpeed API requests."""
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def generate_media(
        self, input_data: Input, credentials: WaveSpeedCredentials
    ) -> list[str]:
        """Run the specified WaveSpeed model and return the output URLs."""
        api_key = credentials.api_key.get_secret_value()
        headers = self._get_headers(api_key)

        # Submit the generation request
        submit_url = f"{WAVESPEED_API_BASE}/{input_data.model}"
        submit_body: dict[str, Any] = {
            **input_data.extra_inputs,
            "prompt": input_data.prompt,
        }

        try:
            submit_response = await Requests().post(
                submit_url, headers=headers, json=submit_body
            )
            submit_data = submit_response.json()
        except ClientResponseError as e:
            logger.error(f"WaveSpeed API request failed: {str(e)}")
            raise RuntimeError(f"Failed to submit request: {str(e)}")

        prediction_id = (submit_data.get("data") or {}).get("id")
        if not prediction_id or not isinstance(prediction_id, str):
            raise ValueError(
                f"Missing prediction ID in submission response: "
                f"{submit_data.get('message') or submit_data}"
            )

        # Poll for the result until a terminal status is reached
        result_url = f"{WAVESPEED_API_BASE}/predictions/{prediction_id}/result"
        max_attempts = 120
        poll_interval = 2

        for _attempt in range(max_attempts):
            try:
                result_response = await Requests().get(result_url, headers=headers)
                result_data = result_response.json()
            except ClientResponseError as e:
                logger.error(f"Failed to get prediction result: {str(e)}")
                raise RuntimeError(f"Failed to get prediction result: {str(e)}")

            prediction = result_data.get("data") or {}
            status = prediction.get("status")

            if status == "completed":
                outputs = [
                    output
                    for output in prediction.get("outputs") or []
                    if isinstance(output, str)
                ]
                if not outputs:
                    raise ValueError("No valid output URLs in response")
                return outputs
            elif status in ("failed", "cancelled", "timeout"):
                error_msg = prediction.get("error") or "No error details provided"
                raise RuntimeError(f"Generation {status}: {error_msg}")
            else:
                logger.debug(f"[WaveSpeed Generation] Status: {status}")

            await asyncio.sleep(poll_interval)

        raise RuntimeError("Maximum polling attempts reached")

    async def run(
        self,
        input_data: Input,
        *,
        credentials: WaveSpeedCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        try:
            output_urls = await self.generate_media(input_data, credentials)
            # Store the generated media to the user's workspace for persistence
            stored_urls = [
                await store_media_file(
                    file=MediaFileType(url),
                    execution_context=execution_context,
                    return_format="for_block_output",
                )
                for url in output_urls
            ]
            yield "media_url", stored_urls[0]
            yield "media_urls", stored_urls
        except Exception as e:
            error_message = str(e)
            yield "error", error_message
