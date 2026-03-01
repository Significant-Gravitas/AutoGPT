"""
ModelsLab AI Image Generator Block for AutoGPT Platform.

Generates images using the ModelsLab API (https://modelslab.com).
Supports Flux, SDXL, and 100+ community models via text-to-image.
"""
import asyncio
import logging
from enum import Enum

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.modelslab._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    ModelsLabCredentials,
    ModelsLabCredentialsField,
    ModelsLabCredentialsInput,
)
from backend.data.execution import ExecutionContext
from backend.data.model import SchemaField
from backend.util.file import store_media_file
from backend.util.request import ClientResponseError, Requests
from backend.util.type import MediaFileType

logger = logging.getLogger(__name__)

MODELSLAB_IMAGE_URL = "https://modelslab.com/api/v6/images/text2img"
MODELSLAB_FETCH_URL = "https://modelslab.com/api/v6/images/fetch"
POLL_INTERVAL = 5
POLL_TIMEOUT = 300


class ModelsLabImageModel(str, Enum):
    FLUX = "flux"
    FLUX_PRO = "fluxpro"
    SDXL = "sdxl"
    SD_3_5 = "sd3.5"
    REALISTIC_VISION = "realistic-vision-v6"
    JUGGERNAUT_XL = "juggernautxl-v10"


class ModelsLabImageSize(str, Enum):
    SQUARE_512 = "512x512"
    SQUARE_1024 = "1024x1024"
    LANDSCAPE_HD = "1344x768"
    PORTRAIT_HD = "768x1344"
    WIDE = "1536x640"


class ModelsLabImageGeneratorBlock(Block):
    class Input(BlockSchemaInput):
        prompt: str = SchemaField(
            description="Text description of the image to generate.",
            placeholder="A majestic mountain landscape at sunset, photorealistic.",
        )
        model: ModelsLabImageModel = SchemaField(
            title="Model",
            default=ModelsLabImageModel.FLUX,
            description="The ModelsLab model to use for image generation.",
        )
        size: ModelsLabImageSize = SchemaField(
            title="Image Size",
            default=ModelsLabImageSize.SQUARE_1024,
            description="The dimensions of the generated image.",
        )
        negative_prompt: str = SchemaField(
            title="Negative Prompt",
            default="blurry, low quality, watermark, ugly, distorted",
            description="What to exclude from the generated image.",
        )
        num_images: int = SchemaField(
            title="Number of Images",
            default=1,
            ge=1,
            le=4,
            description="Number of images to generate (1–4).",
        )
        safety_checker: bool = SchemaField(
            title="Safety Checker",
            default=False,
            description="Enable NSFW content filtering. When enabled, may filter some outputs.",
        )
        credentials: ModelsLabCredentialsInput = ModelsLabCredentialsField()

    class Output(BlockSchemaOutput):
        image_url: str = SchemaField(description="URL of the generated image.")
        error: str = SchemaField(description="Error message if generation failed.")

    def __init__(self):
        super().__init__(
            id="93180aca-aa2b-4857-990e-81cf9ac66ff8",
            description=(
                "Generate images using ModelsLab AI. "
                "Supports Flux, SDXL, and 100+ community models."
            ),
            categories={BlockCategory.AI},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "prompt": "A cozy coffee shop interior, warm lighting, photorealistic.",
                "model": ModelsLabImageModel.FLUX,
                "size": ModelsLabImageSize.SQUARE_1024,
                "negative_prompt": "blurry, low quality",
                "num_images": 1,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("image_url", lambda x: x.startswith(("workspace://", "data:", "https://"))),
            ],
            test_mock={
                "generate_image": lambda *args, **kwargs: "data:image/png;base64,AAAA"
            },
        )

    def _build_request_body(
        self,
        api_key: str,
        prompt: str,
        model: ModelsLabImageModel,
        size: ModelsLabImageSize,
        negative_prompt: str,
        num_images: int,
        safety_checker: bool = False,
    ) -> dict:
        width, height = map(int, size.value.split("x"))
        return {
            "key": api_key,
            "model_id": model.value,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "samples": num_images,
            "num_inference_steps": 30,
            "safety_checker": "yes" if safety_checker else "no",
            "enhance_prompt": "yes",
        }

    async def generate_image(
        self,
        api_key: str,
        prompt: str,
        model: ModelsLabImageModel,
        size: ModelsLabImageSize,
        negative_prompt: str,
        num_images: int,
        safety_checker: bool = False,
    ) -> str:
        """Generate an image and return the URL. Handles async polling."""
        body = self._build_request_body(
            api_key, prompt, model, size, negative_prompt, num_images, safety_checker
        )

        try:
            response = await Requests().post(
                MODELSLAB_IMAGE_URL,
                json=body,
                headers={"Content-Type": "application/json"},
            )
            data = response.json()
        except ClientResponseError as e:
            raise RuntimeError(f"ModelsLab API error: {e}") from e

        status = data.get("status", "")

        if status == "error":
            raise RuntimeError(f"ModelsLab image generation failed: {data.get('message', 'Unknown error')}")

        if status == "processing":
            request_id = data.get("id")
            if not request_id:
                raise RuntimeError("ModelsLab returned processing status without request ID")
            data = await self._poll_until_ready(api_key, str(request_id))

        output = data.get("output", [])
        if not output:
            raise RuntimeError("ModelsLab returned no image output")

        return output[0]

    async def _poll_until_ready(self, api_key: str, request_id: str) -> dict:
        """Poll the ModelsLab fetch endpoint until the image is ready."""
        deadline = asyncio.get_event_loop().time() + POLL_TIMEOUT
        fetch_body = {"key": api_key}

        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(POLL_INTERVAL)
            try:
                response = await Requests().post(
                    f"{MODELSLAB_FETCH_URL}/{request_id}",
                    json=fetch_body,
                    headers={"Content-Type": "application/json"},
                )
                data = response.json()
                if data.get("status") in ("success", "error"):
                    return data
            except ClientResponseError as e:
                logger.warning(f"ModelsLab poll error: {e}")

        raise RuntimeError(
            f"ModelsLab image generation timed out after {POLL_TIMEOUT}s (request_id={request_id})"
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: ModelsLabCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        image_url = await self.generate_image(
            api_key=api_key,
            prompt=input_data.prompt,
            model=input_data.model,
            size=input_data.size,
            negative_prompt=input_data.negative_prompt,
            num_images=input_data.num_images,
            safety_checker=input_data.safety_checker,
        )

        workspace_url = await store_media_file(
            file=MediaFileType(image_url),
            execution_context=execution_context,
            return_format="for_block_output",
        )
        yield "image_url", workspace_url
