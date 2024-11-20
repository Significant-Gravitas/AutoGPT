from enum import Enum
from typing import Literal

import replicate
from pydantic import SecretStr
from replicate.helpers import FileOutput

from backend.data.block import Block, BlockCategory, BlockSchema
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName


class ImageSize(str, Enum):
    """
    Semantic sizes that map reliably across all models
    """

    SQUARE = "square"  # For profile pictures, icons, etc.
    LANDSCAPE = "landscape"  # For traditional photos, scenes
    PORTRAIT = "portrait"  # For vertical photos, portraits
    WIDE = "wide"  # For cinematic, desktop wallpapers
    TALL = "tall"  # For mobile wallpapers, stories


# Mapping semantic sizes to model-specific formats
SIZE_TO_SD_RATIO = {
    ImageSize.SQUARE: "1:1",
    ImageSize.LANDSCAPE: "4:3",
    ImageSize.PORTRAIT: "3:4",
    ImageSize.WIDE: "16:9",
    ImageSize.TALL: "9:16",
}

SIZE_TO_FLUX_RATIO = {
    ImageSize.SQUARE: "1:1",
    ImageSize.LANDSCAPE: "4:3",
    ImageSize.PORTRAIT: "3:4",
    ImageSize.WIDE: "16:9",
    ImageSize.TALL: "9:16",
}

SIZE_TO_FLUX_DIMENSIONS = {
    ImageSize.SQUARE: (1024, 1024),
    ImageSize.LANDSCAPE: (1365, 1024),
    ImageSize.PORTRAIT: (1024, 1365),
    ImageSize.WIDE: (1440, 810),  # Adjusted to maintain 16:9 within 1440 limit
    ImageSize.TALL: (810, 1440),  # Adjusted to maintain 9:16 within 1440 limit
}

SIZE_TO_RECRAFT_DIMENSIONS = {
    ImageSize.SQUARE: "1024x1024",
    ImageSize.LANDSCAPE: "1365x1024",
    ImageSize.PORTRAIT: "1024x1365",
    ImageSize.WIDE: "1536x1024",
    ImageSize.TALL: "1024x1536",
}


class ImageStyle(str, Enum):
    """
    Complete set of supported styles
    """

    ANY = "any"
    # Realistic image styles
    REALISTIC = "realistic_image"
    REALISTIC_BW = "realistic_image/b_and_w"
    REALISTIC_HDR = "realistic_image/hdr"
    REALISTIC_NATURAL = "realistic_image/natural_light"
    REALISTIC_STUDIO = "realistic_image/studio_portrait"
    REALISTIC_ENTERPRISE = "realistic_image/enterprise"
    REALISTIC_HARD_FLASH = "realistic_image/hard_flash"
    REALISTIC_MOTION_BLUR = "realistic_image/motion_blur"
    # Digital illustration styles
    DIGITAL_ART = "digital_illustration"
    PIXEL_ART = "digital_illustration/pixel_art"
    HAND_DRAWN = "digital_illustration/hand_drawn"
    GRAIN = "digital_illustration/grain"
    SKETCH = "digital_illustration/infantile_sketch"
    POSTER = "digital_illustration/2d_art_poster"
    POSTER_2 = "digital_illustration/2d_art_poster_2"
    HANDMADE_3D = "digital_illustration/handmade_3d"
    HAND_DRAWN_OUTLINE = "digital_illustration/hand_drawn_outline"
    ENGRAVING_COLOR = "digital_illustration/engraving_color"


class ImageGenModel(str, Enum):
    """
    Available model providers
    """

    FLUX = "Flux 1.1 Pro"
    FLUX_ULTRA = "Flux 1.1 Pro Ultra"
    RECRAFT = "Recraft v3"
    SD3_5 = "Stable Diffusion 3.5 Medium"


class AIImageGeneratorBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.REPLICATE], Literal["api_key"]
        ] = CredentialsField(
            description="Enter your Replicate API key to access the image generation API. You can obtain an API key from https://replicate.com/account/api-tokens.",
        )
        prompt: str = SchemaField(
            description="Text prompt for image generation",
            placeholder="e.g., 'A red panda using a laptop in a snowy forest'",
            title="Prompt",
        )
        model: ImageGenModel = SchemaField(
            description="The AI model to use for image generation",
            default=ImageGenModel.SD3_5,
            title="Model",
        )
        size: ImageSize = SchemaField(
            description=(
                "Format of the generated image:\n"
                "- Square: Perfect for profile pictures, icons\n"
                "- Landscape: Traditional photo format\n"
                "- Portrait: Vertical photos, portraits\n"
                "- Wide: Cinematic format, desktop wallpapers\n"
                "- Tall: Mobile wallpapers, social media stories"
            ),
            default=ImageSize.SQUARE,
            title="Image Format",
        )
        style: ImageStyle = SchemaField(
            description="Visual style for the generated image",
            default=ImageStyle.ANY,
            title="Image Style",
        )

    class Output(BlockSchema):
        image_url: str = SchemaField(description="URL of the generated image")
        error: str = SchemaField(description="Error message if generation failed")

    def __init__(self):
        super().__init__(
            id="ed1ae7a0-b770-4089-b520-1f0005fad19a",
            description="Generate images using various AI models through a unified interface",
            categories={BlockCategory.AI},
            input_schema=AIImageGeneratorBlock.Input,
            output_schema=AIImageGeneratorBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "prompt": "An octopus using a laptop in a snowy forest with 'AutoGPT' clearly visible on the screen",
                "model": ImageGenModel.RECRAFT,
                "size": ImageSize.SQUARE,
                "style": ImageStyle.REALISTIC,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "image_url",
                    "https://replicate.delivery/generated-image.webp",
                ),
            ],
            test_mock={
                "_run_client": lambda *args, **kwargs: "https://replicate.delivery/generated-image.webp"
            },
        )

    def _run_client(
        self, credentials: APIKeyCredentials, model_name: str, input_params: dict
    ):
        try:
            # Initialize Replicate client
            client = replicate.Client(api_token=credentials.api_key.get_secret_value())

            # Run the model with input parameters
            output = client.run(model_name, input=input_params, wait=False)

            # Process output
            if isinstance(output, list) and len(output) > 0:
                if isinstance(output[0], FileOutput):
                    result_url = output[0].url
                else:
                    result_url = output[0]
            elif isinstance(output, FileOutput):
                result_url = output.url
            elif isinstance(output, str):
                result_url = output
            else:
                result_url = None

            return result_url

        except TypeError as e:
            raise TypeError(f"Error during model execution: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during model execution: {e}")

    def generate_image(self, input_data: Input, credentials: APIKeyCredentials):
        try:
            # Handle style-based prompt modification for models without native style support
            modified_prompt = input_data.prompt
            if input_data.model not in [ImageGenModel.RECRAFT]:
                style_prefix = self._style_to_prompt_prefix(input_data.style)
                modified_prompt = f"{style_prefix} {modified_prompt}".strip()

            if input_data.model == ImageGenModel.SD3_5:
                # Use Stable Diffusion 3.5 with aspect ratio
                input_params = {
                    "prompt": modified_prompt,
                    "aspect_ratio": SIZE_TO_SD_RATIO[input_data.size],
                    "output_format": "webp",
                    "output_quality": 90,
                    "steps": 40,
                    "cfg_scale": 7.0,
                }
                output = self._run_client(
                    credentials,
                    "stability-ai/stable-diffusion-3.5-medium",
                    input_params,
                )
                return output

            elif input_data.model == ImageGenModel.FLUX:
                # Use Flux-specific dimensions with 'jpg' format to avoid ReplicateError
                width, height = SIZE_TO_FLUX_DIMENSIONS[input_data.size]
                input_params = {
                    "prompt": modified_prompt,
                    "width": width,
                    "height": height,
                    "aspect_ratio": SIZE_TO_FLUX_RATIO[input_data.size],
                    "output_format": "jpg",  # Set to jpg for Flux models
                    "output_quality": 90,
                }
                output = self._run_client(
                    credentials, "black-forest-labs/flux-1.1-pro", input_params
                )
                return output

            elif input_data.model == ImageGenModel.FLUX_ULTRA:
                width, height = SIZE_TO_FLUX_DIMENSIONS[input_data.size]
                input_params = {
                    "prompt": modified_prompt,
                    "width": width,
                    "height": height,
                    "aspect_ratio": SIZE_TO_FLUX_RATIO[input_data.size],
                    "output_format": "jpg",
                    "output_quality": 90,
                }
                output = self._run_client(
                    credentials, "black-forest-labs/flux-1.1-pro-ultra", input_params
                )
                return output

            elif input_data.model == ImageGenModel.RECRAFT:
                input_params = {
                    "prompt": input_data.prompt,
                    "size": SIZE_TO_RECRAFT_DIMENSIONS[input_data.size],
                    "style": input_data.style.value,
                }
                output = self._run_client(
                    credentials, "recraft-ai/recraft-v3", input_params
                )
                return output

        except Exception as e:
            raise RuntimeError(f"Failed to generate image: {str(e)}")

    def _style_to_prompt_prefix(self, style: ImageStyle) -> str:
        """
        Convert a style enum to a prompt prefix for models without native style support.
        """
        if style == ImageStyle.ANY:
            return ""

        style_map = {
            ImageStyle.REALISTIC: "photorealistic",
            ImageStyle.REALISTIC_BW: "black and white photograph",
            ImageStyle.REALISTIC_HDR: "HDR photograph",
            ImageStyle.REALISTIC_NATURAL: "natural light photograph",
            ImageStyle.REALISTIC_STUDIO: "studio portrait photograph",
            ImageStyle.REALISTIC_ENTERPRISE: "enterprise photograph",
            ImageStyle.REALISTIC_HARD_FLASH: "hard flash photograph",
            ImageStyle.REALISTIC_MOTION_BLUR: "motion blur photograph",
            ImageStyle.DIGITAL_ART: "digital art",
            ImageStyle.PIXEL_ART: "pixel art",
            ImageStyle.HAND_DRAWN: "hand drawn illustration",
            ImageStyle.GRAIN: "grainy digital illustration",
            ImageStyle.SKETCH: "sketchy illustration",
            ImageStyle.POSTER: "2D art poster",
            ImageStyle.POSTER_2: "alternate 2D art poster",
            ImageStyle.HANDMADE_3D: "handmade 3D illustration",
            ImageStyle.HAND_DRAWN_OUTLINE: "hand drawn outline illustration",
            ImageStyle.ENGRAVING_COLOR: "color engraving illustration",
        }

        style_text = style_map.get(style, "")
        return f"{style_text} of" if style_text else ""

    def run(self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs):
        try:
            url = self.generate_image(input_data, credentials)
            if url:
                yield "image_url", url
            else:
                yield "error", "Image generation returned an empty result."
        except Exception as e:
            # Capture and return only the message of the exception, avoiding serialization of non-serializable objects
            yield "error", str(e)


# Test credentials stay the same
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
