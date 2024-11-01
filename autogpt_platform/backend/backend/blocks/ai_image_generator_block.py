import uuid
from enum import Enum
from typing import Literal, Optional

import replicate
from autogpt_libs.supabase_integration_credentials_store.types import APIKeyCredentials
from pydantic import SecretStr

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import CredentialsField, CredentialsMetaInput, SchemaField


class ImageSize(str, Enum):
    """
    Universal sizes that work across all models via aspect ratio
    """

    SQUARE = "1:1"
    LANDSCAPE = "3:2"
    PORTRAIT = "2:3"
    WIDE = "16:9"
    TALL = "9:16"


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
    # Digital illustration styles
    DIGITAL_ART = "digital_illustration"
    PIXEL_ART = "digital_illustration/pixel_art"
    HAND_DRAWN = "digital_illustration/hand_drawn"
    SKETCH = "digital_illustration/infantile_sketch"
    POSTER = "digital_illustration/2d_art_poster"


class ModelProvider(str, Enum):
    """
    Available model providers
    """

    FLUX = "flux"
    RECRAFT = "recraft"
    SD3_5 = "sd3.5"


# Size to dimension mappings for models that need explicit dimensions
SIZE_DIMENSIONS = {
    ImageSize.SQUARE: (1024, 1024),
    ImageSize.LANDSCAPE: (1365, 1024),
    ImageSize.PORTRAIT: (1024, 1365),
    ImageSize.WIDE: (1536, 1024),
    ImageSize.TALL: (1024, 1536),
}

# Test credentials for the block
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


class AIImageGeneratorBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput[Literal["replicate"], Literal["api_key"]] = (
            CredentialsField(
                provider="replicate",
                supported_credential_types={"api_key"},
                description="The Replicate integration can be used with "
                "any API key with sufficient permissions for the blocks it is used on.",
            )
        )
        prompt: str = SchemaField(
            description="Text prompt for image generation",
            placeholder="e.g., 'A red panda using a laptop in a snowy forest'",
            title="Prompt",
        )
        model: ModelProvider = SchemaField(
            description="The AI model to use for image generation",
            default=ModelProvider.SD3_5,
            title="Model Provider",
        )
        size: ImageSize = SchemaField(
            description="Aspect ratio/size for the generated image",
            default=ImageSize.SQUARE,
            title="Image Size",
        )
        style: ImageStyle = SchemaField(
            description="Visual style for the generated image",
            default=ImageStyle.ANY,
            title="Image Style",
        )
        steps: int = SchemaField(
            description="Number of diffusion steps",
            default=40,
            title="Steps",
            advanced=True,
        )
        cfg: float = SchemaField(
            description="Classifier Free Guidance scale - higher values make image match prompt more closely",
            default=5.0,
            title="CFG Scale",
            advanced=True,
        )
        output_format: str = SchemaField(
            description="Output image format",
            default="webp",
            title="Output Format",
            advanced=True,
        )
        output_quality: int = SchemaField(
            description="Output image quality (0-100)",
            default=90,
            title="Output Quality",
            advanced=True,
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
                "prompt": "A red panda using a laptop in a snowy forest",
                "model": ModelProvider.SD3_5,
                "size": ImageSize.SQUARE,
                "style": ImageStyle.PIXEL_ART,
                "steps": 40,
                "cfg": 5.0,
                "output_format": "webp",
                "output_quality": 90,
            },
            test_output=[
                (
                    "image_url",
                    "https://replicate.delivery/generated-image.webp",
                ),
            ],
            test_credentials=TEST_CREDENTIALS,
        )

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
            ImageStyle.DIGITAL_ART: "digital art",
            ImageStyle.PIXEL_ART: "pixel art",
            ImageStyle.HAND_DRAWN: "hand drawn illustration",
            ImageStyle.SKETCH: "sketchy illustration",
            ImageStyle.POSTER: "2D art poster",
        }

        style_text = style_map.get(style, "")
        return f"{style_text} of" if style_text else ""

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            # Initialize Replicate client
            client = replicate.Client(api_token=credentials.api_key.get_secret_value())

            # Handle style-based prompt modification for models without native style support
            modified_prompt = input_data.prompt
            if (
                input_data.model != ModelProvider.RECRAFT
            ):  # Recraft has native style support
                style_prefix = self._style_to_prompt_prefix(input_data.style)
                modified_prompt = f"{style_prefix} {modified_prompt}".strip()

            # Get dimensions for the selected size
            width, height = SIZE_DIMENSIONS[input_data.size]

            if input_data.model == ModelProvider.SD3_5:
                # Use Stable Diffusion 3.5
                output = client.run(
                    "stability-ai/stable-diffusion-3.5-medium",
                    input={
                        "prompt": modified_prompt,
                        "aspect_ratio": input_data.size.value,
                        "output_format": input_data.output_format,
                        "output_quality": input_data.output_quality,
                        "steps": input_data.steps,
                        "cfg_scale": input_data.cfg,
                    },
                )
                # Convert output to list if it's an iterator
                output_list = list(output) if hasattr(output, "__iter__") else [output]
                # SD3.5 returns a list of URLs, take the first one
                yield "image_url", output_list[0]

            elif input_data.model == ModelProvider.FLUX:
                # Use Flux
                output = client.run(
                    "black-forest-labs/flux-1.1-pro",
                    input={
                        "prompt": modified_prompt,
                        "width": width,
                        "height": height,
                        "output_format": input_data.output_format,
                        "output_quality": input_data.output_quality,
                        "steps": input_data.steps,
                        "guidance": input_data.cfg,
                    },
                )
                yield "image_url", output

            elif input_data.model == ModelProvider.RECRAFT:
                # Use Recraft
                output = client.run(
                    "recraft-ai/recraft-v3",
                    input={
                        "prompt": input_data.prompt,  # Use original prompt
                        "size": f"{width}x{height}",
                        "style": input_data.style.value,  # Use native style support
                        "steps": input_data.steps,
                        "guidance_scale": input_data.cfg,
                    },
                )
                yield "image_url", output

        except Exception as e:
            yield "error", str(e)
