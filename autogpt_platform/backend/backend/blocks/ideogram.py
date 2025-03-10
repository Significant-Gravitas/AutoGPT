from enum import Enum
from typing import Any, Dict, Literal, Optional

from pydantic import SecretStr
from requests.exceptions import RequestException

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util.request import requests

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="ideogram",
    api_key=SecretStr("mock-ideogram-api-key"),
    title="Mock Ideogram API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.type,
}


class IdeogramModelName(str, Enum):
    V2 = "V_2"
    V1 = "V_1"
    V1_TURBO = "V_1_TURBO"
    V2_TURBO = "V_2_TURBO"


class MagicPromptOption(str, Enum):
    AUTO = "AUTO"
    ON = "ON"
    OFF = "OFF"


class StyleType(str, Enum):
    AUTO = "AUTO"
    GENERAL = "GENERAL"
    REALISTIC = "REALISTIC"
    DESIGN = "DESIGN"
    RENDER_3D = "RENDER_3D"
    ANIME = "ANIME"


class ColorPalettePreset(str, Enum):
    NONE = "NONE"
    EMBER = "EMBER"
    FRESH = "FRESH"
    JUNGLE = "JUNGLE"
    MAGIC = "MAGIC"
    MELON = "MELON"
    MOSAIC = "MOSAIC"
    PASTEL = "PASTEL"
    ULTRAMARINE = "ULTRAMARINE"


class AspectRatio(str, Enum):
    ASPECT_10_16 = "ASPECT_10_16"
    ASPECT_16_10 = "ASPECT_16_10"
    ASPECT_9_16 = "ASPECT_9_16"
    ASPECT_16_9 = "ASPECT_16_9"
    ASPECT_3_2 = "ASPECT_3_2"
    ASPECT_2_3 = "ASPECT_2_3"
    ASPECT_4_3 = "ASPECT_4_3"
    ASPECT_3_4 = "ASPECT_3_4"
    ASPECT_1_1 = "ASPECT_1_1"
    ASPECT_1_3 = "ASPECT_1_3"
    ASPECT_3_1 = "ASPECT_3_1"


class UpscaleOption(str, Enum):
    AI_UPSCALE = "AI Upscale"
    NO_UPSCALE = "No Upscale"


class IdeogramModelBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.IDEOGRAM], Literal["api_key"]
        ] = CredentialsField(
            description="The Ideogram integration can be used with any API key with sufficient permissions for the blocks it is used on.",
        )
        prompt: str = SchemaField(
            description="Text prompt for image generation",
            placeholder="e.g., 'A futuristic cityscape at sunset'",
            title="Prompt",
        )
        ideogram_model_name: IdeogramModelName = SchemaField(
            description="The name of the Image Generation Model, e.g., V_2",
            default=IdeogramModelName.V2,
            title="Image Generation Model",
            advanced=False,
        )
        aspect_ratio: AspectRatio = SchemaField(
            description="Aspect ratio for the generated image",
            default=AspectRatio.ASPECT_1_1,
            title="Aspect Ratio",
            advanced=False,
        )
        upscale: UpscaleOption = SchemaField(
            description="Upscale the generated image",
            default=UpscaleOption.NO_UPSCALE,
            title="Upscale Image",
            advanced=False,
        )
        magic_prompt_option: MagicPromptOption = SchemaField(
            description="Whether to use MagicPrompt for enhancing the request",
            default=MagicPromptOption.AUTO,
            title="Magic Prompt Option",
            advanced=True,
        )
        seed: Optional[int] = SchemaField(
            description="Random seed. Set for reproducible generation",
            default=None,
            title="Seed",
            advanced=True,
        )
        style_type: StyleType = SchemaField(
            description="Style type to apply, applicable for V_2 and above",
            default=StyleType.AUTO,
            title="Style Type",
            advanced=True,
        )
        negative_prompt: Optional[str] = SchemaField(
            description="Description of what to exclude from the image",
            default=None,
            title="Negative Prompt",
            advanced=True,
        )
        color_palette_name: ColorPalettePreset = SchemaField(
            description="Color palette preset name, choose 'None' to skip",
            default=ColorPalettePreset.NONE,
            title="Color Palette Preset",
            advanced=True,
        )
        custom_color_palette: Optional[list[str]] = SchemaField(
            description=(
                "Only available for model version V_2 or V_2_TURBO. Provide one or more color hex codes "
                "(e.g., ['#000030', '#1C0C47', '#9900FF', '#4285F4', '#FFFFFF']) to define a custom color "
                "palette. Only used if 'color_palette_name' is 'NONE'."
            ),
            default=None,
            title="Custom Color Palette",
            advanced=True,
        )

    class Output(BlockSchema):
        result: str = SchemaField(description="Generated image URL")
        error: str = SchemaField(description="Error message if the model run failed")

    def __init__(self):
        super().__init__(
            id="6ab085e2-20b3-4055-bc3e-08036e01eca6",
            description="This block runs Ideogram models with both simple and advanced settings.",
            categories={BlockCategory.AI, BlockCategory.MULTIMEDIA},
            input_schema=IdeogramModelBlock.Input,
            output_schema=IdeogramModelBlock.Output,
            test_input={
                "ideogram_model_name": IdeogramModelName.V2,
                "prompt": "A futuristic cityscape at sunset",
                "aspect_ratio": AspectRatio.ASPECT_1_1,
                "upscale": UpscaleOption.NO_UPSCALE,
                "magic_prompt_option": MagicPromptOption.AUTO,
                "seed": None,
                "style_type": StyleType.AUTO,
                "negative_prompt": None,
                "color_palette_name": ColorPalettePreset.NONE,
                "custom_color_palette": [
                    "#000030",
                    "#1C0C47",
                    "#9900FF",
                    "#4285F4",
                    "#FFFFFF",
                ],
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                (
                    "result",
                    "https://ideogram.ai/api/images/test-generated-image-url.png",
                ),
            ],
            test_mock={
                "run_model": lambda api_key, model_name, prompt, seed, aspect_ratio, magic_prompt_option, style_type, negative_prompt, color_palette_name, custom_colors: "https://ideogram.ai/api/images/test-generated-image-url.png",
                "upscale_image": lambda api_key, image_url: "https://ideogram.ai/api/images/test-upscaled-image-url.png",
            },
            test_credentials=TEST_CREDENTIALS,
        )

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        seed = input_data.seed

        # Step 1: Generate the image
        result = self.run_model(
            api_key=credentials.api_key,
            model_name=input_data.ideogram_model_name.value,
            prompt=input_data.prompt,
            seed=seed,
            aspect_ratio=input_data.aspect_ratio.value,
            magic_prompt_option=input_data.magic_prompt_option.value,
            style_type=input_data.style_type.value,
            negative_prompt=input_data.negative_prompt,
            color_palette_name=input_data.color_palette_name.value,
            custom_colors=input_data.custom_color_palette,
        )

        # Step 2: Upscale the image if requested
        if input_data.upscale == UpscaleOption.AI_UPSCALE:
            result = self.upscale_image(
                api_key=credentials.api_key,
                image_url=result,
            )

        yield "result", result

    def run_model(
        self,
        api_key: SecretStr,
        model_name: str,
        prompt: str,
        seed: Optional[int],
        aspect_ratio: str,
        magic_prompt_option: str,
        style_type: str,
        negative_prompt: Optional[str],
        color_palette_name: str,
        custom_colors: Optional[list[str]],
    ):
        url = "https://api.ideogram.ai/generate"
        headers = {
            "Api-Key": api_key.get_secret_value(),
            "Content-Type": "application/json",
        }

        data: Dict[str, Any] = {
            "image_request": {
                "prompt": prompt,
                "model": model_name,
                "aspect_ratio": aspect_ratio,
                "magic_prompt_option": magic_prompt_option,
                "style_type": style_type,
            }
        }

        if seed is not None:
            data["image_request"]["seed"] = seed

        if negative_prompt:
            data["image_request"]["negative_prompt"] = negative_prompt

        if color_palette_name != "NONE":
            data["color_palette"] = {"name": color_palette_name}
        elif custom_colors:
            data["color_palette"] = {
                "members": [{"color_hex": color} for color in custom_colors]
            }

        try:
            response = requests.post(url, json=data, headers=headers)
            return response.json()["data"][0]["url"]
        except RequestException as e:
            raise Exception(f"Failed to fetch image: {str(e)}")

    def upscale_image(self, api_key: SecretStr, image_url: str):
        url = "https://api.ideogram.ai/upscale"
        headers = {
            "Api-Key": api_key.get_secret_value(),
        }

        try:
            # Step 1: Download the image from the provided URL
            image_response = requests.get(image_url)

            # Step 2: Send the downloaded image to the upscale API
            files = {
                "image_file": ("image.png", image_response.content, "image/png"),
            }

            response = requests.post(
                url,
                headers=headers,
                data={"image_request": "{}"},
                files=files,
            )

            return response.json()["data"][0]["url"]

        except RequestException as e:
            raise Exception(f"Failed to upscale image: {str(e)}")
