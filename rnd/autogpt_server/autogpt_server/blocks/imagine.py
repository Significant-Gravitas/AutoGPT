import logging
from enum import Enum

from openai import OpenAI
from pydantic import HttpUrl

from autogpt_server.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from autogpt_server.data.model import BlockSecret, SchemaField, SecretField

logger = logging.getLogger(__name__)


class ImageSize(str, Enum):
    SMALL = "256x256"
    MEDIUM = "512x512"
    LARGE = "1024x1024"


class ImageQuality(str, Enum):
    STANDARD = "standard"
    HD = "hd"


class ImagineWithDallEBlock(Block):
    class Input(BlockSchema):
        prompt: str = SchemaField(description="The prompt to generate the image from.")
        n: int = SchemaField(
            default=1, description="The number of images to generate.", ge=1, le=10
        )
        size: ImageSize = SchemaField(
            default=ImageSize.MEDIUM, description="The size of the generated image(s)."
        )
        quality: ImageQuality = SchemaField(
            default=ImageQuality.STANDARD,
            description="The quality of the generated image(s).",
        )
        api_key: BlockSecret = SecretField(
            value="", description="OpenAI API key for DALL-E."
        )

    class Output(BlockSchema):
        images: HttpUrl = SchemaField(
            description="One or more URLs of generated images."
        )
        error: str = SchemaField(
            description="Error message if the image generation failed."
        )

    def __init__(self):
        super().__init__(
            id="7b6ce609-adac-4d27-81e9-6dd2f30d9977",
            description="Generate images using DALL-E based on a text prompt.",
            categories={BlockCategory.AI, BlockCategory.IMAGE},
            input_schema=ImagineWithDallEBlock.Input,
            output_schema=ImagineWithDallEBlock.Output,
            test_input={
                "prompt": "A futuristic city skyline at sunset",
                "n": 1,
                "size": ImageSize.MEDIUM,
                "quality": ImageQuality.STANDARD,
                "api_key": "test_api_key",
            },
            test_output=("images", ["https://example.com/generated_image.png"]),
            test_mock={
                "generate_image": lambda *args, **kwargs: [
                    "https://example.com/generated_image.png"
                ]
            },
        )

    @staticmethod
    def generate_image(
        api_key: str, prompt: str, n: int, size: ImageSize, quality: ImageQuality
    ) -> list[str]:
        response = OpenAI(api_key=api_key).images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=n,
            size=size.value,
            quality=quality.value,
            response_format="url",
        )
        return [image.url for image in response.data]

    def run(self, input_data: Input) -> BlockOutput:
        try:
            api_key = input_data.api_key.get_secret_value()
            image_urls = self.generate_image(
                api_key=api_key,
                prompt=input_data.prompt,
                n=input_data.n,
                size=input_data.size,
                quality=input_data.quality,
            )
            for url in image_urls:
                yield "images", url
        except Exception as e:
            logger.error(f"Error generating DALL-E image: {e}")
            yield "error", f"Error generating DALL-E image: {str(e)}"
