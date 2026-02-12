import asyncio
from enum import Enum
from typing import Literal

from pydantic import SecretStr
from replicate.client import Client as ReplicateClient
from replicate.helpers import FileOutput

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
from backend.util.file import MediaFileType, store_media_file


class GeminiImageModel(str, Enum):
    NANO_BANANA = "google/nano-banana"
    NANO_BANANA_PRO = "google/nano-banana-pro"


class AspectRatio(str, Enum):
    MATCH_INPUT_IMAGE = "match_input_image"
    ASPECT_1_1 = "1:1"
    ASPECT_2_3 = "2:3"
    ASPECT_3_2 = "3:2"
    ASPECT_3_4 = "3:4"
    ASPECT_4_3 = "4:3"
    ASPECT_4_5 = "4:5"
    ASPECT_5_4 = "5:4"
    ASPECT_9_16 = "9:16"
    ASPECT_16_9 = "16:9"
    ASPECT_21_9 = "21:9"


class OutputFormat(str, Enum):
    JPG = "jpg"
    PNG = "png"


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


class AIImageCustomizerBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.REPLICATE], Literal["api_key"]
        ] = CredentialsField(
            description="Replicate API key with permissions for Google Gemini image models",
        )
        prompt: str = SchemaField(
            description="A text description of the image you want to generate",
            title="Prompt",
        )
        model: GeminiImageModel = SchemaField(
            description="The AI model to use for image generation and editing",
            default=GeminiImageModel.NANO_BANANA,
            title="Model",
        )
        images: list[MediaFileType] = SchemaField(
            description="Optional list of input images to reference or modify",
            default=[],
            title="Input Images",
        )
        aspect_ratio: AspectRatio = SchemaField(
            description="Aspect ratio of the generated image",
            default=AspectRatio.MATCH_INPUT_IMAGE,
            title="Aspect Ratio",
        )
        output_format: OutputFormat = SchemaField(
            description="Format of the output image",
            default=OutputFormat.PNG,
            title="Output Format",
        )

    class Output(BlockSchemaOutput):
        image_url: MediaFileType = SchemaField(description="URL of the generated image")

    def __init__(self):
        super().__init__(
            id="d76bbe4c-930e-4894-8469-b66775511f71",
            description=(
                "Generate and edit custom images using Google's Nano-Banana model from Gemini 2.5. "
                "Provide a prompt and optional reference images to create or modify images."
            ),
            categories={BlockCategory.AI, BlockCategory.MULTIMEDIA},
            input_schema=AIImageCustomizerBlock.Input,
            output_schema=AIImageCustomizerBlock.Output,
            test_input={
                "prompt": "Make the scene more vibrant and colorful",
                "model": GeminiImageModel.NANO_BANANA,
                "images": [],
                "aspect_ratio": AspectRatio.MATCH_INPUT_IMAGE,
                "output_format": OutputFormat.JPG,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                # Output will be a workspace ref or data URI depending on context
                ("image_url", lambda x: x.startswith(("workspace://", "data:"))),
            ],
            test_mock={
                # Use data URI to avoid HTTP requests during tests
                "run_model": lambda *args, **kwargs: MediaFileType(
                    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAgAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigD//2Q=="
                ),
            },
            test_credentials=TEST_CREDENTIALS,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        try:
            # Convert local file paths to Data URIs (base64) so Replicate can access them
            processed_images = await asyncio.gather(
                *(
                    store_media_file(
                        file=img,
                        execution_context=execution_context,
                        return_format="for_external_api",  # Get content for Replicate API
                    )
                    for img in input_data.images
                )
            )

            result = await self.run_model(
                api_key=credentials.api_key,
                model_name=input_data.model.value,
                prompt=input_data.prompt,
                images=processed_images,
                aspect_ratio=input_data.aspect_ratio.value,
                output_format=input_data.output_format.value,
            )

            # Store the generated image to the user's workspace for persistence
            stored_url = await store_media_file(
                file=result,
                execution_context=execution_context,
                return_format="for_block_output",
            )
            yield "image_url", stored_url
        except Exception as e:
            yield "error", str(e)

    async def run_model(
        self,
        api_key: SecretStr,
        model_name: str,
        prompt: str,
        images: list[MediaFileType],
        aspect_ratio: str,
        output_format: str,
    ) -> MediaFileType:
        client = ReplicateClient(api_token=api_key.get_secret_value())

        input_params: dict = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format,
        }

        # Add images to input if provided (API expects "image_input" parameter)
        if images:
            input_params["image_input"] = [str(img) for img in images]

        output: FileOutput | str = await client.async_run(  # type: ignore
            model_name,
            input=input_params,
            wait=False,
        )

        if isinstance(output, FileOutput):
            return MediaFileType(output.url)
        if isinstance(output, str):
            return MediaFileType(output)

        raise ValueError("No output received from the model")
