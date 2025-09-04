from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import SecretStr
from replicate.client import Client as ReplicateClient

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util.file import MediaFileType, store_media_file

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


class GeminiImageModel(str, Enum):
    GEMINI_2_5_FLASH_IMAGE = "google/gemini-2.5-flash-image"

    @property
    def api_name(self) -> str:
        return self.value


class OutputFormat(str, Enum):
    JPG = "jpg"
    PNG = "png"


class AIImageCustomizerBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.REPLICATE], Literal["api_key"]
        ] = CredentialsField(
            description="Replicate API key with permissions for Google Gemini models",
        )
        prompt: str = SchemaField(
            description="Text description of the image you want to generate",
            placeholder="e.g., 'A golden retriever, the Dalai Lama, and Taylor Swift ring the bell at the New York stock exchange for their new company'",
            title="Prompt",
        )
        model: GeminiImageModel = SchemaField(
            description="The AI model to use for image generation",
            default=GeminiImageModel.GEMINI_2_5_FLASH_IMAGE,
            title="Model",
        )
        images: Optional[List[MediaFileType]] = SchemaField(
            description="List of input images (optional)",
            default=None,
            title="Images",
        )
        output_format: OutputFormat = SchemaField(
            description="Format of the output image",
            default=OutputFormat.PNG,
            title="Output Format",
        )

    class Output(BlockSchema):
        image_url: str = SchemaField(description="URL of the generated image")
        error: str = SchemaField(description="Error message if generation failed")

    def __init__(self):
        super().__init__(
            id="8b3e9f2a-4d7c-4a1b-9e5f-6c8d2a9b3e7f",
            description="Generate images using Google's Gemini 2.5 Flash Image model via Replicate API",
            categories={BlockCategory.AI, BlockCategory.MULTIMEDIA},
            input_schema=AIImageCustomizerBlock.Input,
            output_schema=AIImageCustomizerBlock.Output,
            test_input={
                "prompt": "A golden retriever, the Dalai Lama, and Taylor Swift ring the bell at the New York stock exchange for their new company",
                "model": GeminiImageModel.GEMINI_2_5_FLASH_IMAGE,
                "images": None,
                "output_format": OutputFormat.PNG,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                ("image_url", "https://replicate.delivery/generated-image.jpg"),
            ],
            test_mock={
                "run_model": lambda *args, **kwargs: "https://replicate.delivery/generated-image.jpg",
            },
            test_credentials=TEST_CREDENTIALS,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        graph_exec_id: str,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        try:
            # Process input images if provided
            processed_images = None
            if input_data.images:
                processed_images = []
                for image in input_data.images:
                    image_b64 = await store_media_file(
                        graph_exec_id=graph_exec_id,
                        file=image,
                        user_id=user_id,
                        return_content=True,
                    )
                    processed_images.append(image_b64)

            result = await self.run_model(
                api_key=credentials.api_key,
                model_name=input_data.model.api_name,
                prompt=input_data.prompt,
                images=processed_images,
                output_format=input_data.output_format.value,
            )
            yield "image_url", result
        except Exception as e:
            yield "error", str(e)

    async def run_model(
        self,
        api_key: SecretStr,
        model_name: str,
        prompt: str,
        images: Optional[List[str]],
        output_format: str,
    ) -> str:
        client = ReplicateClient(api_token=api_key.get_secret_value())

        input_params: Dict[str, Any] = {
            "prompt": prompt,
            "output_format": output_format,
        }

        # Add images to input if provided
        if images:
            input_params["images"] = images

        output = await client.async_run(
            model_name,
            input=input_params,
            wait=True,
        )

        # Handle different output types
        if isinstance(output, str):
            return output
        elif isinstance(output, list) and output:
            return output[0] if isinstance(output[0], str) else str(output[0])
        else:
            raise ValueError("No valid output received from the model")
