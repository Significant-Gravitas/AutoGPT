import uuid
from enum import Enum
from typing import Literal

from pydantic import SecretStr
from replicate.client import Client as ReplicateClient
from replicate.helpers import FileOutput

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util.file import MediaFileType


class GeminiImageModel(str, Enum):
    GEMINI_2_5_FLASH_IMAGE = "google/gemini-2.5-flash-image"


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
    class Input(BlockSchema):
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
            description="The AI model to use for image generation",
            default=GeminiImageModel.GEMINI_2_5_FLASH_IMAGE,
            title="Model",
        )
        images: list[MediaFileType] = SchemaField(
            description="Optional list of input images to reference or modify",
            default_factory=list,
            title="Input Images",
        )
        output_format: OutputFormat = SchemaField(
            description="Format of the output image",
            default=OutputFormat.PNG,
            title="Output Format",
        )

    class Output(BlockSchema):
        image_url: MediaFileType = SchemaField(description="URL of the generated image")
        error: str = SchemaField(description="Error message if generation failed")

    def __init__(self):
        super().__init__(
            id=str(uuid.uuid4()),
            description=(
                "Generate custom images using Google's Gemini 2.5 Flash Image model. "
                "Provide a prompt and optional reference images to create new images."
            ),
            categories={BlockCategory.AI, BlockCategory.MULTIMEDIA},
            input_schema=AIImageCustomizerBlock.Input,
            output_schema=AIImageCustomizerBlock.Output,
            test_input={
                "prompt": "A golden retriever, the Dalai Lama, and Taylor Swift ring the bell at the New York stock exchange for their new company",
                "model": GeminiImageModel.GEMINI_2_5_FLASH_IMAGE,
                "images": [],
                "output_format": OutputFormat.JPG,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                ("image_url", "https://replicate.delivery/generated-image.jpg"),
            ],
            test_mock={
                "run_model": lambda *args, **kwargs: MediaFileType("https://replicate.delivery/generated-image.jpg"),
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
            result = await self.run_model(
                api_key=credentials.api_key,
                model_name=input_data.model.value,
                prompt=input_data.prompt,
                images=input_data.images,
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
        images: list[MediaFileType],
        output_format: str,
    ) -> MediaFileType:
        client = ReplicateClient(api_token=api_key.get_secret_value())

        input_params = {
            "prompt": prompt,
            "output_format": output_format,
        }
        
        # Add images to input if provided
        if images:
            # Convert MediaFileType objects to base64 strings for the API
            input_params["images"] = [str(img) for img in images]

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
