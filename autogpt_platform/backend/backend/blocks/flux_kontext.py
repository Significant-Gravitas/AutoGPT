from enum import Enum
from io import BytesIO
import base64
from typing import Literal, Optional, cast

import openai
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
from backend.util.exceptions import ModerationError
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


class ImageEditorModel(str, Enum):
    FLUX_KONTEXT_PRO = "Flux Kontext Pro"
    FLUX_KONTEXT_MAX = "Flux Kontext Max"
    NANO_BANANA_PRO = "Nano Banana Pro"
    NANO_BANANA_2 = "Nano Banana 2"
    GPT_IMAGE_1 = "gpt-image-1"
    GPT_IMAGE_1_5 = "gpt-image-1.5"
    GPT_IMAGE_2 = "gpt-image-2"
    GPT_IMAGE_1_MINI = "gpt-image-1-mini"

    @property
    def api_name(self) -> str:
        _map = {
            "FLUX_KONTEXT_PRO": "black-forest-labs/flux-kontext-pro",
            "FLUX_KONTEXT_MAX": "black-forest-labs/flux-kontext-max",
            "NANO_BANANA_PRO": "google/nano-banana-pro",
            "NANO_BANANA_2": "google/nano-banana-2",
        }
        return _map[self.name]


FluxKontextModelName = ImageEditorModel


class AspectRatio(str, Enum):
    MATCH_INPUT_IMAGE = "match_input_image"
    ASPECT_1_1 = "1:1"
    ASPECT_16_9 = "16:9"
    ASPECT_9_16 = "9:16"
    ASPECT_4_3 = "4:3"
    ASPECT_3_4 = "3:4"
    ASPECT_3_2 = "3:2"
    ASPECT_2_3 = "2:3"
    ASPECT_4_5 = "4:5"
    ASPECT_5_4 = "5:4"
    ASPECT_21_9 = "21:9"
    ASPECT_9_21 = "9:21"
    ASPECT_2_1 = "2:1"
    ASPECT_1_2 = "1:2"


ASPECT_TO_OPENAI_SIZE = {
    AspectRatio.MATCH_INPUT_IMAGE: "auto",
    AspectRatio.ASPECT_1_1: "1024x1024",
    AspectRatio.ASPECT_16_9: "1536x1024",
    AspectRatio.ASPECT_9_16: "1024x1536",
    AspectRatio.ASPECT_4_3: "1536x1024",
    AspectRatio.ASPECT_3_4: "1024x1536",
    AspectRatio.ASPECT_3_2: "1536x1024",
    AspectRatio.ASPECT_2_3: "1024x1536",
    AspectRatio.ASPECT_4_5: "1024x1536",
    AspectRatio.ASPECT_5_4: "1536x1024",
    AspectRatio.ASPECT_21_9: "1536x1024",
    AspectRatio.ASPECT_9_21: "1024x1536",
    AspectRatio.ASPECT_2_1: "1536x1024",
    AspectRatio.ASPECT_1_2: "1024x1536",
}


class AIImageEditorBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.REPLICATE, ProviderName.OPENAI],
            Literal["api_key"],
        ] = CredentialsField(
            description="Replicate or OpenAI API key with permissions for image editing models",
        )
        prompt: str = SchemaField(
            description="Text instruction describing the desired edit",
            title="Prompt",
        )
        input_image: Optional[MediaFileType] = SchemaField(
            description="Reference image URI (jpeg, png, gif, webp)",
            default=None,
            title="Input Image",
        )
        aspect_ratio: AspectRatio = SchemaField(
            description="Aspect ratio of the generated image",
            default=AspectRatio.MATCH_INPUT_IMAGE,
            title="Aspect Ratio",
            advanced=False,
        )
        seed: Optional[int] = SchemaField(
            description="Random seed. Set for reproducible generation (Flux Kontext only; ignored by other models)",
            default=None,
            title="Seed",
            advanced=True,
        )
        model: ImageEditorModel = SchemaField(
            description="Model variant to use",
            default=ImageEditorModel.NANO_BANANA_2,
            title="Model",
        )

    class Output(BlockSchemaOutput):
        output_image: MediaFileType = SchemaField(
            description="URL of the transformed image"
        )

    def __init__(self):
        super().__init__(
            id="3fd9c73d-4370-4925-a1ff-1b86b99fabfa",
            description=(
                "Edit images using Flux Kontext, Google Nano Banana, or OpenAI GPT-image models. "
                "Provide a prompt and optional reference image to generate a modified image."
            ),
            categories={BlockCategory.AI, BlockCategory.MULTIMEDIA},
            input_schema=AIImageEditorBlock.Input,
            output_schema=AIImageEditorBlock.Output,
            test_input={
                "prompt": "Add a hat to the cat",
                "input_image": "data:image/png;base64,MQ==",
                "aspect_ratio": AspectRatio.MATCH_INPUT_IMAGE,
                "seed": None,
                "model": ImageEditorModel.NANO_BANANA_2,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                ("output_image", lambda x: x.startswith(("workspace://", "data:"))),
            ],
            test_mock={
                "run_model": lambda *args, **kwargs: (
                    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAhKmMIQAAAABJRU5ErkJggg=="
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
        result = await self.run_model(
            api_key=credentials.api_key,
            model=input_data.model,
            prompt=input_data.prompt,
            input_image_b64=(
                await store_media_file(
                    file=input_data.input_image,
                    execution_context=execution_context,
                    return_format="for_external_api",
                )
                if input_data.input_image
                else None
            ),
            aspect_ratio=input_data.aspect_ratio.value,
            seed=input_data.seed,
            user_id=execution_context.user_id or "",
            graph_exec_id=execution_context.graph_exec_id or "",
        )
        stored_url = await store_media_file(
            file=result,
            execution_context=execution_context,
            return_format="for_block_output",
        )
        yield "output_image", stored_url

    async def _edit_with_openai(
        self,
        api_key: SecretStr,
        model: ImageEditorModel,
        prompt: str,
        input_image_b64: Optional[str],
        aspect_ratio: str,
    ) -> MediaFileType:
        if not input_image_b64:
            raise ValueError("OpenAI image editing requires an input image.")

        client = openai.AsyncOpenAI(api_key=api_key.get_secret_value())

        data_uri = str(input_image_b64)
        if "," not in data_uri:
            raise ValueError("Expected a data-URI for the input image.")
        _, encoded = data_uri.split(",", 1)
        image_bytes = BytesIO(base64.b64decode(encoded))

        size = ASPECT_TO_OPENAI_SIZE.get(aspect_ratio, "1024x1024")
        size_literal = cast(
            Literal["1024x1024", "1536x1024", "1024x1536", "auto"], size
        )

        response = await client.images.edit(
            model=model.value,
            image=image_bytes,
            prompt=prompt,
            n=1,
            size=size_literal,
        )
        if not response.data or not response.data[0].b64_json:
            raise ValueError("OpenAI image edit returned empty result")
        return MediaFileType(f"data:image/png;base64,{response.data[0].b64_json}")

    async def run_model(
        self,
        api_key: SecretStr,
        model: ImageEditorModel,
        prompt: str,
        input_image_b64: Optional[str],
        aspect_ratio: str,
        seed: Optional[int],
        user_id: str,
        graph_exec_id: str,
    ) -> MediaFileType:
        if model.value.startswith("gpt-image"):
            return await self._edit_with_openai(
                api_key, model, prompt, input_image_b64, aspect_ratio
            )

        client = ReplicateClient(api_token=api_key.get_secret_value())
        model_name = model.api_name

        is_nano_banana = model in (
            ImageEditorModel.NANO_BANANA_PRO,
            ImageEditorModel.NANO_BANANA_2,
        )
        if is_nano_banana:
            input_params: dict = {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "output_format": "jpg",
                "safety_filter_level": "block_only_high",
            }
            if input_image_b64:
                input_params["image_input"] = [input_image_b64]
        else:
            input_params = {
                "prompt": prompt,
                "input_image": input_image_b64,
                "aspect_ratio": aspect_ratio,
                **({"seed": seed} if seed is not None else {}),
            }

        try:
            output: FileOutput | list[FileOutput] = await client.async_run(  # type: ignore
                model_name,
                input=input_params,
                wait=False,
            )
        except Exception as e:
            if "flagged as sensitive" in str(e).lower():
                raise ModerationError(
                    message="Content was flagged as sensitive by the model provider",
                    user_id=user_id,
                    graph_exec_id=graph_exec_id,
                    moderation_type="model_provider",
                )
            raise ValueError(f"Model execution failed: {e}") from e

        if isinstance(output, list) and output:
            output = output[0]

        if isinstance(output, FileOutput):
            return MediaFileType(output.url)
        if isinstance(output, str):
            return MediaFileType(output)

        raise ValueError("No output received")
