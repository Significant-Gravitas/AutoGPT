from enum import Enum
from typing import Literal, Optional

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
    "title": TEST_CREDENTIALS.type,
}


class FluxKontextModelName(str, Enum):
    PRO = "Flux Kontext Pro"
    MAX = "Flux Kontext Max"

    @property
    def api_name(self) -> str:
        return f"black-forest-labs/flux-kontext-{self.name.lower()}"


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


class AIImageEditorBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.REPLICATE], Literal["api_key"]
        ] = CredentialsField(
            description="Replicate API key with permissions for Flux Kontext models",
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
            description="Random seed. Set for reproducible generation",
            default=None,
            title="Seed",
            advanced=True,
        )
        model: FluxKontextModelName = SchemaField(
            description="Model variant to use",
            default=FluxKontextModelName.PRO,
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
                "Edit images using BlackForest Labs' Flux Kontext models. Provide a prompt "
                "and optional reference image to generate a modified image."
            ),
            categories={BlockCategory.AI, BlockCategory.MULTIMEDIA},
            input_schema=AIImageEditorBlock.Input,
            output_schema=AIImageEditorBlock.Output,
            test_input={
                "prompt": "Add a hat to the cat",
                "input_image": "data:image/png;base64,MQ==",
                "aspect_ratio": AspectRatio.MATCH_INPUT_IMAGE,
                "seed": None,
                "model": FluxKontextModelName.PRO,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                # Output will be a workspace ref or data URI depending on context
                ("output_image", lambda x: x.startswith(("workspace://", "data:"))),
            ],
            test_mock={
                # Use data URI to avoid HTTP requests during tests
                "run_model": lambda *args, **kwargs: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
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
            model_name=input_data.model.api_name,
            prompt=input_data.prompt,
            input_image_b64=(
                await store_media_file(
                    file=input_data.input_image,
                    execution_context=execution_context,
                    return_format="for_external_api",  # Get content for Replicate API
                )
                if input_data.input_image
                else None
            ),
            aspect_ratio=input_data.aspect_ratio.value,
            seed=input_data.seed,
            user_id=execution_context.user_id or "",
            graph_exec_id=execution_context.graph_exec_id or "",
        )
        # Store the generated image to the user's workspace for persistence
        stored_url = await store_media_file(
            file=result,
            execution_context=execution_context,
            return_format="for_block_output",
        )
        yield "output_image", stored_url

    async def run_model(
        self,
        api_key: SecretStr,
        model_name: str,
        prompt: str,
        input_image_b64: Optional[str],
        aspect_ratio: str,
        seed: Optional[int],
        user_id: str,
        graph_exec_id: str,
    ) -> MediaFileType:
        client = ReplicateClient(api_token=api_key.get_secret_value())
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
