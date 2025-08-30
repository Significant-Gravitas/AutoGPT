import uuid
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    pass

from pydantic import SecretStr

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    Requests,
    SchemaField,
)

from ._config import bannerbear

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="bannerbear",
    api_key=SecretStr("mock-bannerbear-api-key"),
    title="Mock Bannerbear API Key",
)


class TextModification(BlockSchema):
    name: str = SchemaField(
        description="The name of the layer to modify in the template"
    )
    text: str = SchemaField(description="The text content to add to this layer")
    color: str = SchemaField(
        description="Hex color code for the text (e.g., '#FF0000')",
        default="",
        advanced=True,
    )
    font_family: str = SchemaField(
        description="Font family to use for the text",
        default="",
        advanced=True,
    )
    font_size: int = SchemaField(
        description="Font size in pixels",
        default=0,
        advanced=True,
    )
    font_weight: str = SchemaField(
        description="Font weight (e.g., 'bold', 'normal')",
        default="",
        advanced=True,
    )
    text_align: str = SchemaField(
        description="Text alignment (left, center, right)",
        default="",
        advanced=True,
    )


class BannerbearTextOverlayBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = bannerbear.credentials_field(
            description="API credentials for Bannerbear"
        )
        template_id: str = SchemaField(
            description="The unique ID of your Bannerbear template"
        )
        project_id: str = SchemaField(
            description="Optional: Project ID (required when using Master API Key)",
            default="",
            advanced=True,
        )
        text_modifications: List[TextModification] = SchemaField(
            description="List of text layers to modify in the template"
        )
        image_url: str = SchemaField(
            description="Optional: URL of an image to use in the template",
            default="",
            advanced=True,
        )
        image_layer_name: str = SchemaField(
            description="Optional: Name of the image layer in the template",
            default="photo",
            advanced=True,
        )
        webhook_url: str = SchemaField(
            description="Optional: URL to receive webhook notification when image is ready",
            default="",
            advanced=True,
        )
        metadata: str = SchemaField(
            description="Optional: Custom metadata to attach to the image",
            default="",
            advanced=True,
        )

    class Output(BlockSchema):
        success: bool = SchemaField(
            description="Whether the image generation was successfully initiated"
        )
        image_url: str = SchemaField(
            description="URL of the generated image (if synchronous) or placeholder"
        )
        uid: str = SchemaField(description="Unique identifier for the generated image")
        status: str = SchemaField(description="Status of the image generation")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="c7d3a5c2-05fc-450e-8dce-3b0e04626009",
            description="Add text overlay to images using Bannerbear templates. Perfect for creating social media graphics, marketing materials, and dynamic image content.",
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.AI},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "template_id": "jJWBKNELpQPvbX5R93Gk",
                "text_modifications": [
                    {
                        "name": "headline",
                        "text": "Amazing Product Launch!",
                        "color": "#FF0000",
                    },
                    {
                        "name": "subtitle",
                        "text": "50% OFF Today Only",
                    },
                ],
                "credentials": {
                    "provider": "bannerbear",
                    "id": str(uuid.uuid4()),
                    "type": "api_key",
                },
            },
            test_output=[
                ("success", True),
                ("image_url", "https://cdn.bannerbear.com/test-image.jpg"),
                ("uid", "test-uid-123"),
                ("status", "completed"),
            ],
            test_mock={
                "_make_api_request": lambda *args, **kwargs: {
                    "uid": "test-uid-123",
                    "status": "completed",
                    "image_url": "https://cdn.bannerbear.com/test-image.jpg",
                }
            },
            test_credentials=TEST_CREDENTIALS,
        )

    async def _make_api_request(self, payload: dict, api_key: str) -> dict:
        """Make the actual API request to Bannerbear. This is separated for easy mocking in tests."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response = await Requests().post(
            "https://sync.api.bannerbear.com/v2/images",
            headers=headers,
            json=payload,
        )

        if response.status in [200, 201, 202]:
            return response.json()
        else:
            error_msg = f"API request failed with status {response.status}"
            if response.text:
                try:
                    error_data = response.json()
                    error_msg = (
                        f"{error_msg}: {error_data.get('message', response.text)}"
                    )
                except Exception:
                    error_msg = f"{error_msg}: {response.text}"
            raise Exception(error_msg)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        # Build the modifications array
        modifications = []

        # Add text modifications
        for text_mod in input_data.text_modifications:
            mod_data: Dict[str, Any] = {
                "name": text_mod.name,
                "text": text_mod.text,
            }

            # Add optional text styling parameters only if they have values
            if text_mod.color and text_mod.color.strip():
                mod_data["color"] = text_mod.color
            if text_mod.font_family and text_mod.font_family.strip():
                mod_data["font_family"] = text_mod.font_family
            if text_mod.font_size and text_mod.font_size > 0:
                mod_data["font_size"] = text_mod.font_size
            if text_mod.font_weight and text_mod.font_weight.strip():
                mod_data["font_weight"] = text_mod.font_weight
            if text_mod.text_align and text_mod.text_align.strip():
                mod_data["text_align"] = text_mod.text_align

            modifications.append(mod_data)

        # Add image modification if provided and not empty
        if input_data.image_url and input_data.image_url.strip():
            modifications.append(
                {
                    "name": input_data.image_layer_name,
                    "image_url": input_data.image_url,
                }
            )

        # Build the request payload - only include non-empty optional fields
        payload = {
            "template": input_data.template_id,
            "modifications": modifications,
        }

        # Add project_id if provided (required for Master API keys)
        if input_data.project_id and input_data.project_id.strip():
            payload["project_id"] = input_data.project_id

        if input_data.webhook_url and input_data.webhook_url.strip():
            payload["webhook_url"] = input_data.webhook_url
        if input_data.metadata and input_data.metadata.strip():
            payload["metadata"] = input_data.metadata

        # Make the API request using the private method
        data = await self._make_api_request(
            payload, credentials.api_key.get_secret_value()
        )

        # Synchronous request - image should be ready
        yield "success", True
        yield "image_url", data.get("image_url", "")
        yield "uid", data.get("uid", "")
        yield "status", data.get("status", "completed")
