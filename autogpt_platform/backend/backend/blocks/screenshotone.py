from base64 import b64encode
from typing import Literal
from backend.data.block import Block, BlockCategory, BlockSchema, BlockOutput
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util.request import Requests
from pydantic import SecretStr


class ScreenshotOneBlock(Block):
    """Block for taking screenshots using ScreenshotOne API"""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.SCREENSHOTONE], Literal["api_key"]
        ] = CredentialsField(description="The ScreenshotOne API key")
        url: str = SchemaField(
            description="URL of the website to screenshot",
            placeholder="https://example.com",
        )
        viewport_width: int = SchemaField(
            description="Width of the viewport in pixels", default=1920
        )
        viewport_height: int = SchemaField(
            description="Height of the viewport in pixels", default=1080
        )
        full_page: bool = SchemaField(
            description="Whether to capture the full page length", default=False
        )
        format: str = SchemaField(
            description="Output format (png, jpeg, webp)", default="png"
        )
        block_ads: bool = SchemaField(description="Whether to block ads", default=True)
        block_cookie_banners: bool = SchemaField(
            description="Whether to block cookie banners", default=True
        )
        block_chats: bool = SchemaField(
            description="Whether to block chat widgets", default=True
        )
        cache: bool = SchemaField(
            description="Whether to enable caching", default=False
        )

    class Output(BlockSchema):
        image_data: bytes = SchemaField(description="The screenshot image data")
        content_type: str = SchemaField(description="The MIME type of the image")
        error: str = SchemaField(description="Error message if the screenshot failed")

    def __init__(self):
        super().__init__(
            id="3a7c4b8d-6e2f-4a5d-b9c1-f8d23c5a9b0e",  # Generated UUID
            description="Takes a screenshot of a specified website using ScreenshotOne API",
            categories={BlockCategory.DATA},
            input_schema=ScreenshotOneBlock.Input,
            output_schema=ScreenshotOneBlock.Output,
            test_input={
                "url": "https://example.com",
                "viewport_width": 1920,
                "viewport_height": 1080,
                "full_page": False,
                "format": "png",
                "block_ads": True,
                "block_cookie_banners": True,
                "block_chats": True,
                "cache": False,
                "credentials": {
                    "provider": "screenshotone",
                    "type": "api_key",
                    "id": "test-id",
                    "title": "Test API Key",
                },
            },
            test_credentials=APIKeyCredentials(
                id="test-id",
                provider="screenshotone",
                api_key=SecretStr("test-key"),
                title="Test API Key",
                expires_at=None,
            ),
            test_output=[
                ("image_data", b"test-image-data"),
                ("content_type", "image/png"),
            ],
            test_mock={
                "take_screenshot": lambda *args, **kwargs: {
                    "image_data": b"test-image-data",
                    "content_type": "image/png",
                }
            },
        )

    @staticmethod
    def take_screenshot(
        credentials: APIKeyCredentials,
        url: str,
        viewport_width: int,
        viewport_height: int,
        full_page: bool,
        format: str,
        block_ads: bool,
        block_cookie_banners: bool,
        block_chats: bool,
        cache: bool,
    ) -> dict:
        """
        Takes a screenshot using the ScreenshotOne API
        """
        api = Requests(trusted_origins=["https://api.screenshotone.com"])

        # Build API URL with parameters
        params = {
            "access_key": credentials.api_key.get_secret_value(),
            "url": url,
            "viewport_width": viewport_width,
            "viewport_height": viewport_height,
            "full_page": str(full_page).lower(),
            "format": format,
            "block_ads": str(block_ads).lower(),
            "block_cookie_banners": str(block_cookie_banners).lower(),
            "block_chats": str(block_chats).lower(),
            "cache": str(cache).lower(),
        }

        response = api.get("https://api.screenshotone.com/take", params=params)

        return {
            "image_data": b64encode(response.content).decode("utf-8"),
            "content_type": f"image/{format}",
        }

    def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            screenshot_data = self.take_screenshot(
                credentials=credentials,
                url=input_data.url,
                viewport_width=input_data.viewport_width,
                viewport_height=input_data.viewport_height,
                full_page=input_data.full_page,
                format=input_data.format,
                block_ads=input_data.block_ads,
                block_cookie_banners=input_data.block_cookie_banners,
                block_chats=input_data.block_chats,
                cache=input_data.cache,
            )
            yield "image_data", screenshot_data["image_data"]
            yield "content_type", screenshot_data["content_type"]
        except Exception as e:
            yield "error", str(e)
