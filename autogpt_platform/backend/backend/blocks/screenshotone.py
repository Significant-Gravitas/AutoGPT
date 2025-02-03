from base64 import b64encode
from enum import Enum
from typing import Literal

from pydantic import SecretStr

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util.file import MediaFile, store_media_file
from backend.util.request import Requests


class Format(str, Enum):
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"


class ScreenshotWebPageBlock(Block):
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
        format: Format = SchemaField(
            description="Output format (png, jpeg, webp)", default=Format.PNG
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
        image: MediaFile = SchemaField(description="The screenshot image data")
        error: str = SchemaField(description="Error message if the screenshot failed")

    def __init__(self):
        super().__init__(
            id="3a7c4b8d-6e2f-4a5d-b9c1-f8d23c5a9b0e",  # Generated UUID
            description="Takes a screenshot of a specified website using ScreenshotOne API",
            categories={BlockCategory.DATA},
            input_schema=ScreenshotWebPageBlock.Input,
            output_schema=ScreenshotWebPageBlock.Output,
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
                (
                    "image",
                    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAB5JREFUOE9jZPjP8J+BAsA4agDDaBgwjIYBw7AIAwCV5B/xAsMbygAAAABJRU5ErkJggg==",
                ),
            ],
            test_mock={
                "take_screenshot": lambda *args, **kwargs: {
                    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAB5JREFUOE9jZPjP8J+BAsA4agDDaBgwjIYBw7AIAwCV5B/xAsMbygAAAABJRU5ErkJggg==",
                }
            },
        )

    @staticmethod
    def take_screenshot(
        credentials: APIKeyCredentials,
        graph_exec_id: str,
        url: str,
        viewport_width: int,
        viewport_height: int,
        full_page: bool,
        format: Format,
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
            "format": format.value,
            "block_ads": str(block_ads).lower(),
            "block_cookie_banners": str(block_cookie_banners).lower(),
            "block_chats": str(block_chats).lower(),
            "cache": str(cache).lower(),
        }

        response = api.get("https://api.screenshotone.com/take", params=params)

        return {
            "image": store_media_file(
                graph_exec_id=graph_exec_id,
                file=f"data:image/{format.value};base64,{b64encode(response.content).decode('utf-8')}",
                return_content=True,
            )
        }

    def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        graph_exec_id: str,
        **kwargs,
    ) -> BlockOutput:
        try:
            screenshot_data = self.take_screenshot(
                credentials=credentials,
                graph_exec_id=graph_exec_id,
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
            yield "image", screenshot_data["image"]
        except Exception as e:
            yield "error", str(e)
