"""
Instagram Like Blocks for AutoGPT Platform.
"""

from instagrapi import Client

from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField

from .auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    InstagramCredentials,
    InstagramCredentialsField,
    InstagramCredentialsInput,
)


class InstagramLikePostBlock(Block):
    """
    Likes an Instagram post.

    This block likes a post using either the post URL or media ID.
    """

    class Input(BlockSchemaInput):
        credentials: InstagramCredentialsInput = InstagramCredentialsField()

        media_id: str = SchemaField(
            description="Instagram media ID or post URL to like",
            placeholder="1234567890123456789 or https://www.instagram.com/p/ABC123/",
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(description="Whether the like was successful")
        error: str = SchemaField(
            description="Error message if liking failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="d4e5f6a7-b8c9-0123-def0-234567890123",
            description="Like an Instagram post by media ID or URL",
            categories={BlockCategory.SOCIAL},
            input_schema=InstagramLikePostBlock.Input,
            output_schema=InstagramLikePostBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "media_id": "1234567890123456789",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
            ],
            test_mock={"like_post": lambda *args, **kwargs: (True, None)},
        )

    @staticmethod
    def like_post(credentials: InstagramCredentials, media_id: str):
        """Like an Instagram post."""
        try:
            api_key = credentials.api_key.get_secret_value()
            if ":" not in api_key:
                return False, "Invalid credentials format"

            username, password = api_key.split(":", 1)

            client = Client()
            client.login(username, password)

            # Handle both media ID and URL
            if media_id.startswith("http://") or media_id.startswith("https://"):
                # Validate it's actually an Instagram URL
                from urllib.parse import urlparse

                parsed = urlparse(media_id)
                if parsed.netloc not in (
                    "instagram.com",
                    "www.instagram.com",
                    "m.instagram.com",
                ):
                    return False, "Invalid URL: must be an Instagram URL"
                media_id = client.media_pk_from_url(media_id)

            client.media_like(media_id)
            return True, None

        except Exception as e:
            return False, f"Failed to like post: {str(e)}"

    async def run(
        self,
        input_data: Input,
        *,
        credentials: InstagramCredentials,
        **kwargs,
    ) -> BlockOutput:
        """Execute the like action."""
        success, error = self.like_post(credentials, input_data.media_id)

        yield "success", success
        if error:
            yield "error", error


class InstagramUnlikePostBlock(Block):
    """
    Unlikes an Instagram post.

    This block removes a like from a post using either the post URL or media ID.
    """

    class Input(BlockSchemaInput):
        credentials: InstagramCredentialsInput = InstagramCredentialsField()

        media_id: str = SchemaField(
            description="Instagram media ID or post URL to unlike",
            placeholder="1234567890123456789 or https://www.instagram.com/p/ABC123/",
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(description="Whether the unlike was successful")
        error: str = SchemaField(
            description="Error message if unliking failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="e5f6a7b8-c9d0-1234-e0f1-345678901234",
            description="Unlike an Instagram post by media ID or URL",
            categories={BlockCategory.SOCIAL},
            input_schema=InstagramUnlikePostBlock.Input,
            output_schema=InstagramUnlikePostBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "media_id": "1234567890123456789",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
            ],
            test_mock={"unlike_post": lambda *args, **kwargs: (True, None)},
        )

    @staticmethod
    def unlike_post(credentials: InstagramCredentials, media_id: str):
        """Unlike an Instagram post."""
        try:
            api_key = credentials.api_key.get_secret_value()
            if ":" not in api_key:
                return False, "Invalid credentials format"

            username, password = api_key.split(":", 1)

            client = Client()
            client.login(username, password)

            if media_id.startswith("http://") or media_id.startswith("https://"):
                # Validate it's actually an Instagram URL
                from urllib.parse import urlparse

                parsed = urlparse(media_id)
                if parsed.netloc not in (
                    "instagram.com",
                    "www.instagram.com",
                    "m.instagram.com",
                ):
                    return False, "Invalid URL: must be an Instagram URL"
                media_id = client.media_pk_from_url(media_id)

            client.media_unlike(media_id)
            return True, None

        except Exception as e:
            return False, f"Failed to unlike post: {str(e)}"

    async def run(
        self,
        input_data: Input,
        *,
        credentials: InstagramCredentials,
        **kwargs,
    ) -> BlockOutput:
        """Execute the unlike action."""
        success, error = self.unlike_post(credentials, input_data.media_id)

        yield "success", success
        if error:
            yield "error", error
