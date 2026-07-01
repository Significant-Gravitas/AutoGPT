"""
Instagram Comment Block for AutoGPT Platform.
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


class InstagramCommentBlock(Block):
    """
    Comments on an Instagram post.

    This block adds a comment to a post using either the post URL or media ID.
    """

    class Input(BlockSchemaInput):
        credentials: InstagramCredentialsInput = InstagramCredentialsField()

        media_id: str = SchemaField(
            description="Instagram media ID or post URL to comment on",
            placeholder="1234567890123456789 or https://www.instagram.com/p/ABC123/",
        )

        comment_text: str = SchemaField(
            description="Comment text to post (max 2,200 characters)",
            placeholder="Nice post! 👍",
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(
            description="Whether the comment was posted successfully"
        )
        comment_id: str = SchemaField(description="ID of the posted comment")
        error: str = SchemaField(
            description="Error message if commenting failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="b8c9d0e1-f2a3-4567-b3c4-678901234567",
            description="Comment on an Instagram post by media ID or URL",
            categories={BlockCategory.SOCIAL},
            input_schema=InstagramCommentBlock.Input,
            output_schema=InstagramCommentBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "media_id": "1234567890123456789",
                "comment_text": "Great post! 🔥",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
                ("comment_id", "17890123456789012"),
            ],
            test_mock={
                "comment_post": lambda *args, **kwargs: (
                    True,
                    "17890123456789012",
                    None,
                )
            },
        )

    @staticmethod
    def comment_post(
        credentials: InstagramCredentials, media_id: str, comment_text: str
    ):
        """Comment on an Instagram post."""
        try:
            api_key = credentials.api_key.get_secret_value()
            if ":" not in api_key:
                return False, None, "Invalid credentials format"

            username, password = api_key.split(":", 1)

            client = Client()
            client.login(username, password)

            # Validate comment length
            if len(comment_text) > 2200:
                return False, None, f"Comment too long ({len(comment_text)}/2200 chars)"

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
                    return False, None, "Invalid URL: must be an Instagram URL"
                media_id = client.media_pk_from_url(media_id)

            # Post the comment
            comment = client.media_comment(media_id, comment_text)
            comment_id = str(comment.pk)

            return True, comment_id, None

        except Exception as e:
            return False, None, f"Failed to comment on post: {str(e)}"

    async def run(
        self,
        input_data: Input,
        *,
        credentials: InstagramCredentials,
        **kwargs,
    ) -> BlockOutput:
        """Execute the comment action."""
        success, comment_id, error = self.comment_post(
            credentials,
            input_data.media_id,
            input_data.comment_text,
        )

        yield "success", success
        if comment_id:
            yield "comment_id", comment_id
        if error:
            yield "error", error
