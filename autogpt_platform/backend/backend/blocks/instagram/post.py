"""
Instagram Post Blocks for AutoGPT Platform.
"""

from instagrapi import Client
from instagrapi.types import Media

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchemaInput, BlockSchemaOutput
from backend.data.model import SchemaField

from .auth import (
    InstagramCredentials,
    InstagramCredentialsField,
    InstagramCredentialsInput,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
)


class InstagramPostPhotoBlock(Block):
    """
    Posts a photo to Instagram with a caption.

    This block uploads a photo to Instagram along with an optional caption.
    The photo can be provided as a URL or local file path.
    """

    class Input(BlockSchemaInput):
        credentials: InstagramCredentialsInput = InstagramCredentialsField()

        photo_url: str = SchemaField(
            description="URL or local path to the photo to post",
            placeholder="https://example.com/photo.jpg or /path/to/photo.jpg",
        )

        caption: str = SchemaField(
            description="Caption for the Instagram post (max 2,200 characters)",
            placeholder="Your caption here #hashtag",
            default="",
        )

        location_name: str | None = SchemaField(
            description="Optional location name to tag in the post",
            placeholder="New York, USA",
            default=None,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(
            description="Whether the post was successful"
        )
        media_id: str = SchemaField(
            description="Instagram media ID of the posted photo"
        )
        media_code: str = SchemaField(
            description="Instagram media code (short code in URL)"
        )
        post_url: str = SchemaField(
            description="URL to the Instagram post"
        )
        error: str = SchemaField(
            description="Error message if posting failed",
            default=""
        )

    def __init__(self):
        super().__init__(
            id="b2c3d4e5-f6a7-8901-bcde-f12345678901",
            description="Post a photo to Instagram with caption and optional location",
            categories={BlockCategory.SOCIAL},
            input_schema=InstagramPostPhotoBlock.Input,
            output_schema=InstagramPostPhotoBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "photo_url": "https://example.com/test.jpg",
                "caption": "Test post from AutoGPT! #automation",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
                ("media_id", "1234567890123456789"),
                ("media_code", "ABC123def"),
                ("post_url", "https://www.instagram.com/p/ABC123def/"),
            ],
            test_mock={
                "post_photo": lambda *args, **kwargs: (
                    True,
                    "1234567890123456789",
                    "ABC123def",
                    "https://www.instagram.com/p/ABC123def/",
                    None,
                )
            },
        )

    @staticmethod
    def post_photo(
        credentials: InstagramCredentials,
        photo_url: str,
        caption: str,
        location_name: str | None = None,
    ):
        """
        Post a photo to Instagram.

        Args:
            credentials: Instagram credentials
            photo_url: URL or path to photo
            caption: Post caption
            location_name: Optional location name

        Returns:
            Tuple of (success, media_id, media_code, post_url, error)
        """
        try:
            # Extract username and password
            api_key = credentials.api_key.get_secret_value()
            if ":" not in api_key:
                return False, None, None, None, "Invalid credentials format"

            username, password = api_key.split(":", 1)

            # Login to Instagram
            client = Client()
            client.login(username, password)

            # Validate caption length
            if len(caption) > 2200:
                return False, None, None, None, f"Caption too long ({len(caption)}/2200 chars)"

            # Post the photo
            media: Media = client.photo_upload(
                photo_url,
                caption=caption,
            )

            # Get media details
            media_id = str(media.pk)
            media_code = media.code
            post_url = f"https://www.instagram.com/p/{media_code}/"

            return True, media_id, media_code, post_url, None

        except Exception as e:
            return False, None, None, None, f"Failed to post photo: {str(e)}"

    async def run(
        self,
        input_data: Input,
        *,
        credentials: InstagramCredentials,
        **kwargs,
    ) -> BlockOutput:
        """Execute the photo post."""
        success, media_id, media_code, post_url, error = self.post_photo(
            credentials,
            input_data.photo_url,
            input_data.caption,
            input_data.location_name,
        )

        if success:
            yield "success", True
            if media_id:
                yield "media_id", media_id
            if media_code:
                yield "media_code", media_code
            if post_url:
                yield "post_url", post_url
        else:
            yield "success", False
            yield "error", error or "Unknown error occurred"


class InstagramPostReelBlock(Block):
    """
    Posts a video reel to Instagram.

    This block uploads a video to Instagram as a Reel with an optional caption.
    """

    class Input(BlockSchemaInput):
        credentials: InstagramCredentialsInput = InstagramCredentialsField()

        video_url: str = SchemaField(
            description="URL or local path to the video to post as a Reel",
            placeholder="https://example.com/video.mp4",
        )

        caption: str = SchemaField(
            description="Caption for the Instagram Reel (max 2,200 characters)",
            placeholder="Your caption here #reels",
            default="",
        )

        thumbnail_url: str | None = SchemaField(
            description="Optional thumbnail image URL for the Reel",
            default=None,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(description="Whether the reel was posted successfully")
        media_id: str = SchemaField(description="Instagram media ID of the posted reel")
        media_code: str = SchemaField(description="Instagram media code")
        post_url: str = SchemaField(description="URL to the Instagram reel")
        error: str = SchemaField(description="Error message if posting failed", default="")

    def __init__(self):
        super().__init__(
            id="c3d4e5f6-a7b8-9012-cdef-123456789012",
            description="Post a video Reel to Instagram with caption and optional thumbnail",
            categories={BlockCategory.SOCIAL},
            input_schema=InstagramPostReelBlock.Input,
            output_schema=InstagramPostReelBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "video_url": "https://example.com/test.mp4",
                "caption": "Test reel from AutoGPT! #reels #automation",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
                ("media_id", "9876543210987654321"),
                ("media_code", "XYZ987xyz"),
                ("post_url", "https://www.instagram.com/p/XYZ987xyz/"),
            ],
            test_mock={
                "post_reel": lambda *args, **kwargs: (
                    True,
                    "9876543210987654321",
                    "XYZ987xyz",
                    "https://www.instagram.com/p/XYZ987xyz/",
                    None,
                )
            },
        )

    @staticmethod
    def post_reel(
        credentials: InstagramCredentials,
        video_url: str,
        caption: str,
        thumbnail_url: str | None = None,
    ):
        """Post a reel to Instagram."""
        try:
            api_key = credentials.api_key.get_secret_value()
            if ":" not in api_key:
                return False, None, None, None, "Invalid credentials format"

            username, password = api_key.split(":", 1)

            client = Client()
            client.login(username, password)

            if len(caption) > 2200:
                return False, None, None, None, f"Caption too long ({len(caption)}/2200 chars)"

            # Post the reel
            media: Media = client.clip_upload(
                video_url,
                caption=caption,
                thumbnail=thumbnail_url,
            )

            media_id = str(media.pk)
            media_code = media.code
            post_url = f"https://www.instagram.com/p/{media_code}/"

            return True, media_id, media_code, post_url, None

        except Exception as e:
            return False, None, None, None, f"Failed to post reel: {str(e)}"

    async def run(
        self,
        input_data: Input,
        *,
        credentials: InstagramCredentials,
        **kwargs,
    ) -> BlockOutput:
        """Execute the reel post."""
        success, media_id, media_code, post_url, error = self.post_reel(
            credentials,
            input_data.video_url,
            input_data.caption,
            input_data.thumbnail_url,
        )

        if success:
            yield "success", True
            if media_id:
                yield "media_id", media_id
            if media_code:
                yield "media_code", media_code
            if post_url:
                yield "post_url", post_url
        else:
            yield "success", False
            yield "error", error or "Unknown error occurred"
