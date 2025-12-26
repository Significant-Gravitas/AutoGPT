"""
Instagram Search Blocks for AutoGPT Platform.
"""

from instagrapi import Client
from instagrapi.types import Media

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


class InstagramGetUserInfoBlock(Block):
    """
    Gets information about an Instagram user.

    This block retrieves detailed information about a user including
    follower count, biography, profile picture, and more.
    """

    class Input(BlockSchemaInput):
        credentials: InstagramCredentialsInput = InstagramCredentialsField()

        username: str = SchemaField(
            description="Instagram username to get information about",
            placeholder="autogpt",
        )

    class Output(BlockSchemaOutput):
        user_id: str = SchemaField(description="User's Instagram ID")
        username: str = SchemaField(description="User's username")
        full_name: str = SchemaField(description="User's full name")
        biography: str = SchemaField(description="User's biography/bio")
        follower_count: int = SchemaField(description="Number of followers")
        following_count: int = SchemaField(description="Number of accounts following")
        media_count: int = SchemaField(description="Number of posts")
        is_private: bool = SchemaField(description="Whether the account is private")
        is_verified: bool = SchemaField(description="Whether the account is verified")
        profile_pic_url: str = SchemaField(description="URL to profile picture")
        error: str = SchemaField(
            description="Error message if request failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="c9d0e1f2-a3b4-5678-c4d5-789012345678",
            description="Get detailed information about an Instagram user by username",
            categories={BlockCategory.SOCIAL},
            input_schema=InstagramGetUserInfoBlock.Input,
            output_schema=InstagramGetUserInfoBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "username": "autogpt",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("user_id", "123456789"),
                ("username", "autogpt"),
                ("full_name", "AutoGPT"),
                ("biography", "Official AutoGPT account"),
                ("follower_count", 10000),
                ("following_count", 100),
                ("media_count", 50),
                ("is_private", False),
                ("is_verified", True),
                ("profile_pic_url", "https://example.com/profile.jpg"),
            ],
            test_mock={
                "get_user_info": lambda *args, **kwargs: (
                    "123456789",
                    "autogpt",
                    "AutoGPT",
                    "Official AutoGPT account",
                    10000,
                    100,
                    50,
                    False,
                    True,
                    "https://example.com/profile.jpg",
                    None,
                )
            },
        )

    @staticmethod
    def get_user_info(credentials: InstagramCredentials, username: str):
        """Get Instagram user information."""
        try:
            api_key = credentials.api_key.get_secret_value()
            if ":" not in api_key:
                return (
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "Invalid credentials format",
                )

            user, pwd = api_key.split(":", 1)

            client = Client()
            client.login(user, pwd)

            # Get user information
            user_info = client.user_info_by_username(username)

            return (
                str(user_info.pk),
                user_info.username,
                user_info.full_name or "",
                user_info.biography or "",
                user_info.follower_count,
                user_info.following_count,
                user_info.media_count,
                user_info.is_private,
                user_info.is_verified,
                str(user_info.profile_pic_url),
                None,
            )

        except Exception as e:
            return (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                f"Failed to get user info: {str(e)}",
            )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: InstagramCredentials,
        **kwargs,
    ) -> BlockOutput:
        """Execute the user info retrieval."""
        (
            user_id,
            username,
            full_name,
            biography,
            follower_count,
            following_count,
            media_count,
            is_private,
            is_verified,
            profile_pic_url,
            error,
        ) = self.get_user_info(credentials, input_data.username)

        if error:
            yield "error", error
        else:
            if user_id:
                yield "user_id", user_id
            if username:
                yield "username", username
            if full_name:
                yield "full_name", full_name
            if biography:
                yield "biography", biography
            if follower_count is not None:
                yield "follower_count", follower_count
            if following_count is not None:
                yield "following_count", following_count
            if media_count is not None:
                yield "media_count", media_count
            if is_private is not None:
                yield "is_private", is_private
            if is_verified is not None:
                yield "is_verified", is_verified
            if profile_pic_url:
                yield "profile_pic_url", profile_pic_url


class InstagramSearchHashtagBlock(Block):
    """
    Searches for posts by hashtag on Instagram.

    This block searches for recent posts with a specific hashtag
    and returns information about the top posts.
    """

    class Input(BlockSchemaInput):
        credentials: InstagramCredentialsInput = InstagramCredentialsField()

        hashtag: str = SchemaField(
            description="Hashtag to search for (without the # symbol)",
            placeholder="autogpt",
        )

        amount: int = SchemaField(
            description="Number of posts to retrieve (max 50)",
            default=10,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        post_ids: list[str] = SchemaField(description="List of media IDs found")
        post_urls: list[str] = SchemaField(description="List of post URLs")
        captions: list[str] = SchemaField(description="List of post captions")
        like_counts: list[int] = SchemaField(
            description="List of like counts for each post"
        )
        error: str = SchemaField(
            description="Error message if search failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="d0e1f2a3-b4c5-6789-d5e6-890123456789",
            description="Search Instagram posts by hashtag and retrieve top posts",
            categories={BlockCategory.SOCIAL},
            input_schema=InstagramSearchHashtagBlock.Input,
            output_schema=InstagramSearchHashtagBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "hashtag": "autogpt",
                "amount": 5,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("post_ids", ["123", "456", "789"]),
                (
                    "post_urls",
                    [
                        "https://www.instagram.com/p/ABC1/",
                        "https://www.instagram.com/p/ABC2/",
                        "https://www.instagram.com/p/ABC3/",
                    ],
                ),
                ("captions", ["Caption 1", "Caption 2", "Caption 3"]),
                ("like_counts", [100, 200, 300]),
            ],
            test_mock={
                "search_hashtag": lambda *args, **kwargs: (
                    ["123", "456", "789"],
                    [
                        "https://www.instagram.com/p/ABC1/",
                        "https://www.instagram.com/p/ABC2/",
                        "https://www.instagram.com/p/ABC3/",
                    ],
                    ["Caption 1", "Caption 2", "Caption 3"],
                    [100, 200, 300],
                    None,
                )
            },
        )

    @staticmethod
    def search_hashtag(
        credentials: InstagramCredentials, hashtag: str, amount: int = 10
    ):
        """Search Instagram posts by hashtag."""
        try:
            api_key = credentials.api_key.get_secret_value()
            if ":" not in api_key:
                return None, None, None, None, "Invalid credentials format"

            username, password = api_key.split(":", 1)

            client = Client()
            client.login(username, password)

            # Validate amount
            if amount > 50:
                amount = 50
            if amount < 1:
                amount = 1

            # Remove # if user included it
            hashtag = hashtag.lstrip("#")

            # Search for posts with hashtag
            medias: list[Media] = client.hashtag_medias_recent(hashtag, amount)

            post_ids = [str(media.pk) for media in medias]
            post_urls = [
                f"https://www.instagram.com/p/{media.code}/" for media in medias
            ]
            captions = [media.caption_text or "" for media in medias]
            like_counts = [media.like_count for media in medias]

            return post_ids, post_urls, captions, like_counts, None

        except Exception as e:
            return None, None, None, None, f"Failed to search hashtag: {str(e)}"

    async def run(
        self,
        input_data: Input,
        *,
        credentials: InstagramCredentials,
        **kwargs,
    ) -> BlockOutput:
        """Execute the hashtag search."""
        post_ids, post_urls, captions, like_counts, error = self.search_hashtag(
            credentials,
            input_data.hashtag,
            input_data.amount,
        )

        if error:
            yield "error", error
        else:
            if post_ids:
                yield "post_ids", post_ids
            if post_urls:
                yield "post_urls", post_urls
            if captions:
                yield "captions", captions
            if like_counts:
                yield "like_counts", like_counts
