import logging
from typing import List, Optional, Union

from pydantic import BaseModel, Field
from datetime import datetime
from backend.blocks.aryshare._api import (
    AutoHashtag,
    AutoRepost,
    AutoSchedule,
    AyrshareClient,
    FirstComment,
    SocialPlatform,
)
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.integrations.credentials_store import IntegrationCredentialsStore

logger = logging.getLogger(__name__)

creads_store = IntegrationCredentialsStore()


class RequestOutput(BaseModel):
    """Base output model for Ayrshare social media posts."""

    status: str = Field(..., description="Status of the post")
    id: str = Field(..., description="ID of the post")
    refId: str = Field(..., description="Reference ID of the post")
    profileTitle: str = Field(..., description="Title of the profile")
    post: str = Field(..., description="The post text")
    postIds: Optional[List[dict]] = Field(
        description="IDs of the posts on each platform"
    )
    scheduleDate: Optional[str] = Field(description="Scheduled date of the post")
    errors: Optional[List[str]] = Field(description="Any errors that occurred")


class BaseAyrsharePostBlock(Block):
    """Base class for Ayrshare social media posting blocks."""

    class Input(BlockSchema):
        """Base input model for Ayrshare social media posts."""

        post: str = SchemaField(
            description="The post text to be published", default="", advanced=False
        )
        media_urls: List[str] = SchemaField(
            description="Optional list of media URLs to include. Set is_video in advanced settings to true if you want to upload videos.",
            default_factory=list,
            advanced=False,
        )
        is_video: bool = SchemaField(
            description="Whether the media is a video", default=False, advanced=True
        )
        schedule_date: Optional[datetime] = SchemaField(
            description="UTC datetime for scheduling (YYYY-MM-DDThh:mm:ssZ)",
            default=None,
            advanced=True,
        )
        disable_comments: bool = SchemaField(
            description="Whether to disable comments", default=False, advanced=True
        )
        shorten_links: bool = SchemaField(
            description="Whether to shorten links", default=False, advanced=True
        )

        unsplash: Optional[str] = SchemaField(
            description="Unsplash image configuration", default=None, advanced=True
        )
        requires_approval: bool = SchemaField(
            description="Whether to enable approval workflow",
            default=False,
            advanced=True,
        )
        random_post: bool= SchemaField(
            description="Whether to generate random post text",
            default=False,
            advanced=True,
        )
        random_media_url: bool = SchemaField(
            description="Whether to generate random media", default=False, advanced=True
        )
        notes: Optional[str] = SchemaField(
            description="Additional notes for the post", default=None, advanced=True
        )

    class Output(BlockSchema):
        post_result: RequestOutput = SchemaField(description="The result of the post")

    def __init__(
        self,
        id="b3a7b3b9-5169-410a-9d5c-fd625460fb14",
        description="Ayrshare Post",
        test_output=[
            (
                "post_result",
                RequestOutput(
                    status="success",
                    id="12345",
                    refId="abc123",
                    profileTitle="Test Profile",
                    post="Hello, world! This is a test post.",
                    postIds=[{"platform": "facebook", "id": "fb_123456"}],
                    scheduleDate=None,
                    errors=None,
                ),
            ),
        ],
    ):
        super().__init__(
            # The unique identifier for the block, this value will be persisted in the DB.
            # It should be unique and constant across the application run.
            # Use the UUID format for the ID.
            id=id,
            # The description of the block, explaining what the block does.
            description=description,
            # The set of categories that the block belongs to.
            # Each category is an instance of BlockCategory Enum.
            categories={BlockCategory.SOCIAL},
            # The schema, defined as a Pydantic model, for the input data.
            input_schema=BaseAyrsharePostBlock.Input,
            # The schema, defined as a Pydantic model, for the output data.
            output_schema=BaseAyrsharePostBlock.Output,
            # This is an instance of the Input schema with sample values.
            test_input={
                "post": "Hello, world! This is a test post.",
                "media_urls": ["https://example.com/image.jpg"],
                "is_video": False,
            },
            # The list or single expected output if the test_input is run.
            # Each output is a tuple of (output_name, output_data).
            test_output=test_output,
            # Function names on the block implementation to mock on test run.
            # Each mock is a dictionary with function names as keys and mock implementations as values.
            test_mock={
                "_create_post": lambda *args, **kwargs: RequestOutput(
                    status="success",
                    id="12345",
                    refId="abc123",
                    profileTitle="Test Profile",
                    post="Hello, world! This is a test post.",
                    postIds=[{"platform": "facebook", "id": "fb_123456"}],
                    scheduleDate=None,
                    errors=None,
                ),
                "_get_profile_key": lambda user_id: ("profile_key", "mock_profile_key"),
            },
        )

    @staticmethod
    def create_client():
        return AyrshareClient()

    def _create_post(
        self,
        input_data: "BaseAyrsharePostBlock.Input",
        platforms: List[SocialPlatform],
        profile_key: Optional[str] = None,
    ) -> RequestOutput:
        client = self.create_client()
        """Create a post on the specified platforms."""
        iso_date = input_data.schedule_date.isoformat() if input_data.schedule_date else None
        response = client.create_post(
            post=input_data.post,
            platforms=platforms,
            media_urls=input_data.media_urls,
            is_video=input_data.is_video,
            schedule_date=iso_date,
            disable_comments=input_data.disable_comments,
            shorten_links=input_data.shorten_links,
            unsplash=input_data.unsplash,
            requires_approval=input_data.requires_approval,
            random_post=input_data.random_post,
            random_media_url=input_data.random_media_url,
            notes=input_data.notes,
            profile_key=profile_key,
        )
        return RequestOutput(**response.__dict__)

    def _get_profile_key(self, user_id: str) -> tuple[str, str]:
        creds_store = IntegrationCredentialsStore()
        profile_key = creds_store.get_ayrshare_profile_key(user_id)
        if profile_key:
            return "profile_key", profile_key.get_secret_value()
        else:
            return (
                "error",
                "You need to connect your social media profile to Ayrshare first.",
            )

    def run(
        self,
        input_data: "BaseAyrsharePostBlock.Input",
        **kwargs,
    ) -> BlockOutput:
        """Run the block."""
        platforms = [SocialPlatform.FACEBOOK]

        yield "post_result", self._create_post(input_data, platforms=platforms)


class PostToFacebookBlock(BaseAyrsharePostBlock):
    """Block for posting to Facebook."""

    def __init__(self):
        super().__init__(
            id="3352f512-3524-49ed-a08f-003042da2fc1",
            description="Post to Facebook using Ayrshare",
        )

    def run(
        self,
        input_data: BaseAyrsharePostBlock.Input,
        *,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        """Post to Facebook."""
        profile_key, profile_key_value = self._get_profile_key(user_id)
        if profile_key == "error":
            yield "error", profile_key_value
            return

        post_result = self._create_post(
            input_data,
            [SocialPlatform.FACEBOOK],
            profile_key=profile_key_value,
        )
        yield "post_result", post_result


class PostToXBlock(BaseAyrsharePostBlock):
    """Block for posting to X / Twitter."""

    def __init__(self):
        super().__init__(
            id="9e8f844e-b4a5-4b25-80f2-9e1dd7d67625",
            description="Post to X / Twitter using Ayrshare",
        )

    def run(
        self,
        input_data: BaseAyrsharePostBlock.Input,
        *,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        """Post to Twitter."""
        profile_key, profile_key_value = self._get_profile_key(user_id)
        if profile_key == "error":
            yield "error", profile_key_value
            return

        post_result = self._create_post(
            input_data,
            [SocialPlatform.TWITTER],
            profile_key=profile_key_value,
        )
        yield "post_result", post_result


class PostToLinkedInBlock(BaseAyrsharePostBlock):
    """Block for posting to LinkedIn."""

    def __init__(self):
        super().__init__(
            id="589af4e4-507f-42fd-b9ac-a67ecef25811",
            description="Post to LinkedIn using Ayrshare",
        )

    def run(
        self,
        input_data: BaseAyrsharePostBlock.Input,
        *,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        """Post to LinkedIn."""
        profile_key, profile_key_value = self._get_profile_key(user_id)
        if profile_key == "error":
            yield "error", profile_key_value
            return

        post_result = self._create_post(
            input_data,
            [SocialPlatform.LINKEDIN],
            profile_key=profile_key_value,
        )
        yield "post_result", post_result


class PostToInstagramBlock(BaseAyrsharePostBlock):
    """Block for posting to Instagram."""

    def __init__(self):
        super().__init__(
            id="89b02b96-a7cb-46f4-9900-c48b32fe1552",
            description="Post to Instagram using Ayrshare",
        )

    def run(
        self,
        input_data: BaseAyrsharePostBlock.Input,
        *,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        """Post to Instagram."""
        profile_key, profile_key_value = self._get_profile_key(user_id)
        if profile_key == "error":
            yield "error", profile_key_value
            return

        post_result = self._create_post(
            input_data,
            [SocialPlatform.INSTAGRAM],
            profile_key=profile_key_value,
        )
        yield "post_result", post_result


class PostToYouTubeBlock(BaseAyrsharePostBlock):
    """Block for posting to YouTube."""

    def __init__(self):
        super().__init__(
            id="0082d712-ff1b-4c3d-8a8d-6c7721883b83",
            description="Post to YouTube using Ayrshare",
        )

    def run(
        self,
        input_data: BaseAyrsharePostBlock.Input,
        *,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        """Post to YouTube."""
        profile_key, profile_key_value = self._get_profile_key(user_id)
        if profile_key == "error":
            yield "error", profile_key_value
            return

        post_result = self._create_post(
            input_data,
            [SocialPlatform.YOUTUBE],
            profile_key=profile_key_value,
        )
        yield "post_result", post_result


class PostToRedditBlock(BaseAyrsharePostBlock):
    """Block for posting to Reddit."""

    def __init__(self):
        super().__init__(
            id="c7733580-3c72-483e-8e47-a8d58754d853",
            description="Post to Reddit using Ayrshare",
        )

    def run(
        self,
        input_data: BaseAyrsharePostBlock.Input,
        *,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        """Post to Reddit."""
        profile_key, profile_key_value = self._get_profile_key(user_id)
        if profile_key == "error":
            yield "error", profile_key_value
            return

        post_result = self._create_post(
            input_data,
            [SocialPlatform.REDDIT],
            profile_key=profile_key_value,
        )
        yield "post_result", post_result


class PostToTelegramBlock(BaseAyrsharePostBlock):
    """Block for posting to Telegram."""

    def __init__(self):
        super().__init__(
            id="47bc74eb-4af2-452c-b933-af377c7287df",
            description="Post to Telegram using Ayrshare",
        )

    def run(
        self,
        input_data: BaseAyrsharePostBlock.Input,
        *,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        """Post to Telegram."""
        profile_key, profile_key_value = self._get_profile_key(user_id)
        if profile_key == "error":
            yield "error", profile_key_value
            return

        post_result = self._create_post(
            input_data,
            [SocialPlatform.TELEGRAM],
            profile_key=profile_key_value,
        )
        yield "post_result", post_result


class PostToGMBBlock(BaseAyrsharePostBlock):
    """Block for posting to Google My Business."""

    def __init__(self):
        super().__init__(
            id="2c38c783-c484-4503-9280-ef5d1d345a7e",
            description="Post to Google My Business using Ayrshare",
        )

    def run(
        self,
        input_data: BaseAyrsharePostBlock.Input,
        *,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        """Post to Google My Business."""
        profile_key, profile_key_value = self._get_profile_key(user_id)
        if profile_key == "error":
            yield "error", profile_key_value
            return

        post_result = self._create_post(
            input_data,
            [SocialPlatform.GMB],
            profile_key=profile_key_value,
        )
        yield "post_result", post_result


class PostToPinterestBlock(BaseAyrsharePostBlock):
    """Block for posting to Pinterest."""

    def __init__(self):
        super().__init__(
            id="3ca46e05-dbaa-4afb-9e95-5a429c4177e6",
            description="Post to Pinterest using Ayrshare",
        )

    def run(
        self,
        input_data: BaseAyrsharePostBlock.Input,
        *,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        """Post to Pinterest."""
        profile_key, profile_key_value = self._get_profile_key(user_id)
        if profile_key == "error":
            yield "error", profile_key_value
            return

        post_result = self._create_post(
            input_data,
            [SocialPlatform.PINTEREST],
            profile_key=profile_key_value,
        )
        yield "post_result", post_result


class PostToTikTokBlock(BaseAyrsharePostBlock):
    """Block for posting to TikTok."""

    def __init__(self):
        super().__init__(
            id="7faf4b27-96b0-4f05-bf64-e0de54ae74e1",
            description="Post to TikTok using Ayrshare",
        )

    def run(
        self,
        input_data: BaseAyrsharePostBlock.Input,
        *,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        """Post to TikTok."""
        profile_key, profile_key_value = self._get_profile_key(user_id)
        if profile_key == "error":
            yield "error", profile_key_value
            return

        post_result = self._create_post(
            input_data,
            [SocialPlatform.TIKTOK],
            profile_key=profile_key_value,
        )
        yield "post_result", post_result


class PostToBlueskyBlock(BaseAyrsharePostBlock):
    """Block for posting to Bluesky."""

    def __init__(self):
        super().__init__(
            id="cbd52c2a-06d2-43ed-9560-6576cc163283",
            description="Post to Bluesky using Ayrshare",
        )

    def run(
        self,
        input_data: BaseAyrsharePostBlock.Input,
        *,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        """Post to Bluesky."""
        profile_key, profile_key_value = self._get_profile_key(user_id)
        if profile_key == "error":
            yield "error", profile_key_value
            return

        post_result = self._create_post(
            input_data,
            [SocialPlatform.BLUESKY],
            profile_key=profile_key_value,
        )
        yield "post_result", post_result
