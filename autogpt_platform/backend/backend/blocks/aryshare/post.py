import logging
from typing import List, Optional, Union

from pydantic import BaseModel, Field

from backend.blocks.aryshare._api import (
    AutoHashtag,
    AutoRepost,
    AutoSchedule,
    AyrshareClient,
    FirstComment,
    SocialPlatform,
)
from backend.blocks.aryshare._auth import AYRSHARE_CREDENTIALS
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import APIKeyCredentials
from backend.integrations.credentials_store import IntegrationCredentialsStore

from ._auth import AYRSHARE_CREDENTIALS_INPUT

logger = logging.getLogger(__name__)

creads_store = IntegrationCredentialsStore()


class AyrsharePostInput(BlockSchema):
    """Base input model for Ayrshare social media posts."""

    post: str = Field(..., description="The post text to be published")
    media_urls: Optional[List[str]] = Field(
        None, description="Optional list of media URLs to include"
    )
    is_video: Optional[bool] = Field(None, description="Whether the media is a video")
    schedule_date: Optional[str] = Field(
        None, description="UTC datetime for scheduling (YYYY-MM-DDThh:mm:ssZ)"
    )
    first_comment: Optional[FirstComment] = Field(
        None, description="Configuration for first comment"
    )
    disable_comments: Optional[bool] = Field(
        None, description="Whether to disable comments"
    )
    shorten_links: Optional[bool] = Field(None, description="Whether to shorten links")
    auto_schedule: Optional[AutoSchedule] = Field(
        None, description="Configuration for automatic scheduling"
    )
    auto_repost: Optional[AutoRepost] = Field(
        None, description="Configuration for automatic reposting"
    )
    auto_hashtag: Optional[Union[AutoHashtag, bool]] = Field(
        None, description="Configuration for automatic hashtags"
    )
    unsplash: Optional[str] = Field(None, description="Unsplash image configuration")
    requires_approval: Optional[bool] = Field(
        None, description="Whether to enable approval workflow"
    )
    random_post: Optional[bool] = Field(
        None, description="Whether to generate random post text"
    )
    random_media_url: Optional[bool] = Field(
        None, description="Whether to generate random media"
    )
    idempotency_key: Optional[str] = Field(None, description="Unique ID for the post")
    notes: Optional[str] = Field(None, description="Additional notes for the post")


class RequestOutput(BaseModel):
    """Base output model for Ayrshare social media posts."""

    status: str = Field(..., description="Status of the post")
    id: str = Field(..., description="ID of the post")
    refId: str = Field(..., description="Reference ID of the post")
    profileTitle: str = Field(..., description="Title of the profile")
    post: str = Field(..., description="The post text")
    postIds: Optional[List[dict]] = Field(
        None, description="IDs of the posts on each platform"
    )
    scheduleDate: Optional[str] = Field(None, description="Scheduled date of the post")
    errors: Optional[List[str]] = Field(None, description="Any errors that occurred")


class AyrsharePostOutput(BlockSchema):
    post_result: RequestOutput = Field(..., description="The result of the post")


class BaseAyrsharePostBlock(Block):
    """Base class for Ayrshare social media posting blocks."""

    def __init__(self):
        super().__init__(
            # The unique identifier for the block, this value will be persisted in the DB.
            # It should be unique and constant across the application run.
            # Use the UUID format for the ID.
            id="380694d5-3b2e-4130-bced-b43752b70de9",
            # The description of the block, explaining what the block does.
            description="Base class for Ayrshare social media posting blocks",
            # The set of categories that the block belongs to.
            # Each category is an instance of BlockCategory Enum.
            categories={BlockCategory.SOCIAL},
            # The schema, defined as a Pydantic model, for the input data.
            input_schema=AyrsharePostInput,
            # The schema, defined as a Pydantic model, for the output data.
            output_schema=AyrsharePostOutput,
            # The credentials required for testing the block.
            # This is an instance of APIKeyCredentials with sample values.
            test_credentials=AYRSHARE_CREDENTIALS,
            # The list or single sample input data for the block, for testing.
            # This is an instance of the Input schema with sample values.
            test_input={
                "post": "Hello, world! This is a test post.",
                "media_urls": ["https://example.com/image.jpg"],
                "is_video": False,
                "credentials": AYRSHARE_CREDENTIALS_INPUT,
            },
            # The list or single expected output if the test_input is run.
            # Each output is a tuple of (output_name, output_data).
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
                )
            },
        )

    @staticmethod
    def create_client(credentials):
        return AyrshareClient(
            api_key=credentials.api_key.get_secret_value(),
        )

    def _create_post(
        self,
        input_data: AyrsharePostInput,
        platforms: List[SocialPlatform],
        profile_key: Optional[str] = None,
        credentials: Optional[APIKeyCredentials] = None,
    ) -> AyrsharePostOutput:
        client = self.create_client(credentials)
        """Create a post on the specified platforms."""
        response = client.create_post(
            post=input_data.post,
            platforms=platforms,
            media_urls=input_data.media_urls,
            is_video=input_data.is_video,
            schedule_date=input_data.schedule_date,
            first_comment=input_data.first_comment,
            disable_comments=input_data.disable_comments,
            shorten_links=input_data.shorten_links,
            auto_schedule=input_data.auto_schedule,
            auto_repost=input_data.auto_repost,
            auto_hashtag=input_data.auto_hashtag,
            unsplash=input_data.unsplash,
            requires_approval=input_data.requires_approval,
            random_post=input_data.random_post,
            random_media_url=input_data.random_media_url,
            idempotency_key=input_data.idempotency_key,
            notes=input_data.notes,
            profile_key=profile_key,
        )
        return AyrsharePostOutput(**response.__dict__)
    
    def run(
        self,
        input_data: AyrsharePostInput,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        """Run the block."""
        platforms = [SocialPlatform.FACEBOOK]
        
        yield "post_result", self._create_post(input_data, platforms=platforms, credentials=credentials)


class PostToFacebookBlock(BaseAyrsharePostBlock):
    """Block for posting to Facebook."""

    def __init__(self):
        super().__init__()

    def run(
        self,
        input_data: AyrsharePostInput,
        *,
        user_id: str,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        """Post to Facebook."""
        creds_store = IntegrationCredentialsStore()
        profile_key = creds_store.get_ayrshare_profile_key(user_id)
        if profile_key:
            logger.info(f"Profile key: {profile_key}")
        else:
            yield "error", "Profile key not found"

        post_result = self._create_post(
            input_data,
            [SocialPlatform.FACEBOOK],
            profile_key=profile_key.get_secret_value() if profile_key else None,
            credentials=credentials,
        )
        yield "post_result", post_result


class PostToTwitterBlock(BaseAyrsharePostBlock):
    """Block for posting to Twitter."""

    def __init__(self):
        super().__init__()

    def run(
        self, input_data: AyrsharePostInput, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        """Post to Twitter."""
        post_result = self._create_post(
            input_data, [SocialPlatform.TWITTER], credentials=credentials
        )
        yield "post_result", post_result


class PostToLinkedInBlock(BaseAyrsharePostBlock):
    """Block for posting to LinkedIn."""

    def __init__(self):
        super().__init__()

    def run(
        self, input_data: AyrsharePostInput, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        """Post to LinkedIn."""
        post_result = self._create_post(
            input_data, [SocialPlatform.LINKEDIN], credentials=credentials
        )
        yield "post_result", post_result


class PostToInstagramBlock(BaseAyrsharePostBlock):
    """Block for posting to Instagram."""

    def __init__(self):
        super().__init__()

    def run(
        self, input_data: AyrsharePostInput, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        """Post to Instagram."""
        post_result = self._create_post(
            input_data, [SocialPlatform.INSTAGRAM], credentials=credentials
        )
        yield "post_result", post_result


class PostToYouTubeBlock(BaseAyrsharePostBlock):
    """Block for posting to YouTube."""

    def __init__(self):
        super().__init__()

    def run(
        self, input_data: AyrsharePostInput, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        """Post to YouTube."""
        post_result = self._create_post(
            input_data, [SocialPlatform.YOUTUBE], credentials=credentials
        )
        yield "post_result", post_result


class PostToRedditBlock(BaseAyrsharePostBlock):
    """Block for posting to Reddit."""

    def __init__(self):
        super().__init__()

    def run(
        self, input_data: AyrsharePostInput, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        """Post to Reddit."""
        post_result = self._create_post(
            input_data, [SocialPlatform.REDDIT], credentials=credentials
        )
        yield "post_result", post_result


class PostToTelegramBlock(BaseAyrsharePostBlock):
    """Block for posting to Telegram."""

    def __init__(self):
        super().__init__()

    def run(
        self, input_data: AyrsharePostInput, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        """Post to Telegram."""
        post_result = self._create_post(
            input_data, [SocialPlatform.TELEGRAM], credentials=credentials
        )
        yield "post_result", post_result


class PostToGMBBlock(BaseAyrsharePostBlock):
    """Block for posting to Google My Business."""

    def __init__(self):
        super().__init__()

    def run(
        self, input_data: AyrsharePostInput, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        """Post to Google My Business."""
        post_result = self._create_post(
            input_data, [SocialPlatform.GMB], credentials=credentials
        )
        yield "post_result", post_result


class PostToPinterestBlock(BaseAyrsharePostBlock):
    """Block for posting to Pinterest."""

    def __init__(self):
        super().__init__()

    def run(
        self, input_data: AyrsharePostInput, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        """Post to Pinterest."""
        post_result = self._create_post(
            input_data, [SocialPlatform.PINTEREST], credentials=credentials
        )
        yield "post_result", post_result


class PostToTikTokBlock(BaseAyrsharePostBlock):
    """Block for posting to TikTok."""

    def __init__(self):
        super().__init__()

    def run(
        self,
        input_data: AyrsharePostInput,
        *,
        user_id: str,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        """Post to TikTok."""
        post_result = self._create_post(
            input_data, [SocialPlatform.TIKTOK], credentials=credentials
        )
        yield "post_result", post_result


class PostToBlueskyBlock(BaseAyrsharePostBlock):
    """Block for posting to Bluesky."""

    def __init__(self):
        super().__init__()

    def run(
        self, input_data: AyrsharePostInput, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        """Post to Bluesky."""
        post_result = self._create_post(
            input_data, [SocialPlatform.BLUESKY], credentials=credentials
        )
        yield "post_result", post_result
