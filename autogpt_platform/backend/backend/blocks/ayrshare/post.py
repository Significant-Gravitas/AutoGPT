import logging
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, SecretStr

from backend.blocks.ayrshare._api import (
    AyrshareClient,
    PostError,
    PostResponse,
    SocialPlatform,
)
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema, BlockType
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


class AyrsharePostBlockBase(Block):
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
        random_post: bool = SchemaField(
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
            # The type of block, this is used to determine the block type in the UI.
            block_type=BlockType.AYRSHARE,
            # The schema, defined as a Pydantic model, for the input data.
            input_schema=AyrsharePostBlockBase.Input,
            # The schema, defined as a Pydantic model, for the output data.
            output_schema=AyrsharePostBlockBase.Output,
        )

    @staticmethod
    def create_client():
        return AyrshareClient()

    def _create_post(
        self,
        input_data: "AyrsharePostBlockBase.Input",
        platforms: List[SocialPlatform],
        profile_key: Optional[str] = None,
    ) -> PostResponse | PostError:
        client = self.create_client()
        """Create a post on the specified platforms."""
        iso_date = (
            input_data.schedule_date.isoformat() if input_data.schedule_date else None
        )
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
        return response

    def run(
        self,
        input_data: "AyrsharePostBlockBase.Input",
        *,
        profile_key: SecretStr,
        **kwargs,
    ) -> BlockOutput:
        """Run the block."""
        platforms = [SocialPlatform.FACEBOOK]

        if not profile_key:
            yield "error", "Please Link a social account via Ayrshare"
            return

        post_result = self._create_post(
            input_data, platforms=platforms, profile_key=profile_key.get_secret_value()
        )
        if isinstance(post_result, PostError):
            yield "error", post_result.message
            return
        yield "post_result", post_result


class PostToFacebookBlock(AyrsharePostBlockBase):
    """Block for posting to Facebook."""

    def __init__(self):
        super().__init__(
            id="3352f512-3524-49ed-a08f-003042da2fc1",
            description="Post to Facebook using Ayrshare",
        )

    def run(
        self,
        input_data: AyrsharePostBlockBase.Input,
        *,
        profile_key: SecretStr,
        **kwargs,
    ) -> BlockOutput:
        """Post to Facebook."""
        if not profile_key:
            yield "error", "Please Link a social account via Ayrshare"
            return

        post_result = self._create_post(
            input_data,
            [SocialPlatform.FACEBOOK],
            profile_key=profile_key.get_secret_value(),
        )
        if isinstance(post_result, PostError):
            yield "error", post_result.message
            return
        yield "post_result", post_result


class PostToXBlock(AyrsharePostBlockBase):
    """Block for posting to X / Twitter."""

    def __init__(self):
        super().__init__(
            id="9e8f844e-b4a5-4b25-80f2-9e1dd7d67625",
            description="Post to X / Twitter using Ayrshare",
        )

    def run(
        self,
        input_data: AyrsharePostBlockBase.Input,
        *,
        profile_key: SecretStr,
        **kwargs,
    ) -> BlockOutput:
        """Post to Twitter."""
        if not profile_key:
            yield "error", "Please Link a social account via Ayrshare"
            return

        post_result = self._create_post(
            input_data,
            [SocialPlatform.TWITTER],
            profile_key=profile_key.get_secret_value(),
        )
        if isinstance(post_result, PostError):
            yield "error", post_result.message
            return
        yield "post_result", post_result


class PostToLinkedInBlock(AyrsharePostBlockBase):
    """Block for posting to LinkedIn."""

    def __init__(self):
        super().__init__(
            id="589af4e4-507f-42fd-b9ac-a67ecef25811",
            description="Post to LinkedIn using Ayrshare",
        )

    def run(
        self,
        input_data: AyrsharePostBlockBase.Input,
        *,
        profile_key: SecretStr,
        **kwargs,
    ) -> BlockOutput:
        """Post to LinkedIn."""
        if not profile_key:
            yield "error", "Please Link a social account via Ayrshare"
            return

        post_result = self._create_post(
            input_data,
            [SocialPlatform.LINKEDIN],
            profile_key=profile_key.get_secret_value(),
        )
        if isinstance(post_result, PostError):
            yield "error", post_result.message
            return
        yield "post_result", post_result


class PostToInstagramBlock(AyrsharePostBlockBase):
    """Block for posting to Instagram."""

    def __init__(self):
        super().__init__(
            id="89b02b96-a7cb-46f4-9900-c48b32fe1552",
            description="Post to Instagram using Ayrshare",
        )

    def run(
        self,
        input_data: AyrsharePostBlockBase.Input,
        *,
        profile_key: SecretStr,
        **kwargs,
    ) -> BlockOutput:
        """Post to Instagram."""
        if not profile_key:
            yield "error", "Please Link a social account via Ayrshare"
            return

        post_result = self._create_post(
            input_data,
            [SocialPlatform.INSTAGRAM],
            profile_key=profile_key.get_secret_value(),
        )
        if isinstance(post_result, PostError):
            yield "error", post_result.message
            return
        yield "post_result", post_result


class PostToYouTubeBlock(AyrsharePostBlockBase):
    """Block for posting to YouTube."""

    def __init__(self):
        super().__init__(
            id="0082d712-ff1b-4c3d-8a8d-6c7721883b83",
            description="Post to YouTube using Ayrshare",
        )

    def run(
        self,
        input_data: AyrsharePostBlockBase.Input,
        *,
        profile_key: SecretStr,
        **kwargs,
    ) -> BlockOutput:
        """Post to YouTube."""
        if not profile_key:
            yield "error", "Please Link a social account via Ayrshare"
            return

        post_result = self._create_post(
            input_data,
            [SocialPlatform.YOUTUBE],
            profile_key=profile_key.get_secret_value(),
        )
        if isinstance(post_result, PostError):
            yield "error", post_result.message
            return
        yield "post_result", post_result


class PostToRedditBlock(AyrsharePostBlockBase):
    """Block for posting to Reddit."""

    def __init__(self):
        super().__init__(
            id="c7733580-3c72-483e-8e47-a8d58754d853",
            description="Post to Reddit using Ayrshare",
        )

    def run(
        self,
        input_data: AyrsharePostBlockBase.Input,
        *,
        profile_key: SecretStr,
        **kwargs,
    ) -> BlockOutput:
        """Post to Reddit."""
        if not profile_key:
            yield "error", "Please Link a social account via Ayrshare"
            return

        post_result = self._create_post(
            input_data,
            [SocialPlatform.REDDIT],
            profile_key=profile_key.get_secret_value(),
        )
        if isinstance(post_result, PostError):
            yield "error", post_result.message
            return
        yield "post_result", post_result


class PostToTelegramBlock(AyrsharePostBlockBase):
    """Block for posting to Telegram."""

    def __init__(self):
        super().__init__(
            id="47bc74eb-4af2-452c-b933-af377c7287df",
            description="Post to Telegram using Ayrshare",
        )

    def run(
        self,
        input_data: AyrsharePostBlockBase.Input,
        *,
        profile_key: SecretStr,
        **kwargs,
    ) -> BlockOutput:
        """Post to Telegram."""
        if not profile_key:
            yield "error", "Please Link a social account via Ayrshare"
            return

        post_result = self._create_post(
            input_data,
            [SocialPlatform.TELEGRAM],
            profile_key=profile_key.get_secret_value(),
        )
        if isinstance(post_result, PostError):
            yield "error", post_result.message
            return
        yield "post_result", post_result


class PostToGMBBlock(AyrsharePostBlockBase):
    """Block for posting to Google My Business."""

    def __init__(self):
        super().__init__(
            id="2c38c783-c484-4503-9280-ef5d1d345a7e",
            description="Post to Google My Business using Ayrshare",
        )

    def run(
        self,
        input_data: AyrsharePostBlockBase.Input,
        *,
        profile_key: SecretStr,
        **kwargs,
    ) -> BlockOutput:
        """Post to Google My Business."""
        if not profile_key:
            yield "error", "Please Link a social account via Ayrshare"
            return

        post_result = self._create_post(
            input_data,
            [SocialPlatform.GMB],
            profile_key=profile_key.get_secret_value(),
        )
        if isinstance(post_result, PostError):
            yield "error", post_result.message
            return
        yield "post_result", post_result


class PostToPinterestBlock(AyrsharePostBlockBase):
    """Block for posting to Pinterest."""

    def __init__(self):
        super().__init__(
            id="3ca46e05-dbaa-4afb-9e95-5a429c4177e6",
            description="Post to Pinterest using Ayrshare",
        )

    def run(
        self,
        input_data: AyrsharePostBlockBase.Input,
        *,
        profile_key: SecretStr,
        **kwargs,
    ) -> BlockOutput:
        """Post to Pinterest."""
        if not profile_key:
            yield "error", "Please Link a social account via Ayrshare"
            return

        post_result = self._create_post(
            input_data,
            [SocialPlatform.PINTEREST],
            profile_key=profile_key.get_secret_value(),
        )
        if isinstance(post_result, PostError):
            yield "error", post_result.message
            return
        yield "post_result", post_result


class PostToTikTokBlock(AyrsharePostBlockBase):
    """Block for posting to TikTok."""

    def __init__(self):
        super().__init__(
            id="7faf4b27-96b0-4f05-bf64-e0de54ae74e1",
            description="Post to TikTok using Ayrshare",
        )

    def run(
        self,
        input_data: AyrsharePostBlockBase.Input,
        *,
        profile_key: SecretStr,
        **kwargs,
    ) -> BlockOutput:
        """Post to TikTok."""
        if not profile_key:
            yield "error", "Please Link a social account via Ayrshare"
            return

        post_result = self._create_post(
            input_data,
            [SocialPlatform.TIKTOK],
            profile_key=profile_key.get_secret_value(),
        )
        if isinstance(post_result, PostError):
            yield "error", post_result.message
            return
        yield "post_result", post_result


class PostToBlueskyBlock(AyrsharePostBlockBase):
    """Block for posting to Bluesky."""

    def __init__(self):
        super().__init__(
            id="cbd52c2a-06d2-43ed-9560-6576cc163283",
            description="Post to Bluesky using Ayrshare",
        )

    def run(
        self,
        input_data: AyrsharePostBlockBase.Input,
        *,
        profile_key: SecretStr,
        **kwargs,
    ) -> BlockOutput:
        """Post to Bluesky."""
        if not profile_key:
            yield "error", "Please Link a social account via Ayrshare"
            return

        post_result = self._create_post(
            input_data,
            [SocialPlatform.BLUESKY],
            profile_key=profile_key.get_secret_value(),
        )
        if isinstance(post_result, PostError):
            yield "error", post_result.message
            return
        yield "post_result", post_result


AYRSHARE_NODE_IDS = [
    PostToBlueskyBlock().id,
    PostToFacebookBlock().id,
    PostToXBlock().id,
    PostToLinkedInBlock().id,
    PostToInstagramBlock().id,
    PostToYouTubeBlock().id,
    PostToRedditBlock().id,
    PostToTelegramBlock().id,
    PostToGMBBlock().id,
    PostToPinterestBlock().id,
    PostToTikTokBlock().id,
]
