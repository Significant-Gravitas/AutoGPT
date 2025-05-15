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
from backend.data.model import APIKeyCredentials
from backend.integrations.credentials_store import IntegrationCredentialsStore

logger = logging.getLogger(__name__)

creads_store = IntegrationCredentialsStore()


class AyrsharePostInput(BaseModel):
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


class AyrsharePostOutput(BaseModel):
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


class BaseAyrsharePost:
    """Base class for Ayrshare social media posting blocks."""

    def __init__(self, credentials: Optional[APIKeyCredentials] = None):
        self.credentials = credentials or AYRSHARE_CREDENTIALS
        self.client = AyrshareClient(
            api_key=self.credentials.api_key.get_secret_value(),
        )

    def _create_post(
        self,
        input_data: AyrsharePostInput,
        platforms: List[SocialPlatform],
        profile_key: Optional[str] = None,
    ) -> AyrsharePostOutput:
        """Create a post on the specified platforms."""
        response = self.client.create_post(
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


class PostToFacebook(BaseAyrsharePost):
    """Block for posting to Facebook."""

    def __init__(self, credentials: Optional[APIKeyCredentials] = None):
        super().__init__(credentials)

    def run(self, input_data: AyrsharePostInput, *, user_id: str) -> AyrsharePostOutput:
        creds_store = IntegrationCredentialsStore()
        profile_key = creds_store.get_ayrshare_profile_key(user_id)
        if profile_key:
            logger.info(f"Profile key: {profile_key}")
        """Post to Facebook."""
        return self._create_post(
            input_data,
            [SocialPlatform.FACEBOOK],
            profile_key=profile_key.get_secret_value() if profile_key else None,
        )


class PostToTwitter(BaseAyrsharePost):
    """Block for posting to Twitter."""

    def __init__(self, credentials: Optional[APIKeyCredentials] = None):
        super().__init__(credentials)

    def run(self, input_data: AyrsharePostInput) -> AyrsharePostOutput:
        """Post to Twitter."""
        return self._create_post(input_data, [SocialPlatform.TWITTER])


class PostToLinkedIn(BaseAyrsharePost):
    """Block for posting to LinkedIn."""

    def __init__(self, credentials: Optional[APIKeyCredentials] = None):
        super().__init__(credentials)

    def run(self, input_data: AyrsharePostInput) -> AyrsharePostOutput:
        """Post to LinkedIn."""
        return self._create_post(input_data, [SocialPlatform.LINKEDIN])


class PostToInstagram(BaseAyrsharePost):
    """Block for posting to Instagram."""

    def __init__(self, credentials: Optional[APIKeyCredentials] = None):
        super().__init__(credentials)

    def run(self, input_data: AyrsharePostInput) -> AyrsharePostOutput:
        """Post to Instagram."""
        return self._create_post(input_data, [SocialPlatform.INSTAGRAM])


class PostToYouTube(BaseAyrsharePost):
    """Block for posting to YouTube."""

    def __init__(self, credentials: Optional[APIKeyCredentials] = None):
        super().__init__(credentials)

    def run(self, input_data: AyrsharePostInput) -> AyrsharePostOutput:
        """Post to YouTube."""
        return self._create_post(input_data, [SocialPlatform.YOUTUBE])


class PostToReddit(BaseAyrsharePost):
    """Block for posting to Reddit."""

    def __init__(self, credentials: Optional[APIKeyCredentials] = None):
        super().__init__(credentials)

    def run(self, input_data: AyrsharePostInput) -> AyrsharePostOutput:
        """Post to Reddit."""
        return self._create_post(input_data, [SocialPlatform.REDDIT])


class PostToTelegram(BaseAyrsharePost):
    """Block for posting to Telegram."""

    def __init__(self, credentials: Optional[APIKeyCredentials] = None):
        super().__init__(credentials)

    def run(self, input_data: AyrsharePostInput) -> AyrsharePostOutput:
        """Post to Telegram."""
        return self._create_post(input_data, [SocialPlatform.TELEGRAM])


class PostToGMB(BaseAyrsharePost):
    """Block for posting to Google My Business."""

    def __init__(self, credentials: Optional[APIKeyCredentials] = None):
        super().__init__(credentials)

    def run(self, input_data: AyrsharePostInput) -> AyrsharePostOutput:
        """Post to Google My Business."""
        return self._create_post(input_data, [SocialPlatform.GMB])


class PostToPinterest(BaseAyrsharePost):
    """Block for posting to Pinterest."""

    def __init__(self, credentials: Optional[APIKeyCredentials] = None):
        super().__init__(credentials)

    def run(self, input_data: AyrsharePostInput) -> AyrsharePostOutput:
        """Post to Pinterest."""
        return self._create_post(input_data, [SocialPlatform.PINTEREST])


class PostToTikTok(BaseAyrsharePost):
    """Block for posting to TikTok."""

    def __init__(self, credentials: Optional[APIKeyCredentials] = None):
        super().__init__(credentials)

    def run(self, input_data: AyrsharePostInput, *, user_id: str) -> AyrsharePostOutput:
        """Post to TikTok."""
        return self._create_post(input_data, [SocialPlatform.TIKTOK])


class PostToBluesky(BaseAyrsharePost):
    """Block for posting to Bluesky."""

    def __init__(self, credentials: Optional[APIKeyCredentials] = None):
        super().__init__(credentials)

    def run(self, input_data: AyrsharePostInput) -> AyrsharePostOutput:
        """Post to Bluesky."""
        return self._create_post(input_data, [SocialPlatform.BLUESKY])
