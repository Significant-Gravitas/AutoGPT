from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel

from backend.util.exceptions import MissingConfigError
from backend.util.request import Requests
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()


class AyrshareAPIException(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class SocialPlatform(str, Enum):
    BLUESKY = "bluesky"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"
    REDDIT = "reddit"
    TELEGRAM = "telegram"
    GOOGLE_MY_BUSINESS = "gmb"
    PINTEREST = "pinterest"
    TIKTOK = "tiktok"
    SNAPCHAT = "snapchat"
    THREADS = "threads"


class EmailConfig(BaseModel):
    to: str
    subject: Optional[str] = None
    body: Optional[str] = None
    from_name: Optional[str] = None
    from_email: Optional[str] = None


class JWTResponse(BaseModel):
    status: str
    title: str
    token: str
    url: str
    emailSent: Optional[bool] = None
    expiresIn: Optional[str] = None


class ProfileResponse(BaseModel):
    status: str
    title: str
    refId: str
    profileKey: str
    messagingActive: Optional[bool] = None


class PostResponse(BaseModel):
    status: str
    id: str
    refId: str
    profileTitle: str
    post: str
    postIds: Optional[list[PostIds]] = None
    scheduleDate: Optional[str] = None
    errors: Optional[list[str]] = None


class PostIds(BaseModel):
    status: str
    id: str
    postUrl: str
    platform: str


class AutoHashtag(BaseModel):
    max: Optional[int] = None
    position: Optional[str] = None


class FirstComment(BaseModel):
    text: str
    platforms: Optional[list[SocialPlatform]] = None


class AutoSchedule(BaseModel):
    interval: str
    platforms: Optional[list[SocialPlatform]] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None


class AutoRepost(BaseModel):
    interval: str
    platforms: Optional[list[SocialPlatform]] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None


class AyrshareClient:
    """Client for the Ayrshare Social Media Post API"""

    API_URL = "https://api.ayrshare.com/api"
    POST_ENDPOINT = f"{API_URL}/post"
    PROFILES_ENDPOINT = f"{API_URL}/profiles"
    JWT_ENDPOINT = f"{PROFILES_ENDPOINT}/generateJWT"

    def __init__(
        self,
        custom_requests: Optional[Requests] = None,
    ):
        if not settings.secrets.ayrshare_api_key:
            raise MissingConfigError("AYRSHARE_API_KEY is not configured")

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.secrets.ayrshare_api_key}",
        }
        self.headers = headers

        if custom_requests:
            self._requests = custom_requests
        else:
            self._requests = Requests(
                extra_headers=headers,
                trusted_origins=["https://api.ayrshare.com"],
            )

    async def generate_jwt(
        self,
        private_key: str,
        profile_key: str,
        logout: Optional[bool] = None,
        redirect: Optional[str] = None,
        allowed_social: Optional[list[SocialPlatform]] = None,
        verify: Optional[bool] = None,
        base64: Optional[bool] = None,
        expires_in: Optional[int] = None,
        email: Optional[EmailConfig] = None,
    ) -> JWTResponse:
        """
        Generate a JSON Web Token (JWT) for use with single sign on.

        Docs: https://www.ayrshare.com/docs/apis/profiles/generate-jwt-overview

        Args:
            domain: Domain of app. Must match the domain given during onboarding.
            private_key: Private Key used for encryption.
            profile_key: User Profile Key (not the API Key).
            logout: Automatically logout the current session.
            redirect: URL to redirect to when the "Done" button or logo is clicked.
            allowed_social: List of social networks to display in the linking page.
            verify: Verify that the generated token is valid (recommended for non-production).
            base64: Whether the private key is base64 encoded.
            expires_in: Token longevity in minutes (1-2880).
            email: Configuration for sending Connect Accounts email.

        Returns:
            JWTResponse object containing the JWT token and URL.

        Raises:
            AyrshareAPIException: If the API request fails or private key is invalid.
        """
        payload: dict[str, Any] = {
            "domain": "id-pojeg",
            "privateKey": private_key,
            "profileKey": profile_key,
        }

        headers = self.headers
        headers["Profile-Key"] = profile_key
        if logout is not None:
            payload["logout"] = logout
        if redirect is not None:
            payload["redirect"] = redirect
        if allowed_social is not None:
            payload["allowedSocial"] = [p.value for p in allowed_social]
        if verify is not None:
            payload["verify"] = verify
        if base64 is not None:
            payload["base64"] = base64
        if expires_in is not None:
            payload["expiresIn"] = expires_in
        if email is not None:
            payload["email"] = email.model_dump(exclude_none=True)

        response = await self._requests.post(
            self.JWT_ENDPOINT, json=payload, headers=headers
        )

        if not response.ok:
            try:
                error_data = response.json()
                error_message = error_data.get("message", "Unknown error")
            except json.JSONDecodeError:
                error_message = response.text()

            raise AyrshareAPIException(
                f"Ayrshare API request failed ({response.status}): {error_message}",
                response.status,
            )

        response_data = response.json()
        if response_data.get("status") != "success":
            raise AyrshareAPIException(
                f"Ayrshare API returned error: {response_data.get('message', 'Unknown error')}",
                response.status,
            )

        return JWTResponse(**response_data)

    async def create_profile(
        self,
        title: str,
        messaging_active: Optional[bool] = None,
        hide_top_header: Optional[bool] = None,
        top_header: Optional[str] = None,
        disable_social: Optional[list[SocialPlatform]] = None,
        team: Optional[bool] = None,
        email: Optional[str] = None,
        sub_header: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> ProfileResponse:
        """
        Create a new User Profile under your Primary Profile.

        Docs: https://www.ayrshare.com/docs/apis/profiles/create-profile

        Args:
            title: Title of the new profile. Must be unique.
            messaging_active: Set to true to activate messaging for this user profile.
            hide_top_header: Hide the top header on the social accounts linkage page.
            top_header: Change the header on the social accounts linkage page.
            disable_social: Array of social networks that are disabled for this user's profile.
            team: Create a new user profile as a team member.
            email: Email address for team member invite (required if team is true).
            sub_header: Change the sub header on the social accounts linkage page.
            tags: Array of strings to tag user profiles.

        Returns:
            ProfileResponse object containing the profile details and profile key.

        Raises:
            AyrshareAPIException: If the API request fails or profile title already exists.
        """
        payload: dict[str, Any] = {
            "title": title,
        }

        if messaging_active is not None:
            payload["messagingActive"] = messaging_active
        if hide_top_header is not None:
            payload["hideTopHeader"] = hide_top_header
        if top_header is not None:
            payload["topHeader"] = top_header
        if disable_social is not None:
            payload["disableSocial"] = [p.value for p in disable_social]
        if team is not None:
            payload["team"] = team
        if email is not None:
            payload["email"] = email
        if sub_header is not None:
            payload["subHeader"] = sub_header
        if tags is not None:
            payload["tags"] = tags

        response = await self._requests.post(self.PROFILES_ENDPOINT, json=payload)

        if not response.ok:
            try:
                error_data = response.json()
                error_message = error_data.get("message", "Unknown error")
            except json.JSONDecodeError:
                error_message = response.text()

            raise AyrshareAPIException(
                f"Ayrshare API request failed ({response.status}): {error_message}",
                response.status,
            )

        response_data = response.json()
        if response_data.get("status") != "success":
            raise AyrshareAPIException(
                f"Ayrshare API returned error: {response_data.get('message', 'Unknown error')}",
                response.status,
            )

        return ProfileResponse(**response_data)

    async def create_post(
        self,
        post: str,
        platforms: list[SocialPlatform],
        *,
        media_urls: Optional[list[str]] = None,
        is_video: Optional[bool] = None,
        schedule_date: Optional[str] = None,
        validate_schedule: Optional[bool] = None,
        first_comment: Optional[FirstComment] = None,
        disable_comments: Optional[bool] = None,
        shorten_links: Optional[bool] = None,
        auto_schedule: Optional[AutoSchedule] = None,
        auto_repost: Optional[AutoRepost] = None,
        auto_hashtag: Optional[AutoHashtag | bool] = None,
        unsplash: Optional[str] = None,
        bluesky_options: Optional[dict[str, Any]] = None,
        facebook_options: Optional[dict[str, Any]] = None,
        gmb_options: Optional[dict[str, Any]] = None,
        instagram_options: Optional[dict[str, Any]] = None,
        linkedin_options: Optional[dict[str, Any]] = None,
        pinterest_options: Optional[dict[str, Any]] = None,
        reddit_options: Optional[dict[str, Any]] = None,
        snapchat_options: Optional[dict[str, Any]] = None,
        telegram_options: Optional[dict[str, Any]] = None,
        threads_options: Optional[dict[str, Any]] = None,
        tiktok_options: Optional[dict[str, Any]] = None,
        twitter_options: Optional[dict[str, Any]] = None,
        youtube_options: Optional[dict[str, Any]] = None,
        requires_approval: Optional[bool] = None,
        random_post: Optional[bool] = None,
        random_media_url: Optional[bool] = None,
        idempotency_key: Optional[str] = None,
        notes: Optional[str] = None,
        profile_key: Optional[str] = None,
    ) -> PostResponse:
        """
        Create a post across multiple social media platforms.

        Docs: https://www.ayrshare.com/docs/apis/post/post

        Args:
            post: The post text to be published - required
            platforms: List of platforms to post to (e.g. [SocialPlatform.TWITTER, SocialPlatform.FACEBOOK]) - required
            media_urls: Optional list of media URLs to include - required if is_video is true
            is_video: Whether the media is a video - default is false (in api docs)
            schedule_date: UTC datetime for scheduling (YYYY-MM-DDThh:mm:ssZ) - default is None (in api docs)
            validate_schedule: Whether to validate the schedule date - default is false (in api docs)
            first_comment: Configuration for first comment - default is None (in api docs)
            disable_comments: Whether to disable comments - default is false (in api docs)
            shorten_links: Whether to shorten links - default is false (in api docs)
            auto_schedule: Configuration for automatic scheduling - default is None (in api docs https://www.ayrshare.com/docs/apis/auto-schedule/overview)
            auto_repost: Configuration for automatic reposting - default is None (in api docs https://www.ayrshare.com/docs/apis/post/overview#auto-repost)
            auto_hashtag: Configuration for automatic hashtags - default is None (in api docs https://www.ayrshare.com/docs/apis/post/overview#auto-hashtags)
            unsplash: Unsplash image configuration - default is None (in api docs https://www.ayrshare.com/docs/apis/post/overview#unsplash)

            ------------------------------------------------------------

            bluesky_options: Bluesky-specific options - https://www.ayrshare.com/docs/apis/post/social-networks/bluesky
            facebook_options: Facebook-specific options - https://www.ayrshare.com/docs/apis/post/social-networks/facebook
            gmb_options: Google Business Profile options - https://www.ayrshare.com/docs/apis/post/social-networks/google
            instagram_options: Instagram-specific options - https://www.ayrshare.com/docs/apis/post/social-networks/instagram
            linkedin_options: LinkedIn-specific options - https://www.ayrshare.com/docs/apis/post/social-networks/linkedin
            pinterest_options: Pinterest-specific options - https://www.ayrshare.com/docs/apis/post/social-networks/pinterest
            reddit_options: Reddit-specific options - https://www.ayrshare.com/docs/apis/post/social-networks/reddit
            snapchat_options: Snapchat-specific options - https://www.ayrshare.com/docs/apis/post/social-networks/snapchat
            telegram_options: Telegram-specific options - https://www.ayrshare.com/docs/apis/post/social-networks/telegram
            threads_options: Threads-specific options - https://www.ayrshare.com/docs/apis/post/social-networks/threads
            tiktok_options: TikTok-specific options - https://www.ayrshare.com/docs/apis/post/social-networks/tiktok
            twitter_options: Twitter-specific options - https://www.ayrshare.com/docs/apis/post/social-networks/twitter
            youtube_options: YouTube-specific options - https://www.ayrshare.com/docs/apis/post/social-networks/youtube

            ------------------------------------------------------------


            requires_approval: Whether to enable approval workflow - default is false (in api docs)
            random_post: Whether to generate random post text - default is false (in api docs)
            random_media_url: Whether to generate random media - default is false (in api docs)
            idempotency_key: Unique ID for the post - default is None (in api docs)
            notes: Additional notes for the post - default is None (in api docs)

        Returns:
            PostResponse object containing the post details and status

        Raises:
            AyrshareAPIException: If the API request fails
        """

        payload: dict[str, Any] = {
            "post": post,
            "platforms": [p.value for p in platforms],
        }

        # Add optional parameters if provided
        if media_urls:
            payload["mediaUrls"] = media_urls
        if is_video is not None:
            payload["isVideo"] = is_video
        if schedule_date:
            payload["scheduleDate"] = schedule_date
        if validate_schedule is not None:
            payload["validateSchedule"] = validate_schedule
        if first_comment:
            first_comment_dict = first_comment.model_dump(exclude_none=True)
            if first_comment.platforms:
                first_comment_dict["platforms"] = [
                    p.value for p in first_comment.platforms
                ]
            payload["firstComment"] = first_comment_dict
        if disable_comments is not None:
            payload["disableComments"] = disable_comments
        if shorten_links is not None:
            payload["shortenLinks"] = shorten_links
        if auto_schedule:
            auto_schedule_dict = auto_schedule.model_dump(exclude_none=True)
            if auto_schedule.platforms:
                auto_schedule_dict["platforms"] = [
                    p.value for p in auto_schedule.platforms
                ]
            payload["autoSchedule"] = auto_schedule_dict
        if auto_repost:
            auto_repost_dict = auto_repost.model_dump(exclude_none=True)
            if auto_repost.platforms:
                auto_repost_dict["platforms"] = [p.value for p in auto_repost.platforms]
            payload["autoRepost"] = auto_repost_dict
        if auto_hashtag:
            payload["autoHashtag"] = (
                auto_hashtag.model_dump(exclude_none=True)
                if isinstance(auto_hashtag, AutoHashtag)
                else auto_hashtag
            )
        if unsplash:
            payload["unsplash"] = unsplash
        if bluesky_options:
            payload["blueskyOptions"] = bluesky_options
        if facebook_options:
            payload["faceBookOptions"] = facebook_options
        if gmb_options:
            payload["gmbOptions"] = gmb_options
        if instagram_options:
            payload["instagramOptions"] = instagram_options
        if linkedin_options:
            payload["linkedInOptions"] = linkedin_options
        if pinterest_options:
            payload["pinterestOptions"] = pinterest_options
        if reddit_options:
            payload["redditOptions"] = reddit_options
        if snapchat_options:
            payload["snapchatOptions"] = snapchat_options
        if telegram_options:
            payload["telegramOptions"] = telegram_options
        if threads_options:
            payload["threadsOptions"] = threads_options
        if tiktok_options:
            payload["tikTokOptions"] = tiktok_options
        if twitter_options:
            payload["twitterOptions"] = twitter_options
        if youtube_options:
            payload["youTubeOptions"] = youtube_options
        if requires_approval is not None:
            payload["requiresApproval"] = requires_approval
        if random_post is not None:
            payload["randomPost"] = random_post
        if random_media_url is not None:
            payload["randomMediaUrl"] = random_media_url
        if idempotency_key:
            payload["idempotencyKey"] = idempotency_key
        if notes:
            payload["notes"] = notes

        headers = self.headers
        if profile_key:
            headers["Profile-Key"] = profile_key

        response = await self._requests.post(
            self.POST_ENDPOINT, json=payload, headers=headers
        )
        logger.warning(f"Ayrshare request: {payload} and headers: {headers}")
        if not response.ok:
            logger.error(
                f"Ayrshare API request failed ({response.status}): {response.text()}"
            )
            try:
                error_data = response.json()
                error_message = error_data.get("message", "Unknown error")
            except json.JSONDecodeError:
                error_message = response.text()

            raise AyrshareAPIException(
                f"Ayrshare API request failed ({response.status}): {error_message}",
                response.status,
            )

        response_data = response.json()
        if response_data.get("status") != "success":
            logger.error(
                f"Ayrshare API returned error: {response_data.get('message', 'Unknown error')}"
            )
            raise AyrshareAPIException(
                f"Ayrshare API returned error: {response_data.get('message', 'Unknown error')}",
                response.status,
            )

        # Ayrshare returns an array of posts even for single posts
        # It seems like there is only ever one post in the array, and within that
        # there are multiple postIds

        # There is a seperate endpoint for bulk posting, so feels safe to just take
        # the first post from the array

        if len(response_data["posts"]) == 0:
            logger.error("Ayrshare API returned no posts")
            raise AyrshareAPIException(
                "Ayrshare API returned no posts",
                response.status,
            )
        logger.warn(f"Ayrshare API returned posts: {response_data['posts']}")
        return PostResponse(**response_data["posts"][0])
