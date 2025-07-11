from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

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


@dataclass
class EmailConfig:
    to: str
    subject: Optional[str] = None
    body: Optional[str] = None
    from_name: Optional[str] = None
    from_email: Optional[str] = None


@dataclass
class JWTResponse:
    status: str
    title: str
    token: str
    url: str
    emailSent: Optional[bool] = None
    expiresIn: Optional[str] = None


@dataclass
class ProfileResponse:
    status: str
    title: str
    refId: str
    profileKey: str
    messagingActive: Optional[bool] = None


@dataclass
class PostResponse:
    status: str
    id: str
    refId: str
    profileTitle: str
    post: str
    postIds: Optional[list[dict[str, Any]]] = None
    scheduleDate: Optional[str] = None
    errors: Optional[list[str]] = None


@dataclass
class AutoHashtag:
    max: Optional[int] = None
    position: Optional[str] = None


@dataclass
class FirstComment:
    text: str
    platforms: Optional[list[SocialPlatform]] = None


@dataclass
class AutoSchedule:
    interval: str
    platforms: Optional[list[SocialPlatform]] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None


@dataclass
class AutoRepost:
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
                raise_for_status=False,
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
            payload["email"] = email.__dict__

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

        Args:
            post: The post text to be published
            platforms: List of platforms to post to (e.g. [SocialPlatform.TWITTER, SocialPlatform.FACEBOOK])
            media_urls: Optional list of media URLs to include
            is_video: Whether the media is a video
            schedule_date: UTC datetime for scheduling (YYYY-MM-DDThh:mm:ssZ)
            first_comment: Configuration for first comment
            disable_comments: Whether to disable comments
            shorten_links: Whether to shorten links
            auto_schedule: Configuration for automatic scheduling
            auto_repost: Configuration for automatic reposting
            auto_hashtag: Configuration for automatic hashtags
            unsplash: Unsplash image configuration
            bluesky_options: Bluesky-specific options
            facebook_options: Facebook-specific options
            gmb_options: Google Business Profile options
            instagram_options: Instagram-specific options
            linkedin_options: LinkedIn-specific options
            pinterest_options: Pinterest-specific options
            reddit_options: Reddit-specific options
            snapchat_options: Snapchat-specific options
            telegram_options: Telegram-specific options
            threads_options: Threads-specific options
            tiktok_options: TikTok-specific options
            twitter_options: Twitter-specific options
            youtube_options: YouTube-specific options
            requires_approval: Whether to enable approval workflow
            random_post: Whether to generate random post text
            random_media_url: Whether to generate random media
            idempotency_key: Unique ID for the post
            notes: Additional notes for the post

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
        if first_comment:
            first_comment_dict = first_comment.__dict__.copy()
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
            auto_schedule_dict = auto_schedule.__dict__.copy()
            if auto_schedule.platforms:
                auto_schedule_dict["platforms"] = [
                    p.value for p in auto_schedule.platforms
                ]
            payload["autoSchedule"] = auto_schedule_dict
        if auto_repost:
            auto_repost_dict = auto_repost.__dict__.copy()
            if auto_repost.platforms:
                auto_repost_dict["platforms"] = [p.value for p in auto_repost.platforms]
            payload["autoRepost"] = auto_repost_dict
        if auto_hashtag:
            payload["autoHashtag"] = (
                auto_hashtag.__dict__
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

        # Return the first post from the response
        # This is because Ayrshare returns an array of posts even for single posts
        return PostResponse(**response_data["posts"][0])
