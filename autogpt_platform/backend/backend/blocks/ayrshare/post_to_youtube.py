from enum import Enum
from typing import Any

from backend.integrations.ayrshare import PostIds, PostResponse, SocialPlatform
from backend.sdk import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaOutput,
    BlockType,
    SchemaField,
)

from ._util import BaseAyrshareInput, create_ayrshare_client, get_profile_key


class YouTubeVisibility(str, Enum):
    PRIVATE = "private"
    PUBLIC = "public"
    UNLISTED = "unlisted"


class PostToYouTubeBlock(Block):
    """Block for posting to YouTube with YouTube-specific options."""

    class Input(BaseAyrshareInput):
        """Input schema for YouTube posts."""

        # Override post field to include YouTube-specific information
        post: str = SchemaField(
            description="Video description (max 5,000 chars, empty string allowed). Cannot contain < or > characters.",
            advanced=False,
        )

        # Override media_urls to include YouTube-specific constraints
        media_urls: list[str] = SchemaField(
            description="Required video URL. YouTube only supports 1 video per post.",
            default_factory=list,
            advanced=False,
        )

        # YouTube-specific required options
        title: str = SchemaField(
            description="Video title (max 100 chars, required). Cannot contain < or > characters.",
            advanced=False,
        )

        # YouTube-specific optional options
        visibility: YouTubeVisibility = SchemaField(
            description="Video visibility: 'private' (default), 'public' , or 'unlisted'",
            default=YouTubeVisibility.PRIVATE,
            advanced=False,
        )
        thumbnail: str | None = SchemaField(
            description="Thumbnail URL (JPEG/PNG under 2MB, must end in .png/.jpg/.jpeg). Requires phone verification.",
            default=None,
            advanced=True,
        )
        playlist_id: str | None = SchemaField(
            description="Playlist ID to add video (user must own playlist)",
            default=None,
            advanced=True,
        )
        tags: list[str] | None = SchemaField(
            description="Video tags (min 2 chars each, max 500 chars total)",
            default=None,
            advanced=True,
        )
        made_for_kids: bool | None = SchemaField(
            description="Self-declared kids content", default=None, advanced=True
        )
        is_shorts: bool | None = SchemaField(
            description="Post as YouTube Short (max 3 minutes, adds #shorts)",
            default=None,
            advanced=True,
        )
        notify_subscribers: bool | None = SchemaField(
            description="Send notification to subscribers", default=None, advanced=True
        )
        category_id: int | None = SchemaField(
            description="Video category ID (e.g., 24 = Entertainment)",
            default=None,
            advanced=True,
        )
        contains_synthetic_media: bool | None = SchemaField(
            description="Disclose realistic AI/synthetic content",
            default=None,
            advanced=True,
        )
        publish_at: str | None = SchemaField(
            description="UTC publish time (YouTube controlled, format: 2022-10-08T21:18:36Z)",
            default=None,
            advanced=True,
        )
        # YouTube targeting options (flattened from YouTubeTargeting object)
        targeting_block_countries: list[str] | None = SchemaField(
            description="Country codes to block from viewing (e.g., ['US', 'CA'])",
            default=None,
            advanced=True,
        )
        targeting_allow_countries: list[str] | None = SchemaField(
            description="Country codes to allow viewing (e.g., ['GB', 'AU'])",
            default=None,
            advanced=True,
        )
        subtitle_url: str | None = SchemaField(
            description="URL to SRT or SBV subtitle file (must be HTTPS and end in .srt/.sbv, under 100MB)",
            default=None,
            advanced=True,
        )
        subtitle_language: str | None = SchemaField(
            description="Language code for subtitles (default: 'en')",
            default=None,
            advanced=True,
        )
        subtitle_name: str | None = SchemaField(
            description="Name of caption track (max 150 chars, default: 'English')",
            default=None,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        post_result: PostResponse = SchemaField(description="The result of the post")
        post: PostIds = SchemaField(description="The result of the post")

    def __init__(self):
        super().__init__(
            id="0082d712-ff1b-4c3d-8a8d-6c7721883b83",
            description="Post to YouTube using Ayrshare",
            categories={BlockCategory.SOCIAL},
            block_type=BlockType.AYRSHARE,
            input_schema=PostToYouTubeBlock.Input,
            output_schema=PostToYouTubeBlock.Output,
        )

    async def run(
        self,
        input_data: "PostToYouTubeBlock.Input",
        *,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        """Post to YouTube with YouTube-specific validation and options."""

        profile_key = await get_profile_key(user_id)
        if not profile_key:
            yield "error", "Please link a social account via Ayrshare"
            return

        client = create_ayrshare_client()
        if not client:
            yield "error", "Ayrshare integration is not configured. Please set up the AYRSHARE_API_KEY."
            return

        # Validate YouTube constraints
        if not input_data.title:
            yield "error", "YouTube requires a video title"
            return

        if len(input_data.title) > 100:
            yield "error", f"YouTube title exceeds 100 character limit ({len(input_data.title)} characters)"
            return

        if len(input_data.post) > 5000:
            yield "error", f"YouTube description exceeds 5,000 character limit ({len(input_data.post)} characters)"
            return

        # Check for forbidden characters
        forbidden_chars = ["<", ">"]
        for char in forbidden_chars:
            if char in input_data.title:
                yield "error", f"YouTube title cannot contain '{char}' character"
                return
            if char in input_data.post:
                yield "error", f"YouTube description cannot contain '{char}' character"
                return

        if not input_data.media_urls:
            yield "error", "YouTube requires exactly one video URL"
            return

        if len(input_data.media_urls) > 1:
            yield "error", "YouTube supports only 1 video per post"
            return

        # Validate visibility option
        valid_visibility = ["private", "public", "unlisted"]
        if input_data.visibility not in valid_visibility:
            yield "error", f"YouTube visibility must be one of: {', '.join(valid_visibility)}"
            return

        # Validate thumbnail URL format
        if input_data.thumbnail:
            valid_extensions = [".png", ".jpg", ".jpeg"]
            if not any(
                input_data.thumbnail.lower().endswith(ext) for ext in valid_extensions
            ):
                yield "error", "YouTube thumbnail must end in .png, .jpg, or .jpeg"
                return

        # Validate tags
        if input_data.tags:
            total_tag_length = sum(len(tag) for tag in input_data.tags)
            if total_tag_length > 500:
                yield "error", f"YouTube tags total length exceeds 500 characters ({total_tag_length} characters)"
                return

            for tag in input_data.tags:
                if len(tag) < 2:
                    yield "error", f"YouTube tag '{tag}' is too short (minimum 2 characters)"
                    return

        # Validate subtitle URL
        if input_data.subtitle_url:
            if not input_data.subtitle_url.startswith("https://"):
                yield "error", "YouTube subtitle URL must start with https://"
                return

            valid_subtitle_extensions = [".srt", ".sbv"]
            if not any(
                input_data.subtitle_url.lower().endswith(ext)
                for ext in valid_subtitle_extensions
            ):
                yield "error", "YouTube subtitle URL must end in .srt or .sbv"
                return

        if input_data.subtitle_name and len(input_data.subtitle_name) > 150:
            yield "error", f"YouTube subtitle name exceeds 150 character limit ({len(input_data.subtitle_name)} characters)"
            return

        # Validate publish_at format if provided
        if input_data.publish_at and input_data.schedule_date:
            yield "error", "Cannot use both 'publish_at' and 'schedule_date'. Use 'publish_at' for YouTube-controlled publishing."
            return

        # Convert datetime to ISO format if provided (only if not using publish_at)
        iso_date = None
        if not input_data.publish_at and input_data.schedule_date:
            iso_date = input_data.schedule_date.isoformat()

        # Build YouTube-specific options
        youtube_options: dict[str, Any] = {"title": input_data.title}

        # Basic options
        if input_data.visibility != "private":
            youtube_options["visibility"] = input_data.visibility

        if input_data.thumbnail:
            youtube_options["thumbNail"] = input_data.thumbnail

        if input_data.playlist_id:
            youtube_options["playListId"] = input_data.playlist_id

        if input_data.tags:
            youtube_options["tags"] = input_data.tags

        if input_data.made_for_kids:
            youtube_options["madeForKids"] = True

        if input_data.is_shorts:
            youtube_options["shorts"] = True

        if not input_data.notify_subscribers:
            youtube_options["notifySubscribers"] = False

        if input_data.category_id and input_data.category_id > 0:
            youtube_options["categoryId"] = input_data.category_id

        if input_data.contains_synthetic_media:
            youtube_options["containsSyntheticMedia"] = True

        if input_data.publish_at:
            youtube_options["publishAt"] = input_data.publish_at

        # Country targeting (from flattened fields)
        targeting_dict = {}
        if input_data.targeting_block_countries:
            targeting_dict["block"] = input_data.targeting_block_countries
        if input_data.targeting_allow_countries:
            targeting_dict["allow"] = input_data.targeting_allow_countries

        if targeting_dict:
            youtube_options["targeting"] = targeting_dict

        # Subtitle options
        if input_data.subtitle_url:
            youtube_options["subTitleUrl"] = input_data.subtitle_url
            youtube_options["subTitleLanguage"] = input_data.subtitle_language
            youtube_options["subTitleName"] = input_data.subtitle_name

        response = await client.create_post(
            post=input_data.post,
            platforms=[SocialPlatform.YOUTUBE],
            media_urls=input_data.media_urls,
            is_video=True,  # YouTube only supports videos
            schedule_date=iso_date,
            disable_comments=input_data.disable_comments,
            shorten_links=input_data.shorten_links,
            unsplash=input_data.unsplash,
            requires_approval=input_data.requires_approval,
            random_post=input_data.random_post,
            random_media_url=input_data.random_media_url,
            notes=input_data.notes,
            youtube_options=youtube_options,
            profile_key=profile_key.get_secret_value(),
        )
        yield "post_result", response
        if response.postIds:
            for p in response.postIds:
                yield "post", p
