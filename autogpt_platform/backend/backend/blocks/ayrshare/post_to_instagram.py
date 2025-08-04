from typing import Any

from backend.integrations.ayrshare import PostIds, PostResponse, SocialPlatform
from backend.sdk import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    BlockType,
    SchemaField,
)

from ._util import (
    BaseAyrshareInput,
    InstagramUserTag,
    create_ayrshare_client,
    get_profile_key,
)


class PostToInstagramBlock(Block):
    """Block for posting to Instagram with Instagram-specific options."""

    class Input(BaseAyrshareInput):
        """Input schema for Instagram posts."""

        # Override post field to include Instagram-specific information
        post: str = SchemaField(
            description="The post text (max 2,200 chars, up to 30 hashtags, 3 @mentions)",
            default="",
            advanced=False,
        )

        # Override media_urls to include Instagram-specific constraints
        media_urls: list[str] = SchemaField(
            description="Optional list of media URLs. Instagram supports up to 10 images/videos in a carousel.",
            default_factory=list,
            advanced=False,
        )

        # Instagram-specific options
        is_story: bool | None = SchemaField(
            description="Whether to post as Instagram Story (24-hour expiration)",
            default=None,
            advanced=True,
        )

        # ------- REELS OPTIONS -------
        share_reels_feed: bool | None = SchemaField(
            description="Whether Reel should appear in both Feed and Reels tabs",
            default=None,
            advanced=True,
        )
        audio_name: str | None = SchemaField(
            description="Audio name for Reels (e.g., 'The Weeknd - Blinding Lights')",
            default=None,
            advanced=True,
        )
        thumbnail: str | None = SchemaField(
            description="Thumbnail URL for Reel video", default=None, advanced=True
        )
        thumbnail_offset: int | None = SchemaField(
            description="Thumbnail frame offset in milliseconds (default: 0)",
            default=0,
            advanced=True,
        )

        # ------- POST OPTIONS -------

        alt_text: list[str] = SchemaField(
            description="Alt text for each media item (up to 1,000 chars each, accessibility feature), each item in the list corresponds to a media item in the media_urls list",
            default_factory=list,
            advanced=True,
        )

        location_id: str | None = SchemaField(
            description="Facebook Page ID or name for location tagging (e.g., '7640348500' or '@guggenheimmuseum')",
            default=None,
            advanced=True,
        )
        user_tags: list[dict[str, Any]] = SchemaField(
            description="List of users to tag with coordinates for images",
            default_factory=list,
            advanced=True,
        )
        collaborators: list[str] = SchemaField(
            description="Instagram usernames to invite as collaborators (max 3, public accounts only)",
            default_factory=list,
            advanced=True,
        )
        auto_resize: bool | None = SchemaField(
            description="Auto-resize images to 1080x1080px for Instagram",
            default=None,
            advanced=True,
        )

    class Output(BlockSchema):
        post_result: PostResponse = SchemaField(description="The result of the post")
        post: PostIds = SchemaField(description="The result of the post")

    def __init__(self):
        super().__init__(
            id="89b02b96-a7cb-46f4-9900-c48b32fe1552",
            description="Post to Instagram using Ayrshare. Requires a Business or Creator Instagram Account connected with a Facebook Page",
            categories={BlockCategory.SOCIAL},
            block_type=BlockType.AYRSHARE,
            input_schema=PostToInstagramBlock.Input,
            output_schema=PostToInstagramBlock.Output,
        )

    async def run(
        self,
        input_data: "PostToInstagramBlock.Input",
        *,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        """Post to Instagram with Instagram-specific options."""
        profile_key = await get_profile_key(user_id)
        if not profile_key:
            yield "error", "Please link a social account via Ayrshare"
            return

        client = create_ayrshare_client()
        if not client:
            yield "error", "Ayrshare integration is not configured. Please set up the AYRSHARE_API_KEY."
            return

        # Validate Instagram constraints
        if len(input_data.post) > 2200:
            yield "error", f"Instagram post text exceeds 2,200 character limit ({len(input_data.post)} characters)"
            return

        if len(input_data.media_urls) > 10:
            yield "error", "Instagram supports a maximum of 10 images/videos in a carousel"
            return

        if len(input_data.collaborators) > 3:
            yield "error", "Instagram supports a maximum of 3 collaborators"
            return

        # Validate that if any reel option is set, all required reel options are set
        reel_options = [
            input_data.share_reels_feed,
            input_data.audio_name,
            input_data.thumbnail,
        ]

        if any(reel_options) and not all(reel_options):
            yield "error", "When posting a reel, all reel options must be set: share_reels_feed, audio_name, and either thumbnail or thumbnail_offset"
            return

        # Count hashtags and mentions
        hashtag_count = input_data.post.count("#")
        mention_count = input_data.post.count("@")

        if hashtag_count > 30:
            yield "error", f"Instagram allows maximum 30 hashtags ({hashtag_count} found)"
            return

        if mention_count > 3:
            yield "error", f"Instagram allows maximum 3 @mentions ({mention_count} found)"
            return

        # Convert datetime to ISO format if provided
        iso_date = (
            input_data.schedule_date.isoformat() if input_data.schedule_date else None
        )

        # Build Instagram-specific options
        instagram_options = {}

        # Stories
        if input_data.is_story:
            instagram_options["stories"] = True

        # Reels options
        if input_data.share_reels_feed is not None:
            instagram_options["shareReelsFeed"] = input_data.share_reels_feed

        if input_data.audio_name:
            instagram_options["audioName"] = input_data.audio_name

        if input_data.thumbnail:
            instagram_options["thumbNail"] = input_data.thumbnail
        elif input_data.thumbnail_offset and input_data.thumbnail_offset > 0:
            instagram_options["thumbNailOffset"] = input_data.thumbnail_offset

        # Alt text
        if input_data.alt_text:
            # Validate alt text length
            for i, alt in enumerate(input_data.alt_text):
                if len(alt) > 1000:
                    yield "error", f"Alt text {i+1} exceeds 1,000 character limit ({len(alt)} characters)"
                    return
            instagram_options["altText"] = input_data.alt_text

        # Location
        if input_data.location_id:
            instagram_options["locationId"] = input_data.location_id

        # User tags
        if input_data.user_tags:
            user_tags_list = []
            for tag in input_data.user_tags:
                try:
                    tag_obj = InstagramUserTag(**tag)
                except Exception as e:
                    yield "error", f"Invalid user tag: {e}, tages need to be a dictionary with a 3 items: username (str), x (float) and y (float)"
                    return
                tag_dict: dict[str, float | str] = {"username": tag_obj.username}
                if tag_obj.x is not None and tag_obj.y is not None:
                    # Validate coordinates
                    if not (0.0 <= tag_obj.x <= 1.0) or not (0.0 <= tag_obj.y <= 1.0):
                        yield "error", f"User tag coordinates must be between 0.0 and 1.0 (user: {tag_obj.username})"
                        return
                    tag_dict["x"] = tag_obj.x
                    tag_dict["y"] = tag_obj.y
                user_tags_list.append(tag_dict)
            instagram_options["userTags"] = user_tags_list

        # Collaborators
        if input_data.collaborators:
            instagram_options["collaborators"] = input_data.collaborators

        # Auto resize
        if input_data.auto_resize:
            instagram_options["autoResize"] = True

        response = await client.create_post(
            post=input_data.post,
            platforms=[SocialPlatform.INSTAGRAM],
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
            instagram_options=instagram_options if instagram_options else None,
            profile_key=profile_key.get_secret_value(),
        )
        yield "post_result", response
        if response.postIds:
            for p in response.postIds:
                yield "post", p
