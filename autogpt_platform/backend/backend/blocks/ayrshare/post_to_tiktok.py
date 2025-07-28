from backend.integrations.ayrshare import PostIds, PostResponse, SocialPlatform
from backend.sdk import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    BlockType,
    SchemaField,
    SecretStr,
)

from ._util import BaseAyrshareInput, create_ayrshare_client


class PostToTikTokBlock(Block):
    """Block for posting to TikTok with TikTok-specific options."""

    class Input(BaseAyrshareInput):
        """Input schema for TikTok posts."""

        # Override post field to include TikTok-specific information
        post: str = SchemaField(
            description="The post text (max 2,200 chars, empty string allowed). Use @handle to mention users. Line breaks will be ignored.",
            default="",
            advanced=False,
        )

        # Override media_urls to include TikTok-specific constraints
        media_urls: list[str] = SchemaField(
            description="Required media URLs. Either 1 video OR up to 35 images (JPG/JPEG/WEBP only). Cannot mix video and images.",
            default_factory=list,
            advanced=False,
        )

        # TikTok-specific options
        auto_add_music: bool = SchemaField(
            description="Automatically add recommended music to image posts",
            default=False,
            advanced=True,
        )
        disable_comments: bool = SchemaField(
            description="Disable comments on the published post",
            default=False,
            advanced=True,
        )
        disable_duet: bool = SchemaField(
            description="Disable duets on published video (video only)",
            default=False,
            advanced=True,
        )
        disable_stitch: bool = SchemaField(
            description="Disable stitch on published video (video only)",
            default=False,
            advanced=True,
        )
        is_ai_generated: bool = SchemaField(
            description="Label content as AI-generated (video only)",
            default=False,
            advanced=True,
        )
        is_branded_content: bool = SchemaField(
            description="Label as branded content (paid partnership)",
            default=False,
            advanced=True,
        )
        is_brand_organic: bool = SchemaField(
            description="Label as brand organic content (promotional)",
            default=False,
            advanced=True,
        )
        image_cover_index: int = SchemaField(
            description="Index of image to use as cover (0-based, image posts only)",
            default=0,
            advanced=True,
        )
        title: str = SchemaField(
            description="Title for image posts", default="", advanced=True
        )
        thumbnail_offset: int = SchemaField(
            description="Video thumbnail frame offset in milliseconds (video only)",
            default=0,
            advanced=True,
        )
        visibility: str = SchemaField(
            description="Post visibility: 'public', 'private', 'followers', or 'friends'",
            default="public",
            advanced=True,
        )
        draft: bool = SchemaField(
            description="Create as draft post (video only)",
            default=False,
            advanced=True,
        )

    class Output(BlockSchema):
        post_result: PostResponse = SchemaField(description="The result of the post")
        post: PostIds = SchemaField(description="The result of the post")

    def __init__(self):
        super().__init__(
            disabled=True,
            id="7faf4b27-96b0-4f05-bf64-e0de54ae74e1",
            description="Post to TikTok using Ayrshare",
            categories={BlockCategory.SOCIAL},
            block_type=BlockType.AYRSHARE,
            input_schema=PostToTikTokBlock.Input,
            output_schema=PostToTikTokBlock.Output,
        )

    async def run(
        self, input_data: "PostToTikTokBlock.Input", *, profile_key: SecretStr, **kwargs
    ) -> BlockOutput:
        """Post to TikTok with TikTok-specific validation and options."""
        if not profile_key:
            yield "error", "Please link a social account via Ayrshare"
            return

        client = create_ayrshare_client()
        if not client:
            yield "error", "Ayrshare integration is not configured. Please set up the AYRSHARE_API_KEY."
            return

        # Validate TikTok constraints
        if len(input_data.post) > 2200:
            yield "error", f"TikTok post text exceeds 2,200 character limit ({len(input_data.post)} characters)"
            return

        if not input_data.media_urls:
            yield "error", "TikTok requires at least one media URL (either 1 video or up to 35 images)"
            return

        # Check for video vs image constraints
        video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm"]
        image_extensions = [".jpg", ".jpeg", ".webp"]

        has_video = input_data.is_video or any(
            any(url.lower().endswith(ext) for ext in video_extensions)
            for url in input_data.media_urls
        )

        has_images = any(
            any(url.lower().endswith(ext) for ext in image_extensions)
            for url in input_data.media_urls
        )

        if has_video and has_images:
            yield "error", "TikTok does not support mixing video and images in the same post"
            return

        if has_video and len(input_data.media_urls) > 1:
            yield "error", "TikTok supports only 1 video per post"
            return

        if has_images and len(input_data.media_urls) > 35:
            yield "error", "TikTok supports a maximum of 35 images per post"
            return

        # Validate image cover index
        if has_images and input_data.image_cover_index >= len(input_data.media_urls):
            yield "error", f"Image cover index {input_data.image_cover_index} is out of range (max: {len(input_data.media_urls) - 1})"
            return

        # Validate visibility option
        valid_visibility = ["public", "private", "followers", "friends"]
        if input_data.visibility not in valid_visibility:
            yield "error", f"TikTok visibility must be one of: {', '.join(valid_visibility)}"
            return

        # Check for PNG files (not supported)
        has_png = any(url.lower().endswith(".png") for url in input_data.media_urls)
        if has_png:
            yield "error", "TikTok does not support PNG files. Please use JPG, JPEG, or WEBP for images."
            return

        # Convert datetime to ISO format if provided
        iso_date = (
            input_data.schedule_date.isoformat() if input_data.schedule_date else None
        )

        # Build TikTok-specific options
        tiktok_options = {}

        # Common options
        if input_data.auto_add_music and has_images:
            tiktok_options["autoAddMusic"] = True

        if input_data.disable_comments:
            tiktok_options["disableComments"] = True

        if input_data.is_branded_content:
            tiktok_options["isBrandedContent"] = True

        if input_data.is_brand_organic:
            tiktok_options["isBrandOrganic"] = True

        # Video-specific options
        if has_video:
            if input_data.disable_duet:
                tiktok_options["disableDuet"] = True

            if input_data.disable_stitch:
                tiktok_options["disableStitch"] = True

            if input_data.is_ai_generated:
                tiktok_options["isAIGenerated"] = True

            if input_data.thumbnail_offset > 0:
                tiktok_options["thumbNailOffset"] = input_data.thumbnail_offset

            if input_data.draft:
                tiktok_options["draft"] = True

        # Image-specific options
        if has_images:
            if input_data.image_cover_index > 0:
                tiktok_options["imageCoverIndex"] = input_data.image_cover_index

            if input_data.title:
                tiktok_options["title"] = input_data.title

            if input_data.visibility != "public":
                tiktok_options["visibility"] = input_data.visibility

        response = await client.create_post(
            post=input_data.post,
            platforms=[SocialPlatform.TIKTOK],
            media_urls=input_data.media_urls,
            is_video=has_video,
            schedule_date=iso_date,
            disable_comments=input_data.disable_comments,
            shorten_links=input_data.shorten_links,
            unsplash=input_data.unsplash,
            requires_approval=input_data.requires_approval,
            random_post=input_data.random_post,
            random_media_url=input_data.random_media_url,
            notes=input_data.notes,
            tiktok_options=tiktok_options if tiktok_options else None,
            profile_key=profile_key.get_secret_value(),
        )
        yield "post_result", response
        if response.postIds:
            for p in response.postIds:
                yield "post", p
