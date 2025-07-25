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


class PostToTelegramBlock(Block):
    """Block for posting to Telegram with Telegram-specific options."""

    class Input(BaseAyrshareInput):
        """Input schema for Telegram posts."""

        # Override post field to include Telegram-specific information
        post: str = SchemaField(
            description="The post text (empty string allowed). Use @handle to mention other Telegram users.",
            default="",
            advanced=False,
        )

        # Override media_urls to include Telegram-specific constraints
        media_urls: list[str] = SchemaField(
            description="Optional list of media URLs. For animated GIFs, only one URL is allowed. Telegram will auto-preview links unless image/video is included.",
            default_factory=list,
            advanced=False,
        )

        # Override is_video to include GIF-specific information
        is_video: bool = SchemaField(
            description="Whether the media is a video. Set to true for animated GIFs that don't end in .gif/.GIF extension.",
            default=False,
            advanced=True,
        )

    class Output(BlockSchema):
        post_result: PostResponse = SchemaField(description="The result of the post")
        post: PostIds = SchemaField(description="The result of the post")

    def __init__(self):
        super().__init__(
            disabled=True,
            id="47bc74eb-4af2-452c-b933-af377c7287df",
            description="Post to Telegram using Ayrshare",
            categories={BlockCategory.SOCIAL},
            block_type=BlockType.AYRSHARE,
            input_schema=PostToTelegramBlock.Input,
            output_schema=PostToTelegramBlock.Output,
        )

    async def run(
        self,
        input_data: "PostToTelegramBlock.Input",
        *,
        profile_key: SecretStr,
        **kwargs,
    ) -> BlockOutput:
        """Post to Telegram with Telegram-specific validation."""
        if not profile_key:
            yield "error", "Please link a social account via Ayrshare"
            return

        client = create_ayrshare_client()
        if not client:
            yield "error", "Ayrshare integration is not configured. Please set up the AYRSHARE_API_KEY."
            return

        # Validate Telegram constraints
        # Check for animated GIFs - only one URL allowed
        gif_extensions = [".gif", ".GIF"]
        has_gif = any(
            any(url.endswith(ext) for ext in gif_extensions)
            for url in input_data.media_urls
        )

        if has_gif and len(input_data.media_urls) > 1:
            yield "error", "Telegram animated GIFs support only one URL per post"
            return

        # Auto-detect if we need to set is_video for GIFs without proper extension
        detected_is_video = input_data.is_video
        if input_data.media_urls and not has_gif and not input_data.is_video:
            # Check if this might be a GIF without proper extension
            # This is just informational - user needs to set is_video manually
            pass

        # Convert datetime to ISO format if provided
        iso_date = (
            input_data.schedule_date.isoformat() if input_data.schedule_date else None
        )

        response = await client.create_post(
            post=input_data.post,
            platforms=[SocialPlatform.TELEGRAM],
            media_urls=input_data.media_urls,
            is_video=detected_is_video,
            schedule_date=iso_date,
            disable_comments=input_data.disable_comments,
            shorten_links=input_data.shorten_links,
            unsplash=input_data.unsplash,
            requires_approval=input_data.requires_approval,
            random_post=input_data.random_post,
            random_media_url=input_data.random_media_url,
            notes=input_data.notes,
            profile_key=profile_key.get_secret_value(),
        )
        yield "post_result", response
        if response.postIds:
            for p in response.postIds:
                yield "post", p
