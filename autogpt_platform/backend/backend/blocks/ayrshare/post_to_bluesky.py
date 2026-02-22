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


class PostToBlueskyBlock(Block):
    """Block for posting to Bluesky with Bluesky-specific options."""

    class Input(BaseAyrshareInput):
        """Input schema for Bluesky posts."""

        # Override post field to include character limit information
        post: str = SchemaField(
            description="The post text to be published (max 300 characters for Bluesky)",
            default="",
            advanced=False,
        )

        # Override media_urls to include Bluesky-specific constraints
        media_urls: list[str] = SchemaField(
            description="Optional list of media URLs to include. Bluesky supports up to 4 images or 1 video.",
            default_factory=list,
            advanced=False,
        )

        # Bluesky-specific options
        alt_text: list[str] = SchemaField(
            description="Alt text for each media item (accessibility)",
            default_factory=list,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        post_result: PostResponse = SchemaField(description="The result of the post")
        post: PostIds = SchemaField(description="The result of the post")

    def __init__(self):
        super().__init__(
            disabled=True,
            id="cbd52c2a-06d2-43ed-9560-6576cc163283",
            description="Post to Bluesky using Ayrshare",
            categories={BlockCategory.SOCIAL},
            block_type=BlockType.AYRSHARE,
            input_schema=PostToBlueskyBlock.Input,
            output_schema=PostToBlueskyBlock.Output,
        )

    async def run(
        self,
        input_data: "PostToBlueskyBlock.Input",
        *,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        """Post to Bluesky with Bluesky-specific options."""

        profile_key = await get_profile_key(user_id)
        if not profile_key:
            yield "error", "Please link a social account via Ayrshare"
            return

        client = create_ayrshare_client()
        if not client:
            yield "error", "Ayrshare integration is not configured. Please set up the AYRSHARE_API_KEY."
            return

        # Validate character limit for Bluesky
        if len(input_data.post) > 300:
            yield "error", f"Post text exceeds Bluesky's 300 character limit ({len(input_data.post)} characters)"
            return

        # Validate media constraints for Bluesky
        if len(input_data.media_urls) > 4:
            yield "error", "Bluesky supports a maximum of 4 images or 1 video"
            return

        # Convert datetime to ISO format if provided
        iso_date = (
            input_data.schedule_date.isoformat() if input_data.schedule_date else None
        )

        # Build Bluesky-specific options
        bluesky_options = {}
        if input_data.alt_text:
            bluesky_options["altText"] = input_data.alt_text

        response = await client.create_post(
            post=input_data.post,
            platforms=[SocialPlatform.BLUESKY],
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
            bluesky_options=bluesky_options if bluesky_options else None,
            profile_key=profile_key.get_secret_value(),
        )
        yield "post_result", response
        if response.postIds:
            for p in response.postIds:
                yield "post", p
