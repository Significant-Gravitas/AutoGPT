from backend.integrations.ayrshare import PostIds, PostResponse, SocialPlatform
from backend.sdk import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    BlockType,
    SchemaField,
)

from ._util import BaseAyrshareInput, create_ayrshare_client, get_profile_key


class PostToSnapchatBlock(Block):
    """Block for posting to Snapchat with Snapchat-specific options."""

    class Input(BaseAyrshareInput):
        """Input schema for Snapchat posts."""

        # Override post field to include Snapchat-specific information
        post: str = SchemaField(
            description="The post text (optional for video-only content)",
            default="",
            advanced=False,
        )

        # Override media_urls to include Snapchat-specific constraints
        media_urls: list[str] = SchemaField(
            description="Required video URL for Snapchat posts. Snapchat only supports video content.",
            default_factory=list,
            advanced=False,
        )

        # Snapchat-specific options
        story_type: str = SchemaField(
            description="Type of Snapchat content: 'story' (24-hour Stories), 'saved_story' (Saved Stories), or 'spotlight' (Spotlight posts)",
            default="story",
            advanced=True,
        )
        video_thumbnail: str = SchemaField(
            description="Thumbnail URL for video content (optional, auto-generated if not provided)",
            default="",
            advanced=True,
        )

    class Output(BlockSchema):
        post_result: PostResponse = SchemaField(description="The result of the post")
        post: PostIds = SchemaField(description="The result of the post")

    def __init__(self):
        super().__init__(
            disabled=True,
            id="a9d7f854-2c83-4e96-b3a1-7f2e9c5d4b8e",
            description="Post to Snapchat using Ayrshare",
            categories={BlockCategory.SOCIAL},
            block_type=BlockType.AYRSHARE,
            input_schema=PostToSnapchatBlock.Input,
            output_schema=PostToSnapchatBlock.Output,
        )

    async def run(
        self,
        input_data: "PostToSnapchatBlock.Input",
        *,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        """Post to Snapchat with Snapchat-specific options."""
        profile_key = await get_profile_key(user_id)
        if not profile_key:
            yield "error", "Please link a social account via Ayrshare"
            return

        client = create_ayrshare_client()
        if not client:
            yield "error", "Ayrshare integration is not configured. Please set up the AYRSHARE_API_KEY."
            return

        # Validate Snapchat constraints
        if not input_data.media_urls:
            yield "error", "Snapchat requires at least one video URL"
            return

        if len(input_data.media_urls) > 1:
            yield "error", "Snapchat supports only one video per post"
            return

        # Validate story type
        valid_story_types = ["story", "saved_story", "spotlight"]
        if input_data.story_type not in valid_story_types:
            yield "error", f"Snapchat story type must be one of: {', '.join(valid_story_types)}"
            return

        # Convert datetime to ISO format if provided
        iso_date = (
            input_data.schedule_date.isoformat() if input_data.schedule_date else None
        )

        # Build Snapchat-specific options
        snapchat_options = {}

        # Story type
        if input_data.story_type != "story":
            snapchat_options["storyType"] = input_data.story_type

        # Video thumbnail
        if input_data.video_thumbnail:
            snapchat_options["videoThumbnail"] = input_data.video_thumbnail

        response = await client.create_post(
            post=input_data.post,
            platforms=[SocialPlatform.SNAPCHAT],
            media_urls=input_data.media_urls,
            is_video=True,  # Snapchat only supports video
            schedule_date=iso_date,
            disable_comments=input_data.disable_comments,
            shorten_links=input_data.shorten_links,
            unsplash=input_data.unsplash,
            requires_approval=input_data.requires_approval,
            random_post=input_data.random_post,
            random_media_url=input_data.random_media_url,
            notes=input_data.notes,
            snapchat_options=snapchat_options if snapchat_options else None,
            profile_key=profile_key.get_secret_value(),
        )
        yield "post_result", response
        if response.postIds:
            for p in response.postIds:
                yield "post", p
