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


class PostToRedditBlock(Block):
    """Block for posting to Reddit."""

    class Input(BaseAyrshareInput):
        """Input schema for Reddit posts."""

        pass  # Uses all base fields

    class Output(BlockSchema):
        post_result: PostResponse = SchemaField(description="The result of the post")
        post: PostIds = SchemaField(description="The result of the post")

    def __init__(self):
        super().__init__(
            disabled=True,
            id="c7733580-3c72-483e-8e47-a8d58754d853",
            description="Post to Reddit using Ayrshare",
            categories={BlockCategory.SOCIAL},
            block_type=BlockType.AYRSHARE,
            input_schema=PostToRedditBlock.Input,
            output_schema=PostToRedditBlock.Output,
        )

    async def run(
        self, input_data: "PostToRedditBlock.Input", *, user_id: str, **kwargs
    ) -> BlockOutput:
        profile_key = await get_profile_key(user_id)
        if not profile_key:
            yield "error", "Please link a social account via Ayrshare"
            return
        client = create_ayrshare_client()
        if not client:
            yield "error", "Ayrshare integration is not configured."
            return
        iso_date = (
            input_data.schedule_date.isoformat() if input_data.schedule_date else None
        )
        response = await client.create_post(
            post=input_data.post,
            platforms=[SocialPlatform.REDDIT],
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
            profile_key=profile_key.get_secret_value(),
        )
        yield "post_result", response
        if response.postIds:
            for p in response.postIds:
                yield "post", p
