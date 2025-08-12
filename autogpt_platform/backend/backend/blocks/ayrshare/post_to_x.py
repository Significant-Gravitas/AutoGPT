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


class PostToXBlock(Block):
    """Block for posting to X / Twitter with Twitter-specific options."""

    class Input(BaseAyrshareInput):
        """Input schema for X / Twitter posts."""

        # Override post field to include X-specific information
        post: str = SchemaField(
            description="The post text (max 280 chars, up to 25,000 for Premium users). Use @handle to mention users. Use \\n\\n for thread breaks.",
            advanced=False,
        )

        # Override media_urls to include X-specific constraints
        media_urls: list[str] = SchemaField(
            description="Optional list of media URLs. X supports up to 4 images or videos per tweet. Auto-preview links unless media is included.",
            default_factory=list,
            advanced=False,
        )

        # X-specific options
        reply_to_id: str | None = SchemaField(
            description="ID of the tweet to reply to",
            default=None,
            advanced=True,
        )
        quote_tweet_id: str | None = SchemaField(
            description="ID of the tweet to quote (low-level Tweet ID)",
            default=None,
            advanced=True,
        )
        poll_options: list[str] = SchemaField(
            description="Poll options (2-4 choices)",
            default_factory=list,
            advanced=True,
        )
        poll_duration: int = SchemaField(
            description="Poll duration in minutes (1-10080)",
            default=1440,
            advanced=True,
        )
        alt_text: list[str] = SchemaField(
            description="Alt text for each image (max 1,000 chars each, not supported for videos)",
            default_factory=list,
            advanced=True,
        )
        is_thread: bool = SchemaField(
            description="Whether to automatically break post into thread based on line breaks",
            default=False,
            advanced=True,
        )
        thread_number: bool = SchemaField(
            description="Add thread numbers (1/n format) to each thread post",
            default=False,
            advanced=True,
        )
        thread_media_urls: list[str] = SchemaField(
            description="Media URLs for thread posts (one per thread, use 'null' to skip)",
            default_factory=list,
            advanced=True,
        )
        long_post: bool = SchemaField(
            description="Force long form post (requires Premium X account)",
            default=False,
            advanced=True,
        )
        long_video: bool = SchemaField(
            description="Enable long video upload (requires approval and Business/Enterprise plan)",
            default=False,
            advanced=True,
        )
        subtitle_url: str = SchemaField(
            description="URL to SRT subtitle file for videos (must be HTTPS and end in .srt)",
            default="",
            advanced=True,
        )
        subtitle_language: str = SchemaField(
            description="Language code for subtitles (default: 'en')",
            default="en",
            advanced=True,
        )
        subtitle_name: str = SchemaField(
            description="Name of caption track (max 150 chars, default: 'English')",
            default="English",
            advanced=True,
        )

    class Output(BlockSchema):
        post_result: PostResponse = SchemaField(description="The result of the post")
        post: PostIds = SchemaField(description="The result of the post")

    def __init__(self):
        super().__init__(
            id="9e8f844e-b4a5-4b25-80f2-9e1dd7d67625",
            description="Post to X / Twitter using Ayrshare",
            categories={BlockCategory.SOCIAL},
            block_type=BlockType.AYRSHARE,
            input_schema=PostToXBlock.Input,
            output_schema=PostToXBlock.Output,
        )

    async def run(
        self,
        input_data: "PostToXBlock.Input",
        *,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        """Post to X / Twitter with enhanced X-specific options."""
        profile_key = await get_profile_key(user_id)
        if not profile_key:
            yield "error", "Please link a social account via Ayrshare"
            return

        client = create_ayrshare_client()
        if not client:
            yield "error", "Ayrshare integration is not configured. Please set up the AYRSHARE_API_KEY."
            return

        # Validate X constraints
        if not input_data.long_post and len(input_data.post) > 280:
            yield "error", f"X post text exceeds 280 character limit ({len(input_data.post)} characters). Enable 'long_post' for Premium accounts."
            return

        if input_data.long_post and len(input_data.post) > 25000:
            yield "error", f"X long post text exceeds 25,000 character limit ({len(input_data.post)} characters)"
            return

        if len(input_data.media_urls) > 4:
            yield "error", "X supports a maximum of 4 images or videos per tweet"
            return

        # Validate poll options
        if input_data.poll_options:
            if len(input_data.poll_options) < 2 or len(input_data.poll_options) > 4:
                yield "error", "X polls require 2-4 options"
                return

            if input_data.poll_duration < 1 or input_data.poll_duration > 10080:
                yield "error", "X poll duration must be between 1 and 10,080 minutes (7 days)"
                return

        # Validate alt text
        if input_data.alt_text:
            for i, alt in enumerate(input_data.alt_text):
                if len(alt) > 1000:
                    yield "error", f"X alt text {i+1} exceeds 1,000 character limit ({len(alt)} characters)"
                    return

        # Validate subtitle settings
        if input_data.subtitle_url:
            if not input_data.subtitle_url.startswith(
                "https://"
            ) or not input_data.subtitle_url.endswith(".srt"):
                yield "error", "Subtitle URL must start with https:// and end with .srt"
                return

            if len(input_data.subtitle_name) > 150:
                yield "error", f"Subtitle name exceeds 150 character limit ({len(input_data.subtitle_name)} characters)"
                return

        # Convert datetime to ISO format if provided
        iso_date = (
            input_data.schedule_date.isoformat() if input_data.schedule_date else None
        )

        # Build X-specific options
        twitter_options = {}

        # Basic options
        if input_data.reply_to_id:
            twitter_options["replyToId"] = input_data.reply_to_id

        if input_data.quote_tweet_id:
            twitter_options["quoteTweetId"] = input_data.quote_tweet_id

        if input_data.long_post:
            twitter_options["longPost"] = True

        if input_data.long_video:
            twitter_options["longVideo"] = True

        # Poll options
        if input_data.poll_options:
            twitter_options["poll"] = {
                "duration": input_data.poll_duration,
                "options": input_data.poll_options,
            }

        # Alt text for images
        if input_data.alt_text:
            twitter_options["altText"] = input_data.alt_text

        # Thread options
        if input_data.is_thread:
            twitter_options["thread"] = True

            if input_data.thread_number:
                twitter_options["threadNumber"] = True

            if input_data.thread_media_urls:
                twitter_options["mediaUrls"] = input_data.thread_media_urls

        # Video subtitle options
        if input_data.subtitle_url:
            twitter_options["subTitleUrl"] = input_data.subtitle_url
            twitter_options["subTitleLanguage"] = input_data.subtitle_language
            twitter_options["subTitleName"] = input_data.subtitle_name

        response = await client.create_post(
            post=input_data.post,
            platforms=[SocialPlatform.TWITTER],
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
            twitter_options=twitter_options if twitter_options else None,
            profile_key=profile_key.get_secret_value(),
        )
        yield "post_result", response
        if response.postIds:
            for p in response.postIds:
                yield "post", p
