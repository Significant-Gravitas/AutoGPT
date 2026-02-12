from backend.integrations.ayrshare import PostIds, PostResponse, SocialPlatform
from backend.sdk import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaOutput,
    BlockType,
    SchemaField,
)

from ._util import (
    BaseAyrshareInput,
    PinterestCarouselOption,
    create_ayrshare_client,
    get_profile_key,
)


class PostToPinterestBlock(Block):
    """Block for posting to Pinterest with Pinterest-specific options."""

    class Input(BaseAyrshareInput):
        """Input schema for Pinterest posts."""

        # Override post field to include Pinterest-specific information
        post: str = SchemaField(
            description="Pin description (max 500 chars, links not clickable - use link field instead)",
            default="",
            advanced=False,
        )

        # Override media_urls to include Pinterest-specific constraints
        media_urls: list[str] = SchemaField(
            description="Required image/video URLs. Pinterest requires at least one image. Videos need thumbnail. Up to 5 images for carousel.",
            default_factory=list,
            advanced=False,
        )

        # Pinterest-specific options
        pin_title: str = SchemaField(
            description="Pin title displayed in 'Add your title' section (max 100 chars)",
            default="",
            advanced=True,
        )
        link: str = SchemaField(
            description="Clickable destination URL when users click the pin (max 2048 chars)",
            default="",
            advanced=True,
        )
        board_id: str = SchemaField(
            description="Pinterest Board ID to post to (from /user/details endpoint, uses default board if not specified)",
            default="",
            advanced=True,
        )
        note: str = SchemaField(
            description="Private note for the pin (only visible to you and board collaborators)",
            default="",
            advanced=True,
        )
        thumbnail: str = SchemaField(
            description="Required thumbnail URL for video pins (must have valid image Content-Type)",
            default="",
            advanced=True,
        )
        carousel_options: list[PinterestCarouselOption] = SchemaField(
            description="Options for each image in carousel (title, link, description per image)",
            default_factory=list,
            advanced=True,
        )
        alt_text: list[str] = SchemaField(
            description="Alt text for each image/video (max 500 chars each, accessibility feature)",
            default_factory=list,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        post_result: PostResponse = SchemaField(description="The result of the post")
        post: PostIds = SchemaField(description="The result of the post")

    def __init__(self):
        super().__init__(
            disabled=True,
            id="3ca46e05-dbaa-4afb-9e95-5a429c4177e6",
            description="Post to Pinterest using Ayrshare",
            categories={BlockCategory.SOCIAL},
            block_type=BlockType.AYRSHARE,
            input_schema=PostToPinterestBlock.Input,
            output_schema=PostToPinterestBlock.Output,
        )

    async def run(
        self,
        input_data: "PostToPinterestBlock.Input",
        *,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        """Post to Pinterest with Pinterest-specific options."""
        profile_key = await get_profile_key(user_id)
        if not profile_key:
            yield "error", "Please link a social account via Ayrshare"
            return

        client = create_ayrshare_client()
        if not client:
            yield "error", "Ayrshare integration is not configured. Please set up the AYRSHARE_API_KEY."
            return

        # Validate Pinterest constraints
        if len(input_data.post) > 500:
            yield "error", f"Pinterest pin description exceeds 500 character limit ({len(input_data.post)} characters)"
            return

        if len(input_data.pin_title) > 100:
            yield "error", f"Pinterest pin title exceeds 100 character limit ({len(input_data.pin_title)} characters)"
            return

        if len(input_data.link) > 2048:
            yield "error", f"Pinterest link URL exceeds 2048 character limit ({len(input_data.link)} characters)"
            return

        if len(input_data.media_urls) == 0:
            yield "error", "Pinterest requires at least one image or video"
            return

        if len(input_data.media_urls) > 5:
            yield "error", "Pinterest supports a maximum of 5 images in a carousel"
            return

        # Check if video is included and thumbnail is provided
        video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm"]
        has_video = any(
            any(url.lower().endswith(ext) for ext in video_extensions)
            for url in input_data.media_urls
        )

        if (has_video or input_data.is_video) and not input_data.thumbnail:
            yield "error", "Pinterest video pins require a thumbnail URL"
            return

        # Validate alt text length
        for i, alt in enumerate(input_data.alt_text):
            if len(alt) > 500:
                yield "error", f"Pinterest alt text {i+1} exceeds 500 character limit ({len(alt)} characters)"
                return

        # Convert datetime to ISO format if provided
        iso_date = (
            input_data.schedule_date.isoformat() if input_data.schedule_date else None
        )

        # Build Pinterest-specific options
        pinterest_options = {}

        # Pin title
        if input_data.pin_title:
            pinterest_options["title"] = input_data.pin_title

        # Clickable link
        if input_data.link:
            pinterest_options["link"] = input_data.link

        # Board ID
        if input_data.board_id:
            pinterest_options["boardId"] = input_data.board_id

        # Private note
        if input_data.note:
            pinterest_options["note"] = input_data.note

        # Video thumbnail
        if input_data.thumbnail:
            pinterest_options["thumbNail"] = input_data.thumbnail

        # Carousel options
        if input_data.carousel_options:
            carousel_list = []
            for option in input_data.carousel_options:
                carousel_dict = {}
                if option.title:
                    carousel_dict["title"] = option.title
                if option.link:
                    carousel_dict["link"] = option.link
                if option.description:
                    carousel_dict["description"] = option.description
                if carousel_dict:  # Only add if not empty
                    carousel_list.append(carousel_dict)
            if carousel_list:
                pinterest_options["carouselOptions"] = carousel_list

        # Alt text
        if input_data.alt_text:
            pinterest_options["altText"] = input_data.alt_text

        response = await client.create_post(
            post=input_data.post,
            platforms=[SocialPlatform.PINTEREST],
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
            pinterest_options=pinterest_options if pinterest_options else None,
            profile_key=profile_key.get_secret_value(),
        )
        yield "post_result", response
        if response.postIds:
            for p in response.postIds:
                yield "post", p
