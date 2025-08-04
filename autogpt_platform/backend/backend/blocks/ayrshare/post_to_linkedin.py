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


class PostToLinkedInBlock(Block):
    """Block for posting to LinkedIn with LinkedIn-specific options."""

    class Input(BaseAyrshareInput):
        """Input schema for LinkedIn posts."""

        # Override post field to include LinkedIn-specific information
        post: str = SchemaField(
            description="The post text (max 3,000 chars, hashtags supported with #)",
            default="",
            advanced=False,
        )

        # Override media_urls to include LinkedIn-specific constraints
        media_urls: list[str] = SchemaField(
            description="Optional list of media URLs. LinkedIn supports up to 9 images, videos, or documents (PPT, PPTX, DOC, DOCX, PDF <100MB, <300 pages).",
            default_factory=list,
            advanced=False,
        )

        # LinkedIn-specific options
        visibility: str = SchemaField(
            description="Post visibility: 'public' (default), 'connections' (personal only), 'loggedin'",
            default="public",
            advanced=True,
        )
        alt_text: list[str] = SchemaField(
            description="Alt text for each image (accessibility feature, not supported for videos/documents)",
            default_factory=list,
            advanced=True,
        )
        titles: list[str] = SchemaField(
            description="Title/caption for each image or video",
            default_factory=list,
            advanced=True,
        )
        document_title: str = SchemaField(
            description="Title for document posts (max 400 chars, uses filename if not specified)",
            default="",
            advanced=True,
        )
        thumbnail: str = SchemaField(
            description="Thumbnail URL for video (PNG/JPG, same dimensions as video, <10MB)",
            default="",
            advanced=True,
        )
        # LinkedIn targeting options (flattened from LinkedInTargeting object)
        targeting_countries: list[str] | None = SchemaField(
            description="Country codes for targeting (e.g., ['US', 'IN', 'DE', 'GB']). Requires 300+ followers in target audience.",
            default=None,
            advanced=True,
        )
        targeting_seniorities: list[str] | None = SchemaField(
            description="Seniority levels for targeting (e.g., ['Senior', 'VP']). Requires 300+ followers in target audience.",
            default=None,
            advanced=True,
        )
        targeting_degrees: list[str] | None = SchemaField(
            description="Education degrees for targeting. Requires 300+ followers in target audience.",
            default=None,
            advanced=True,
        )
        targeting_fields_of_study: list[str] | None = SchemaField(
            description="Fields of study for targeting. Requires 300+ followers in target audience.",
            default=None,
            advanced=True,
        )
        targeting_industries: list[str] | None = SchemaField(
            description="Industry categories for targeting. Requires 300+ followers in target audience.",
            default=None,
            advanced=True,
        )
        targeting_job_functions: list[str] | None = SchemaField(
            description="Job function categories for targeting. Requires 300+ followers in target audience.",
            default=None,
            advanced=True,
        )
        targeting_staff_count_ranges: list[str] | None = SchemaField(
            description="Company size ranges for targeting. Requires 300+ followers in target audience.",
            default=None,
            advanced=True,
        )

    class Output(BlockSchema):
        post_result: PostResponse = SchemaField(description="The result of the post")
        post: PostIds = SchemaField(description="The result of the post")

    def __init__(self):
        super().__init__(
            id="589af4e4-507f-42fd-b9ac-a67ecef25811",
            description="Post to LinkedIn using Ayrshare",
            categories={BlockCategory.SOCIAL},
            block_type=BlockType.AYRSHARE,
            input_schema=PostToLinkedInBlock.Input,
            output_schema=PostToLinkedInBlock.Output,
        )

    async def run(
        self,
        input_data: "PostToLinkedInBlock.Input",
        *,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        """Post to LinkedIn with LinkedIn-specific options."""
        profile_key = await get_profile_key(user_id)
        if not profile_key:
            yield "error", "Please link a social account via Ayrshare"
            return

        client = create_ayrshare_client()
        if not client:
            yield "error", "Ayrshare integration is not configured. Please set up the AYRSHARE_API_KEY."
            return

        # Validate LinkedIn constraints
        if len(input_data.post) > 3000:
            yield "error", f"LinkedIn post text exceeds 3,000 character limit ({len(input_data.post)} characters)"
            return

        if len(input_data.media_urls) > 9:
            yield "error", "LinkedIn supports a maximum of 9 images/videos/documents"
            return

        if input_data.document_title and len(input_data.document_title) > 400:
            yield "error", f"LinkedIn document title exceeds 400 character limit ({len(input_data.document_title)} characters)"
            return

        # Validate visibility option
        valid_visibility = ["public", "connections", "loggedin"]
        if input_data.visibility not in valid_visibility:
            yield "error", f"LinkedIn visibility must be one of: {', '.join(valid_visibility)}"
            return

        # Check for document extensions
        document_extensions = [".ppt", ".pptx", ".doc", ".docx", ".pdf"]
        has_documents = any(
            any(url.lower().endswith(ext) for ext in document_extensions)
            for url in input_data.media_urls
        )

        # Convert datetime to ISO format if provided
        iso_date = (
            input_data.schedule_date.isoformat() if input_data.schedule_date else None
        )

        # Build LinkedIn-specific options
        linkedin_options = {}

        # Visibility
        if input_data.visibility != "public":
            linkedin_options["visibility"] = input_data.visibility

        # Alt text (not supported for videos or documents)
        if input_data.alt_text and not has_documents:
            linkedin_options["altText"] = input_data.alt_text

        # Titles/captions
        if input_data.titles:
            linkedin_options["titles"] = input_data.titles

        # Document title
        if input_data.document_title and has_documents:
            linkedin_options["title"] = input_data.document_title

        # Video thumbnail
        if input_data.thumbnail:
            linkedin_options["thumbNail"] = input_data.thumbnail

        # Audience targeting (from flattened fields)
        targeting_dict = {}
        if input_data.targeting_countries:
            targeting_dict["countries"] = input_data.targeting_countries
        if input_data.targeting_seniorities:
            targeting_dict["seniorities"] = input_data.targeting_seniorities
        if input_data.targeting_degrees:
            targeting_dict["degrees"] = input_data.targeting_degrees
        if input_data.targeting_fields_of_study:
            targeting_dict["fieldsOfStudy"] = input_data.targeting_fields_of_study
        if input_data.targeting_industries:
            targeting_dict["industries"] = input_data.targeting_industries
        if input_data.targeting_job_functions:
            targeting_dict["jobFunctions"] = input_data.targeting_job_functions
        if input_data.targeting_staff_count_ranges:
            targeting_dict["staffCountRanges"] = input_data.targeting_staff_count_ranges

        if targeting_dict:
            linkedin_options["targeting"] = targeting_dict

        response = await client.create_post(
            post=input_data.post,
            platforms=[SocialPlatform.LINKEDIN],
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
            linkedin_options=linkedin_options if linkedin_options else None,
            profile_key=profile_key.get_secret_value(),
        )
        yield "post_result", response
        if response.postIds:
            for p in response.postIds:
                yield "post", p
