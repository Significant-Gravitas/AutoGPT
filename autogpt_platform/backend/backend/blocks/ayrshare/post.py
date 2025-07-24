from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, SecretStr

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema, BlockType
from backend.data.model import SchemaField
from backend.integrations.ayrshare import AyrshareClient, SocialPlatform
from backend.util.exceptions import MissingConfigError


class RequestOutput(BaseModel):
    """Base output model for Ayrshare social media posts."""

    status: str = Field(..., description="Status of the post")
    id: str = Field(..., description="ID of the post")
    refId: str = Field(..., description="Reference ID of the post")
    profileTitle: str = Field(..., description="Title of the profile")
    post: str = Field(..., description="The post text")
    postIds: Optional[list[dict]] = Field(
        description="IDs of the posts on each platform"
    )
    scheduleDate: Optional[str] = Field(description="Scheduled date of the post")
    errors: Optional[list[str]] = Field(description="Any errors that occurred")


class BaseAyrshareInput(BlockSchema):
    """Base input model for Ayrshare social media posts with common fields."""

    post: str = SchemaField(
        description="The post text to be published", default="", advanced=False
    )
    media_urls: list[str] = SchemaField(
        description="Optional list of media URLs to include. Set is_video in advanced settings to true if you want to upload videos.",
        default_factory=list,
        advanced=False,
    )
    is_video: bool = SchemaField(
        description="Whether the media is a video", default=False, advanced=True
    )
    schedule_date: Optional[datetime] = SchemaField(
        description="UTC datetime for scheduling (YYYY-MM-DDThh:mm:ssZ)",
        default=None,
        advanced=True,
    )
    disable_comments: bool = SchemaField(
        description="Whether to disable comments", default=False, advanced=True
    )
    shorten_links: bool = SchemaField(
        description="Whether to shorten links", default=False, advanced=True
    )
    unsplash: Optional[str] = SchemaField(
        description="Unsplash image configuration", default=None, advanced=True
    )
    requires_approval: bool = SchemaField(
        description="Whether to enable approval workflow",
        default=False,
        advanced=True,
    )
    random_post: bool = SchemaField(
        description="Whether to generate random post text",
        default=False,
        advanced=True,
    )
    random_media_url: bool = SchemaField(
        description="Whether to generate random media", default=False, advanced=True
    )
    notes: Optional[str] = SchemaField(
        description="Additional notes for the post", default=None, advanced=True
    )


class CarouselItem(BaseModel):
    """Model for Facebook carousel items."""

    name: str = Field(..., description="The name of the item")
    link: str = Field(..., description="The link of the item")
    picture: str = Field(..., description="The picture URL of the item")


class CallToAction(BaseModel):
    """Model for Google My Business Call to Action."""

    action_type: str = Field(
        ..., description="Type of action (book, order, shop, learn_more, sign_up, call)"
    )
    url: Optional[str] = Field(
        description="URL for the action (not required for 'call' action)"
    )


class EventDetails(BaseModel):
    """Model for Google My Business Event details."""

    title: str = Field(..., description="Event title")
    start_date: str = Field(..., description="Event start date (ISO format)")
    end_date: str = Field(..., description="Event end date (ISO format)")


class OfferDetails(BaseModel):
    """Model for Google My Business Offer details."""

    title: str = Field(..., description="Offer title")
    start_date: str = Field(..., description="Offer start date (ISO format)")
    end_date: str = Field(..., description="Offer end date (ISO format)")
    coupon_code: str = Field(..., description="Coupon code (max 58 characters)")
    redeem_online_url: str = Field(..., description="URL to redeem the offer")
    terms_conditions: str = Field(..., description="Terms and conditions")


class InstagramUserTag(BaseModel):
    """Model for Instagram user tags."""

    username: str = Field(..., description="Instagram username (without @)")
    x: Optional[float] = Field(description="X coordinate (0.0-1.0) for image posts")
    y: Optional[float] = Field(description="Y coordinate (0.0-1.0) for image posts")


class LinkedInTargeting(BaseModel):
    """Model for LinkedIn audience targeting."""

    countries: Optional[list[str]] = Field(
        description="Country codes (e.g., ['US', 'IN', 'DE', 'GB'])"
    )
    seniorities: Optional[list[str]] = Field(
        description="Seniority levels (e.g., ['Senior', 'VP'])"
    )
    degrees: Optional[list[str]] = Field(description="Education degrees")
    fields_of_study: Optional[list[str]] = Field(description="Fields of study")
    industries: Optional[list[str]] = Field(description="Industry categories")
    job_functions: Optional[list[str]] = Field(description="Job function categories")
    staff_count_ranges: Optional[list[str]] = Field(description="Company size ranges")


class PinterestCarouselOption(BaseModel):
    """Model for Pinterest carousel image options."""

    title: Optional[str] = Field(description="Image title")
    link: Optional[str] = Field(description="External destination link for the image")
    description: Optional[str] = Field(description="Image description")


class YouTubeTargeting(BaseModel):
    """Model for YouTube country targeting."""

    block: Optional[list[str]] = Field(
        description="Country codes to block (e.g., ['US', 'CA'])"
    )
    allow: Optional[list[str]] = Field(
        description="Country codes to allow (e.g., ['GB', 'AU'])"
    )


def create_ayrshare_client():
    """Create an Ayrshare client instance."""
    try:
        return AyrshareClient()
    except MissingConfigError:
        return None


# class PostToFacebookBlock(Block):
#     """Block for posting to Facebook with Facebook-specific options."""

#     class Input(BaseAyrshareInput):
#         """Input schema for Facebook posts."""

#         # Facebook-specific options
#         is_carousel: bool = SchemaField(
#             description="Whether to post a carousel", default=False, advanced=True
#         )
#         carousel_link: str = SchemaField(
#             description="The URL for the 'See More At' button in the carousel",
#             default="",
#             advanced=True,
#         )
#         carousel_items: list[CarouselItem] = SchemaField(
#             description="List of carousel items with name, link and picture URLs. Min 2, max 10 items.",
#             default_factory=list,
#             advanced=True,
#         )
#         is_reels: bool = SchemaField(
#             description="Whether to post to Facebook Reels",
#             default=False,
#             advanced=True,
#         )
#         reels_title: str = SchemaField(
#             description="Title for the Reels video (max 255 chars)",
#             default="",
#             advanced=True,
#         )
#         reels_thumbnail: str = SchemaField(
#             description="Thumbnail URL for Reels video (JPEG/PNG, <10MB)",
#             default="",
#             advanced=True,
#         )
#         is_story: bool = SchemaField(
#             description="Whether to post as a Facebook Story",
#             default=False,
#             advanced=True,
#         )
#         media_captions: list[str] = SchemaField(
#             description="Captions for each media item",
#             default_factory=list,
#             advanced=True,
#         )
#         location_id: str = SchemaField(
#             description="Facebook Page ID or name for location tagging",
#             default="",
#             advanced=True,
#         )
#         age_min: int = SchemaField(
#             description="Minimum age for audience targeting (13,15,18,21,25)",
#             default=0,
#             advanced=True,
#         )
#         target_countries: list[str] = SchemaField(
#             description="List of country codes to target (max 25)",
#             default_factory=list,
#             advanced=True,
#         )
#         alt_text: list[str] = SchemaField(
#             description="Alt text for each media item",
#             default_factory=list,
#             advanced=True,
#         )
#         video_title: str = SchemaField(
#             description="Title for video post", default="", advanced=True
#         )
#         video_thumbnail: str = SchemaField(
#             description="Thumbnail URL for video post", default="", advanced=True
#         )
#         is_draft: bool = SchemaField(
#             description="Save as draft in Meta Business Suite",
#             default=False,
#             advanced=True,
#         )
#         scheduled_publish_date: str = SchemaField(
#             description="Schedule publish time in Meta Business Suite (UTC)",
#             default="",
#             advanced=True,
#         )
#         preview_link: str = SchemaField(
#             description="URL for custom link preview", default="", advanced=True
#         )

#     class Output(BlockSchema):
#         post_result: RequestOutput = SchemaField(description="The result of the post")

#     def __init__(self):
#         super().__init__(
#             id="3352f512-3524-49ed-a08f-003042da2fc1",
#             description="Post to Facebook using Ayrshare",
#             categories={BlockCategory.SOCIAL},
#             block_type=BlockType.AYRSHARE,
#             input_schema=PostToFacebookBlock.Input,
#             output_schema=PostToFacebookBlock.Output,
#         )

#     async def run(
#         self,
#         input_data: "PostToFacebookBlock.Input",
#         *,
#         profile_key: SecretStr,
#         **kwargs,
#     ) -> BlockOutput:
#         """Post to Facebook with Facebook-specific options."""
#         if not profile_key:
#             yield "error", "Please link a social account via Ayrshare"
#             return

#         client = create_ayrshare_client()
#         if not client:
#             yield "error", "Ayrshare integration is not configured. Please set up the AYRSHARE_API_KEY."
#             return

#         # Convert datetime to ISO format if provided
#         iso_date = (
#             input_data.schedule_date.isoformat() if input_data.schedule_date else None
#         )

#         # Build Facebook-specific options
#         facebook_options = {}
#         if input_data.is_carousel:
#             facebook_options["isCarousel"] = True
#             if input_data.carousel_link:
#                 facebook_options["carouselLink"] = input_data.carousel_link
#             if input_data.carousel_items:
#                 facebook_options["carouselItems"] = [
#                     item.dict() for item in input_data.carousel_items
#                 ]

#         if input_data.is_reels:
#             facebook_options["isReels"] = True
#             if input_data.reels_title:
#                 facebook_options["reelsTitle"] = input_data.reels_title
#             if input_data.reels_thumbnail:
#                 facebook_options["reelsThumbnail"] = input_data.reels_thumbnail

#         if input_data.is_story:
#             facebook_options["isStory"] = True

#         if input_data.media_captions:
#             facebook_options["mediaCaptions"] = input_data.media_captions

#         if input_data.location_id:
#             facebook_options["locationId"] = input_data.location_id

#         if input_data.age_min > 0:
#             facebook_options["ageMin"] = input_data.age_min

#         if input_data.target_countries:
#             facebook_options["targetCountries"] = input_data.target_countries

#         if input_data.alt_text:
#             facebook_options["altText"] = input_data.alt_text

#         if input_data.video_title:
#             facebook_options["videoTitle"] = input_data.video_title

#         if input_data.video_thumbnail:
#             facebook_options["videoThumbnail"] = input_data.video_thumbnail

#         if input_data.is_draft:
#             facebook_options["isDraft"] = True

#         if input_data.scheduled_publish_date:
#             facebook_options["scheduledPublishDate"] = input_data.scheduled_publish_date

#         if input_data.preview_link:
#             facebook_options["previewLink"] = input_data.preview_link

#         try:
#             response = client.create_post(
#                 post=input_data.post,
#                 platforms=[SocialPlatform.FACEBOOK],
#                 media_urls=input_data.media_urls,
#                 is_video=input_data.is_video,
#                 schedule_date=iso_date,
#                 disable_comments=input_data.disable_comments,
#                 shorten_links=input_data.shorten_links,
#                 unsplash=input_data.unsplash,
#                 requires_approval=input_data.requires_approval,
#                 random_post=input_data.random_post,
#                 random_media_url=input_data.random_media_url,
#                 notes=input_data.notes,
#                 facebook_options=facebook_options if facebook_options else None,
#                 profile_key=profile_key.get_secret_value(),
#             )
#             yield "post_result", response
#         except Exception as e:
#             yield "error", str(e)


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
        reply_to_id: str = SchemaField(
            description="ID of the tweet to reply to", advanced=True
        )
        quote_tweet_id: str = SchemaField(
            description="ID of the tweet to quote (low-level Tweet ID)",
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
        post_result: RequestOutput = SchemaField(description="The result of the post")

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
        profile_key: SecretStr,
        **kwargs,
    ) -> BlockOutput:
        """Post to X / Twitter with enhanced X-specific options."""
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

        try:
            response = client.create_post(
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
        except Exception as e:
            yield "error", str(e)


# class PostToLinkedInBlock(Block):
#     """Block for posting to LinkedIn with LinkedIn-specific options."""

#     class Input(BaseAyrshareInput):
#         """Input schema for LinkedIn posts."""

#         # Override post field to include LinkedIn-specific information
#         post: str = SchemaField(
#             description="The post text (max 3,000 chars, hashtags supported with #)",
#             default="",
#             advanced=False,
#         )

#         # Override media_urls to include LinkedIn-specific constraints
#         media_urls: list[str] = SchemaField(
#             description="Optional list of media URLs. LinkedIn supports up to 9 images, videos, or documents (PPT, PPTX, DOC, DOCX, PDF <100MB, <300 pages).",
#             default_factory=list,
#             advanced=False,
#         )

#         # LinkedIn-specific options
#         visibility: str = SchemaField(
#             description="Post visibility: 'public' (default), 'connections' (personal only), 'loggedin'",
#             default="public",
#             advanced=True,
#         )
#         alt_text: list[str] = SchemaField(
#             description="Alt text for each image (accessibility feature, not supported for videos/documents)",
#             default_factory=list,
#             advanced=True,
#         )
#         titles: list[str] = SchemaField(
#             description="Title/caption for each image or video",
#             default_factory=list,
#             advanced=True,
#         )
#         document_title: str = SchemaField(
#             description="Title for document posts (max 400 chars, uses filename if not specified)",
#             default="",
#             advanced=True,
#         )
#         thumbnail: str = SchemaField(
#             description="Thumbnail URL for video (PNG/JPG, same dimensions as video, <10MB)",
#             default="",
#             advanced=True,
#         )
#         # LinkedIn targeting options (flattened from LinkedInTargeting object)
#         targeting_countries: Optional[list[str]] = SchemaField(
#             description="Country codes for targeting (e.g., ['US', 'IN', 'DE', 'GB']). Requires 300+ followers in target audience.",
#             default=None,
#             advanced=True,
#         )
#         targeting_seniorities: Optional[list[str]] = SchemaField(
#             description="Seniority levels for targeting (e.g., ['Senior', 'VP']). Requires 300+ followers in target audience.",
#             default=None,
#             advanced=True,
#         )
#         targeting_degrees: Optional[list[str]] = SchemaField(
#             description="Education degrees for targeting. Requires 300+ followers in target audience.",
#             default=None,
#             advanced=True,
#         )
#         targeting_fields_of_study: Optional[list[str]] = SchemaField(
#             description="Fields of study for targeting. Requires 300+ followers in target audience.",
#             default=None,
#             advanced=True,
#         )
#         targeting_industries: Optional[list[str]] = SchemaField(
#             description="Industry categories for targeting. Requires 300+ followers in target audience.",
#             default=None,
#             advanced=True,
#         )
#         targeting_job_functions: Optional[list[str]] = SchemaField(
#             description="Job function categories for targeting. Requires 300+ followers in target audience.",
#             default=None,
#             advanced=True,
#         )
#         targeting_staff_count_ranges: Optional[list[str]] = SchemaField(
#             description="Company size ranges for targeting. Requires 300+ followers in target audience.",
#             default=None,
#             advanced=True,
#         )

#     class Output(BlockSchema):
#         post_result: RequestOutput = SchemaField(description="The result of the post")

#     def __init__(self):
#         super().__init__(
#             id="589af4e4-507f-42fd-b9ac-a67ecef25811",
#             description="Post to LinkedIn using Ayrshare",
#             categories={BlockCategory.SOCIAL},
#             block_type=BlockType.AYRSHARE,
#             input_schema=PostToLinkedInBlock.Input,
#             output_schema=PostToLinkedInBlock.Output,
#         )

#     async def run(
#         self,
#         input_data: "PostToLinkedInBlock.Input",
#         *,
#         profile_key: SecretStr,
#         **kwargs,
#     ) -> BlockOutput:
#         """Post to LinkedIn with LinkedIn-specific options."""
#         if not profile_key:
#             yield "error", "Please link a social account via Ayrshare"
#             return

#         client = create_ayrshare_client()
#         if not client:
#             yield "error", "Ayrshare integration is not configured. Please set up the AYRSHARE_API_KEY."
#             return

#         # Validate LinkedIn constraints
#         if len(input_data.post) > 3000:
#             yield "error", f"LinkedIn post text exceeds 3,000 character limit ({len(input_data.post)} characters)"
#             return

#         if len(input_data.media_urls) > 9:
#             yield "error", "LinkedIn supports a maximum of 9 images/videos/documents"
#             return

#         if input_data.document_title and len(input_data.document_title) > 400:
#             yield "error", f"LinkedIn document title exceeds 400 character limit ({len(input_data.document_title)} characters)"
#             return

#         # Validate visibility option
#         valid_visibility = ["public", "connections", "loggedin"]
#         if input_data.visibility not in valid_visibility:
#             yield "error", f"LinkedIn visibility must be one of: {', '.join(valid_visibility)}"
#             return

#         # Check for document extensions
#         document_extensions = [".ppt", ".pptx", ".doc", ".docx", ".pdf"]
#         has_documents = any(
#             any(url.lower().endswith(ext) for ext in document_extensions)
#             for url in input_data.media_urls
#         )

#         # Convert datetime to ISO format if provided
#         iso_date = (
#             input_data.schedule_date.isoformat() if input_data.schedule_date else None
#         )

#         # Build LinkedIn-specific options
#         linkedin_options = {}

#         # Visibility
#         if input_data.visibility != "public":
#             linkedin_options["visibility"] = input_data.visibility

#         # Alt text (not supported for videos or documents)
#         if input_data.alt_text and not has_documents:
#             linkedin_options["altText"] = input_data.alt_text

#         # Titles/captions
#         if input_data.titles:
#             linkedin_options["titles"] = input_data.titles

#         # Document title
#         if input_data.document_title and has_documents:
#             linkedin_options["title"] = input_data.document_title

#         # Video thumbnail
#         if input_data.thumbnail:
#             linkedin_options["thumbNail"] = input_data.thumbnail

#         # Audience targeting (from flattened fields)
#         targeting_dict = {}
#         if input_data.targeting_countries:
#             targeting_dict["countries"] = input_data.targeting_countries
#         if input_data.targeting_seniorities:
#             targeting_dict["seniorities"] = input_data.targeting_seniorities
#         if input_data.targeting_degrees:
#             targeting_dict["degrees"] = input_data.targeting_degrees
#         if input_data.targeting_fields_of_study:
#             targeting_dict["fieldsOfStudy"] = input_data.targeting_fields_of_study
#         if input_data.targeting_industries:
#             targeting_dict["industries"] = input_data.targeting_industries
#         if input_data.targeting_job_functions:
#             targeting_dict["jobFunctions"] = input_data.targeting_job_functions
#         if input_data.targeting_staff_count_ranges:
#             targeting_dict["staffCountRanges"] = input_data.targeting_staff_count_ranges

#         if targeting_dict:
#             linkedin_options["targeting"] = targeting_dict

#         try:
#             response = client.create_post(
#                 post=input_data.post,
#                 platforms=[SocialPlatform.LINKEDIN],
#                 media_urls=input_data.media_urls,
#                 is_video=input_data.is_video,
#                 schedule_date=iso_date,
#                 disable_comments=input_data.disable_comments,
#                 shorten_links=input_data.shorten_links,
#                 unsplash=input_data.unsplash,
#                 requires_approval=input_data.requires_approval,
#                 random_post=input_data.random_post,
#                 random_media_url=input_data.random_media_url,
#                 notes=input_data.notes,
#                 linkedin_options=linkedin_options if linkedin_options else None,
#                 profile_key=profile_key.get_secret_value(),
#             )
#             yield "post_result", response
#         except Exception as e:
#             yield "error", str(e)


# class PostToInstagramBlock(Block):
#     """Block for posting to Instagram with Instagram-specific options."""

#     class Input(BaseAyrshareInput):
#         """Input schema for Instagram posts."""

#         # Override post field to include Instagram-specific information
#         post: str = SchemaField(
#             description="The post text (max 2,200 chars, up to 30 hashtags, 3 @mentions)",
#             default="",
#             advanced=False,
#         )

#         # Override media_urls to include Instagram-specific constraints
#         media_urls: list[str] = SchemaField(
#             description="Optional list of media URLs. Instagram supports up to 10 images/videos in a carousel.",
#             default_factory=list,
#             advanced=False,
#         )

#         # Instagram-specific options
#         is_story: bool = SchemaField(
#             description="Whether to post as Instagram Story (24-hour expiration)",
#             default=False,
#             advanced=True,
#         )
#         share_reels_feed: bool = SchemaField(
#             description="Whether Reel should appear in both Feed and Reels tabs",
#             default=True,
#             advanced=True,
#         )
#         audio_name: str = SchemaField(
#             description="Audio name for Reels (e.g., 'The Weeknd - Blinding Lights')",
#             default="",
#             advanced=True,
#         )
#         thumbnail: str = SchemaField(
#             description="Thumbnail URL for Reel video", default="", advanced=True
#         )
#         thumbnail_offset: int = SchemaField(
#             description="Thumbnail frame offset in milliseconds (default: 0)",
#             default=0,
#             advanced=True,
#         )
#         alt_text: list[str] = SchemaField(
#             description="Alt text for each media item (up to 1,000 chars each, accessibility feature)",
#             default_factory=list,
#             advanced=True,
#         )
#         location_id: str = SchemaField(
#             description="Facebook Page ID or name for location tagging (e.g., '7640348500' or '@guggenheimmuseum')",
#             default="",
#             advanced=True,
#         )
#         user_tags: list[InstagramUserTag] = SchemaField(
#             description="List of users to tag with coordinates for images",
#             default_factory=list,
#             advanced=True,
#         )
#         collaborators: list[str] = SchemaField(
#             description="Instagram usernames to invite as collaborators (max 3, public accounts only)",
#             default_factory=list,
#             advanced=True,
#         )
#         auto_resize: bool = SchemaField(
#             description="Auto-resize images to 1080x1080px for Instagram",
#             default=False,
#             advanced=True,
#         )

#     class Output(BlockSchema):
#         post_result: RequestOutput = SchemaField(description="The result of the post")

#     def __init__(self):
#         super().__init__(
#             id="89b02b96-a7cb-46f4-9900-c48b32fe1552",
#             description="Post to Instagram using Ayrshare",
#             categories={BlockCategory.SOCIAL},
#             block_type=BlockType.AYRSHARE,
#             input_schema=PostToInstagramBlock.Input,
#             output_schema=PostToInstagramBlock.Output,
#         )

#     async def run(
#         self,
#         input_data: "PostToInstagramBlock.Input",
#         *,
#         profile_key: SecretStr,
#         **kwargs,
#     ) -> BlockOutput:
#         """Post to Instagram with Instagram-specific options."""
#         if not profile_key:
#             yield "error", "Please link a social account via Ayrshare"
#             return

#         client = create_ayrshare_client()
#         if not client:
#             yield "error", "Ayrshare integration is not configured. Please set up the AYRSHARE_API_KEY."
#             return

#         # Validate Instagram constraints
#         if len(input_data.post) > 2200:
#             yield "error", f"Instagram post text exceeds 2,200 character limit ({len(input_data.post)} characters)"
#             return

#         if len(input_data.media_urls) > 10:
#             yield "error", "Instagram supports a maximum of 10 images/videos in a carousel"
#             return

#         if len(input_data.collaborators) > 3:
#             yield "error", "Instagram supports a maximum of 3 collaborators"
#             return

#         # Count hashtags and mentions
#         hashtag_count = input_data.post.count("#")
#         mention_count = input_data.post.count("@")

#         if hashtag_count > 30:
#             yield "error", f"Instagram allows maximum 30 hashtags ({hashtag_count} found)"
#             return

#         if mention_count > 3:
#             yield "error", f"Instagram allows maximum 3 @mentions ({mention_count} found)"
#             return

#         # Convert datetime to ISO format if provided
#         iso_date = (
#             input_data.schedule_date.isoformat() if input_data.schedule_date else None
#         )

#         # Build Instagram-specific options
#         instagram_options = {}

#         # Stories
#         if input_data.is_story:
#             instagram_options["stories"] = True

#         # Reels options
#         if input_data.share_reels_feed is not None:
#             instagram_options["shareReelsFeed"] = input_data.share_reels_feed

#         if input_data.audio_name:
#             instagram_options["audioName"] = input_data.audio_name

#         if input_data.thumbnail:
#             instagram_options["thumbNail"] = input_data.thumbnail
#         elif input_data.thumbnail_offset > 0:
#             instagram_options["thumbNailOffset"] = input_data.thumbnail_offset

#         # Alt text
#         if input_data.alt_text:
#             # Validate alt text length
#             for i, alt in enumerate(input_data.alt_text):
#                 if len(alt) > 1000:
#                     yield "error", f"Alt text {i+1} exceeds 1,000 character limit ({len(alt)} characters)"
#                     return
#             instagram_options["altText"] = input_data.alt_text

#         # Location
#         if input_data.location_id:
#             instagram_options["locationId"] = input_data.location_id

#         # User tags
#         if input_data.user_tags:
#             user_tags_list = []
#             for tag in input_data.user_tags:
#                 tag_dict: dict[str, float | str] = {"username": tag.username}
#                 if tag.x is not None and tag.y is not None:
#                     # Validate coordinates
#                     if not (0.0 <= tag.x <= 1.0) or not (0.0 <= tag.y <= 1.0):
#                         yield "error", f"User tag coordinates must be between 0.0 and 1.0 (user: {tag.username})"
#                         return
#                     tag_dict["x"] = tag.x
#                     tag_dict["y"] = tag.y
#                 user_tags_list.append(tag_dict)
#             instagram_options["userTags"] = user_tags_list

#         # Collaborators
#         if input_data.collaborators:
#             instagram_options["collaborators"] = input_data.collaborators

#         # Auto resize
#         if input_data.auto_resize:
#             instagram_options["autoResize"] = True

#         try:
#             response = client.create_post(
#                 post=input_data.post,
#                 platforms=[SocialPlatform.INSTAGRAM],
#                 media_urls=input_data.media_urls,
#                 is_video=input_data.is_video,
#                 schedule_date=iso_date,
#                 disable_comments=input_data.disable_comments,
#                 shorten_links=input_data.shorten_links,
#                 unsplash=input_data.unsplash,
#                 requires_approval=input_data.requires_approval,
#                 random_post=input_data.random_post,
#                 random_media_url=input_data.random_media_url,
#                 notes=input_data.notes,
#                 instagram_options=instagram_options if instagram_options else None,
#                 profile_key=profile_key.get_secret_value(),
#             )
#             yield "post_result", response
#         except Exception as e:
#             yield "error", str(e)


# class PostToYouTubeBlock(Block):
#     """Block for posting to YouTube with YouTube-specific options."""

#     class Input(BaseAyrshareInput):
#         """Input schema for YouTube posts."""

#         # Override post field to include YouTube-specific information
#         post: str = SchemaField(
#             description="Video description (max 5,000 chars, empty string allowed). Cannot contain < or > characters.",
#             default="",
#             advanced=False,
#         )

#         # Override media_urls to include YouTube-specific constraints
#         media_urls: list[str] = SchemaField(
#             description="Required video URL. YouTube only supports 1 video per post.",
#             default_factory=list,
#             advanced=False,
#         )

#         # YouTube-specific required options
#         title: str = SchemaField(
#             description="Video title (max 100 chars, required). Cannot contain < or > characters.",
#             default="",
#             advanced=False,
#         )

#         # YouTube-specific optional options
#         visibility: str = SchemaField(
#             description="Video visibility: 'private' (default), 'public', or 'unlisted'",
#             default="private",
#             advanced=True,
#         )
#         thumbnail: str = SchemaField(
#             description="Thumbnail URL (JPEG/PNG under 2MB, must end in .png/.jpg/.jpeg). Requires phone verification.",
#             default="",
#             advanced=True,
#         )
#         playlist_id: str = SchemaField(
#             description="Playlist ID to add video (user must own playlist)",
#             default="",
#             advanced=True,
#         )
#         tags: list[str] = SchemaField(
#             description="Video tags (min 2 chars each, max 500 chars total)",
#             default_factory=list,
#             advanced=True,
#         )
#         made_for_kids: bool = SchemaField(
#             description="Self-declared kids content", default=False, advanced=True
#         )
#         is_shorts: bool = SchemaField(
#             description="Post as YouTube Short (max 3 minutes, adds #shorts)",
#             default=False,
#             advanced=True,
#         )
#         notify_subscribers: bool = SchemaField(
#             description="Send notification to subscribers", default=True, advanced=True
#         )
#         category_id: int = SchemaField(
#             description="Video category ID (e.g., 24 = Entertainment)",
#             default=0,
#             advanced=True,
#         )
#         contains_synthetic_media: bool = SchemaField(
#             description="Disclose realistic AI/synthetic content",
#             default=False,
#             advanced=True,
#         )
#         publish_at: str = SchemaField(
#             description="UTC publish time (YouTube controlled, format: 2022-10-08T21:18:36Z)",
#             default="",
#             advanced=True,
#         )
#         # YouTube targeting options (flattened from YouTubeTargeting object)
#         targeting_block_countries: Optional[list[str]] = SchemaField(
#             description="Country codes to block from viewing (e.g., ['US', 'CA'])",
#             default=None,
#             advanced=True,
#         )
#         targeting_allow_countries: Optional[list[str]] = SchemaField(
#             description="Country codes to allow viewing (e.g., ['GB', 'AU'])",
#             default=None,
#             advanced=True,
#         )
#         subtitle_url: str = SchemaField(
#             description="URL to SRT or SBV subtitle file (must be HTTPS and end in .srt/.sbv, under 100MB)",
#             default="",
#             advanced=True,
#         )
#         subtitle_language: str = SchemaField(
#             description="Language code for subtitles (default: 'en')",
#             default="en",
#             advanced=True,
#         )
#         subtitle_name: str = SchemaField(
#             description="Name of caption track (max 150 chars, default: 'English')",
#             default="English",
#             advanced=True,
#         )

#     class Output(BlockSchema):
#         post_result: RequestOutput = SchemaField(description="The result of the post")

#     def __init__(self):
#         super().__init__(
#             id="0082d712-ff1b-4c3d-8a8d-6c7721883b83",
#             description="Post to YouTube using Ayrshare",
#             categories={BlockCategory.SOCIAL},
#             block_type=BlockType.AYRSHARE,
#             input_schema=PostToYouTubeBlock.Input,
#             output_schema=PostToYouTubeBlock.Output,
#         )

#     async def run(
#         self,
#         input_data: "PostToYouTubeBlock.Input",
#         *,
#         profile_key: SecretStr,
#         **kwargs,
#     ) -> BlockOutput:
#         """Post to YouTube with YouTube-specific validation and options."""
#         if not profile_key:
#             yield "error", "Please link a social account via Ayrshare"
#             return

#         client = create_ayrshare_client()
#         if not client:
#             yield "error", "Ayrshare integration is not configured. Please set up the AYRSHARE_API_KEY."
#             return

#         # Validate YouTube constraints
#         if not input_data.title:
#             yield "error", "YouTube requires a video title"
#             return

#         if len(input_data.title) > 100:
#             yield "error", f"YouTube title exceeds 100 character limit ({len(input_data.title)} characters)"
#             return

#         if len(input_data.post) > 5000:
#             yield "error", f"YouTube description exceeds 5,000 character limit ({len(input_data.post)} characters)"
#             return

#         # Check for forbidden characters
#         forbidden_chars = ["<", ">"]
#         for char in forbidden_chars:
#             if char in input_data.title:
#                 yield "error", f"YouTube title cannot contain '{char}' character"
#                 return
#             if char in input_data.post:
#                 yield "error", f"YouTube description cannot contain '{char}' character"
#                 return

#         if not input_data.media_urls:
#             yield "error", "YouTube requires exactly one video URL"
#             return

#         if len(input_data.media_urls) > 1:
#             yield "error", "YouTube supports only 1 video per post"
#             return

#         # Validate visibility option
#         valid_visibility = ["private", "public", "unlisted"]
#         if input_data.visibility not in valid_visibility:
#             yield "error", f"YouTube visibility must be one of: {', '.join(valid_visibility)}"
#             return

#         # Validate thumbnail URL format
#         if input_data.thumbnail:
#             valid_extensions = [".png", ".jpg", ".jpeg"]
#             if not any(
#                 input_data.thumbnail.lower().endswith(ext) for ext in valid_extensions
#             ):
#                 yield "error", "YouTube thumbnail must end in .png, .jpg, or .jpeg"
#                 return

#         # Validate tags
#         if input_data.tags:
#             total_tag_length = sum(len(tag) for tag in input_data.tags)
#             if total_tag_length > 500:
#                 yield "error", f"YouTube tags total length exceeds 500 characters ({total_tag_length} characters)"
#                 return

#             for tag in input_data.tags:
#                 if len(tag) < 2:
#                     yield "error", f"YouTube tag '{tag}' is too short (minimum 2 characters)"
#                     return

#         # Validate subtitle URL
#         if input_data.subtitle_url:
#             if not input_data.subtitle_url.startswith("https://"):
#                 yield "error", "YouTube subtitle URL must start with https://"
#                 return

#             valid_subtitle_extensions = [".srt", ".sbv"]
#             if not any(
#                 input_data.subtitle_url.lower().endswith(ext)
#                 for ext in valid_subtitle_extensions
#             ):
#                 yield "error", "YouTube subtitle URL must end in .srt or .sbv"
#                 return

#         if len(input_data.subtitle_name) > 150:
#             yield "error", f"YouTube subtitle name exceeds 150 character limit ({len(input_data.subtitle_name)} characters)"
#             return

#         # Validate publish_at format if provided
#         if input_data.publish_at and input_data.schedule_date:
#             yield "error", "Cannot use both 'publish_at' and 'schedule_date'. Use 'publish_at' for YouTube-controlled publishing."
#             return

#         # Convert datetime to ISO format if provided (only if not using publish_at)
#         iso_date = None
#         if not input_data.publish_at and input_data.schedule_date:
#             iso_date = input_data.schedule_date.isoformat()

#         # Build YouTube-specific options
#         youtube_options: dict[str, Any] = {"title": input_data.title}

#         # Basic options
#         if input_data.visibility != "private":
#             youtube_options["visibility"] = input_data.visibility

#         if input_data.thumbnail:
#             youtube_options["thumbNail"] = input_data.thumbnail

#         if input_data.playlist_id:
#             youtube_options["playListId"] = input_data.playlist_id

#         if input_data.tags:
#             youtube_options["tags"] = input_data.tags

#         if input_data.made_for_kids:
#             youtube_options["madeForKids"] = True

#         if input_data.is_shorts:
#             youtube_options["shorts"] = True

#         if not input_data.notify_subscribers:
#             youtube_options["notifySubscribers"] = False

#         if input_data.category_id > 0:
#             youtube_options["categoryId"] = input_data.category_id

#         if input_data.contains_synthetic_media:
#             youtube_options["containsSyntheticMedia"] = True

#         if input_data.publish_at:
#             youtube_options["publishAt"] = input_data.publish_at

#         # Country targeting (from flattened fields)
#         targeting_dict = {}
#         if input_data.targeting_block_countries:
#             targeting_dict["block"] = input_data.targeting_block_countries
#         if input_data.targeting_allow_countries:
#             targeting_dict["allow"] = input_data.targeting_allow_countries

#         if targeting_dict:
#             youtube_options["targeting"] = targeting_dict

#         # Subtitle options
#         if input_data.subtitle_url:
#             youtube_options["subTitleUrl"] = input_data.subtitle_url
#             youtube_options["subTitleLanguage"] = input_data.subtitle_language
#             youtube_options["subTitleName"] = input_data.subtitle_name

#         try:
#             response = client.create_post(
#                 post=input_data.post,
#                 platforms=[SocialPlatform.YOUTUBE],
#                 media_urls=input_data.media_urls,
#                 is_video=True,  # YouTube only supports videos
#                 schedule_date=iso_date,
#                 disable_comments=input_data.disable_comments,
#                 shorten_links=input_data.shorten_links,
#                 unsplash=input_data.unsplash,
#                 requires_approval=input_data.requires_approval,
#                 random_post=input_data.random_post,
#                 random_media_url=input_data.random_media_url,
#                 notes=input_data.notes,
#                 youtube_options=youtube_options,
#                 profile_key=profile_key.get_secret_value(),
#             )
#             yield "post_result", response
#         except Exception as e:
#             yield "error", str(e)


# class PostToRedditBlock(Block):
#     """Block for posting to Reddit."""

#     class Input(BaseAyrshareInput):
#         """Input schema for Reddit posts."""

#         pass  # Uses all base fields

#     class Output(BlockSchema):
#         post_result: RequestOutput = SchemaField(description="The result of the post")

#     def __init__(self):
#         super().__init__(
#             id="c7733580-3c72-483e-8e47-a8d58754d853",
#             description="Post to Reddit using Ayrshare",
#             categories={BlockCategory.SOCIAL},
#             block_type=BlockType.AYRSHARE,
#             input_schema=PostToRedditBlock.Input,
#             output_schema=PostToRedditBlock.Output,
#         )

#     async def run(
#         self, input_data: "PostToRedditBlock.Input", *, profile_key: SecretStr, **kwargs
#     ) -> BlockOutput:
#         if not profile_key:
#             yield "error", "Please link a social account via Ayrshare"
#             return
#         client = create_ayrshare_client()
#         if not client:
#             yield "error", "Ayrshare integration is not configured."
#             return
#         iso_date = (
#             input_data.schedule_date.isoformat() if input_data.schedule_date else None
#         )
#         try:
#             response = client.create_post(
#                 post=input_data.post,
#                 platforms=[SocialPlatform.REDDIT],
#                 media_urls=input_data.media_urls,
#                 is_video=input_data.is_video,
#                 schedule_date=iso_date,
#                 disable_comments=input_data.disable_comments,
#                 shorten_links=input_data.shorten_links,
#                 unsplash=input_data.unsplash,
#                 requires_approval=input_data.requires_approval,
#                 random_post=input_data.random_post,
#                 random_media_url=input_data.random_media_url,
#                 notes=input_data.notes,
#                 profile_key=profile_key.get_secret_value(),
#             )
#             yield "post_result", response
#         except Exception as e:
#             yield "error", str(e)


# class PostToTelegramBlock(Block):
#     """Block for posting to Telegram with Telegram-specific options."""

#     class Input(BaseAyrshareInput):
#         """Input schema for Telegram posts."""

#         # Override post field to include Telegram-specific information
#         post: str = SchemaField(
#             description="The post text (empty string allowed). Use @handle to mention other Telegram users.",
#             default="",
#             advanced=False,
#         )

#         # Override media_urls to include Telegram-specific constraints
#         media_urls: list[str] = SchemaField(
#             description="Optional list of media URLs. For animated GIFs, only one URL is allowed. Telegram will auto-preview links unless image/video is included.",
#             default_factory=list,
#             advanced=False,
#         )

#         # Override is_video to include GIF-specific information
#         is_video: bool = SchemaField(
#             description="Whether the media is a video. Set to true for animated GIFs that don't end in .gif/.GIF extension.",
#             default=False,
#             advanced=True,
#         )

#     class Output(BlockSchema):
#         post_result: RequestOutput = SchemaField(description="The result of the post")

#     def __init__(self):
#         super().__init__(
#             id="47bc74eb-4af2-452c-b933-af377c7287df",
#             description="Post to Telegram using Ayrshare",
#             categories={BlockCategory.SOCIAL},
#             block_type=BlockType.AYRSHARE,
#             input_schema=PostToTelegramBlock.Input,
#             output_schema=PostToTelegramBlock.Output,
#         )

#     async def run(
#         self,
#         input_data: "PostToTelegramBlock.Input",
#         *,
#         profile_key: SecretStr,
#         **kwargs,
#     ) -> BlockOutput:
#         """Post to Telegram with Telegram-specific validation."""
#         if not profile_key:
#             yield "error", "Please link a social account via Ayrshare"
#             return

#         client = create_ayrshare_client()
#         if not client:
#             yield "error", "Ayrshare integration is not configured. Please set up the AYRSHARE_API_KEY."
#             return

#         # Validate Telegram constraints
#         # Check for animated GIFs - only one URL allowed
#         gif_extensions = [".gif", ".GIF"]
#         has_gif = any(
#             any(url.endswith(ext) for ext in gif_extensions)
#             for url in input_data.media_urls
#         )

#         if has_gif and len(input_data.media_urls) > 1:
#             yield "error", "Telegram animated GIFs support only one URL per post"
#             return

#         # Auto-detect if we need to set is_video for GIFs without proper extension
#         detected_is_video = input_data.is_video
#         if input_data.media_urls and not has_gif and not input_data.is_video:
#             # Check if this might be a GIF without proper extension
#             # This is just informational - user needs to set is_video manually
#             pass

#         # Convert datetime to ISO format if provided
#         iso_date = (
#             input_data.schedule_date.isoformat() if input_data.schedule_date else None
#         )

#         try:
#             response = client.create_post(
#                 post=input_data.post,
#                 platforms=[SocialPlatform.TELEGRAM],
#                 media_urls=input_data.media_urls,
#                 is_video=detected_is_video,
#                 schedule_date=iso_date,
#                 disable_comments=input_data.disable_comments,
#                 shorten_links=input_data.shorten_links,
#                 unsplash=input_data.unsplash,
#                 requires_approval=input_data.requires_approval,
#                 random_post=input_data.random_post,
#                 random_media_url=input_data.random_media_url,
#                 notes=input_data.notes,
#                 profile_key=profile_key.get_secret_value(),
#             )
#             yield "post_result", response
#         except Exception as e:
#             yield "error", str(e)


# class PostToGMBBlock(Block):
#     """Block for posting to Google My Business with GMB-specific options."""

#     class Input(BaseAyrshareInput):
#         """Input schema for Google My Business posts."""

#         # Override media_urls to include GMB-specific constraints
#         media_urls: list[str] = SchemaField(
#             description="Optional list of media URLs. GMB supports only one image or video per post.",
#             default_factory=list,
#             advanced=False,
#         )

#         # GMB-specific options
#         is_photo_video: bool = SchemaField(
#             description="Whether this is a photo/video post (appears in Photos section)",
#             default=False,
#             advanced=True,
#         )
#         photo_category: str = SchemaField(
#             description="Category for photo/video: cover, profile, logo, exterior, interior, product, at_work, food_and_drink, menu, common_area, rooms, teams",
#             default="",
#             advanced=True,
#         )
#         # Call to action options (flattened from CallToAction object)
#         call_to_action_type: str = SchemaField(
#             description="Type of action button: 'book', 'order', 'shop', 'learn_more', 'sign_up', or 'call'",
#             default="",
#             advanced=True,
#         )
#         call_to_action_url: str = SchemaField(
#             description="URL for the action button (not required for 'call' action)",
#             default="",
#             advanced=True,
#         )
#         # Event details options (flattened from EventDetails object)
#         event_title: str = SchemaField(
#             description="Event title for event posts",
#             default="",
#             advanced=True,
#         )
#         event_start_date: str = SchemaField(
#             description="Event start date in ISO format (e.g., '2024-03-15T09:00:00Z')",
#             default="",
#             advanced=True,
#         )
#         event_end_date: str = SchemaField(
#             description="Event end date in ISO format (e.g., '2024-03-15T17:00:00Z')",
#             default="",
#             advanced=True,
#         )
#         # Offer details options (flattened from OfferDetails object)
#         offer_title: str = SchemaField(
#             description="Offer title for promotional posts",
#             default="",
#             advanced=True,
#         )
#         offer_start_date: str = SchemaField(
#             description="Offer start date in ISO format (e.g., '2024-03-15T00:00:00Z')",
#             default="",
#             advanced=True,
#         )
#         offer_end_date: str = SchemaField(
#             description="Offer end date in ISO format (e.g., '2024-04-15T23:59:59Z')",
#             default="",
#             advanced=True,
#         )
#         offer_coupon_code: str = SchemaField(
#             description="Coupon code for the offer (max 58 characters)",
#             default="",
#             advanced=True,
#         )
#         offer_redeem_online_url: str = SchemaField(
#             description="URL where customers can redeem the offer online",
#             default="",
#             advanced=True,
#         )
#         offer_terms_conditions: str = SchemaField(
#             description="Terms and conditions for the offer",
#             default="",
#             advanced=True,
#         )

#     class Output(BlockSchema):
#         post_result: RequestOutput = SchemaField(description="The result of the post")

#     def __init__(self):
#         super().__init__(
#             id="2c38c783-c484-4503-9280-ef5d1d345a7e",
#             description="Post to Google My Business using Ayrshare",
#             categories={BlockCategory.SOCIAL},
#             block_type=BlockType.AYRSHARE,
#             input_schema=PostToGMBBlock.Input,
#             output_schema=PostToGMBBlock.Output,
#         )

#     async def run(
#         self, input_data: "PostToGMBBlock.Input", *, profile_key: SecretStr, **kwargs
#     ) -> BlockOutput:
#         """Post to Google My Business with GMB-specific options."""
#         if not profile_key:
#             yield "error", "Please link a social account via Ayrshare"
#             return

#         client = create_ayrshare_client()
#         if not client:
#             yield "error", "Ayrshare integration is not configured. Please set up the AYRSHARE_API_KEY."
#             return

#         # Validate GMB constraints
#         if len(input_data.media_urls) > 1:
#             yield "error", "Google My Business supports only one image or video per post"
#             return

#         # Validate offer coupon code length
#         if input_data.offer_coupon_code and len(input_data.offer_coupon_code) > 58:
#             yield "error", "GMB offer coupon code cannot exceed 58 characters"
#             return

#         # Convert datetime to ISO format if provided
#         iso_date = (
#             input_data.schedule_date.isoformat() if input_data.schedule_date else None
#         )

#         # Build GMB-specific options
#         gmb_options = {}

#         # Photo/Video post options
#         if input_data.is_photo_video:
#             gmb_options["isPhotoVideo"] = True
#             if input_data.photo_category:
#                 gmb_options["category"] = input_data.photo_category

#         # Call to Action (from flattened fields)
#         if input_data.call_to_action_type:
#             cta_dict = {"actionType": input_data.call_to_action_type}
#             # URL not required for 'call' action type
#             if (
#                 input_data.call_to_action_type != "call"
#                 and input_data.call_to_action_url
#             ):
#                 cta_dict["url"] = input_data.call_to_action_url
#             gmb_options["callToAction"] = cta_dict

#         # Event details (from flattened fields)
#         if (
#             input_data.event_title
#             and input_data.event_start_date
#             and input_data.event_end_date
#         ):
#             gmb_options["event"] = {
#                 "title": input_data.event_title,
#                 "startDate": input_data.event_start_date,
#                 "endDate": input_data.event_end_date,
#             }

#         # Offer details (from flattened fields)
#         if (
#             input_data.offer_title
#             and input_data.offer_start_date
#             and input_data.offer_end_date
#             and input_data.offer_coupon_code
#             and input_data.offer_redeem_online_url
#             and input_data.offer_terms_conditions
#         ):
#             gmb_options["offer"] = {
#                 "title": input_data.offer_title,
#                 "startDate": input_data.offer_start_date,
#                 "endDate": input_data.offer_end_date,
#                 "couponCode": input_data.offer_coupon_code,
#                 "redeemOnlineUrl": input_data.offer_redeem_online_url,
#                 "termsConditions": input_data.offer_terms_conditions,
#             }

#         try:
#             response = client.create_post(
#                 post=input_data.post,
#                 platforms=[SocialPlatform.GOOGLE_MY_BUSINESS],
#                 media_urls=input_data.media_urls,
#                 is_video=input_data.is_video,
#                 schedule_date=iso_date,
#                 disable_comments=input_data.disable_comments,
#                 shorten_links=input_data.shorten_links,
#                 unsplash=input_data.unsplash,
#                 requires_approval=input_data.requires_approval,
#                 random_post=input_data.random_post,
#                 random_media_url=input_data.random_media_url,
#                 notes=input_data.notes,
#                 gmb_options=gmb_options if gmb_options else None,
#                 profile_key=profile_key.get_secret_value(),
#             )
#             yield "post_result", response
#         except Exception as e:
#             yield "error", str(e)


# class PostToPinterestBlock(Block):
#     """Block for posting to Pinterest with Pinterest-specific options."""

#     class Input(BaseAyrshareInput):
#         """Input schema for Pinterest posts."""

#         # Override post field to include Pinterest-specific information
#         post: str = SchemaField(
#             description="Pin description (max 500 chars, links not clickable - use link field instead)",
#             default="",
#             advanced=False,
#         )

#         # Override media_urls to include Pinterest-specific constraints
#         media_urls: list[str] = SchemaField(
#             description="Required image/video URLs. Pinterest requires at least one image. Videos need thumbnail. Up to 5 images for carousel.",
#             default_factory=list,
#             advanced=False,
#         )

#         # Pinterest-specific options
#         pin_title: str = SchemaField(
#             description="Pin title displayed in 'Add your title' section (max 100 chars)",
#             default="",
#             advanced=True,
#         )
#         link: str = SchemaField(
#             description="Clickable destination URL when users click the pin (max 2048 chars)",
#             default="",
#             advanced=True,
#         )
#         board_id: str = SchemaField(
#             description="Pinterest Board ID to post to (from /user/details endpoint, uses default board if not specified)",
#             default="",
#             advanced=True,
#         )
#         note: str = SchemaField(
#             description="Private note for the pin (only visible to you and board collaborators)",
#             default="",
#             advanced=True,
#         )
#         thumbnail: str = SchemaField(
#             description="Required thumbnail URL for video pins (must have valid image Content-Type)",
#             default="",
#             advanced=True,
#         )
#         carousel_options: list[PinterestCarouselOption] = SchemaField(
#             description="Options for each image in carousel (title, link, description per image)",
#             default_factory=list,
#             advanced=True,
#         )
#         alt_text: list[str] = SchemaField(
#             description="Alt text for each image/video (max 500 chars each, accessibility feature)",
#             default_factory=list,
#             advanced=True,
#         )

#     class Output(BlockSchema):
#         post_result: RequestOutput = SchemaField(description="The result of the post")

#     def __init__(self):
#         super().__init__(
#             id="3ca46e05-dbaa-4afb-9e95-5a429c4177e6",
#             description="Post to Pinterest using Ayrshare",
#             categories={BlockCategory.SOCIAL},
#             block_type=BlockType.AYRSHARE,
#             input_schema=PostToPinterestBlock.Input,
#             output_schema=PostToPinterestBlock.Output,
#         )

#     async def run(
#         self,
#         input_data: "PostToPinterestBlock.Input",
#         *,
#         profile_key: SecretStr,
#         **kwargs,
#     ) -> BlockOutput:
#         """Post to Pinterest with Pinterest-specific options."""
#         if not profile_key:
#             yield "error", "Please link a social account via Ayrshare"
#             return

#         client = create_ayrshare_client()
#         if not client:
#             yield "error", "Ayrshare integration is not configured. Please set up the AYRSHARE_API_KEY."
#             return

#         # Validate Pinterest constraints
#         if len(input_data.post) > 500:
#             yield "error", f"Pinterest pin description exceeds 500 character limit ({len(input_data.post)} characters)"
#             return

#         if len(input_data.pin_title) > 100:
#             yield "error", f"Pinterest pin title exceeds 100 character limit ({len(input_data.pin_title)} characters)"
#             return

#         if len(input_data.link) > 2048:
#             yield "error", f"Pinterest link URL exceeds 2048 character limit ({len(input_data.link)} characters)"
#             return

#         if len(input_data.media_urls) == 0:
#             yield "error", "Pinterest requires at least one image or video"
#             return

#         if len(input_data.media_urls) > 5:
#             yield "error", "Pinterest supports a maximum of 5 images in a carousel"
#             return

#         # Check if video is included and thumbnail is provided
#         video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm"]
#         has_video = any(
#             any(url.lower().endswith(ext) for ext in video_extensions)
#             for url in input_data.media_urls
#         )

#         if (has_video or input_data.is_video) and not input_data.thumbnail:
#             yield "error", "Pinterest video pins require a thumbnail URL"
#             return

#         # Validate alt text length
#         for i, alt in enumerate(input_data.alt_text):
#             if len(alt) > 500:
#                 yield "error", f"Pinterest alt text {i+1} exceeds 500 character limit ({len(alt)} characters)"
#                 return

#         # Convert datetime to ISO format if provided
#         iso_date = (
#             input_data.schedule_date.isoformat() if input_data.schedule_date else None
#         )

#         # Build Pinterest-specific options
#         pinterest_options = {}

#         # Pin title
#         if input_data.pin_title:
#             pinterest_options["title"] = input_data.pin_title

#         # Clickable link
#         if input_data.link:
#             pinterest_options["link"] = input_data.link

#         # Board ID
#         if input_data.board_id:
#             pinterest_options["boardId"] = input_data.board_id

#         # Private note
#         if input_data.note:
#             pinterest_options["note"] = input_data.note

#         # Video thumbnail
#         if input_data.thumbnail:
#             pinterest_options["thumbNail"] = input_data.thumbnail

#         # Carousel options
#         if input_data.carousel_options:
#             carousel_list = []
#             for option in input_data.carousel_options:
#                 carousel_dict = {}
#                 if option.title:
#                     carousel_dict["title"] = option.title
#                 if option.link:
#                     carousel_dict["link"] = option.link
#                 if option.description:
#                     carousel_dict["description"] = option.description
#                 if carousel_dict:  # Only add if not empty
#                     carousel_list.append(carousel_dict)
#             if carousel_list:
#                 pinterest_options["carouselOptions"] = carousel_list

#         # Alt text
#         if input_data.alt_text:
#             pinterest_options["altText"] = input_data.alt_text

#         try:
#             response = client.create_post(
#                 post=input_data.post,
#                 platforms=[SocialPlatform.PINTEREST],
#                 media_urls=input_data.media_urls,
#                 is_video=input_data.is_video,
#                 schedule_date=iso_date,
#                 disable_comments=input_data.disable_comments,
#                 shorten_links=input_data.shorten_links,
#                 unsplash=input_data.unsplash,
#                 requires_approval=input_data.requires_approval,
#                 random_post=input_data.random_post,
#                 random_media_url=input_data.random_media_url,
#                 notes=input_data.notes,
#                 pinterest_options=pinterest_options if pinterest_options else None,
#                 profile_key=profile_key.get_secret_value(),
#             )
#             yield "post_result", response
#         except Exception as e:
#             yield "error", str(e)


# class PostToTikTokBlock(Block):
#     """Block for posting to TikTok with TikTok-specific options."""

#     class Input(BaseAyrshareInput):
#         """Input schema for TikTok posts."""

#         # Override post field to include TikTok-specific information
#         post: str = SchemaField(
#             description="The post text (max 2,200 chars, empty string allowed). Use @handle to mention users. Line breaks will be ignored.",
#             default="",
#             advanced=False,
#         )

#         # Override media_urls to include TikTok-specific constraints
#         media_urls: list[str] = SchemaField(
#             description="Required media URLs. Either 1 video OR up to 35 images (JPG/JPEG/WEBP only). Cannot mix video and images.",
#             default_factory=list,
#             advanced=False,
#         )

#         # TikTok-specific options
#         auto_add_music: bool = SchemaField(
#             description="Automatically add recommended music to image posts",
#             default=False,
#             advanced=True,
#         )
#         disable_comments: bool = SchemaField(
#             description="Disable comments on the published post",
#             default=False,
#             advanced=True,
#         )
#         disable_duet: bool = SchemaField(
#             description="Disable duets on published video (video only)",
#             default=False,
#             advanced=True,
#         )
#         disable_stitch: bool = SchemaField(
#             description="Disable stitch on published video (video only)",
#             default=False,
#             advanced=True,
#         )
#         is_ai_generated: bool = SchemaField(
#             description="Label content as AI-generated (video only)",
#             default=False,
#             advanced=True,
#         )
#         is_branded_content: bool = SchemaField(
#             description="Label as branded content (paid partnership)",
#             default=False,
#             advanced=True,
#         )
#         is_brand_organic: bool = SchemaField(
#             description="Label as brand organic content (promotional)",
#             default=False,
#             advanced=True,
#         )
#         image_cover_index: int = SchemaField(
#             description="Index of image to use as cover (0-based, image posts only)",
#             default=0,
#             advanced=True,
#         )
#         title: str = SchemaField(
#             description="Title for image posts", default="", advanced=True
#         )
#         thumbnail_offset: int = SchemaField(
#             description="Video thumbnail frame offset in milliseconds (video only)",
#             default=0,
#             advanced=True,
#         )
#         visibility: str = SchemaField(
#             description="Post visibility: 'public', 'private', 'followers', or 'friends'",
#             default="public",
#             advanced=True,
#         )
#         draft: bool = SchemaField(
#             description="Create as draft post (video only)",
#             default=False,
#             advanced=True,
#         )

#     class Output(BlockSchema):
#         post_result: RequestOutput = SchemaField(description="The result of the post")

#     def __init__(self):
#         super().__init__(
#             id="7faf4b27-96b0-4f05-bf64-e0de54ae74e1",
#             description="Post to TikTok using Ayrshare",
#             categories={BlockCategory.SOCIAL},
#             block_type=BlockType.AYRSHARE,
#             input_schema=PostToTikTokBlock.Input,
#             output_schema=PostToTikTokBlock.Output,
#         )

#     async def run(
#         self, input_data: "PostToTikTokBlock.Input", *, profile_key: SecretStr, **kwargs
#     ) -> BlockOutput:
#         """Post to TikTok with TikTok-specific validation and options."""
#         if not profile_key:
#             yield "error", "Please link a social account via Ayrshare"
#             return

#         client = create_ayrshare_client()
#         if not client:
#             yield "error", "Ayrshare integration is not configured. Please set up the AYRSHARE_API_KEY."
#             return

#         # Validate TikTok constraints
#         if len(input_data.post) > 2200:
#             yield "error", f"TikTok post text exceeds 2,200 character limit ({len(input_data.post)} characters)"
#             return

#         if not input_data.media_urls:
#             yield "error", "TikTok requires at least one media URL (either 1 video or up to 35 images)"
#             return

#         # Check for video vs image constraints
#         video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm"]
#         image_extensions = [".jpg", ".jpeg", ".webp"]

#         has_video = input_data.is_video or any(
#             any(url.lower().endswith(ext) for ext in video_extensions)
#             for url in input_data.media_urls
#         )

#         has_images = any(
#             any(url.lower().endswith(ext) for ext in image_extensions)
#             for url in input_data.media_urls
#         )

#         if has_video and has_images:
#             yield "error", "TikTok does not support mixing video and images in the same post"
#             return

#         if has_video and len(input_data.media_urls) > 1:
#             yield "error", "TikTok supports only 1 video per post"
#             return

#         if has_images and len(input_data.media_urls) > 35:
#             yield "error", "TikTok supports a maximum of 35 images per post"
#             return

#         # Validate image cover index
#         if has_images and input_data.image_cover_index >= len(input_data.media_urls):
#             yield "error", f"Image cover index {input_data.image_cover_index} is out of range (max: {len(input_data.media_urls) - 1})"
#             return

#         # Validate visibility option
#         valid_visibility = ["public", "private", "followers", "friends"]
#         if input_data.visibility not in valid_visibility:
#             yield "error", f"TikTok visibility must be one of: {', '.join(valid_visibility)}"
#             return

#         # Check for PNG files (not supported)
#         has_png = any(url.lower().endswith(".png") for url in input_data.media_urls)
#         if has_png:
#             yield "error", "TikTok does not support PNG files. Please use JPG, JPEG, or WEBP for images."
#             return

#         # Convert datetime to ISO format if provided
#         iso_date = (
#             input_data.schedule_date.isoformat() if input_data.schedule_date else None
#         )

#         # Build TikTok-specific options
#         tiktok_options = {}

#         # Common options
#         if input_data.auto_add_music and has_images:
#             tiktok_options["autoAddMusic"] = True

#         if input_data.disable_comments:
#             tiktok_options["disableComments"] = True

#         if input_data.is_branded_content:
#             tiktok_options["isBrandedContent"] = True

#         if input_data.is_brand_organic:
#             tiktok_options["isBrandOrganic"] = True

#         # Video-specific options
#         if has_video:
#             if input_data.disable_duet:
#                 tiktok_options["disableDuet"] = True

#             if input_data.disable_stitch:
#                 tiktok_options["disableStitch"] = True

#             if input_data.is_ai_generated:
#                 tiktok_options["isAIGenerated"] = True

#             if input_data.thumbnail_offset > 0:
#                 tiktok_options["thumbNailOffset"] = input_data.thumbnail_offset

#             if input_data.draft:
#                 tiktok_options["draft"] = True

#         # Image-specific options
#         if has_images:
#             if input_data.image_cover_index > 0:
#                 tiktok_options["imageCoverIndex"] = input_data.image_cover_index

#             if input_data.title:
#                 tiktok_options["title"] = input_data.title

#             if input_data.visibility != "public":
#                 tiktok_options["visibility"] = input_data.visibility

#         try:
#             response = client.create_post(
#                 post=input_data.post,
#                 platforms=[SocialPlatform.TIKTOK],
#                 media_urls=input_data.media_urls,
#                 is_video=has_video,
#                 schedule_date=iso_date,
#                 disable_comments=input_data.disable_comments,
#                 shorten_links=input_data.shorten_links,
#                 unsplash=input_data.unsplash,
#                 requires_approval=input_data.requires_approval,
#                 random_post=input_data.random_post,
#                 random_media_url=input_data.random_media_url,
#                 notes=input_data.notes,
#                 tiktok_options=tiktok_options if tiktok_options else None,
#                 profile_key=profile_key.get_secret_value(),
#             )
#             yield "post_result", response
#         except Exception as e:
#             yield "error", str(e)


# class PostToThreadsBlock(Block):
#     """Block for posting to Threads with Threads-specific options."""

#     class Input(BaseAyrshareInput):
#         """Input schema for Threads posts."""

#         # Override post field to include Threads-specific information
#         post: str = SchemaField(
#             description="The post text (max 500 chars, empty string allowed). Only 1 hashtag allowed. Use @handle to mention users.",
#             default="",
#             advanced=False,
#         )

#         # Override media_urls to include Threads-specific constraints
#         media_urls: list[str] = SchemaField(
#             description="Optional list of media URLs. Supports up to 20 images/videos in a carousel. Auto-preview links unless media is included.",
#             default_factory=list,
#             advanced=False,
#         )

#     class Output(BlockSchema):
#         post_result: RequestOutput = SchemaField(description="The result of the post")

#     def __init__(self):
#         super().__init__(
#             id="f8c3b2e1-9d4a-4e5f-8c7b-6a9e8d2f1c3b",
#             description="Post to Threads using Ayrshare",
#             categories={BlockCategory.SOCIAL},
#             block_type=BlockType.AYRSHARE,
#             input_schema=PostToThreadsBlock.Input,
#             output_schema=PostToThreadsBlock.Output,
#         )

#     async def run(
#         self,
#         input_data: "PostToThreadsBlock.Input",
#         *,
#         profile_key: SecretStr,
#         **kwargs,
#     ) -> BlockOutput:
#         """Post to Threads with Threads-specific validation."""
#         if not profile_key:
#             yield "error", "Please link a social account via Ayrshare"
#             return

#         client = create_ayrshare_client()
#         if not client:
#             yield "error", "Ayrshare integration is not configured. Please set up the AYRSHARE_API_KEY."
#             return

#         # Validate Threads constraints
#         if len(input_data.post) > 500:
#             yield "error", f"Threads post text exceeds 500 character limit ({len(input_data.post)} characters)"
#             return

#         if len(input_data.media_urls) > 20:
#             yield "error", "Threads supports a maximum of 20 images/videos in a carousel"
#             return

#         # Count hashtags (only 1 allowed)
#         hashtag_count = input_data.post.count("#")
#         if hashtag_count > 1:
#             yield "error", f"Threads allows only 1 hashtag per post ({hashtag_count} found)"
#             return

#         # Convert datetime to ISO format if provided
#         iso_date = (
#             input_data.schedule_date.isoformat() if input_data.schedule_date else None
#         )

#         # Build Threads-specific options
#         threads_options = {}
#         # Note: Based on the documentation, Threads doesn't seem to have specific options
#         # beyond the standard ones. The main constraints are validation-based.

#         try:
#             response = client.create_post(
#                 post=input_data.post,
#                 platforms=[SocialPlatform.THREADS],
#                 media_urls=input_data.media_urls,
#                 is_video=input_data.is_video,
#                 schedule_date=iso_date,
#                 disable_comments=input_data.disable_comments,
#                 shorten_links=input_data.shorten_links,
#                 unsplash=input_data.unsplash,
#                 requires_approval=input_data.requires_approval,
#                 random_post=input_data.random_post,
#                 random_media_url=input_data.random_media_url,
#                 notes=input_data.notes,
#                 threads_options=threads_options if threads_options else None,
#                 profile_key=profile_key.get_secret_value(),
#             )
#             yield "post_result", response
#         except Exception as e:
#             yield "error", str(e)


# class PostToBlueskyBlock(Block):
#     """Block for posting to Bluesky with Bluesky-specific options."""

#     class Input(BaseAyrshareInput):
#         """Input schema for Bluesky posts."""

#         # Override post field to include character limit information
#         post: str = SchemaField(
#             description="The post text to be published (max 300 characters for Bluesky)",
#             default="",
#             advanced=False,
#         )

#         # Override media_urls to include Bluesky-specific constraints
#         media_urls: list[str] = SchemaField(
#             description="Optional list of media URLs to include. Bluesky supports up to 4 images or 1 video.",
#             default_factory=list,
#             advanced=False,
#         )

#         # Bluesky-specific options
#         alt_text: list[str] = SchemaField(
#             description="Alt text for each media item (accessibility)",
#             default_factory=list,
#             advanced=True,
#         )

#     class Output(BlockSchema):
#         post_result: RequestOutput = SchemaField(description="The result of the post")

#     def __init__(self):
#         super().__init__(
#             id="cbd52c2a-06d2-43ed-9560-6576cc163283",
#             description="Post to Bluesky using Ayrshare",
#             categories={BlockCategory.SOCIAL},
#             block_type=BlockType.AYRSHARE,
#             input_schema=PostToBlueskyBlock.Input,
#             output_schema=PostToBlueskyBlock.Output,
#         )

#     async def run(
#         self,
#         input_data: "PostToBlueskyBlock.Input",
#         *,
#         profile_key: SecretStr,
#         **kwargs,
#     ) -> BlockOutput:
#         """Post to Bluesky with Bluesky-specific options."""
#         if not profile_key:
#             yield "error", "Please link a social account via Ayrshare"
#             return

#         client = create_ayrshare_client()
#         if not client:
#             yield "error", "Ayrshare integration is not configured. Please set up the AYRSHARE_API_KEY."
#             return

#         # Validate character limit for Bluesky
#         if len(input_data.post) > 300:
#             yield "error", f"Post text exceeds Bluesky's 300 character limit ({len(input_data.post)} characters)"
#             return

#         # Validate media constraints for Bluesky
#         if len(input_data.media_urls) > 4:
#             yield "error", "Bluesky supports a maximum of 4 images or 1 video"
#             return

#         # Convert datetime to ISO format if provided
#         iso_date = (
#             input_data.schedule_date.isoformat() if input_data.schedule_date else None
#         )

#         # Build Bluesky-specific options
#         bluesky_options = {}
#         if input_data.alt_text:
#             bluesky_options["altText"] = input_data.alt_text

#         try:
#             response = client.create_post(
#                 post=input_data.post,
#                 platforms=[SocialPlatform.BLUESKY],
#                 media_urls=input_data.media_urls,
#                 is_video=input_data.is_video,
#                 schedule_date=iso_date,
#                 disable_comments=input_data.disable_comments,
#                 shorten_links=input_data.shorten_links,
#                 unsplash=input_data.unsplash,
#                 requires_approval=input_data.requires_approval,
#                 random_post=input_data.random_post,
#                 random_media_url=input_data.random_media_url,
#                 notes=input_data.notes,
#                 bluesky_options=bluesky_options if bluesky_options else None,
#                 profile_key=profile_key.get_secret_value(),
#             )
#             yield "post_result", response
#         except Exception as e:
#             yield "error", str(e)


# class PostToSnapchatBlock(Block):
#     """Block for posting to Snapchat with Snapchat-specific options."""

#     class Input(BaseAyrshareInput):
#         """Input schema for Snapchat posts."""

#         # Override post field to include Snapchat-specific information
#         post: str = SchemaField(
#             description="The post text (optional for video-only content)",
#             default="",
#             advanced=False,
#         )

#         # Override media_urls to include Snapchat-specific constraints
#         media_urls: list[str] = SchemaField(
#             description="Required video URL for Snapchat posts. Snapchat only supports video content.",
#             default_factory=list,
#             advanced=False,
#         )

#         # Snapchat-specific options
#         story_type: str = SchemaField(
#             description="Type of Snapchat content: 'story' (24-hour Stories), 'saved_story' (Saved Stories), or 'spotlight' (Spotlight posts)",
#             default="story",
#             advanced=True,
#         )
#         video_thumbnail: str = SchemaField(
#             description="Thumbnail URL for video content (optional, auto-generated if not provided)",
#             default="",
#             advanced=True,
#         )

#     class Output(BlockSchema):
#         post_result: RequestOutput = SchemaField(description="The result of the post")

#     def __init__(self):
#         super().__init__(
#             id="a9d7f854-2c83-4e96-b3a1-7f2e9c5d4b8e",
#             description="Post to Snapchat using Ayrshare",
#             categories={BlockCategory.SOCIAL},
#             block_type=BlockType.AYRSHARE,
#             input_schema=PostToSnapchatBlock.Input,
#             output_schema=PostToSnapchatBlock.Output,
#         )

#     async def run(
#         self,
#         input_data: "PostToSnapchatBlock.Input",
#         *,
#         profile_key: SecretStr,
#         **kwargs,
#     ) -> BlockOutput:
#         """Post to Snapchat with Snapchat-specific options."""
#         if not profile_key:
#             yield "error", "Please link a social account via Ayrshare"
#             return

#         client = create_ayrshare_client()
#         if not client:
#             yield "error", "Ayrshare integration is not configured. Please set up the AYRSHARE_API_KEY."
#             return

#         # Validate Snapchat constraints
#         if not input_data.media_urls:
#             yield "error", "Snapchat requires at least one video URL"
#             return

#         if len(input_data.media_urls) > 1:
#             yield "error", "Snapchat supports only one video per post"
#             return

#         # Validate story type
#         valid_story_types = ["story", "saved_story", "spotlight"]
#         if input_data.story_type not in valid_story_types:
#             yield "error", f"Snapchat story type must be one of: {', '.join(valid_story_types)}"
#             return

#         # Convert datetime to ISO format if provided
#         iso_date = (
#             input_data.schedule_date.isoformat() if input_data.schedule_date else None
#         )

#         # Build Snapchat-specific options
#         snapchat_options = {}

#         # Story type
#         if input_data.story_type != "story":
#             snapchat_options["storyType"] = input_data.story_type

#         # Video thumbnail
#         if input_data.video_thumbnail:
#             snapchat_options["videoThumbnail"] = input_data.video_thumbnail

#         try:
#             response = client.create_post(
#                 post=input_data.post,
#                 platforms=[SocialPlatform.SNAPCHAT],
#                 media_urls=input_data.media_urls,
#                 is_video=True,  # Snapchat only supports video
#                 schedule_date=iso_date,
#                 disable_comments=input_data.disable_comments,
#                 shorten_links=input_data.shorten_links,
#                 unsplash=input_data.unsplash,
#                 requires_approval=input_data.requires_approval,
#                 random_post=input_data.random_post,
#                 random_media_url=input_data.random_media_url,
#                 notes=input_data.notes,
#                 snapchat_options=snapchat_options if snapchat_options else None,
#                 profile_key=profile_key.get_secret_value(),
#             )
#             yield "post_result", response
#         except Exception as e:
#             yield "error", str(e)


AYRSHARE_BLOCK_IDS = [
    # PostToBlueskyBlock().id,
    # PostToFacebookBlock().id,
    PostToXBlock().id,
    # PostToLinkedInBlock().id,
    # PostToInstagramBlock().id,
    # PostToYouTubeBlock().id,
    # PostToRedditBlock().id,
    # PostToTelegramBlock().id,
    # PostToGMBBlock().id,
    # PostToPinterestBlock().id,
    # PostToTikTokBlock().id,
    # PostToThreadsBlock().id,
    # PostToSnapchatBlock().id,
]
