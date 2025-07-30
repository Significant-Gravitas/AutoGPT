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


class PostToGMBBlock(Block):
    """Block for posting to Google My Business with GMB-specific options."""

    class Input(BaseAyrshareInput):
        """Input schema for Google My Business posts."""

        # Override media_urls to include GMB-specific constraints
        media_urls: list[str] = SchemaField(
            description="Optional list of media URLs. GMB supports only one image or video per post.",
            default_factory=list,
            advanced=False,
        )

        # GMB-specific options
        is_photo_video: bool = SchemaField(
            description="Whether this is a photo/video post (appears in Photos section)",
            default=False,
            advanced=True,
        )
        photo_category: str = SchemaField(
            description="Category for photo/video: cover, profile, logo, exterior, interior, product, at_work, food_and_drink, menu, common_area, rooms, teams",
            default="",
            advanced=True,
        )
        # Call to action options (flattened from CallToAction object)
        call_to_action_type: str = SchemaField(
            description="Type of action button: 'book', 'order', 'shop', 'learn_more', 'sign_up', or 'call'",
            default="",
            advanced=True,
        )
        call_to_action_url: str = SchemaField(
            description="URL for the action button (not required for 'call' action)",
            default="",
            advanced=True,
        )
        # Event details options (flattened from EventDetails object)
        event_title: str = SchemaField(
            description="Event title for event posts",
            default="",
            advanced=True,
        )
        event_start_date: str = SchemaField(
            description="Event start date in ISO format (e.g., '2024-03-15T09:00:00Z')",
            default="",
            advanced=True,
        )
        event_end_date: str = SchemaField(
            description="Event end date in ISO format (e.g., '2024-03-15T17:00:00Z')",
            default="",
            advanced=True,
        )
        # Offer details options (flattened from OfferDetails object)
        offer_title: str = SchemaField(
            description="Offer title for promotional posts",
            default="",
            advanced=True,
        )
        offer_start_date: str = SchemaField(
            description="Offer start date in ISO format (e.g., '2024-03-15T00:00:00Z')",
            default="",
            advanced=True,
        )
        offer_end_date: str = SchemaField(
            description="Offer end date in ISO format (e.g., '2024-04-15T23:59:59Z')",
            default="",
            advanced=True,
        )
        offer_coupon_code: str = SchemaField(
            description="Coupon code for the offer (max 58 characters)",
            default="",
            advanced=True,
        )
        offer_redeem_online_url: str = SchemaField(
            description="URL where customers can redeem the offer online",
            default="",
            advanced=True,
        )
        offer_terms_conditions: str = SchemaField(
            description="Terms and conditions for the offer",
            default="",
            advanced=True,
        )

    class Output(BlockSchema):
        post_result: PostResponse = SchemaField(description="The result of the post")
        post: PostIds = SchemaField(description="The result of the post")

    def __init__(self):
        super().__init__(
            disabled=True,
            id="2c38c783-c484-4503-9280-ef5d1d345a7e",
            description="Post to Google My Business using Ayrshare",
            categories={BlockCategory.SOCIAL},
            block_type=BlockType.AYRSHARE,
            input_schema=PostToGMBBlock.Input,
            output_schema=PostToGMBBlock.Output,
        )

    async def run(
        self, input_data: "PostToGMBBlock.Input", *, profile_key: SecretStr, **kwargs
    ) -> BlockOutput:
        """Post to Google My Business with GMB-specific options."""
        if not profile_key:
            yield "error", "Please link a social account via Ayrshare"
            return

        client = create_ayrshare_client()
        if not client:
            yield "error", "Ayrshare integration is not configured. Please set up the AYRSHARE_API_KEY."
            return

        # Validate GMB constraints
        if len(input_data.media_urls) > 1:
            yield "error", "Google My Business supports only one image or video per post"
            return

        # Validate offer coupon code length
        if input_data.offer_coupon_code and len(input_data.offer_coupon_code) > 58:
            yield "error", "GMB offer coupon code cannot exceed 58 characters"
            return

        # Convert datetime to ISO format if provided
        iso_date = (
            input_data.schedule_date.isoformat() if input_data.schedule_date else None
        )

        # Build GMB-specific options
        gmb_options = {}

        # Photo/Video post options
        if input_data.is_photo_video:
            gmb_options["isPhotoVideo"] = True
            if input_data.photo_category:
                gmb_options["category"] = input_data.photo_category

        # Call to Action (from flattened fields)
        if input_data.call_to_action_type:
            cta_dict = {"actionType": input_data.call_to_action_type}
            # URL not required for 'call' action type
            if (
                input_data.call_to_action_type != "call"
                and input_data.call_to_action_url
            ):
                cta_dict["url"] = input_data.call_to_action_url
            gmb_options["callToAction"] = cta_dict

        # Event details (from flattened fields)
        if (
            input_data.event_title
            and input_data.event_start_date
            and input_data.event_end_date
        ):
            gmb_options["event"] = {
                "title": input_data.event_title,
                "startDate": input_data.event_start_date,
                "endDate": input_data.event_end_date,
            }

        # Offer details (from flattened fields)
        if (
            input_data.offer_title
            and input_data.offer_start_date
            and input_data.offer_end_date
            and input_data.offer_coupon_code
            and input_data.offer_redeem_online_url
            and input_data.offer_terms_conditions
        ):
            gmb_options["offer"] = {
                "title": input_data.offer_title,
                "startDate": input_data.offer_start_date,
                "endDate": input_data.offer_end_date,
                "couponCode": input_data.offer_coupon_code,
                "redeemOnlineUrl": input_data.offer_redeem_online_url,
                "termsConditions": input_data.offer_terms_conditions,
            }

        response = await client.create_post(
            post=input_data.post,
            platforms=[SocialPlatform.GOOGLE_MY_BUSINESS],
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
            gmb_options=gmb_options if gmb_options else None,
            profile_key=profile_key.get_secret_value(),
        )
        yield "post_result", response
        if response.postIds:
            for p in response.postIds:
                yield "post", p
