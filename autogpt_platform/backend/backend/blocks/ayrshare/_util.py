from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from backend.data.block import BlockSchema
from backend.data.model import SchemaField, UserIntegrations
from backend.integrations.ayrshare import AyrshareClient
from backend.util.clients import get_database_manager_async_client
from backend.util.exceptions import MissingConfigError


async def get_profile_key(user_id: str):
    user_integrations: UserIntegrations = (
        await get_database_manager_async_client().get_user_integrations(user_id)
    )
    return user_integrations.managed_credentials.ayrshare_profile_key


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
