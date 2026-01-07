import datetime
from typing import List

import prisma.enums
import pydantic

from backend.util.models import Pagination


class ChangelogEntry(pydantic.BaseModel):
    version: str
    changes_summary: str
    date: datetime.datetime


class MyAgent(pydantic.BaseModel):
    agent_id: str
    agent_version: int
    agent_name: str
    agent_image: str | None = None
    description: str
    last_edited: datetime.datetime
    recommended_schedule_cron: str | None = None


class MyAgentsResponse(pydantic.BaseModel):
    agents: list[MyAgent]
    pagination: Pagination


class StoreAgent(pydantic.BaseModel):
    slug: str
    agent_name: str
    agent_image: str
    creator: str
    creator_avatar: str
    sub_heading: str
    description: str
    runs: int
    rating: float


class StoreAgentsResponse(pydantic.BaseModel):
    agents: list[StoreAgent]
    pagination: Pagination


class StoreAgentDetails(pydantic.BaseModel):
    store_listing_version_id: str
    slug: str
    agent_name: str
    agent_video: str
    agent_output_demo: str
    agent_image: list[str]
    creator: str
    creator_avatar: str
    sub_heading: str
    description: str
    instructions: str | None = None
    categories: list[str]
    runs: int
    rating: float
    versions: list[str]
    agentGraphVersions: list[str]
    agentGraphId: str
    last_updated: datetime.datetime
    recommended_schedule_cron: str | None = None

    active_version_id: str | None = None
    has_approved_version: bool = False

    # Optional changelog data when include_changelog=True
    changelog: list[ChangelogEntry] | None = None


class Creator(pydantic.BaseModel):
    name: str
    username: str
    description: str
    avatar_url: str
    num_agents: int
    agent_rating: float
    agent_runs: int
    is_featured: bool


class CreatorsResponse(pydantic.BaseModel):
    creators: List[Creator]
    pagination: Pagination


class CreatorDetails(pydantic.BaseModel):
    name: str
    username: str
    description: str
    links: list[str]
    avatar_url: str
    agent_rating: float
    agent_runs: int
    top_categories: list[str]


class Profile(pydantic.BaseModel):
    name: str
    username: str
    description: str
    links: list[str]
    avatar_url: str
    is_featured: bool = False


class StoreSubmission(pydantic.BaseModel):
    listing_id: str
    agent_id: str
    agent_version: int
    name: str
    sub_heading: str
    slug: str
    description: str
    instructions: str | None = None
    image_urls: list[str]
    date_submitted: datetime.datetime
    status: prisma.enums.SubmissionStatus
    runs: int
    rating: float
    store_listing_version_id: str | None = None
    version: int | None = None  # Actual version number from the database

    reviewer_id: str | None = None
    review_comments: str | None = None  # External comments visible to creator
    internal_comments: str | None = None  # Private notes for admin use only
    reviewed_at: datetime.datetime | None = None
    changes_summary: str | None = None

    # Additional fields for editing
    video_url: str | None = None
    agent_output_demo_url: str | None = None
    categories: list[str] = []


class StoreSubmissionsResponse(pydantic.BaseModel):
    submissions: list[StoreSubmission]
    pagination: Pagination


class StoreListingWithVersions(pydantic.BaseModel):
    """A store listing with its version history"""

    listing_id: str
    slug: str
    agent_id: str
    agent_version: int
    active_version_id: str | None = None
    has_approved_version: bool = False
    creator_email: str | None = None
    latest_version: StoreSubmission | None = None
    versions: list[StoreSubmission] = []


class StoreListingsWithVersionsResponse(pydantic.BaseModel):
    """Response model for listings with version history"""

    listings: list[StoreListingWithVersions]
    pagination: Pagination


class StoreSubmissionRequest(pydantic.BaseModel):
    agent_id: str = pydantic.Field(
        ..., min_length=1, description="Agent ID cannot be empty"
    )
    agent_version: int = pydantic.Field(
        ..., gt=0, description="Agent version must be greater than 0"
    )
    slug: str
    name: str
    sub_heading: str
    video_url: str | None = None
    agent_output_demo_url: str | None = None
    image_urls: list[str] = []
    description: str = ""
    instructions: str | None = None
    categories: list[str] = []
    changes_summary: str | None = None
    recommended_schedule_cron: str | None = None


class StoreSubmissionEditRequest(pydantic.BaseModel):
    name: str
    sub_heading: str
    video_url: str | None = None
    agent_output_demo_url: str | None = None
    image_urls: list[str] = []
    description: str = ""
    instructions: str | None = None
    categories: list[str] = []
    changes_summary: str | None = None
    recommended_schedule_cron: str | None = None


class ProfileDetails(pydantic.BaseModel):
    name: str
    username: str
    description: str
    links: list[str]
    avatar_url: str | None = None


class StoreReview(pydantic.BaseModel):
    score: int
    comments: str | None = None


class StoreReviewCreate(pydantic.BaseModel):
    store_listing_version_id: str
    score: int
    comments: str | None = None


class ReviewSubmissionRequest(pydantic.BaseModel):
    store_listing_version_id: str
    is_approved: bool
    comments: str  # External comments visible to creator
    internal_comments: str | None = None  # Private admin notes
