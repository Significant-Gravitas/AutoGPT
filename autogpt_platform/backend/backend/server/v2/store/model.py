import datetime
from typing import List

import prisma.enums
import pydantic


class Pagination(pydantic.BaseModel):
    total_items: int = pydantic.Field(
        description="Total number of items.", examples=[42]
    )
    total_pages: int = pydantic.Field(
        description="Total number of pages.", examples=[97]
    )
    current_page: int = pydantic.Field(
        description="Current_page page number.", examples=[1]
    )
    page_size: int = pydantic.Field(
        description="Number of items per page.", examples=[25]
    )


class MyAgent(pydantic.BaseModel):
    agent_id: str
    agent_version: int
    agent_name: str
    description: str
    last_edited: datetime.datetime


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
    agent_image: list[str]
    creator: str
    creator_avatar: str
    sub_heading: str
    description: str
    categories: list[str]
    runs: int
    rating: float
    versions: list[str]
    last_updated: datetime.datetime


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
    agent_id: str
    agent_version: int
    name: str
    sub_heading: str
    slug: str
    description: str
    image_urls: list[str]
    date_submitted: datetime.datetime
    status: prisma.enums.SubmissionStatus
    runs: int
    rating: float
    store_listing_version_id: str | None = None


class StoreSubmissionsResponse(pydantic.BaseModel):
    submissions: list[StoreSubmission]
    pagination: Pagination


class StoreSubmissionRequest(pydantic.BaseModel):
    agent_id: str
    agent_version: int
    slug: str
    name: str
    sub_heading: str
    video_url: str | None = None
    image_urls: list[str] = []
    description: str = ""
    categories: list[str] = []


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
    comments: str
