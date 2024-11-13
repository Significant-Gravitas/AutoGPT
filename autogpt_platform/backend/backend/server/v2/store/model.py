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


class StoreSubmission(pydantic.BaseModel):
    name: str
    description: str
    image_urls: list[str]
    date_submitted: datetime.datetime
    status: prisma.enums.SubmissionStatus
    runs: int
    rating: float


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
