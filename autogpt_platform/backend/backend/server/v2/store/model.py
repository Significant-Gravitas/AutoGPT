from typing import List
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
    agentName: str
    agentImage: str
    creator: str
    creatorAvatar: str
    subHeading: str
    description: str
    runs: int
    stars: float


class StoreAgentsResponse(pydantic.BaseModel):
    agents: list[StoreAgent]
    pagination: Pagination


class StoreAgentDetails(pydantic.BaseModel):
    slug: str
    agentName: str
    agentVideo: str
    agentImage: list[str]
    creator: str
    creatorAvatar: str
    subHeading: str
    description: str
    categoires: list[str]
    runs: int
    stars: float
    verions: list[str]


class Creator(pydantic.BaseModel):
    name: str
    username: str
    description: str
    avatarUrl: str
    numAgents: int


class CreatorsResponse(pydantic.BaseModel):
    creators: List[Creator]
    pagination: Pagination


class CreatorDetails(pydantic.BaseModel):
    name: str
    username: str
    description: str
    links: list[str]
    avatarUrl: str
    agentRating: float
    agentRuns: int
    topCategories: list[str]
