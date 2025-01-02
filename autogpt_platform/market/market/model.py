import datetime
import typing
from enum import Enum
from typing import Generic, Literal, TypeVar, Union

import prisma.enums
import pydantic


class InstallationLocation(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"


class AgentInstalledFromMarketplaceEventData(pydantic.BaseModel):
    marketplace_agent_id: str
    installed_agent_id: str
    installation_location: InstallationLocation


class AgentInstalledFromTemplateEventData(pydantic.BaseModel):
    template_id: str
    installed_agent_id: str
    installation_location: InstallationLocation


class AgentInstalledFromMarketplaceEvent(pydantic.BaseModel):
    event_name: Literal["agent_installed_from_marketplace"]
    event_data: AgentInstalledFromMarketplaceEventData


class AgentInstalledFromTemplateEvent(pydantic.BaseModel):
    event_name: Literal["agent_installed_from_template"]
    event_data: AgentInstalledFromTemplateEventData


AnalyticsEvent = Union[
    AgentInstalledFromMarketplaceEvent, AgentInstalledFromTemplateEvent
]


class AnalyticsRequest(pydantic.BaseModel):
    event: AnalyticsEvent


class AddAgentRequest(pydantic.BaseModel):
    graph: dict[str, typing.Any]
    author: str
    keywords: list[str]
    categories: list[str]


class SubmissionReviewRequest(pydantic.BaseModel):
    agent_id: str
    version: int
    status: prisma.enums.SubmissionStatus
    comments: str | None


class AgentResponse(pydantic.BaseModel):
    """
    Represents a response from an agent.

    Attributes:
        id (str): The ID of the agent.
        name (str, optional): The name of the agent.
        description (str, optional): The description of the agent.
        author (str, optional): The author of the agent.
        keywords (list[str]): The keywords associated with the agent.
        categories (list[str]): The categories the agent belongs to.
        version (int): The version of the agent.
        createdAt (str): The creation date of the agent.
        updatedAt (str): The last update date of the agent.
    """

    id: str
    name: typing.Optional[str]
    description: typing.Optional[str]
    author: typing.Optional[str]
    keywords: list[str]
    categories: list[str]
    version: int
    createdAt: datetime.datetime
    updatedAt: datetime.datetime
    submissionStatus: str
    views: int = 0
    downloads: int = 0


class AgentDetailResponse(pydantic.BaseModel):
    """
    Represents the response data for an agent detail.

    Attributes:
        id (str): The ID of the agent.
        name (Optional[str]): The name of the agent.
        description (Optional[str]): The description of the agent.
        author (Optional[str]): The author of the agent.
        keywords (List[str]): The keywords associated with the agent.
        categories (List[str]): The categories the agent belongs to.
        version (int): The version of the agent.
        createdAt (str): The creation date of the agent.
        updatedAt (str): The last update date of the agent.
        graph (Dict[str, Any]): The graph data of the agent.
    """

    id: str
    name: typing.Optional[str]
    description: typing.Optional[str]
    author: typing.Optional[str]
    keywords: list[str]
    categories: list[str]
    version: int
    createdAt: datetime.datetime
    updatedAt: datetime.datetime
    graph: dict[str, typing.Any]


class FeaturedAgentResponse(pydantic.BaseModel):
    """
    Represents the response data for an agent detail.
    """

    agentId: str
    featuredCategories: list[str]
    createdAt: datetime.datetime
    updatedAt: datetime.datetime
    isActive: bool


class CategoriesResponse(pydantic.BaseModel):
    """
    Represents the response data for a list of categories.

    Attributes:
        unique_categories (list[str]): The list of unique categories.
    """

    unique_categories: list[str]


T = TypeVar("T")


class ListResponse(pydantic.BaseModel, Generic[T]):
    """
    Represents a list response.

    Attributes:
        items (list[T]): The list of items.
        total_count (int): The total count of items.
        page (int): The current page number.
        page_size (int): The number of items per page.
        total_pages (int): The total number of pages.
    """

    items: list[T]
    total_count: int
    page: int
    page_size: int
    total_pages: int
