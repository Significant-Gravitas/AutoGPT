import datetime
import typing

import prisma.enums
import pydantic


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
    submission_status: str
    views: int = 0
    downloads: int = 0


class AgentListResponse(pydantic.BaseModel):
    """
    Represents a response containing a list of agents.

    Attributes:
        agents (list[AgentResponse]): The list of agents.
        total_count (int): The total count of agents.
        page (int): The current page number.
        page_size (int): The number of agents per page.
        total_pages (int): The total number of pages.
    """

    agents: list[AgentResponse]
    total_count: int
    page: int
    page_size: int
    total_pages: int


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
