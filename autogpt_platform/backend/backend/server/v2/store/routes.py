from re import _FlagsType
import fastapi
import fastapi.responses
import pydantic

import backend.server.v2.store.model

router = fastapi.APIRouter()

##############################################
############### Agent Endpoints #############
##############################################


@router.get("agents", tags=["store"])
def get_agents(
    featured: bool, top: bool, categories: str, page: int = 1, page_size: int = 20
) -> backend.server.v2.store.model.StoreAgentsResponse:
    """
    This is needed for:
    - Home Page Featured Agents
    - Home Page Top Agents
    - Search Results
    - Agent Details - Other Agents By
    - Agent Details - Similar Agents
    - Creator Details - Agent By x

    ---
    To support all these different usecase we need a bunch of options:
    - featured: bool - Filteres the list to featured only
    - createdBy: username - Returns all agents by that user or group
    - sortedBy: [runs, stars] - For the Top agents
    - searchTerm: string - embedding similarity search based on Name, subheading and description
    - category: string - Filter by category

    In addition we need:
    - page: int - for pagination
    - page_size: int - for pagination
    """

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

    return fastapi.responses.JSONResponse(
        content=[
            {
                "agentName": "Super SEO Optimizer",
                "agentImage": "https://ddz4ak4pa3d19.cloudfront.net/cache/cc/11/cc1172271dcf723a34f488a3344e82b2.jpg",
                "creatorName": "AI Labs",
                "description": "Boost your website's search engine rankings with our advanced AI-powered SEO optimization tool.",
                "runs": 100000,
                "rating": 4.9,
                "featured": True,
            },
            {
                "agentName": "Content Wizard",
                "agentImage": "https://upload.wikimedia.org/wikipedia/commons/c/c5/Big_buck_bunny_poster_big.jpg",
                "creatorName": "WriteRight Inc.",
                "description": "Generate high-quality, engaging content for your blog, social media, or marketing campaigns.",
                "runs": 75000,
                "rating": 4.7,
                "featured": True,
            },
        ]
    )


@router.get("agents/{username}/{agent_name}")
def get_agent(
    username: str, agent_name: int
) -> backend.server.v2.store.model.StoreAgentDetails:
    """
    This is only used on the AgentDetails Page

    It returns the store listing agents details.
    """

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

    pass


##############################################
############# Creator Endpoints #############
##############################################


@router.get("creators", tags=["store"])
def get_creators(
    featured: bool, search: str, sortedBy: str, page: int = 1, page_size: int = 20
) -> backend.server.v2.store.model.CreatorsResponse:
    """
    This is needed for:
    - Home Page Featured Creators
    - Search Results Page

    ---

    To support this functionality we need:
    - featured: bool - to limit the list to just featured agents
    - search: str - vector search based on the creators profile description.
    - sortedBy: [agentRating, agentRuns] -
    """

    class Creator(pydantic.BaseModel):
        name: str
        username: str
        description: str
        avatarUrl: str
        numAgents: int

    pass


@router.get("creator/{username}", tags=["store"])
def get_ceator(username: str) -> backend.server.v2.store.model.CreatorDetails:
    """
    Get the details of a creator
    - Creator Details Page

    """

    class CreatorDetails(pydantic.BaseModel):
        name: str
        username: str
        description: str
        links: list[str]
        avatarUrl: str
        agentRating: float
        agentRuns: int
        topCategories: list[str]
