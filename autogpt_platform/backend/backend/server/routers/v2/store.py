import fastapi
import fastapi.responses
import pydantic

router = fastapi.APIRouter()

##############################################
############### Agent Endpoints #############
##############################################


@router.get("agents", tags=["store"])
def get_agents(
    featured: bool, top: bool, categories: str, page: int = 1, page_size: int = 20
):
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
def get_agent(username: str, agent_name: int):
    """
    This is only used on the AgentDetails Page

    It returns the store listing agents details.
    """

    class StoreAgentDetails(pydantic.BaseModel):
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
def get_creators(page: int = 1, page_size: int = 20):
    """Get a list of creators"""
    pass


@router.get("creator/{username}", tags=["store"])
def get_ceator(username: string):
    """Get the details of a creator"""
