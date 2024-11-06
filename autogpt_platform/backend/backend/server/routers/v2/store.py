import fastapi
import fastapi.responses

router = fastapi.APIRouter()

##############################################
############### Agent Endpoints #############
##############################################


@router.get("agents", tags=["store"])
def get_agents(
    featured: bool, top: bool, categories: str, page: int = 1, page_size: int = 20
):
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


@router.get("agents/{username}/{slug}")
def get_agent(agent_id: str, agent_version: int):
    """Gets the details of an agent"""
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
