import typing
import fastapi
import prisma
import prisma.enums
import prisma.models

import market.db
import market.model
import market.auth

router = fastapi.APIRouter()


@router.post("/agent", response_model=market.model.AgentResponse)
async def create_agent_entry(
    request: market.model.AddAgentRequest,
    user: prisma.models.User = fastapi.Depends(market.auth.get_user),
):
    """
    A basic endpoint to create a new agent entry in the database.

    TODO: Protect endpoint!
    """
    if not user or user.role != "admin":
        raise fastapi.HTTPException(status_code=401, detail="Unauthorized")

    try:
        agent = await market.db.create_agent_entry(
            request.graph["name"],
            request.graph["description"],
            request.author,
            request.keywords,
            request.categories,
            prisma.Json(request.graph),
        )

        return fastapi.responses.PlainTextResponse(agent.model_dump_json())
    except market.db.AgentQueryError as e:
        raise fastapi.HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=str(e))


@router.post("/agent/featured/{agent_id}")
async def set_agent_featured(
    agent_id: str,
    category: str = "featured",
    user: prisma.models.User = fastapi.Depends(market.auth.get_user),
):
    """
    A basic endpoint to set an agent as featured in the database.
    """
    if not user or user.role != "admin":
        raise fastapi.HTTPException(status_code=401, detail="Unauthorized")
    try:
        await market.db.set_agent_featured(
            agent_id, is_featured=True, category=category
        )
        return fastapi.responses.Response(status_code=200)
    except market.db.AgentQueryError as e:
        raise fastapi.HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=str(e))


@router.delete("/agent/featured/{agent_id}")
async def unset_agent_featured(
    agent_id: str,
    category: str = "featured",
    user: prisma.models.User = fastapi.Depends(market.auth.get_user),
):
    """
    A basic endpoint to unset an agent as featured in the database.
    """
    if not user or user.role != "admin":
        raise fastapi.HTTPException(status_code=401, detail="Unauthorized")
    try:
        await market.db.set_agent_featured(
            agent_id, is_featured=False, category=category
        )
        return fastapi.responses.Response(status_code=200)
    except market.db.AgentQueryError as e:
        raise fastapi.HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=str(e))


@router.get("/agent/submissions", response_model=market.model.AgentListResponse)
async def get_agent_submissions(
    page: int = fastapi.Query(1, ge=1, description="Page number"),
    page_size: int = fastapi.Query(
        10, ge=1, le=100, description="Number of items per page"
    ),
    name: typing.Optional[str] = fastapi.Query(
        None, description="Filter by agent name"
    ),
    keyword: typing.Optional[str] = fastapi.Query(
        None, description="Filter by keyword"
    ),
    category: typing.Optional[str] = fastapi.Query(
        None, description="Filter by category"
    ),
    description: typing.Optional[str] = fastapi.Query(
        None, description="Fuzzy search in description"
    ),
    description_threshold: int = fastapi.Query(
        60, ge=0, le=100, description="Fuzzy search threshold"
    ),
    sort_by: str = fastapi.Query("createdAt", description="Field to sort by"),
    sort_order: typing.Literal["asc", "desc"] = fastapi.Query(
        "desc", description="Sort order (asc or desc)"
    ),
    user: prisma.models.User = fastapi.Depends(market.auth.get_user),
):
    if not user or user.role != "admin":
        raise fastapi.HTTPException(status_code=401, detail="Unauthorized")

    try:
        result = await market.db.get_agents(
            page=page,
            page_size=page_size,
            name=name,
            keyword=keyword,
            category=category,
            description=description,
            description_threshold=description_threshold,
            sort_by=sort_by,
            sort_order=sort_order,
            submission_status=prisma.enums.SubmissionStatus.PENDING,
        )

        agents = [
            market.model.AgentResponse(**agent.dict()) for agent in result["agents"]
        ]

        return market.model.AgentListResponse(
            agents=agents,
            total_count=result["total_count"],
            page=result["page"],
            page_size=result["page_size"],
            total_pages=result["total_pages"],
        )

    except market.db.AgentQueryError as e:
        raise fastapi.HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise fastapi.HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )

@router.post("/agent/submissions/{agent_id}")
async def review_submission(
    review_request: market.model.SubmissionReviewRequest,
    user: prisma.models.User = fastapi.Depends(market.auth.get_user),
):
    """
    A basic endpoint to review a submission in the database.
    """
    if not user or user.role != "admin":
        raise fastapi.HTTPException(status_code=401, detail="Unauthorized")

    try:
        await market.db.update_agent_entry(
            agent_id=review_request.agent_id,
            version=review_request.version,
            submission_state=review_request.status,
            comments=review_request.comments,
        )
        return fastapi.responses.Response(status_code=200)
    except market.db.AgentQueryError as e:
        raise fastapi.HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=str(e))