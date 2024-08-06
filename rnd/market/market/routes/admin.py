import fastapi
import prisma
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
