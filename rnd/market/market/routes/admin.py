import fastapi
import prisma

import market.db
import market.model

router = fastapi.APIRouter()


@router.post("/agent", response_model=market.model.AgentResponse)
async def create_agent_entry(request: market.model.AddAgentRequest):
    """
    A basic endpoint to create a new agent entry in the database.

    TODO: Protect endpoint!
    """
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
