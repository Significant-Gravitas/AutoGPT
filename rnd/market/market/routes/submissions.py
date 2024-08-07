import autogpt_libs.auth
import fastapi
import fastapi.responses
import prisma

import market.db
import market.model
import market.utils.analytics

router = fastapi.APIRouter()


@autogpt_libs.auth.require_auth()
@router.post("/agents/submit", response_model=market.model.AgentResponse)
async def submit_agent(
    request: market.model.AddAgentRequest,
):
    """
    A basic endpoint to create a new agent entry in the database.
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
