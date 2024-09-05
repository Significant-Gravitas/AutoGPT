import fastapi

import market.db
import market.model

router = fastapi.APIRouter()


@router.post("/agent-installed")
async def agent_installed_endpoint(
    event_data: market.model.AgentInstalledFromMarketplaceEventData,
):
    """
    Endpoint to track agent installation events from the marketplace.

    Args:
        event_data (market.model.AgentInstalledFromMarketplaceEventData): The event data.
    """
    try:
        await market.db.create_agent_installed_event(event_data)
    except market.db.AgentQueryError as e:
        raise fastapi.HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise fastapi.HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )
