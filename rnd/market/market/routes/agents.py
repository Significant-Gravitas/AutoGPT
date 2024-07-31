import json
from tempfile import NamedTemporaryFile
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import FileResponse
from prisma import Json

import market.model
from market.db import AgentQueryError, get_agent_details, get_agents

router = APIRouter()


@router.get("/agents", response_model=market.model.AgentListResponse)
async def list_agents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page"),
    name: Optional[str] = Query(None, description="Filter by agent name"),
    keyword: Optional[str] = Query(None, description="Filter by keyword"),
    category: Optional[str] = Query(None, description="Filter by category"),
    description: Optional[str] = Query(None, description="Fuzzy search in description"),
    description_threshold: int = Query(
        60, ge=0, le=100, description="Fuzzy search threshold"
    ),
    sort_by: str = Query("createdAt", description="Field to sort by"),
    sort_order: Literal["asc"] | Literal["desc"] = Query(
        "desc", description="Sort order (asc or desc)"
    ),
):
    """
    Retrieve a list of agents based on the provided filters.

    Args:
        page (int): Page number (default: 1).
        page_size (int): Number of items per page (default: 10, min: 1, max: 100).
        name (str, optional): Filter by agent name.
        keyword (str, optional): Filter by keyword.
        category (str, optional): Filter by category.
        description (str, optional): Fuzzy search in description.
        description_threshold (int): Fuzzy search threshold (default: 60, min: 0, max: 100).
        sort_by (str): Field to sort by (default: "createdAt").
        sort_order (str): Sort order (asc or desc) (default: "desc").

    Returns:
        market.model.AgentListResponse: A response containing the list of agents and pagination information.

    Raises:
        HTTPException: If there is a client error (status code 400) or an unexpected error (status code 500).
    """
    try:
        result = await get_agents(
            page=page,
            page_size=page_size,
            name=name,
            keyword=keyword,
            category=category,
            description=description,
            description_threshold=description_threshold,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        # Convert the result to the response model
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

    except AgentQueryError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


@router.get("/agents/{agent_id}", response_model=market.model.AgentDetailResponse)
async def get_agent_details_endpoint(
    agent_id: str = Path(..., description="The ID of the agent to retrieve"),
    version: Optional[int] = Query(None, description="Specific version of the agent"),
):
    """
    Retrieve details of a specific agent.

    Args:
        agent_id (str): The ID of the agent to retrieve.
        version (Optional[int]): Specific version of the agent (default: None).

    Returns:
        market.model.AgentDetailResponse: The response containing the agent details.

    Raises:
        HTTPException: If the agent is not found or an unexpected error occurs.
    """
    try:
        agent = await get_agent_details(agent_id, version)
        return market.model.AgentDetailResponse(**agent.model_dump())

    except AgentQueryError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )


@router.get("/agents/{agent_id}/download")
async def download_agent(
    agent_id: str = Path(..., description="The ID of the agent to download"),
    version: Optional[int] = Query(None, description="Specific version of the agent"),
) -> FileResponse:
    """
    Download the agent file by streaming its content.

    Args:
        agent_id (str): The ID of the agent to download.
        version (Optional[int]): Specific version of the agent to download.

    Returns:
        StreamingResponse: A streaming response containing the agent's graph data.

    Raises:
        HTTPException: If the agent is not found or an unexpected error occurs.
    """
    # try:
    agent = await get_agent_details(agent_id, version)

    # The agent.graph is already a JSON string, no need to parse and re-stringify
    graph_data: Json = agent.graph

    # Prepare the file name for download
    file_name = f"agent_{agent_id}_v{version or 'latest'}.json"

    # Create a temporary file to store the graph data
    with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
        tmp_file.write(json.dumps(graph_data))
        tmp_file.flush()

        # Return the temporary file as a streaming response
        return FileResponse(
            tmp_file.name, filename=file_name, media_type="application/json"
        )
