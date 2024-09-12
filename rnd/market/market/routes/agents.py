import json
import tempfile
import typing

import fastapi
import fastapi.responses
import prisma

import market.db
import market.model
import market.utils.analytics

router = fastapi.APIRouter()


@router.get("/agents", response_model=market.model.AgentListResponse)
async def list_agents(
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


@router.get("/agents/{agent_id}", response_model=market.model.AgentDetailResponse)
async def get_agent_details_endpoint(
    background_tasks: fastapi.BackgroundTasks,
    agent_id: str = fastapi.Path(..., description="The ID of the agent to retrieve"),
    version: typing.Optional[int] = fastapi.Query(
        None, description="Specific version of the agent"
    ),
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
        agent = await market.db.get_agent_details(agent_id, version)
        background_tasks.add_task(market.utils.analytics.track_view, agent_id)
        return market.model.AgentDetailResponse(**agent.model_dump())

    except market.db.AgentQueryError as e:
        raise fastapi.HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise fastapi.HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )


@router.get("/agents/{agent_id}/download")
async def download_agent(
    background_tasks: fastapi.BackgroundTasks,
    agent_id: str = fastapi.Path(..., description="The ID of the agent to retrieve"),
    version: typing.Optional[int] = fastapi.Query(
        None, description="Specific version of the agent"
    ),
):
    """
    Download details of a specific agent.

    NOTE: This is the same as agent details, however it also triggers
    the "download" tracking. We don't actually want to download a file though

    Args:
        agent_id (str): The ID of the agent to retrieve.
        version (Optional[int]): Specific version of the agent (default: None).

    Returns:
        market.model.AgentDetailResponse: The response containing the agent details.

    Raises:
        HTTPException: If the agent is not found or an unexpected error occurs.
    """
    try:
        agent = await market.db.get_agent_details(agent_id, version)
        background_tasks.add_task(market.utils.analytics.track_download, agent_id)
        return market.model.AgentDetailResponse(**agent.model_dump())

    except market.db.AgentQueryError as e:
        raise fastapi.HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise fastapi.HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )


@router.get("/agents/{agent_id}/download-file")
async def download_agent_file(
    background_tasks: fastapi.BackgroundTasks,
    agent_id: str = fastapi.Path(..., description="The ID of the agent to download"),
    version: typing.Optional[int] = fastapi.Query(
        None, description="Specific version of the agent"
    ),
) -> fastapi.responses.FileResponse:
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
    agent = await market.db.get_agent_details(agent_id, version)

    graph_data: prisma.Json = agent.graph

    background_tasks.add_task(market.utils.analytics.track_download, agent_id)

    file_name = f"agent_{agent_id}_v{version or 'latest'}.json"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_file:
        tmp_file.write(json.dumps(graph_data))
        tmp_file.flush()

        return fastapi.responses.FileResponse(
            tmp_file.name, filename=file_name, media_type="application/json"
        )


# top agents by downloads
@router.get("/top-downloads/agents", response_model=market.model.AgentListResponse)
async def top_agents_by_downloads(
    page: int = fastapi.Query(1, ge=1, description="Page number"),
    page_size: int = fastapi.Query(
        10, ge=1, le=100, description="Number of items per page"
    ),
):
    """
    Retrieve a list of top agents based on the number of downloads.

    Args:
        page (int): Page number (default: 1).
        page_size (int): Number of items per page (default: 10, min: 1, max: 100).

    Returns:
        market.model.AgentListResponse: A response containing the list of top agents and pagination information.

    Raises:
        HTTPException: If there is a client error (status code 400) or an unexpected error (status code 500).
    """
    try:
        result = await market.db.get_top_agents_by_downloads(
            page=page,
            page_size=page_size,
        )

        ret = market.model.AgentListResponse(
            total_count=result.total_count,
            page=result.page,
            page_size=result.page_size,
            total_pages=result.total_pages,
            agents=[
                market.model.AgentResponse(
                    id=item.agent.id,
                    name=item.agent.name,
                    description=item.agent.description,
                    author=item.agent.author,
                    keywords=item.agent.keywords,
                    categories=item.agent.categories,
                    version=item.agent.version,
                    createdAt=item.agent.createdAt,
                    updatedAt=item.agent.updatedAt,
                    views=item.views,
                    downloads=item.downloads,
                    submissionStatus=item.agent.submissionStatus,
                )
                for item in result.analytics
                if item.agent is not None
            ],
        )

        return ret

    except market.db.AgentQueryError as e:
        raise fastapi.HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise fastapi.HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        ) from e


@router.get("/featured/agents", response_model=market.model.AgentListResponse)
async def get_featured_agents(
    category: str = fastapi.Query(
        "featured", description="Category of featured agents"
    ),
    page: int = fastapi.Query(1, ge=1, description="Page number"),
    page_size: int = fastapi.Query(
        10, ge=1, le=100, description="Number of items per page"
    ),
):
    """
    Retrieve a list of featured agents based on the provided category.

    Args:
        category (str): Category of featured agents (default: "featured").
        page (int): Page number (default: 1).
        page_size (int): Number of items per page (default: 10, min: 1, max: 100).

    Returns:
        market.model.AgentListResponse: A response containing the list of featured agents and pagination information.

    Raises:
        HTTPException: If there is a client error (status code 400) or an unexpected error (status code 500).
    """
    try:
        result = await market.db.get_featured_agents(
            category=category,
            page=page,
            page_size=page_size,
        )

        ret = market.model.AgentListResponse(
            total_count=result.total_count,
            page=result.page,
            page_size=result.page_size,
            total_pages=result.total_pages,
            agents=[
                market.model.AgentResponse(
                    id=item.agent.id,
                    name=item.agent.name,
                    description=item.agent.description,
                    author=item.agent.author,
                    keywords=item.agent.keywords,
                    categories=item.agent.categories,
                    version=item.agent.version,
                    createdAt=item.agent.createdAt,
                    updatedAt=item.agent.updatedAt,
                    views=(
                        item.agent.AnalyticsTracker[0].views
                        if item.agent.AnalyticsTracker
                        and len(item.agent.AnalyticsTracker) > 0
                        else 0
                    ),
                    downloads=(
                        item.agent.AnalyticsTracker[0].downloads
                        if item.agent.AnalyticsTracker
                        and len(item.agent.AnalyticsTracker) > 0
                        else 0
                    ),
                    submissionStatus=item.agent.submissionStatus,
                )
                for item in result.featured_agents
                if item.agent is not None
            ],
        )

        return ret

    except market.db.AgentQueryError as e:
        raise fastapi.HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise fastapi.HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        ) from e
