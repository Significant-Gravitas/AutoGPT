import logging
import urllib.parse
from collections import defaultdict
from typing import Annotated, Any, Literal, Optional, Sequence

from fastapi import APIRouter, Body, HTTPException, Security
from prisma.enums import AgentExecutionStatus, APIKeyPermission
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

import backend.api.features.store.cache as store_cache
import backend.api.features.store.model as store_model
import backend.data.block
from backend.api.external.middleware import require_permission
from backend.data import execution as execution_db
from backend.data import graph as graph_db
from backend.data import user as user_db
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.block import BlockInput, CompletedBlockOutput
from backend.executor.utils import add_graph_execution
from backend.util.settings import Settings

from .integrations import integrations_router
from .tools import tools_router

settings = Settings()
logger = logging.getLogger(__name__)

v1_router = APIRouter()

v1_router.include_router(integrations_router)
v1_router.include_router(tools_router)


class UserInfoResponse(BaseModel):
    id: str
    name: Optional[str]
    email: str
    timezone: str = Field(
        description="The user's last known timezone (e.g. 'Europe/Amsterdam'), "
        "or 'not-set' if not set"
    )


@v1_router.get(
    path="/me",
    tags=["user", "meta"],
)
async def get_user_info(
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.IDENTITY)
    ),
) -> UserInfoResponse:
    user = await user_db.get_user_by_id(auth.user_id)

    return UserInfoResponse(
        id=user.id,
        name=user.name,
        email=user.email,
        timezone=user.timezone,
    )


@v1_router.get(
    path="/blocks",
    tags=["blocks"],
    dependencies=[Security(require_permission(APIKeyPermission.READ_BLOCK))],
)
async def get_graph_blocks() -> Sequence[dict[Any, Any]]:
    blocks = [block() for block in backend.data.block.get_blocks().values()]
    return [b.to_dict() for b in blocks if not b.disabled]


@v1_router.post(
    path="/blocks/{block_id}/execute",
    tags=["blocks"],
    dependencies=[Security(require_permission(APIKeyPermission.EXECUTE_BLOCK))],
)
async def execute_graph_block(
    block_id: str,
    data: BlockInput,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.EXECUTE_BLOCK)
    ),
) -> CompletedBlockOutput:
    obj = backend.data.block.get_block(block_id)
    if not obj:
        raise HTTPException(status_code=404, detail=f"Block #{block_id} not found.")

    output = defaultdict(list)
    async for name, data in obj.execute(data):
        output[name].append(data)
    return output


@v1_router.post(
    path="/graphs/{graph_id}/execute/{graph_version}",
    tags=["graphs"],
)
async def execute_graph(
    graph_id: str,
    graph_version: int,
    node_input: Annotated[dict[str, Any], Body(..., embed=True, default_factory=dict)],
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.EXECUTE_GRAPH)
    ),
) -> dict[str, Any]:
    try:
        graph_exec = await add_graph_execution(
            graph_id=graph_id,
            user_id=auth.user_id,
            inputs=node_input,
            graph_version=graph_version,
        )
        return {"id": graph_exec.id}
    except Exception as e:
        msg = str(e).encode().decode("unicode_escape")
        raise HTTPException(status_code=400, detail=msg)


class ExecutionNode(TypedDict):
    node_id: str
    input: Any
    output: dict[str, Any]


class GraphExecutionResult(TypedDict):
    execution_id: str
    status: str
    nodes: list[ExecutionNode]
    output: Optional[list[dict[str, str]]]


@v1_router.get(
    path="/graphs/{graph_id}/executions/{graph_exec_id}/results",
    tags=["graphs"],
)
async def get_graph_execution_results(
    graph_id: str,
    graph_exec_id: str,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_GRAPH)
    ),
) -> GraphExecutionResult:
    graph_exec = await execution_db.get_graph_execution(
        user_id=auth.user_id,
        execution_id=graph_exec_id,
        include_node_executions=True,
    )
    if not graph_exec:
        raise HTTPException(
            status_code=404, detail=f"Graph execution #{graph_exec_id} not found."
        )

    if not await graph_db.get_graph(
        graph_id=graph_exec.graph_id,
        version=graph_exec.graph_version,
        user_id=auth.user_id,
    ):
        raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found.")

    return GraphExecutionResult(
        execution_id=graph_exec_id,
        status=graph_exec.status.value,
        nodes=[
            ExecutionNode(
                node_id=node_exec.node_id,
                input=node_exec.input_data.get("value", node_exec.input_data),
                output={k: v for k, v in node_exec.output_data.items()},
            )
            for node_exec in graph_exec.node_executions
        ],
        output=(
            [
                {name: value}
                for name, values in graph_exec.outputs.items()
                for value in values
            ]
            if graph_exec.status == AgentExecutionStatus.COMPLETED
            else None
        ),
    )


##############################################
############### Store Endpoints ##############
##############################################


@v1_router.get(
    path="/store/agents",
    tags=["store"],
    dependencies=[Security(require_permission(APIKeyPermission.READ_STORE))],
    response_model=store_model.StoreAgentsResponse,
)
async def get_store_agents(
    featured: bool = False,
    creator: str | None = None,
    sorted_by: Literal["rating", "runs", "name", "updated_at"] | None = None,
    search_query: str | None = None,
    category: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> store_model.StoreAgentsResponse:
    """
    Get a paginated list of agents from the store with optional filtering and sorting.

    Args:
        featured: Filter to only show featured agents
        creator: Filter agents by creator username
        sorted_by: Sort agents by "runs", "rating", "name", or "updated_at"
        search_query: Search agents by name, subheading and description
        category: Filter agents by category
        page: Page number for pagination (default 1)
        page_size: Number of agents per page (default 20)

    Returns:
        StoreAgentsResponse: Paginated list of agents matching the filters
    """
    if page < 1:
        raise HTTPException(status_code=422, detail="Page must be greater than 0")

    if page_size < 1:
        raise HTTPException(status_code=422, detail="Page size must be greater than 0")

    agents = await store_cache._get_cached_store_agents(
        featured=featured,
        creator=creator,
        sorted_by=sorted_by,
        search_query=search_query,
        category=category,
        page=page,
        page_size=page_size,
    )
    return agents


@v1_router.get(
    path="/store/agents/{username}/{agent_name}",
    tags=["store"],
    dependencies=[Security(require_permission(APIKeyPermission.READ_STORE))],
    response_model=store_model.StoreAgentDetails,
)
async def get_store_agent(
    username: str,
    agent_name: str,
) -> store_model.StoreAgentDetails:
    """
    Get details of a specific store agent by username and agent name.

    Args:
        username: Creator's username
        agent_name: Name/slug of the agent

    Returns:
        StoreAgentDetails: Detailed information about the agent
    """
    username = urllib.parse.unquote(username).lower()
    agent_name = urllib.parse.unquote(agent_name).lower()
    agent = await store_cache._get_cached_agent_details(
        username=username, agent_name=agent_name
    )
    return agent


@v1_router.get(
    path="/store/creators",
    tags=["store"],
    dependencies=[Security(require_permission(APIKeyPermission.READ_STORE))],
    response_model=store_model.CreatorsResponse,
)
async def get_store_creators(
    featured: bool = False,
    search_query: str | None = None,
    sorted_by: Literal["agent_rating", "agent_runs", "num_agents"] | None = None,
    page: int = 1,
    page_size: int = 20,
) -> store_model.CreatorsResponse:
    """
    Get a paginated list of store creators with optional filtering and sorting.

    Args:
        featured: Filter to only show featured creators
        search_query: Search creators by profile description
        sorted_by: Sort by "agent_rating", "agent_runs", or "num_agents"
        page: Page number for pagination (default 1)
        page_size: Number of creators per page (default 20)

    Returns:
        CreatorsResponse: Paginated list of creators matching the filters
    """
    if page < 1:
        raise HTTPException(status_code=422, detail="Page must be greater than 0")

    if page_size < 1:
        raise HTTPException(status_code=422, detail="Page size must be greater than 0")

    creators = await store_cache._get_cached_store_creators(
        featured=featured,
        search_query=search_query,
        sorted_by=sorted_by,
        page=page,
        page_size=page_size,
    )
    return creators


@v1_router.get(
    path="/store/creators/{username}",
    tags=["store"],
    dependencies=[Security(require_permission(APIKeyPermission.READ_STORE))],
    response_model=store_model.CreatorDetails,
)
async def get_store_creator(
    username: str,
) -> store_model.CreatorDetails:
    """
    Get details of a specific store creator by username.

    Args:
        username: Creator's username

    Returns:
        CreatorDetails: Detailed information about the creator
    """
    username = urllib.parse.unquote(username).lower()
    creator = await store_cache._get_cached_creator_details(username=username)
    return creator
