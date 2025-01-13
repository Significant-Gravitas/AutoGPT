import asyncio
import logging
from collections import defaultdict
from typing import Any, Annotated

from fastapi import APIRouter, Depends, HTTPException

import backend.data.block
from autogpt_libs.utils.cache import thread_cached
from backend.data.block import BlockInput, CompletedBlockOutput
from backend.executor import ExecutionManager, ExecutionScheduler
from backend.util.service import get_service_client
from backend.util.settings import Settings
from backend.data.api_key import APIKey
from prisma.enums import APIKeyPermission
from backend.server.external.middleware import require_permission

@thread_cached
def execution_manager_client() -> ExecutionManager:
    return get_service_client(ExecutionManager)

@thread_cached
def execution_scheduler_client() -> ExecutionScheduler:
    return get_service_client(ExecutionScheduler)

settings = Settings()
logger = logging.getLogger(__name__)

v1_router = APIRouter()

@v1_router.post(
    path="/blocks/{block_id}/execute",
    tags=["blocks"],
    dependencies=[Depends(require_permission(APIKeyPermission.EXECUTE_BLOCK))],
)
def execute_graph_block(
    block_id: str,
    data: BlockInput,
    api_key: APIKey = Depends(require_permission(APIKeyPermission.EXECUTE_BLOCK))
) -> CompletedBlockOutput:
    obj = backend.data.block.get_block(block_id)
    if not obj:
        raise HTTPException(status_code=404, detail=f"Block #{block_id} not found.")

    output = defaultdict(list)
    for name, data in obj.execute(data):
        output[name].append(data)
    return output

@v1_router.post(
    path="/graphs/{graph_id}/execute",
    tags=["graphs"],
)
def execute_graph(
    graph_id: str,
    graph_version: int,
    node_input: dict[Any, Any],
    api_key: APIKey = Depends(require_permission(APIKeyPermission.EXECUTE_GRAPH))
) -> dict[str, Any]:
    try:
        graph_exec = execution_manager_client().add_execution(
            graph_id,
            node_input,
            user_id=api_key.user_id,
            graph_version=graph_version
        )
        return {"id": graph_exec.graph_exec_id}
    except Exception as e:
        msg = e.__str__().encode().decode("unicode_escape")
        raise HTTPException(status_code=400, detail=msg)