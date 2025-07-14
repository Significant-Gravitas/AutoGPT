import logging
from collections import defaultdict
from typing import Annotated, Any, Optional, Sequence

from fastapi import APIRouter, Body, Depends, HTTPException
from prisma.enums import AgentExecutionStatus, APIKeyPermission
from typing_extensions import TypedDict

import backend.data.block
from backend.data import execution as execution_db
from backend.data import graph as graph_db
from backend.data.api_key import APIKey
from backend.data.block import BlockInput, CompletedBlockOutput
from backend.executor.utils import add_graph_execution
from backend.server.external.middleware import require_permission
from backend.util.settings import Settings

settings = Settings()
logger = logging.getLogger(__name__)

v1_router = APIRouter()


class NodeOutput(TypedDict):
    key: str
    value: Any


class ExecutionNode(TypedDict):
    node_id: str
    input: Any
    output: dict[str, Any]


class ExecutionNodeOutput(TypedDict):
    node_id: str
    outputs: list[NodeOutput]


class GraphExecutionResult(TypedDict):
    execution_id: str
    status: str
    nodes: list[ExecutionNode]
    output: Optional[list[dict[str, str]]]


@v1_router.get(
    path="/blocks",
    tags=["blocks"],
    dependencies=[Depends(require_permission(APIKeyPermission.READ_BLOCK))],
)
def get_graph_blocks() -> Sequence[dict[Any, Any]]:
    blocks = [block() for block in backend.data.block.get_blocks().values()]
    return [b.to_dict() for b in blocks if not b.disabled]


@v1_router.post(
    path="/blocks/{block_id}/execute",
    tags=["blocks"],
    dependencies=[Depends(require_permission(APIKeyPermission.EXECUTE_BLOCK))],
)
async def execute_graph_block(
    block_id: str,
    data: BlockInput,
    api_key: APIKey = Depends(require_permission(APIKeyPermission.EXECUTE_BLOCK)),
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
    api_key: APIKey = Depends(require_permission(APIKeyPermission.EXECUTE_GRAPH)),
) -> dict[str, Any]:
    try:
        graph_exec = await add_graph_execution(
            graph_id=graph_id,
            user_id=api_key.user_id,
            inputs=node_input,
            graph_version=graph_version,
        )
        return {"id": graph_exec.id}
    except Exception as e:
        msg = str(e).encode().decode("unicode_escape")
        raise HTTPException(status_code=400, detail=msg)


@v1_router.get(
    path="/graphs/{graph_id}/executions/{graph_exec_id}/results",
    tags=["graphs"],
)
async def get_graph_execution_results(
    graph_id: str,
    graph_exec_id: str,
    api_key: APIKey = Depends(require_permission(APIKeyPermission.READ_GRAPH)),
) -> GraphExecutionResult:
    graph = await graph_db.get_graph(graph_id, user_id=api_key.user_id)
    if not graph:
        raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found.")

    graph_exec = await execution_db.get_graph_execution(
        user_id=api_key.user_id,
        execution_id=graph_exec_id,
        include_node_executions=True,
    )
    if not graph_exec:
        raise HTTPException(
            status_code=404, detail=f"Graph execution #{graph_exec_id} not found."
        )

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
