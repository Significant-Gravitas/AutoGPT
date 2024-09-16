from fastapi import APIRouter
from autogpt_server.server.rest_api import AgentServer
from typing import Any

router = APIRouter()

@router.get("/blocks")
async def get_graph_blocks():
    return AgentServer.get_graph_blocks()

@router.get("/blocks/costs")
async def get_graph_block_costs():
    return AgentServer.get_graph_block_costs()

@router.post("/blocks/{block_id}/execute")
async def execute_graph_block(block_id: str, data: dict[str, Any]):
    return AgentServer.execute_graph_block(block_id, data)
