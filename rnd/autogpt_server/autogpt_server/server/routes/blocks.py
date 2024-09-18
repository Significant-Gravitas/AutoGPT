import collections

import fastapi
import autogpt_server.data.block
import autogpt_server.data.credit

router = fastapi.APIRouter()


@router.get("/blocks")
async def get_graph_blocks():
    return [v.to_dict() for v in autogpt_server.data.block.get_blocks().values()]


@router.get("/blocks/costs")
async def get_graph_block_costs():
    return autogpt_server.data.credit.get_block_costs()


@router.post("/blocks/{block_id}/execute")
async def execute_graph_block(
    block_id: str, data: autogpt_server.data.block.BlockInput
) -> autogpt_server.data.block.CompletedBlockOutput:
    obj = autogpt_server.data.block.get_block(block_id)
    if not obj:
        raise fastapi.HTTPException(
            status_code=404, detail=f"Block #{block_id} not found."
        )

    output = collections.defaultdict(list)
    for name, data in obj.execute(data):
        output[name].append(data)
    return output
