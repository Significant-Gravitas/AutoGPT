import pytest
import fastapi.testclient
import autogpt_server.server.routes.blocks
import fastapi

client = fastapi.testclient.TestClient(autogpt_server.server.routes.blocks.router)


@pytest.mark.asyncio
async def test_get_graph_blocks():
    response = client.get("/blocks")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_get_graph_block_costs():
    response = client.get("/blocks/costs")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


@pytest.mark.asyncio
async def test_execute_graph_block_success():
    block_id = "f3b1c1b2-4c4f-4f0d-8d2f-4c4f0d8d2f4c"  
    data = {"text": "hi"} 
    response = client.post(f"/blocks/{block_id}/execute", json=data)
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


@pytest.mark.asyncio
async def test_execute_graph_block_not_found():
    block_id = "invalid_block_id"
    data = {"input": "test data"}
    with pytest.raises(fastapi.HTTPException) as exc_info:
        client.post(f"/blocks/{block_id}/execute", json=data)
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == f"Block #{block_id} not found."
