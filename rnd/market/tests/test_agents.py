from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from market.app import app
from market.db import AgentQueryError


@pytest.fixture
def test_client():
    return TestClient(app)


# Mock data
mock_agents = [
    {
        "id": "1",
        "name": "Agent 1",
        "description": "Description 1",
        "author": "Author 1",
        "keywords": ["AI", "chatbot"],
        "categories": ["general"],
        "version": 1,
        "createdAt": datetime.now(timezone.utc),
        "updatedAt": datetime.now(timezone.utc),
        "graph": {"node1": "value1"},
    },
    {
        "id": "2",
        "name": "Agent 2",
        "description": "Description 2",
        "author": "Author 2",
        "keywords": ["ML", "NLP"],
        "categories": ["specialized"],
        "version": 1,
        "createdAt": datetime.now(timezone.utc),
        "updatedAt": datetime.now(timezone.utc),
        "graph": {"node2": "value2"},
    },
]


# TODO: Need to mock prisma somehow


@pytest.mark.asyncio
async def test_list_agents(test_client):
    response = test_client.get("/agents")
    assert response.status_code == 200
    data = response.json()
    assert len(data["agents"]) == 2
    assert data["total_count"] == 2


@pytest.mark.asyncio
async def test_list_agents_with_filters(test_client):
    response = await test_client.get("/agents?name=Agent 1&keyword=AI&category=general")
    assert response.status_code == 200
    data = response.json()
    assert len(data["agents"]) == 1
    assert data["agents"][0]["name"] == "Agent 1"


@pytest.mark.asyncio
async def test_get_agent_details(test_client, mock_get_agent_details):
    response = await test_client.get("/agents/1")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "1"
    assert data["name"] == "Agent 1"
    assert "graph" in data


@pytest.mark.asyncio
async def test_get_nonexistent_agent(test_client, mock_get_agent_details):
    mock_get_agent_details.side_effect = AgentQueryError("Agent not found")
    response = await test_client.get("/agents/999")
    assert response.status_code == 404
