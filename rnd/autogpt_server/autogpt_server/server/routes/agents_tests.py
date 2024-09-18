import autogpt_server.server
import autogpt_server.server.routes
import fastapi
import fastapi.testclient
import pytest
import unittest.mock

from autogpt_server.server.new_rest_app import app
import autogpt_server.server.utils as utils
import autogpt_server.executor as executor
import autogpt_server.data.graph


client = fastapi.testclient.TestClient(app)

async def override_get_user_id():
    return "test_user_id"

async def override_execution_manager_client():
    class MockExecutionManager:

        def add_execution(self, graph_id, node_input, user_id):
            return {"graph_exec_id": "exec1"}
        
        def cancel_execution(self, graph_exec_id):
            return {"graph_exec_id": "exec1"}
        
        def get_execution_results(self, graph_exec_id):
            return [{"node_id": "node1", "output": "result"}]
        
    
    return MockExecutionManager()

async def override_scheduler_client():
    return executor.ExecutionScheduler()

app.dependency_overrides[utils.get_user_id] = override_get_user_id
app.dependency_overrides[autogpt_server.server.routes.agents.execution_manager_client] = override_execution_manager_client
app.dependency_overrides[autogpt_server.server.routes.agents.execution_scheduler_client] = override_scheduler_client

@pytest.mark.asyncio
async def test_get_graphs():
    mock_get_graphs_meta = unittest.mock.AsyncMock()
    mock_get_graphs_meta.return_value = [
        autogpt_server.data.graph.GraphMeta(
            id="graph1",
            version=1,
            name="Test Graph 1",
            description="Test Graph 1 Description",
            is_active=True,
            is_template=False,
        ),
        autogpt_server.data.graph.GraphMeta(
            id="graph2",
            version=1,
            name="Test Graph 2",
            description="Test Graph 2 Description",
            is_active=True,
            is_template=False,
        ),
        autogpt_server.data.graph.GraphMeta(
            id="graph3",
            version=2,
            name="Test Graph 3",
            description="Test Graph 3 Description",
            is_active=False,
            is_template=True,
        )

    ]

    with unittest.mock.patch('autogpt_server.data.graph.get_graphs_meta', mock_get_graphs_meta):
        response = client.get("/api/v1/graphs")
        assert response.status_code == 200
        assert response.json() == [
            {
                "id": "graph1",
                "version": 1,
                "name": "Test Graph 1",
                "description": "Test Graph 1 Description",
                "is_active": True,
                "is_template": False,
            },
            {
                "id": "graph2",
                "version": 1,
                "name": "Test Graph 2",
                "description": "Test Graph 2 Description",
                "is_active": True,
                "is_template": False,
            },
            {
                "id": "graph3",
                "version": 2,
                "name": "Test Graph 3",
                "description": "Test Graph 3 Description",
                "is_active": False,
                "is_template": True,
            }
        ]

@pytest.mark.asyncio
async def test_create_new_graph():
    sample_graph = autogpt_server.data.graph.Graph(
        version=1,
        is_active=True,
        is_template=False,
        name="New Graph",
        description="New Graph Description",
        nodes=[],
        links=[]
    )

    mock_create_graph = unittest.mock.AsyncMock()
    mock_create_graph.return_value = sample_graph

    with unittest.mock.patch('autogpt_server.data.graph.create_graph', mock_create_graph):
        response = client.post("/api/v1/graphs", json={"graph": sample_graph.model_dump()})
        assert response.status_code == 200
        assert response.json() == sample_graph.model_dump()

@pytest.mark.asyncio
async def test_create_new_graph_invalid_request():
    # Test case for missing both graph and template_id
    response = client.post("/api/v1/graphs", json={})
    assert response.status_code == 400
    assert response.json() == {"detail": "Either graph or template_id must be provided."}

    # Test case for non-existent template
    mock_get_graph = unittest.mock.AsyncMock(return_value=None)
    with unittest.mock.patch('autogpt_server.data.graph.get_graph', mock_get_graph):
        response = client.post("/api/v1/graphs", json={"template_id": "non_existent_template"})
        assert response.status_code == 400
        assert response.json() == {"detail": "Template #non_existent_template not found"}

    # Test case for invalid graph structure
    invalid_graph = {
        "version": 1,
        "is_active": True,
        "is_template": False,
        "name": "Invalid Graph",
        "description": "Invalid Graph Description",
        "nodes": "not a list",  # This should be a list
        "links": []
    }
    response = client.post("/api/v1/graphs", json={"graph": invalid_graph})
    assert response.status_code == 422  # Unprocessable Entity
    assert "status" in response.json()
    assert "errors" in response.json()
    

@pytest.mark.asyncio
async def test_get_graph_details():
    sample_graph = autogpt_server.data.graph.Graph(
        id="graph1",
        version=1,
        is_active=True,
        is_template=False,
        name="Test Graph",
        description="Test Graph Description",
        nodes=[],
        links=[]
    )

    mock_get_graph = unittest.mock.AsyncMock(return_value=sample_graph)

    with unittest.mock.patch('autogpt_server.data.graph.get_graph', mock_get_graph):
        response = client.get("/api/v1/graphs/graph1")
        assert response.status_code == 200
        assert response.json() == sample_graph.model_dump()

    # Test case for non-existent graph
    mock_get_graph.return_value = None
    with unittest.mock.patch('autogpt_server.data.graph.get_graph', mock_get_graph):
        response = client.get("/api/v1/graphs/non_existent_graph")
        assert response.status_code == 404
        assert response.json() == {"detail": "Graph #non_existent_graph not found."}

@pytest.mark.asyncio
async def test_update_graph():
    sample_graph = autogpt_server.data.graph.Graph(
        id="graph1",
        version=1,
        is_active=True,
        is_template=False,
        name="Test Graph",
        description="Test Graph Description",
        nodes=[],
        links=[]
    )

    mock_get_graph_all_versions = unittest.mock.AsyncMock(return_value=[sample_graph])
    mock_create_graph = unittest.mock.AsyncMock(return_value=sample_graph)
    mock_set_graph_active_version = unittest.mock.AsyncMock()

    with unittest.mock.patch('autogpt_server.data.graph.get_graph_all_versions', mock_get_graph_all_versions), \
         unittest.mock.patch('autogpt_server.data.graph.create_graph', mock_create_graph), \
         unittest.mock.patch('autogpt_server.data.graph.set_graph_active_version', mock_set_graph_active_version):
        
        updated_graph = sample_graph.model_copy(update={"name": "Updated Graph", "version": 2})
        response = client.put("/api/v1/graphs/graph1", json=updated_graph.model_dump())
        
        assert response.status_code == 200
        assert response.json()['description'] == updated_graph.model_dump()['description']
    

    non_existent_graph = autogpt_server.data.graph.Graph(
        id="non_existent_graph",
        version=1,
        is_active=True,
        is_template=False,
        name="Test Graph",
        description="Test Graph Description",
        nodes=[],
        links=[]
    )

    # Test case for non-existent graph
    mock_get_graph_all_versions.return_value = []
    with unittest.mock.patch('autogpt_server.data.graph.get_graph_all_versions', mock_get_graph_all_versions):
        response = client.put("/api/v1/graphs/non_existent_graph", json=non_existent_graph.model_dump())
        assert response.status_code == 404
        assert response.json() == {"detail": "Graph #non_existent_graph not found"}

    # Test case for mismatched graph ID
    response = client.put("/api/v1/graphs/graph2", json=updated_graph.model_dump())
    assert response.status_code == 400
    assert response.json() == {"detail": "Graph ID does not match ID in URI"}

@pytest.mark.asyncio
async def test_execute_graph():
    mock_add_execution = unittest.mock.AsyncMock()
    mock_add_execution.return_value = {"graph_exec_id": "exec1"}

    response = client.post("/api/v1/graphs/graph1/execute", json={"node_input": {"text": "hi"}})
    assert response.status_code == 200
    assert response.json() == {"id": "exec1"}
