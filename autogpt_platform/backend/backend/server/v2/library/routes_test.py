import autogpt_libs.auth.depends
import autogpt_libs.auth.middleware
import fastapi
import fastapi.testclient
import pytest
import pytest_mock

import backend.server.v2.library.db
import backend.server.v2.library.model
import backend.server.v2.library.routes

app = fastapi.FastAPI()
app.include_router(backend.server.v2.library.routes.router)

client = fastapi.testclient.TestClient(app)


def override_auth_middleware():
    """Override auth middleware for testing"""
    return {"sub": "test-user-id"}


def override_get_user_id():
    """Override get_user_id for testing"""
    return "test-user-id"


app.dependency_overrides[autogpt_libs.auth.middleware.auth_middleware] = (
    override_auth_middleware
)
app.dependency_overrides[autogpt_libs.auth.depends.get_user_id] = override_get_user_id


def test_get_library_agents_success(mocker: pytest_mock.MockFixture):
    mocked_value = [
        backend.server.v2.library.model.LibraryAgent(
            id="test-agent-1",
            version=1,
            is_active=True,
            name="Test Agent 1",
            description="Test Description 1",
            isCreatedByUser=True,
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
        ),
        backend.server.v2.library.model.LibraryAgent(
            id="test-agent-2",
            version=1,
            is_active=True,
            name="Test Agent 2",
            description="Test Description 2",
            isCreatedByUser=False,
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
        ),
    ]
    mock_db_call = mocker.patch("backend.server.v2.library.db.get_library_agents")
    mock_db_call.return_value = mocked_value

    response = client.get("/agents")
    assert response.status_code == 200

    data = [
        backend.server.v2.library.model.LibraryAgent.model_validate(agent)
        for agent in response.json()
    ]
    assert len(data) == 2
    assert data[0].id == "test-agent-1"
    assert data[0].isCreatedByUser is True
    assert data[1].id == "test-agent-2"
    assert data[1].isCreatedByUser is False
    mock_db_call.assert_called_once_with("test-user-id")


def test_get_library_agents_error(mocker: pytest_mock.MockFixture):
    mock_db_call = mocker.patch("backend.server.v2.library.db.get_library_agents")
    mock_db_call.side_effect = Exception("Test error")

    response = client.get("/agents")
    assert response.status_code == 500
    mock_db_call.assert_called_once_with("test-user-id")


@pytest.mark.skip(reason="Mocker Not implemented")
def test_add_agent_to_library_success(mocker: pytest_mock.MockFixture):
    mock_db_call = mocker.patch("backend.server.v2.library.db.add_agent_to_library")
    mock_db_call.return_value = None

    response = client.post("/agents/test-version-id")
    assert response.status_code == 201
    mock_db_call.assert_called_once_with(
        store_listing_version_id="test-version-id", user_id="test-user-id"
    )


@pytest.mark.skip(reason="Mocker Not implemented")
def test_add_agent_to_library_error(mocker: pytest_mock.MockFixture):
    mock_db_call = mocker.patch("backend.server.v2.library.db.add_agent_to_library")
    mock_db_call.side_effect = Exception("Test error")

    response = client.post("/agents/test-version-id")
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to add agent to library"
    mock_db_call.assert_called_once_with(
        store_listing_version_id="test-version-id", user_id="test-user-id"
    )
