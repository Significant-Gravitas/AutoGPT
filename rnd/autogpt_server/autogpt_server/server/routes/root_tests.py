import fastapi
import fastapi.testclient
import pytest

from autogpt_server.server.new_rest_app import app
import autogpt_server.server.utils as utils
import autogpt_libs.auth.middleware as auth_middleware
import unittest.mock

client = fastapi.testclient.TestClient(app)


async def override_get_user_id():
    return "test_user_id"


async def override_user_data():
    return {"id": "test_user_id", "name": "Test User"}


app.dependency_overrides[utils.get_user_id] = override_get_user_id
app.dependency_overrides[auth_middleware.auth_middleware] = override_user_data


@pytest.mark.asyncio
async def test_root():
    response = client.get("/api/v1/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Autogpt Server API"}

@pytest.mark.asyncio
async def test_get_or_create_user_route():

    # Create a mock for the get_or_create_user function
    mock_get_or_create_user = unittest.mock.AsyncMock()
    mock_get_or_create_user.return_value = unittest.mock.Mock(
        model_dump=lambda: {"id": "test_user_id", "name": "Test User"}
    )

    # Apply the mock using patch
    with unittest.mock.patch('autogpt_server.data.user.get_or_create_user', mock_get_or_create_user):
        response = client.post("/api/v1/auth/user")
        assert response.status_code == 200
        assert response.json() == {"id": "test_user_id", "name": "Test User"}

@pytest.mark.asyncio
async def test_get_user_credits():
    response = client.get("/api/v1/credits")
    assert response.status_code == 200
    assert response.json() == {"credits": 0}