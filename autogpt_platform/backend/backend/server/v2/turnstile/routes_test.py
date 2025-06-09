import fastapi
import fastapi.testclient
import pytest_mock

import backend.server.v2.turnstile.routes as turnstile_routes

app = fastapi.FastAPI()
app.include_router(turnstile_routes.router)

client = fastapi.testclient.TestClient(app)


def test_verify_turnstile_token_no_secret_key(mocker: pytest_mock.MockFixture) -> None:
    """Test token verification without secret key configured"""
    # Mock the settings with no secret key
    mock_settings = mocker.patch("backend.server.v2.turnstile.routes.settings")
    mock_settings.secrets.turnstile_secret_key = None

    request_data = {"token": "test_token", "action": "login"}
    response = client.post("/verify", json=request_data)

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is False
    assert response_data["error"] == "CONFIGURATION_ERROR"


def test_verify_turnstile_token_invalid_request() -> None:
    """Test token verification with invalid request data"""
    # Missing token
    response = client.post("/verify", json={"action": "login"})
    assert response.status_code == 422
