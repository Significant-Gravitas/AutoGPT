from types import SimpleNamespace
from unittest.mock import AsyncMock

import fastapi
from autogpt_libs.auth.dependencies import get_request_context
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from fastapi.testclient import TestClient

from backend.api.features.experts import avatar_routes


def test_generation_requires_authentication():
    app = fastapi.FastAPI()
    app.include_router(avatar_routes.router)
    response = TestClient(app).post("/avatars/generations", json={})
    assert response.status_code in (401, 403)


def test_start_generation_returns_job_and_uses_authenticated_owner(
    monkeypatch, mock_jwt_user
):
    app = fastapi.FastAPI()
    app.include_router(avatar_routes.router)
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    app.dependency_overrides[get_request_context] = mock_jwt_user["get_request_context"]
    monkeypatch.setattr(
        avatar_routes,
        "Settings",
        lambda: SimpleNamespace(secrets=SimpleNamespace(openai_api_key="test-key")),
    )
    reserve, save, run = AsyncMock(), AsyncMock(), AsyncMock()
    monkeypatch.setattr(avatar_routes.avatar_jobs, "reserve_generation", reserve)
    monkeypatch.setattr(avatar_routes.avatar_jobs, "save_job", save)
    monkeypatch.setattr(avatar_routes.avatar_jobs, "run_generation", run)
    client = TestClient(app)
    response = client.post("/avatars/generations", json={"category": "finance"})
    assert response.status_code == 202
    assert response.json()["status"] == "pending"
    assert response.json()["avatar_url"] is None
    owner = reserve.call_args.args[0]
    assert owner
    assert save.call_args.args[0] == owner
    assert run.call_args.args[0] == owner
    assert str(run.call_args.args[1].id) == response.json()["id"]
    invalid = client.post(
        "/avatars/generations", json={"category": "otto", "user_id": "another-user"}
    )
    assert invalid.status_code == 422
    assert reserve.await_count == 1
