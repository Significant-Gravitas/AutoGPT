from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import fastapi.testclient
import pytest
from prisma.enums import APIKeyPermission

import backend.api.external.v1.tools as tools_mod
from backend.api.external.fastapi_app import external_api
from backend.api.external.middleware import require_auth
from backend.api.rest_api import app as rest_app
from backend.data.auth.base import APIAuthorizationInfo

client = fastapi.testclient.TestClient(external_api)


@pytest.fixture(autouse=True)
def setup_auth(test_user_id):
    async def fake_require_auth() -> APIAuthorizationInfo:
        return APIAuthorizationInfo(
            user_id=test_user_id,
            scopes=list(APIKeyPermission),
            type="api_key",
            created_at=datetime.now(timezone.utc),
        )

    external_api.dependency_overrides[require_auth] = fake_require_auth
    yield
    external_api.dependency_overrides.clear()


def test_json_body_without_content_type_is_accepted(monkeypatch: pytest.MonkeyPatch):
    execute = AsyncMock(return_value=MagicMock(model_dump=lambda: {"ok": True}))
    monkeypatch.setattr(tools_mod.find_agent_tool, "_execute", execute)
    monkeypatch.setattr(tools_mod, "_create_ephemeral_session", MagicMock())

    response = client.post("/v1/tools/find-agent", content=b'{"query": "weather"}')

    assert "content-type" not in response.request.headers
    assert response.status_code == 200
    assert execute.await_args.kwargs["query"] == "weather"


def test_rest_api_accepts_json_without_content_type():
    assert rest_app.router.strict_content_type is False
