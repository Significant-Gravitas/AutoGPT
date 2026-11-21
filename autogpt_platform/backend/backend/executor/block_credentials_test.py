"""Both direct block-execute routes run a block on the caller's stored credentials.

These drive the routes against the real credential store: whose credential a run
may use is decided there, and a mocked store would decide it for the test.
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import fastapi
import httpx
import pytest
from autogpt_libs.auth.dependencies import get_request_context
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from autogpt_libs.auth.models import RequestContext
from prisma.enums import APIKeyPermission
from pydantic import SecretStr

from backend.api.external.middleware import require_auth
from backend.api.external.v1.routes import v1_router
from backend.api.features.blocks.routes import router as blocks_router
from backend.api.utils.exceptions import add_exception_handlers
from backend.blocks._base import AnyBlockSchema
from backend.blocks.anysearch.search import AnySearchBlock
from backend.blocks.code_executor import ExecuteCodeBlock
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.model import APIKeyCredentials
from backend.data.user import get_or_create_user
from backend.integrations.credentials_store import IntegrationCredentialsStore
from backend.util.test import SpinTestServer

ROUTES = ["internal", "external"]


@pytest.fixture
async def caller(server: SpinTestServer) -> str:
    return await _user()


@pytest.fixture
def spend(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Every run here costs something, so a run that gets as far as the charge shows."""
    spend_credits = AsyncMock(return_value=95)
    monkeypatch.setattr(
        "backend.executor.utils.block_usage_cost", lambda *_a, **_k: (5, {})
    )
    monkeypatch.setattr(
        "backend.executor.utils.get_user_credit_model",
        AsyncMock(return_value=MagicMock(spend_credits=spend_credits)),
    )
    return spend_credits


@pytest.fixture
def sandbox_keys(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The E2B keys ``ExecuteCodeBlock`` opened a sandbox with."""
    keys: list[str] = []

    async def _execute_code(self, *, api_key: str, **_kwargs):
        keys.append(api_key)
        return [], "ok", "", "", None, []

    monkeypatch.setattr(ExecuteCodeBlock, "execute_code", _execute_code)
    return keys


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("route", ROUTES)
async def test_a_block_that_needs_credentials_runs_on_the_callers_stored_credential(
    route: str, caller: str, spend: AsyncMock, sandbox_keys: list[str]
):
    credential = await _store_e2b_key(caller)

    response = await _execute(
        route, caller, ExecuteCodeBlock(), _code_input(credential)
    )

    assert response.status_code == 200, response.text
    assert response.json()["response"] == ["ok"]
    assert sandbox_keys == [credential.api_key.get_secret_value()]


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("credential", ["missing", "another user's"])
async def test_a_credential_the_caller_cannot_use_is_refused_before_the_charge(
    route: str,
    credential: str,
    caller: str,
    spend: AsyncMock,
    sandbox_keys: list[str],
):
    if credential == "missing":
        block_input = {"code": "print(1)"}
    else:
        block_input = _code_input(await _store_e2b_key(await _user()))

    response = await _execute(route, caller, ExecuteCodeBlock(), block_input)

    assert response.status_code == 400, response.text
    assert "'credentials'" in response.json()["detail"]
    spend.assert_not_awaited()
    assert sandbox_keys == []


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("route", ROUTES)
async def test_a_block_whose_credentials_are_optional_runs_without_any(
    route: str, caller: str, spend: AsyncMock, monkeypatch: pytest.MonkeyPatch
):
    async def _search(self, credentials, payload):
        return {"code": 0, "message": "success", "data": {"results": []}}

    monkeypatch.setattr(AnySearchBlock, "_search", _search)

    response = await _execute(
        route, caller, AnySearchBlock(), {"query": "q", "auth": "anonymous"}
    )

    assert response.status_code == 200, response.text
    assert response.json()["results"] == [[]]


@pytest.mark.asyncio(loop_scope="session")
async def test_an_api_key_that_cannot_build_a_graph_cannot_use_a_stored_credential(
    caller: str, spend: AsyncMock, sandbox_keys: list[str]
):
    credential = await _store_e2b_key(caller)

    response = await _execute(
        "external",
        caller,
        ExecuteCodeBlock(),
        _code_input(credential),
        scopes=(APIKeyPermission.EXECUTE_BLOCK, APIKeyPermission.EXECUTE_GRAPH),
    )

    assert response.status_code == 403, response.text
    assert "WRITE_GRAPH" in response.json()["detail"]
    spend.assert_not_awaited()
    assert sandbox_keys == []


async def _execute(
    route: str,
    user_id: str,
    block: AnyBlockSchema,
    block_input: dict,
    scopes: tuple[APIKeyPermission, ...] = tuple(APIKeyPermission),
) -> httpx.Response:
    app = fastapi.FastAPI()
    add_exception_handlers(app)
    if route == "internal":
        app.include_router(blocks_router)
        app.dependency_overrides[get_jwt_payload] = lambda: {
            "sub": user_id,
            "role": "user",
        }
        app.dependency_overrides[get_request_context] = lambda: _request_context(
            user_id
        )
    else:
        app.include_router(v1_router)
        app.dependency_overrides[require_auth] = lambda: APIAuthorizationInfo(
            user_id=user_id,
            scopes=list(scopes),
            type="api_key",
            created_at=datetime.now(UTC),
        )
    # Without raise_app_exceptions=False a 500 surfaces as the exception, not a response.
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
        return await client.post(f"/blocks/{block.id}/execute", json=block_input)


async def _user() -> str:
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"direct-execute-{uuid.uuid4().hex[:8]}@example.com",
            "name": "Direct Execute",
        }
    )
    return user.id


async def _store_e2b_key(user_id: str) -> APIKeyCredentials:
    credential = APIKeyCredentials(
        id=str(uuid.uuid4()),
        provider="e2b",
        api_key=SecretStr(f"e2b-{uuid.uuid4().hex}"),
        title="E2B",
    )
    await IntegrationCredentialsStore().add_creds(user_id, credential)
    return credential


def _code_input(credential: APIKeyCredentials) -> dict:
    return {
        "code": "print(1)",
        "credentials": {
            "id": credential.id,
            "provider": credential.provider,
            "type": credential.type,
            "title": credential.title,
        },
    }


def _request_context(user_id: str) -> RequestContext:
    return RequestContext(
        user_id=user_id,
        org_id="test-org",
        team_id=None,
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=False,
        is_team_admin=True,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )
