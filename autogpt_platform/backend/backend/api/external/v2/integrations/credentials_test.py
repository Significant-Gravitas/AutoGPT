"""Creating a user/password login through the v2 API checks its scope."""

from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from prisma.enums import APIKeyPermission

from ..errors import add_v2_exception_handlers
from ..tenancy import TenantContext, require_auth
from . import credentials as credentials_mod
from .credentials import credentials_router

app = fastapi.FastAPI()
app.include_router(credentials_router, prefix="/integrations")
add_v2_exception_handlers(app)
client = fastapi.testclient.TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _auth(mocker: pytest_mock.MockFixture):
    async def fake_auth() -> TenantContext:
        return TenantContext(
            user_id="user-1",
            scopes=list(APIKeyPermission),
            type="api_key",
            organization_id="org-1",
        )

    app.dependency_overrides[require_auth] = fake_auth
    mocker.patch(
        "backend.api.external.v2.tenancy.enforce_payment_paywall",
        new_callable=AsyncMock,
    )
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def create(mocker: pytest_mock.MockFixture) -> AsyncMock:
    return mocker.patch.object(
        credentials_mod.creds_manager, "create", new_callable=AsyncMock
    )


def _login(provider: str, scope: str | None) -> dict:
    body = {
        "type": "user_password",
        "provider": provider,
        "title": "t",
        "username": "u",
        "password": "p",
    }
    if scope is not None:
        body["scope"] = scope
    return body


@pytest.mark.parametrize(
    "provider, scope",
    [
        ("smtp", None),
        ("smtp", ""),
        ("smtp", "smtp.example.com"),
        ("smtp", "smtp://*.example.com"),
        ("smtp", "postgresql://smtp.example.com"),
        ("database", None),
        ("database", "db.example.com:5432"),
        ("database", "smtp://db.example.com"),
    ],
)
def test_login_without_a_valid_scope_is_rejected(
    create: AsyncMock, provider: str, scope: str | None
):
    response = client.post("/integrations/credentials", json=_login(provider, scope))

    assert response.status_code == 422
    create.assert_not_awaited()


@pytest.mark.parametrize(
    "provider, scope",
    [
        ("smtp", "smtp://smtp.example.com:587"),
        ("database", "postgresql://db.example.com:5432"),
        ("database", "mysql://db.example.com"),
    ],
)
def test_login_with_a_scope_stores_it(create: AsyncMock, provider: str, scope: str):
    response = client.post("/integrations/credentials", json=_login(provider, scope))

    assert response.status_code == 201
    stored = create.await_args.args[1]
    assert stored.provider == provider
    assert stored.scope == scope
