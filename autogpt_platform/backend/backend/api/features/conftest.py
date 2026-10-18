"""
Shared fixtures and helpers for the OAuth endpoint integration tests.

The OAuth coverage is split across focused modules (oauth_test.py for the
endpoint flows, oauth_concurrency_test.py for the refresh-token races), so the
fixtures live here to be shared instead of duplicated.

Note: the session-scoped user id is named oauth_test_user_id so it does not
shadow the function-scoped test_user_id fixture from the parent conftest.
"""

import asyncio
import base64
import hashlib
import secrets
import uuid
from typing import AsyncGenerator
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import pytest_asyncio
from autogpt_libs.api_key.keysmith import APIKeySmith
from prisma.enums import APIKeyPermission
from prisma.models import OAuthAccessToken as PrismaOAuthAccessToken
from prisma.models import OAuthApplication as PrismaOAuthApplication
from prisma.models import OAuthAuthorizationCode as PrismaOAuthAuthorizationCode
from prisma.models import OAuthRefreshToken as PrismaOAuthRefreshToken
from prisma.models import User as PrismaUser

from backend.api.rest_api import app

keysmith = APIKeySmith()


@pytest.fixture(scope="session")
def oauth_test_user_id() -> str:
    """Test user ID for OAuth tests."""
    return str(uuid.uuid4())


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def test_user(server, oauth_test_user_id: str):
    """Create a test user in the database."""
    await PrismaUser.prisma().create(
        data={
            "id": oauth_test_user_id,
            "email": f"oauth-test-{oauth_test_user_id}@example.com",
            "name": "OAuth Test User",
        }
    )

    yield oauth_test_user_id

    # Cleanup - delete in correct order due to foreign key constraints.
    # Wrap in try/except because the event loop or Prisma engine may already
    # be closed during session teardown on Python 3.12+.
    try:
        await asyncio.gather(
            PrismaOAuthAccessToken.prisma().delete_many(
                where={"userId": oauth_test_user_id}
            ),
            PrismaOAuthRefreshToken.prisma().delete_many(
                where={"userId": oauth_test_user_id}
            ),
            PrismaOAuthAuthorizationCode.prisma().delete_many(
                where={"userId": oauth_test_user_id}
            ),
        )
        await asyncio.gather(
            PrismaOAuthApplication.prisma().delete_many(
                where={"ownerId": oauth_test_user_id}
            ),
            PrismaUser.prisma().delete(where={"id": oauth_test_user_id}),
        )
    except RuntimeError:
        pass


@pytest_asyncio.fixture
async def test_oauth_app(test_user: str):
    """Create a test OAuth application in the database."""
    app_id = str(uuid.uuid4())
    client_id = f"test_client_{secrets.token_urlsafe(8)}"
    # Secret must start with "agpt_" prefix for keysmith verification to work
    client_secret_plaintext = f"agpt_secret_{secrets.token_urlsafe(16)}"
    client_secret_hash, client_secret_salt = keysmith.hash_key(client_secret_plaintext)

    await PrismaOAuthApplication.prisma().create(
        data={
            "id": app_id,
            "name": "Test OAuth App",
            "description": "Test application for integration tests",
            "clientId": client_id,
            "clientSecret": client_secret_hash,
            "clientSecretSalt": client_secret_salt,
            "redirectUris": [
                "https://example.com/callback",
                "http://localhost:3000/callback",
            ],
            "grantTypes": ["authorization_code", "refresh_token"],
            "scopes": [APIKeyPermission.EXECUTE_GRAPH, APIKeyPermission.READ_GRAPH],
            "ownerId": test_user,
            "isActive": True,
        }
    )

    yield {
        "id": app_id,
        "client_id": client_id,
        "client_secret": client_secret_plaintext,
        "redirect_uri": "https://example.com/callback",
    }

    # Cleanup is handled by test_user fixture (cascade delete)


def generate_pkce() -> tuple[str, str]:
    """Generate PKCE code verifier and challenge."""
    verifier = secrets.token_urlsafe(32)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .decode("ascii")
        .rstrip("=")
    )
    return verifier, challenge


async def authorize_and_exchange_code(
    client: httpx.AsyncClient, test_oauth_app: dict, state: str
) -> dict:
    """Run the authorize + authorization_code exchange and return the tokens."""
    verifier, challenge = generate_pkce()
    auth_response = await client.post(
        "/api/oauth/authorize",
        json={
            "client_id": test_oauth_app["client_id"],
            "redirect_uri": test_oauth_app["redirect_uri"],
            "scopes": ["EXECUTE_GRAPH"],
            "state": state,
            "response_type": "code",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        follow_redirects=False,
    )
    auth_code = parse_qs(urlparse(auth_response.json()["redirect_url"]).query)["code"][
        0
    ]
    token_response = await client.post(
        "/api/oauth/token",
        json={
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": test_oauth_app["redirect_uri"],
            "client_id": test_oauth_app["client_id"],
            "client_secret": test_oauth_app["client_secret"],
            "code_verifier": verifier,
        },
    )
    return token_response.json()


@pytest.fixture
def pkce_credentials() -> tuple[str, str]:
    """Generate PKCE code verifier and challenge as a fixture."""
    return generate_pkce()


@pytest_asyncio.fixture
async def client(server, test_user: str) -> AsyncGenerator[httpx.AsyncClient, None]:
    """
    Create an async HTTP client that talks directly to the FastAPI app.

    Uses ASGI transport so we don't need an actual HTTP server running.
    Also overrides get_user_id dependency to return our test user.

    Depends on `server` to ensure the DB is connected and `test_user` to ensure
    the user exists in the database before running tests.
    """
    from autogpt_libs.auth import get_user_id

    # Override get_user_id dependency to return our test user
    def override_get_user_id():
        return test_user

    # Store original override if any
    original_override = app.dependency_overrides.get(get_user_id)

    # Set our override
    app.dependency_overrides[get_user_id] = override_get_user_id

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as http_client:
            yield http_client
    finally:
        # Restore original override
        if original_override is not None:
            app.dependency_overrides[get_user_id] = original_override
        else:
            app.dependency_overrides.pop(get_user_id, None)
