"""
End-to-end integration tests for OAuth 2.0 Provider Endpoints.

These tests hit the actual API endpoints and database, testing the complete
OAuth flow from endpoint to database.

Tests cover:
1. Authorization endpoint - creating authorization codes
2. Token endpoint - exchanging codes for tokens and refreshing
3. Token introspection endpoint - checking token validity
4. Token revocation endpoint - revoking tokens
5. Complete OAuth flow end-to-end
"""

import base64
import hashlib
import secrets
import uuid
from typing import AsyncGenerator

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


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def test_user_id() -> str:
    """Test user ID for OAuth tests."""
    return str(uuid.uuid4())


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def test_user(server, test_user_id: str):
    """Create a test user in the database."""
    await PrismaUser.prisma().create(
        data={
            "id": test_user_id,
            "email": f"oauth-test-{test_user_id}@example.com",
            "name": "OAuth Test User",
        }
    )

    yield test_user_id

    # Cleanup - delete in correct order due to foreign key constraints
    await PrismaOAuthAccessToken.prisma().delete_many(where={"userId": test_user_id})
    await PrismaOAuthRefreshToken.prisma().delete_many(where={"userId": test_user_id})
    await PrismaOAuthAuthorizationCode.prisma().delete_many(
        where={"userId": test_user_id}
    )
    await PrismaOAuthApplication.prisma().delete_many(where={"ownerId": test_user_id})
    await PrismaUser.prisma().delete(where={"id": test_user_id})


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


# ============================================================================
# Authorization Endpoint Integration Tests
# ============================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_authorize_creates_code_in_database(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
    pkce_credentials: tuple[str, str],
):
    """Test that authorization endpoint creates a code in the database."""
    verifier, challenge = pkce_credentials

    response = await client.post(
        "/api/oauth/authorize",
        json={
            "client_id": test_oauth_app["client_id"],
            "redirect_uri": test_oauth_app["redirect_uri"],
            "scopes": ["EXECUTE_GRAPH", "READ_GRAPH"],
            "state": "test_state_123",
            "response_type": "code",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        follow_redirects=False,
    )

    assert response.status_code == 200
    redirect_url = response.json()["redirect_url"]

    # Parse the redirect URL to get the authorization code
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(redirect_url)
    query_params = parse_qs(parsed.query)

    assert "code" in query_params, f"Expected 'code' in query params: {query_params}"
    auth_code = query_params["code"][0]
    assert query_params["state"][0] == "test_state_123"

    # Verify code exists in database
    db_code = await PrismaOAuthAuthorizationCode.prisma().find_unique(
        where={"code": auth_code}
    )

    assert db_code is not None
    assert db_code.userId == test_user
    assert db_code.applicationId == test_oauth_app["id"]
    assert db_code.redirectUri == test_oauth_app["redirect_uri"]
    assert APIKeyPermission.EXECUTE_GRAPH in db_code.scopes
    assert APIKeyPermission.READ_GRAPH in db_code.scopes
    assert db_code.usedAt is None  # Not yet consumed
    assert db_code.codeChallenge == challenge
    assert db_code.codeChallengeMethod == "S256"


@pytest.mark.asyncio(loop_scope="session")
async def test_authorize_with_pkce_stores_challenge(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
    pkce_credentials: tuple[str, str],
):
    """Test that PKCE code challenge is stored correctly."""
    verifier, challenge = pkce_credentials

    response = await client.post(
        "/api/oauth/authorize",
        json={
            "client_id": test_oauth_app["client_id"],
            "redirect_uri": test_oauth_app["redirect_uri"],
            "scopes": ["EXECUTE_GRAPH"],
            "state": "pkce_test_state",
            "response_type": "code",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        follow_redirects=False,
    )

    assert response.status_code == 200

    from urllib.parse import parse_qs, urlparse

    auth_code = parse_qs(urlparse(response.json()["redirect_url"]).query)["code"][0]

    # Verify PKCE challenge is stored
    db_code = await PrismaOAuthAuthorizationCode.prisma().find_unique(
        where={"code": auth_code}
    )

    assert db_code is not None
    assert db_code.codeChallenge == challenge
    assert db_code.codeChallengeMethod == "S256"


@pytest.mark.asyncio(loop_scope="session")
async def test_authorize_invalid_client_returns_error(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
):
    """Test that invalid client_id returns error in redirect."""
    _, challenge = generate_pkce()

    response = await client.post(
        "/api/oauth/authorize",
        json={
            "client_id": "nonexistent_client_id",
            "redirect_uri": test_oauth_app["redirect_uri"],
            "scopes": ["EXECUTE_GRAPH"],
            "state": "error_test",
            "response_type": "code",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        follow_redirects=False,
    )

    assert response.status_code == 200
    from urllib.parse import parse_qs, urlparse

    query_params = parse_qs(urlparse(response.json()["redirect_url"]).query)
    assert query_params["error"][0] == "invalid_client"


@pytest_asyncio.fixture
async def inactive_oauth_app(test_user: str):
    """Create an inactive test OAuth application in the database."""
    app_id = str(uuid.uuid4())
    client_id = f"inactive_client_{secrets.token_urlsafe(8)}"
    client_secret_plaintext = f"agpt_secret_{secrets.token_urlsafe(16)}"
    client_secret_hash, client_secret_salt = keysmith.hash_key(client_secret_plaintext)

    await PrismaOAuthApplication.prisma().create(
        data={
            "id": app_id,
            "name": "Inactive OAuth App",
            "description": "Inactive test application",
            "clientId": client_id,
            "clientSecret": client_secret_hash,
            "clientSecretSalt": client_secret_salt,
            "redirectUris": ["https://example.com/callback"],
            "grantTypes": ["authorization_code", "refresh_token"],
            "scopes": [APIKeyPermission.EXECUTE_GRAPH],
            "ownerId": test_user,
            "isActive": False,  # Inactive!
        }
    )

    yield {
        "id": app_id,
        "client_id": client_id,
        "client_secret": client_secret_plaintext,
        "redirect_uri": "https://example.com/callback",
    }


@pytest.mark.asyncio(loop_scope="session")
async def test_authorize_inactive_app(
    client: httpx.AsyncClient,
    test_user: str,
    inactive_oauth_app: dict,
):
    """Test that authorization with inactive app returns error."""
    _, challenge = generate_pkce()

    response = await client.post(
        "/api/oauth/authorize",
        json={
            "client_id": inactive_oauth_app["client_id"],
            "redirect_uri": inactive_oauth_app["redirect_uri"],
            "scopes": ["EXECUTE_GRAPH"],
            "state": "inactive_test",
            "response_type": "code",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        follow_redirects=False,
    )

    assert response.status_code == 200
    from urllib.parse import parse_qs, urlparse

    query_params = parse_qs(urlparse(response.json()["redirect_url"]).query)
    assert query_params["error"][0] == "invalid_client"


@pytest.mark.asyncio(loop_scope="session")
async def test_authorize_invalid_redirect_uri(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
):
    """Test authorization with unregistered redirect_uri returns HTTP error."""
    _, challenge = generate_pkce()

    response = await client.post(
        "/api/oauth/authorize",
        json={
            "client_id": test_oauth_app["client_id"],
            "redirect_uri": "https://malicious.com/callback",
            "scopes": ["EXECUTE_GRAPH"],
            "state": "invalid_redirect_test",
            "response_type": "code",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        follow_redirects=False,
    )

    # Invalid redirect_uri should return HTTP 400, not a redirect
    assert response.status_code == 400
    assert "redirect_uri" in response.json()["detail"].lower()


@pytest.mark.asyncio(loop_scope="session")
async def test_authorize_invalid_scope(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
):
    """Test authorization with invalid scope value."""
    _, challenge = generate_pkce()

    response = await client.post(
        "/api/oauth/authorize",
        json={
            "client_id": test_oauth_app["client_id"],
            "redirect_uri": test_oauth_app["redirect_uri"],
            "scopes": ["INVALID_SCOPE_NAME"],
            "state": "invalid_scope_test",
            "response_type": "code",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        follow_redirects=False,
    )

    assert response.status_code == 200
    from urllib.parse import parse_qs, urlparse

    query_params = parse_qs(urlparse(response.json()["redirect_url"]).query)
    assert query_params["error"][0] == "invalid_scope"


@pytest.mark.asyncio(loop_scope="session")
async def test_authorize_unauthorized_scope(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
):
    """Test authorization requesting scope not authorized for app."""
    _, challenge = generate_pkce()

    # The test_oauth_app only has EXECUTE_GRAPH and READ_GRAPH scopes
    # DELETE_GRAPH is not in the app's allowed scopes
    response = await client.post(
        "/api/oauth/authorize",
        json={
            "client_id": test_oauth_app["client_id"],
            "redirect_uri": test_oauth_app["redirect_uri"],
            "scopes": ["DELETE_GRAPH"],  # Not authorized for this app
            "state": "unauthorized_scope_test",
            "response_type": "code",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        follow_redirects=False,
    )

    assert response.status_code == 200
    from urllib.parse import parse_qs, urlparse

    query_params = parse_qs(urlparse(response.json()["redirect_url"]).query)
    assert query_params["error"][0] == "invalid_scope"


@pytest.mark.asyncio(loop_scope="session")
async def test_authorize_unsupported_response_type(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
):
    """Test authorization with unsupported response_type."""
    _, challenge = generate_pkce()

    response = await client.post(
        "/api/oauth/authorize",
        json={
            "client_id": test_oauth_app["client_id"],
            "redirect_uri": test_oauth_app["redirect_uri"],
            "scopes": ["EXECUTE_GRAPH"],
            "state": "unsupported_response_test",
            "response_type": "token",  # Implicit flow not supported
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        follow_redirects=False,
    )

    assert response.status_code == 200
    from urllib.parse import parse_qs, urlparse

    query_params = parse_qs(urlparse(response.json()["redirect_url"]).query)
    assert query_params["error"][0] == "unsupported_response_type"


# ============================================================================
# Token Endpoint Integration Tests - Authorization Code Grant
# ============================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_token_exchange_creates_tokens_in_database(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
):
    """Test that token exchange creates access and refresh tokens in database."""
    from urllib.parse import parse_qs, urlparse

    verifier, challenge = generate_pkce()

    # First get an authorization code
    auth_response = await client.post(
        "/api/oauth/authorize",
        json={
            "client_id": test_oauth_app["client_id"],
            "redirect_uri": test_oauth_app["redirect_uri"],
            "scopes": ["EXECUTE_GRAPH", "READ_GRAPH"],
            "state": "token_test_state",
            "response_type": "code",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        follow_redirects=False,
    )

    auth_code = parse_qs(urlparse(auth_response.json()["redirect_url"]).query)["code"][
        0
    ]

    # Exchange code for tokens
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

    assert token_response.status_code == 200
    tokens = token_response.json()

    assert "access_token" in tokens
    assert "refresh_token" in tokens
    assert tokens["token_type"] == "Bearer"
    assert "EXECUTE_GRAPH" in tokens["scopes"]
    assert "READ_GRAPH" in tokens["scopes"]

    # Verify access token exists in database (hashed)
    access_token_hash = hashlib.sha256(tokens["access_token"].encode()).hexdigest()
    db_access_token = await PrismaOAuthAccessToken.prisma().find_unique(
        where={"token": access_token_hash}
    )

    assert db_access_token is not None
    assert db_access_token.userId == test_user
    assert db_access_token.applicationId == test_oauth_app["id"]
    assert db_access_token.revokedAt is None

    # Verify refresh token exists in database (hashed)
    refresh_token_hash = hashlib.sha256(tokens["refresh_token"].encode()).hexdigest()
    db_refresh_token = await PrismaOAuthRefreshToken.prisma().find_unique(
        where={"token": refresh_token_hash}
    )

    assert db_refresh_token is not None
    assert db_refresh_token.userId == test_user
    assert db_refresh_token.applicationId == test_oauth_app["id"]
    assert db_refresh_token.revokedAt is None

    # Verify authorization code is marked as used
    db_code = await PrismaOAuthAuthorizationCode.prisma().find_unique(
        where={"code": auth_code}
    )
    assert db_code is not None
    assert db_code.usedAt is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_authorization_code_cannot_be_reused(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
):
    """Test that authorization code can only be used once."""
    from urllib.parse import parse_qs, urlparse

    verifier, challenge = generate_pkce()

    # Get authorization code
    auth_response = await client.post(
        "/api/oauth/authorize",
        json={
            "client_id": test_oauth_app["client_id"],
            "redirect_uri": test_oauth_app["redirect_uri"],
            "scopes": ["EXECUTE_GRAPH"],
            "state": "reuse_test",
            "response_type": "code",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        follow_redirects=False,
    )

    auth_code = parse_qs(urlparse(auth_response.json()["redirect_url"]).query)["code"][
        0
    ]

    # First exchange - should succeed
    first_response = await client.post(
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
    assert first_response.status_code == 200

    # Second exchange - should fail
    second_response = await client.post(
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
    assert second_response.status_code == 400
    assert "already used" in second_response.json()["detail"]


@pytest.mark.asyncio(loop_scope="session")
async def test_token_exchange_with_invalid_client_secret(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
):
    """Test that token exchange fails with invalid client secret."""
    from urllib.parse import parse_qs, urlparse

    verifier, challenge = generate_pkce()

    # Get authorization code
    auth_response = await client.post(
        "/api/oauth/authorize",
        json={
            "client_id": test_oauth_app["client_id"],
            "redirect_uri": test_oauth_app["redirect_uri"],
            "scopes": ["EXECUTE_GRAPH"],
            "state": "bad_secret_test",
            "response_type": "code",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        follow_redirects=False,
    )

    auth_code = parse_qs(urlparse(auth_response.json()["redirect_url"]).query)["code"][
        0
    ]

    # Try to exchange with wrong secret
    response = await client.post(
        "/api/oauth/token",
        json={
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": test_oauth_app["redirect_uri"],
            "client_id": test_oauth_app["client_id"],
            "client_secret": "wrong_secret",
            "code_verifier": verifier,
        },
    )

    assert response.status_code == 401


@pytest.mark.asyncio(loop_scope="session")
async def test_token_authorization_code_invalid_code(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
):
    """Test token exchange with invalid/nonexistent authorization code."""
    response = await client.post(
        "/api/oauth/token",
        json={
            "grant_type": "authorization_code",
            "code": "nonexistent_invalid_code_xyz",
            "redirect_uri": test_oauth_app["redirect_uri"],
            "client_id": test_oauth_app["client_id"],
            "client_secret": test_oauth_app["client_secret"],
            "code_verifier": "",
        },
    )

    assert response.status_code == 400
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio(loop_scope="session")
async def test_token_authorization_code_expired(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
):
    """Test token exchange with expired authorization code."""
    from datetime import datetime, timedelta, timezone

    # Create an expired authorization code directly in the database
    expired_code = f"expired_code_{secrets.token_urlsafe(16)}"
    now = datetime.now(timezone.utc)

    await PrismaOAuthAuthorizationCode.prisma().create(
        data={
            "code": expired_code,
            "applicationId": test_oauth_app["id"],
            "userId": test_user,
            "scopes": [APIKeyPermission.EXECUTE_GRAPH],
            "redirectUri": test_oauth_app["redirect_uri"],
            "expiresAt": now - timedelta(hours=1),  # Already expired
        }
    )

    response = await client.post(
        "/api/oauth/token",
        json={
            "grant_type": "authorization_code",
            "code": expired_code,
            "redirect_uri": test_oauth_app["redirect_uri"],
            "client_id": test_oauth_app["client_id"],
            "client_secret": test_oauth_app["client_secret"],
            "code_verifier": "",
        },
    )

    assert response.status_code == 400
    assert "expired" in response.json()["detail"].lower()


@pytest.mark.asyncio(loop_scope="session")
async def test_token_authorization_code_redirect_uri_mismatch(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
):
    """Test token exchange with mismatched redirect_uri."""
    from urllib.parse import parse_qs, urlparse

    verifier, challenge = generate_pkce()

    # Get authorization code with one redirect_uri
    auth_response = await client.post(
        "/api/oauth/authorize",
        json={
            "client_id": test_oauth_app["client_id"],
            "redirect_uri": test_oauth_app["redirect_uri"],
            "scopes": ["EXECUTE_GRAPH"],
            "state": "redirect_mismatch_test",
            "response_type": "code",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        follow_redirects=False,
    )

    auth_code = parse_qs(urlparse(auth_response.json()["redirect_url"]).query)["code"][
        0
    ]

    # Try to exchange with different redirect_uri
    # Note: localhost:3000 is in the app's registered redirect_uris
    response = await client.post(
        "/api/oauth/token",
        json={
            "grant_type": "authorization_code",
            "code": auth_code,
            # Different redirect_uri from authorization request
            "redirect_uri": "http://localhost:3000/callback",
            "client_id": test_oauth_app["client_id"],
            "client_secret": test_oauth_app["client_secret"],
            "code_verifier": verifier,
        },
    )

    assert response.status_code == 400
    assert "redirect_uri" in response.json()["detail"].lower()


@pytest.mark.asyncio(loop_scope="session")
async def test_token_authorization_code_pkce_failure(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
    pkce_credentials: tuple[str, str],
):
    """Test token exchange with PKCE verification failure (wrong verifier)."""
    from urllib.parse import parse_qs, urlparse

    verifier, challenge = pkce_credentials

    # Get authorization code with PKCE challenge
    auth_response = await client.post(
        "/api/oauth/authorize",
        json={
            "client_id": test_oauth_app["client_id"],
            "redirect_uri": test_oauth_app["redirect_uri"],
            "scopes": ["EXECUTE_GRAPH"],
            "state": "pkce_failure_test",
            "response_type": "code",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        follow_redirects=False,
    )

    auth_code = parse_qs(urlparse(auth_response.json()["redirect_url"]).query)["code"][
        0
    ]

    # Try to exchange with wrong verifier
    response = await client.post(
        "/api/oauth/token",
        json={
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": test_oauth_app["redirect_uri"],
            "client_id": test_oauth_app["client_id"],
            "client_secret": test_oauth_app["client_secret"],
            "code_verifier": "wrong_verifier_that_does_not_match",
        },
    )

    assert response.status_code == 400
    assert "pkce" in response.json()["detail"].lower()


# ============================================================================
# Token Endpoint Integration Tests - Refresh Token Grant
# ============================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_refresh_token_creates_new_tokens(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
):
    """Test that refresh token grant creates new access and refresh tokens."""
    from urllib.parse import parse_qs, urlparse

    verifier, challenge = generate_pkce()

    # Get initial tokens
    auth_response = await client.post(
        "/api/oauth/authorize",
        json={
            "client_id": test_oauth_app["client_id"],
            "redirect_uri": test_oauth_app["redirect_uri"],
            "scopes": ["EXECUTE_GRAPH"],
            "state": "refresh_test",
            "response_type": "code",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        follow_redirects=False,
    )

    auth_code = parse_qs(urlparse(auth_response.json()["redirect_url"]).query)["code"][
        0
    ]

    initial_response = await client.post(
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
    initial_tokens = initial_response.json()

    # Use refresh token to get new tokens
    refresh_response = await client.post(
        "/api/oauth/token",
        json={
            "grant_type": "refresh_token",
            "refresh_token": initial_tokens["refresh_token"],
            "client_id": test_oauth_app["client_id"],
            "client_secret": test_oauth_app["client_secret"],
        },
    )

    assert refresh_response.status_code == 200
    new_tokens = refresh_response.json()

    # Tokens should be different
    assert new_tokens["access_token"] != initial_tokens["access_token"]
    assert new_tokens["refresh_token"] != initial_tokens["refresh_token"]

    # Old refresh token should be revoked in database
    old_refresh_hash = hashlib.sha256(
        initial_tokens["refresh_token"].encode()
    ).hexdigest()
    old_db_token = await PrismaOAuthRefreshToken.prisma().find_unique(
        where={"token": old_refresh_hash}
    )
    assert old_db_token is not None
    assert old_db_token.revokedAt is not None

    # New tokens should exist and be valid
    new_access_hash = hashlib.sha256(new_tokens["access_token"].encode()).hexdigest()
    new_db_access = await PrismaOAuthAccessToken.prisma().find_unique(
        where={"token": new_access_hash}
    )
    assert new_db_access is not None
    assert new_db_access.revokedAt is None


@pytest.mark.asyncio(loop_scope="session")
async def test_token_refresh_invalid_token(
    client: httpx.AsyncClient,
    test_oauth_app: dict,
):
    """Test token refresh with invalid/nonexistent refresh token."""
    response = await client.post(
        "/api/oauth/token",
        json={
            "grant_type": "refresh_token",
            "refresh_token": "completely_invalid_refresh_token_xyz",
            "client_id": test_oauth_app["client_id"],
            "client_secret": test_oauth_app["client_secret"],
        },
    )

    assert response.status_code == 400
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio(loop_scope="session")
async def test_token_refresh_expired(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
):
    """Test token refresh with expired refresh token."""
    from datetime import datetime, timedelta, timezone

    # Create an expired refresh token directly in the database
    expired_token_value = f"expired_refresh_{secrets.token_urlsafe(16)}"
    expired_token_hash = hashlib.sha256(expired_token_value.encode()).hexdigest()
    now = datetime.now(timezone.utc)

    await PrismaOAuthRefreshToken.prisma().create(
        data={
            "token": expired_token_hash,
            "applicationId": test_oauth_app["id"],
            "userId": test_user,
            "scopes": [APIKeyPermission.EXECUTE_GRAPH],
            "expiresAt": now - timedelta(days=1),  # Already expired
        }
    )

    response = await client.post(
        "/api/oauth/token",
        json={
            "grant_type": "refresh_token",
            "refresh_token": expired_token_value,
            "client_id": test_oauth_app["client_id"],
            "client_secret": test_oauth_app["client_secret"],
        },
    )

    assert response.status_code == 400
    assert "expired" in response.json()["detail"].lower()


@pytest.mark.asyncio(loop_scope="session")
async def test_token_refresh_revoked(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
):
    """Test token refresh with revoked refresh token."""
    from datetime import datetime, timedelta, timezone

    # Create a revoked refresh token directly in the database
    revoked_token_value = f"revoked_refresh_{secrets.token_urlsafe(16)}"
    revoked_token_hash = hashlib.sha256(revoked_token_value.encode()).hexdigest()
    now = datetime.now(timezone.utc)

    await PrismaOAuthRefreshToken.prisma().create(
        data={
            "token": revoked_token_hash,
            "applicationId": test_oauth_app["id"],
            "userId": test_user,
            "scopes": [APIKeyPermission.EXECUTE_GRAPH],
            "expiresAt": now + timedelta(days=30),  # Not expired
            "revokedAt": now - timedelta(hours=1),  # But revoked
        }
    )

    response = await client.post(
        "/api/oauth/token",
        json={
            "grant_type": "refresh_token",
            "refresh_token": revoked_token_value,
            "client_id": test_oauth_app["client_id"],
            "client_secret": test_oauth_app["client_secret"],
        },
    )

    assert response.status_code == 400
    assert "revoked" in response.json()["detail"].lower()


@pytest_asyncio.fixture
async def other_oauth_app(test_user: str):
    """Create a second OAuth application for cross-app tests."""
    app_id = str(uuid.uuid4())
    client_id = f"other_client_{secrets.token_urlsafe(8)}"
    client_secret_plaintext = f"agpt_other_{secrets.token_urlsafe(16)}"
    client_secret_hash, client_secret_salt = keysmith.hash_key(client_secret_plaintext)

    await PrismaOAuthApplication.prisma().create(
        data={
            "id": app_id,
            "name": "Other OAuth App",
            "description": "Second test application",
            "clientId": client_id,
            "clientSecret": client_secret_hash,
            "clientSecretSalt": client_secret_salt,
            "redirectUris": ["https://other.example.com/callback"],
            "grantTypes": ["authorization_code", "refresh_token"],
            "scopes": [APIKeyPermission.EXECUTE_GRAPH],
            "ownerId": test_user,
            "isActive": True,
        }
    )

    yield {
        "id": app_id,
        "client_id": client_id,
        "client_secret": client_secret_plaintext,
        "redirect_uri": "https://other.example.com/callback",
    }


@pytest.mark.asyncio(loop_scope="session")
async def test_token_refresh_wrong_application(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
    other_oauth_app: dict,
):
    """Test token refresh with token from different application."""
    from datetime import datetime, timedelta, timezone

    # Create a refresh token for `test_oauth_app`
    token_value = f"app1_refresh_{secrets.token_urlsafe(16)}"
    token_hash = hashlib.sha256(token_value.encode()).hexdigest()
    now = datetime.now(timezone.utc)

    await PrismaOAuthRefreshToken.prisma().create(
        data={
            "token": token_hash,
            "applicationId": test_oauth_app["id"],  # Belongs to test_oauth_app
            "userId": test_user,
            "scopes": [APIKeyPermission.EXECUTE_GRAPH],
            "expiresAt": now + timedelta(days=30),
        }
    )

    # Try to use it with `other_oauth_app`
    response = await client.post(
        "/api/oauth/token",
        json={
            "grant_type": "refresh_token",
            "refresh_token": token_value,
            "client_id": other_oauth_app["client_id"],
            "client_secret": other_oauth_app["client_secret"],
        },
    )

    assert response.status_code == 400
    assert "does not belong" in response.json()["detail"].lower()


# ============================================================================
# Token Introspection Integration Tests
# ============================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_introspect_valid_access_token(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
):
    """Test introspection returns correct info for valid access token."""
    from urllib.parse import parse_qs, urlparse

    verifier, challenge = generate_pkce()

    # Get tokens
    auth_response = await client.post(
        "/api/oauth/authorize",
        json={
            "client_id": test_oauth_app["client_id"],
            "redirect_uri": test_oauth_app["redirect_uri"],
            "scopes": ["EXECUTE_GRAPH", "READ_GRAPH"],
            "state": "introspect_test",
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
    tokens = token_response.json()

    # Introspect the access token
    introspect_response = await client.post(
        "/api/oauth/introspect",
        json={
            "token": tokens["access_token"],
            "token_type_hint": "access_token",
            "client_id": test_oauth_app["client_id"],
            "client_secret": test_oauth_app["client_secret"],
        },
    )

    assert introspect_response.status_code == 200
    data = introspect_response.json()

    assert data["active"] is True
    assert data["token_type"] == "access_token"
    assert data["user_id"] == test_user
    assert data["client_id"] == test_oauth_app["client_id"]
    assert "EXECUTE_GRAPH" in data["scopes"]
    assert "READ_GRAPH" in data["scopes"]


@pytest.mark.asyncio(loop_scope="session")
async def test_introspect_invalid_token_returns_inactive(
    client: httpx.AsyncClient,
    test_oauth_app: dict,
):
    """Test introspection returns inactive for non-existent token."""
    introspect_response = await client.post(
        "/api/oauth/introspect",
        json={
            "token": "completely_invalid_token_that_does_not_exist",
            "client_id": test_oauth_app["client_id"],
            "client_secret": test_oauth_app["client_secret"],
        },
    )

    assert introspect_response.status_code == 200
    assert introspect_response.json()["active"] is False


@pytest.mark.asyncio(loop_scope="session")
async def test_introspect_active_refresh_token(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
):
    """Test introspection returns correct info for valid refresh token."""
    from urllib.parse import parse_qs, urlparse

    verifier, challenge = generate_pkce()

    # Get tokens via the full flow
    auth_response = await client.post(
        "/api/oauth/authorize",
        json={
            "client_id": test_oauth_app["client_id"],
            "redirect_uri": test_oauth_app["redirect_uri"],
            "scopes": ["EXECUTE_GRAPH", "READ_GRAPH"],
            "state": "introspect_refresh_test",
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
    tokens = token_response.json()

    # Introspect the refresh token
    introspect_response = await client.post(
        "/api/oauth/introspect",
        json={
            "token": tokens["refresh_token"],
            "token_type_hint": "refresh_token",
            "client_id": test_oauth_app["client_id"],
            "client_secret": test_oauth_app["client_secret"],
        },
    )

    assert introspect_response.status_code == 200
    data = introspect_response.json()

    assert data["active"] is True
    assert data["token_type"] == "refresh_token"
    assert data["user_id"] == test_user
    assert data["client_id"] == test_oauth_app["client_id"]


@pytest.mark.asyncio(loop_scope="session")
async def test_introspect_invalid_client(
    client: httpx.AsyncClient,
    test_oauth_app: dict,
):
    """Test introspection with invalid client credentials."""
    introspect_response = await client.post(
        "/api/oauth/introspect",
        json={
            "token": "some_token",
            "client_id": test_oauth_app["client_id"],
            "client_secret": "wrong_secret_value",
        },
    )

    assert introspect_response.status_code == 401


@pytest.mark.asyncio(loop_scope="session")
async def test_validate_access_token_fails_when_app_disabled(
    test_user: str,
):
    """
    Test that validate_access_token raises InvalidClientError when the app is disabled.

    This tests the security feature where disabling an OAuth application
    immediately invalidates all its access tokens.
    """
    from datetime import datetime, timedelta, timezone

    from backend.data.auth.oauth import InvalidClientError, validate_access_token

    # Create an OAuth app
    app_id = str(uuid.uuid4())
    client_id = f"disable_test_{secrets.token_urlsafe(8)}"
    client_secret_plaintext = f"agpt_disable_{secrets.token_urlsafe(16)}"
    client_secret_hash, client_secret_salt = keysmith.hash_key(client_secret_plaintext)

    await PrismaOAuthApplication.prisma().create(
        data={
            "id": app_id,
            "name": "App To Be Disabled",
            "description": "Test app for disabled validation",
            "clientId": client_id,
            "clientSecret": client_secret_hash,
            "clientSecretSalt": client_secret_salt,
            "redirectUris": ["https://example.com/callback"],
            "grantTypes": ["authorization_code"],
            "scopes": [APIKeyPermission.EXECUTE_GRAPH],
            "ownerId": test_user,
            "isActive": True,
        }
    )

    # Create an access token directly in the database
    token_plaintext = f"test_token_{secrets.token_urlsafe(32)}"
    token_hash = hashlib.sha256(token_plaintext.encode()).hexdigest()
    now = datetime.now(timezone.utc)

    await PrismaOAuthAccessToken.prisma().create(
        data={
            "token": token_hash,
            "applicationId": app_id,
            "userId": test_user,
            "scopes": [APIKeyPermission.EXECUTE_GRAPH],
            "expiresAt": now + timedelta(hours=1),
        }
    )

    # Token should be valid while app is active
    token_info, _ = await validate_access_token(token_plaintext)
    assert token_info.user_id == test_user

    # Disable the app
    await PrismaOAuthApplication.prisma().update(
        where={"id": app_id},
        data={"isActive": False},
    )

    # Token should now fail validation with InvalidClientError
    with pytest.raises(InvalidClientError, match="disabled"):
        await validate_access_token(token_plaintext)

    # Cleanup
    await PrismaOAuthApplication.prisma().delete(where={"id": app_id})


# ============================================================================
# Token Revocation Integration Tests
# ============================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_revoke_access_token_updates_database(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
):
    """Test that revoking access token updates database."""
    from urllib.parse import parse_qs, urlparse

    verifier, challenge = generate_pkce()

    # Get tokens
    auth_response = await client.post(
        "/api/oauth/authorize",
        json={
            "client_id": test_oauth_app["client_id"],
            "redirect_uri": test_oauth_app["redirect_uri"],
            "scopes": ["EXECUTE_GRAPH"],
            "state": "revoke_access_test",
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
    tokens = token_response.json()

    # Verify token is not revoked in database
    access_hash = hashlib.sha256(tokens["access_token"].encode()).hexdigest()
    db_token_before = await PrismaOAuthAccessToken.prisma().find_unique(
        where={"token": access_hash}
    )
    assert db_token_before is not None
    assert db_token_before.revokedAt is None

    # Revoke the token
    revoke_response = await client.post(
        "/api/oauth/revoke",
        json={
            "token": tokens["access_token"],
            "token_type_hint": "access_token",
            "client_id": test_oauth_app["client_id"],
            "client_secret": test_oauth_app["client_secret"],
        },
    )

    assert revoke_response.status_code == 200
    assert revoke_response.json()["status"] == "ok"

    # Verify token is now revoked in database
    db_token_after = await PrismaOAuthAccessToken.prisma().find_unique(
        where={"token": access_hash}
    )
    assert db_token_after is not None
    assert db_token_after.revokedAt is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_revoke_unknown_token_returns_ok(
    client: httpx.AsyncClient,
    test_oauth_app: dict,
):
    """Test that revoking unknown token returns 200 (per RFC 7009)."""
    revoke_response = await client.post(
        "/api/oauth/revoke",
        json={
            "token": "unknown_token_that_does_not_exist_anywhere",
            "client_id": test_oauth_app["client_id"],
            "client_secret": test_oauth_app["client_secret"],
        },
    )

    # Per RFC 7009, should return 200 even for unknown tokens
    assert revoke_response.status_code == 200
    assert revoke_response.json()["status"] == "ok"


@pytest.mark.asyncio(loop_scope="session")
async def test_revoke_refresh_token_updates_database(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
):
    """Test that revoking refresh token updates database."""
    from urllib.parse import parse_qs, urlparse

    verifier, challenge = generate_pkce()

    # Get tokens
    auth_response = await client.post(
        "/api/oauth/authorize",
        json={
            "client_id": test_oauth_app["client_id"],
            "redirect_uri": test_oauth_app["redirect_uri"],
            "scopes": ["EXECUTE_GRAPH"],
            "state": "revoke_refresh_test",
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
    tokens = token_response.json()

    # Verify refresh token is not revoked in database
    refresh_hash = hashlib.sha256(tokens["refresh_token"].encode()).hexdigest()
    db_token_before = await PrismaOAuthRefreshToken.prisma().find_unique(
        where={"token": refresh_hash}
    )
    assert db_token_before is not None
    assert db_token_before.revokedAt is None

    # Revoke the refresh token
    revoke_response = await client.post(
        "/api/oauth/revoke",
        json={
            "token": tokens["refresh_token"],
            "token_type_hint": "refresh_token",
            "client_id": test_oauth_app["client_id"],
            "client_secret": test_oauth_app["client_secret"],
        },
    )

    assert revoke_response.status_code == 200
    assert revoke_response.json()["status"] == "ok"

    # Verify refresh token is now revoked in database
    db_token_after = await PrismaOAuthRefreshToken.prisma().find_unique(
        where={"token": refresh_hash}
    )
    assert db_token_after is not None
    assert db_token_after.revokedAt is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_revoke_invalid_client(
    client: httpx.AsyncClient,
    test_oauth_app: dict,
):
    """Test revocation with invalid client credentials."""
    revoke_response = await client.post(
        "/api/oauth/revoke",
        json={
            "token": "some_token",
            "client_id": test_oauth_app["client_id"],
            "client_secret": "wrong_secret_value",
        },
    )

    assert revoke_response.status_code == 401


@pytest.mark.asyncio(loop_scope="session")
async def test_revoke_token_from_different_app_fails_silently(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
):
    """
    Test that an app cannot revoke tokens belonging to a different app.

    Per RFC 7009, the endpoint still returns 200 OK (to prevent token scanning),
    but the token should remain valid in the database.
    """
    from urllib.parse import parse_qs, urlparse

    verifier, challenge = generate_pkce()

    # Get tokens for app 1
    auth_response = await client.post(
        "/api/oauth/authorize",
        json={
            "client_id": test_oauth_app["client_id"],
            "redirect_uri": test_oauth_app["redirect_uri"],
            "scopes": ["EXECUTE_GRAPH"],
            "state": "cross_app_revoke_test",
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
    tokens = token_response.json()

    # Create a second OAuth app
    app2_id = str(uuid.uuid4())
    app2_client_id = f"test_client_app2_{secrets.token_urlsafe(8)}"
    app2_client_secret_plaintext = f"agpt_secret_app2_{secrets.token_urlsafe(16)}"
    app2_client_secret_hash, app2_client_secret_salt = keysmith.hash_key(
        app2_client_secret_plaintext
    )

    await PrismaOAuthApplication.prisma().create(
        data={
            "id": app2_id,
            "name": "Second Test OAuth App",
            "description": "Second test application for cross-app revocation test",
            "clientId": app2_client_id,
            "clientSecret": app2_client_secret_hash,
            "clientSecretSalt": app2_client_secret_salt,
            "redirectUris": ["https://other-app.com/callback"],
            "grantTypes": ["authorization_code", "refresh_token"],
            "scopes": [APIKeyPermission.EXECUTE_GRAPH, APIKeyPermission.READ_GRAPH],
            "ownerId": test_user,
            "isActive": True,
        }
    )

    # App 2 tries to revoke App 1's access token
    revoke_response = await client.post(
        "/api/oauth/revoke",
        json={
            "token": tokens["access_token"],
            "token_type_hint": "access_token",
            "client_id": app2_client_id,
            "client_secret": app2_client_secret_plaintext,
        },
    )

    # Per RFC 7009, returns 200 OK even if token not found/not owned
    assert revoke_response.status_code == 200
    assert revoke_response.json()["status"] == "ok"

    # But the token should NOT be revoked in the database
    access_hash = hashlib.sha256(tokens["access_token"].encode()).hexdigest()
    db_token = await PrismaOAuthAccessToken.prisma().find_unique(
        where={"token": access_hash}
    )
    assert db_token is not None
    assert db_token.revokedAt is None, "Token should NOT be revoked by different app"

    # Now app 1 revokes its own token - should work
    revoke_response2 = await client.post(
        "/api/oauth/revoke",
        json={
            "token": tokens["access_token"],
            "token_type_hint": "access_token",
            "client_id": test_oauth_app["client_id"],
            "client_secret": test_oauth_app["client_secret"],
        },
    )

    assert revoke_response2.status_code == 200

    # Token should now be revoked
    db_token_after = await PrismaOAuthAccessToken.prisma().find_unique(
        where={"token": access_hash}
    )
    assert db_token_after is not None
    assert db_token_after.revokedAt is not None, "Token should be revoked by own app"

    # Cleanup second app
    await PrismaOAuthApplication.prisma().delete(where={"id": app2_id})


# ============================================================================
# Complete End-to-End OAuth Flow Test
# ============================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_complete_oauth_flow_end_to_end(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
    pkce_credentials: tuple[str, str],
):
    """
    Test the complete OAuth 2.0 flow from authorization to token refresh.

    This is a comprehensive integration test that verifies the entire
    OAuth flow works correctly with real API calls and database operations.
    """
    from urllib.parse import parse_qs, urlparse

    verifier, challenge = pkce_credentials

    # Step 1: Authorization request with PKCE
    auth_response = await client.post(
        "/api/oauth/authorize",
        json={
            "client_id": test_oauth_app["client_id"],
            "redirect_uri": test_oauth_app["redirect_uri"],
            "scopes": ["EXECUTE_GRAPH", "READ_GRAPH"],
            "state": "e2e_test_state",
            "response_type": "code",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        follow_redirects=False,
    )

    assert auth_response.status_code == 200

    redirect_url = auth_response.json()["redirect_url"]
    query = parse_qs(urlparse(redirect_url).query)

    assert query["state"][0] == "e2e_test_state"
    auth_code = query["code"][0]

    # Verify authorization code in database
    db_code = await PrismaOAuthAuthorizationCode.prisma().find_unique(
        where={"code": auth_code}
    )
    assert db_code is not None
    assert db_code.codeChallenge == challenge

    # Step 2: Exchange code for tokens
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

    assert token_response.status_code == 200
    tokens = token_response.json()
    assert "access_token" in tokens
    assert "refresh_token" in tokens

    # Verify code is marked as used
    db_code_used = await PrismaOAuthAuthorizationCode.prisma().find_unique_or_raise(
        where={"code": auth_code}
    )
    assert db_code_used.usedAt is not None

    # Step 3: Introspect access token
    introspect_response = await client.post(
        "/api/oauth/introspect",
        json={
            "token": tokens["access_token"],
            "client_id": test_oauth_app["client_id"],
            "client_secret": test_oauth_app["client_secret"],
        },
    )

    assert introspect_response.status_code == 200
    introspect_data = introspect_response.json()
    assert introspect_data["active"] is True
    assert introspect_data["user_id"] == test_user

    # Step 4: Refresh tokens
    refresh_response = await client.post(
        "/api/oauth/token",
        json={
            "grant_type": "refresh_token",
            "refresh_token": tokens["refresh_token"],
            "client_id": test_oauth_app["client_id"],
            "client_secret": test_oauth_app["client_secret"],
        },
    )

    assert refresh_response.status_code == 200
    new_tokens = refresh_response.json()
    assert new_tokens["access_token"] != tokens["access_token"]
    assert new_tokens["refresh_token"] != tokens["refresh_token"]

    # Verify old refresh token is revoked
    old_refresh_hash = hashlib.sha256(tokens["refresh_token"].encode()).hexdigest()
    old_db_refresh = await PrismaOAuthRefreshToken.prisma().find_unique_or_raise(
        where={"token": old_refresh_hash}
    )
    assert old_db_refresh.revokedAt is not None

    # Step 5: Verify new access token works
    new_introspect = await client.post(
        "/api/oauth/introspect",
        json={
            "token": new_tokens["access_token"],
            "client_id": test_oauth_app["client_id"],
            "client_secret": test_oauth_app["client_secret"],
        },
    )

    assert new_introspect.status_code == 200
    assert new_introspect.json()["active"] is True

    # Step 6: Revoke new access token
    revoke_response = await client.post(
        "/api/oauth/revoke",
        json={
            "token": new_tokens["access_token"],
            "token_type_hint": "access_token",
            "client_id": test_oauth_app["client_id"],
            "client_secret": test_oauth_app["client_secret"],
        },
    )

    assert revoke_response.status_code == 200

    # Step 7: Verify revoked token is inactive
    final_introspect = await client.post(
        "/api/oauth/introspect",
        json={
            "token": new_tokens["access_token"],
            "client_id": test_oauth_app["client_id"],
            "client_secret": test_oauth_app["client_secret"],
        },
    )

    assert final_introspect.status_code == 200
    assert final_introspect.json()["active"] is False

    # Verify in database
    new_access_hash = hashlib.sha256(new_tokens["access_token"].encode()).hexdigest()
    db_revoked = await PrismaOAuthAccessToken.prisma().find_unique_or_raise(
        where={"token": new_access_hash}
    )
    assert db_revoked.revokedAt is not None
