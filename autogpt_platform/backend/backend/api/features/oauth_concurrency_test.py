"""
Concurrency tests for the OAuth 2.0 token refresh flow (#15288).

Two concurrent refreshes of one refresh token must not both succeed. These
tests run against the real database and token endpoint. Shared OAuth fixtures
(test_user, test_oauth_app, client) live in the colocated conftest.py.
"""

import asyncio

import httpx
import pytest
from prisma.enums import APIKeyPermission
from prisma.models import OAuthRefreshToken as PrismaOAuthRefreshToken

from backend.data.auth.oauth import (
    InvalidGrantError,
    create_refresh_token,
    refresh_tokens,
)

from .conftest import authorize_and_exchange_code


@pytest.mark.asyncio(loop_scope="session")
async def test_refresh_token_concurrent_refresh_succeeds_once(
    test_user: str,
    test_oauth_app: dict,
):
    """Two concurrent refreshes of one refresh token must not both succeed (#15288).

    The check-then-revoke in refresh_tokens let both callers read revokedAt=NULL
    and both mint token pairs. Runs against the real DB.
    """
    for _ in range(20):
        rt = await create_refresh_token(
            application_id=test_oauth_app["id"],
            user_id=test_user,
            scopes=[APIKeyPermission.EXECUTE_GRAPH],
        )
        results = await asyncio.gather(
            *(
                refresh_tokens(rt.token.get_secret_value(), test_oauth_app["id"])
                for _ in range(2)
            ),
            return_exceptions=True,
        )
        successes = [r for r in results if isinstance(r, tuple)]
        failures = [r for r in results if isinstance(r, InvalidGrantError)]
        assert len(successes) == 1, results
        assert len(failures) == 1, results


@pytest.mark.asyncio(loop_scope="session")
async def test_token_endpoint_concurrent_refresh_issues_one_token_pair(
    client: httpx.AsyncClient,
    test_user: str,
    test_oauth_app: dict,
):
    """Concurrent POST /api/oauth/token with one refresh token mints one pair.

    Drives the real token endpoint (client auth, DB) rather than the data
    layer, and checks the database: a lost race must not leave an extra
    active refresh token.
    """
    initial_tokens = await authorize_and_exchange_code(
        client, test_oauth_app, "concurrent_refresh_test"
    )

    refresh = {
        "grant_type": "refresh_token",
        "refresh_token": initial_tokens["refresh_token"],
        "client_id": test_oauth_app["client_id"],
        "client_secret": test_oauth_app["client_secret"],
    }
    responses = await asyncio.gather(
        *(client.post("/api/oauth/token", json=refresh) for _ in range(3))
    )

    statuses = sorted(r.status_code for r in responses)
    assert statuses == [200, 400, 400], [r.text for r in responses]
    active_refresh_tokens = await PrismaOAuthRefreshToken.prisma().count(
        where={
            "applicationId": test_oauth_app["id"],
            "userId": test_user,
            "revokedAt": None,
        }
    )
    assert active_refresh_tokens == 1
